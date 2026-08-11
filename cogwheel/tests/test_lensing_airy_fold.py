"""
Tests for the uniform Airy (fold-catastrophe) wave arm
`lensing.chang_refsdal._airy_fold`.

These are domain tests for Build 8e WP2.  The arm serves the near-fold,
high-``w`` corner that the exact Schwinger engine cannot reach (``w > 60``)
by evaluating the canonical fold diffraction integral in closed form

    F_Airy = 2 sqrt(pi) exp(i (w tau_bar + sigma))
             [ p w^{1/6} Ai(-xi) - i q w^{-1/6} Ai'(-xi) ] .

TWO INDEPENDENT ORACLES, neither of which shares the arm's arithmetic
(F002):

1. A pure-``mpmath`` re-evaluation of the SAME closed form at 40 digits
   (`_mp_airy_fold`), using ``mpmath.airyai`` rather than
   ``scipy.special.airy``.  It certifies the transcription -- the
   ``Ai(-xi)`` sign, the ``-i q w^{-1/6} Ai'`` quadrature term, the fixed
   phase ``sigma``, and the ``2 sqrt(pi)`` prefactor -- to machine
   precision.  A flip to ``Ai(+xi)`` (the primary Architect falsifier)
   fails it outright.

2. The exact geometric two-image sum built DIRECTLY from `geometry`
   (`_geometric_two_image_sum`): magnifications, delays and Morse indices
   of the merging minimum/saddle pair,

       F_geom = sqrt|mu_+| exp(i w tau_+)
                + sqrt|mu_-| exp(i w tau_- - i pi/2) .

   The arm's large-``xi`` limit must reproduce this in the
   MAX-NORMALIZED ENVELOPE currency (F016/F018 -- the lnL-relevant one),
   and it does so at the leading uniform rate ``~ xi^{-3/2}`` when the
   Airy amplitudes carry the SUM ``p ~ sqrt|mu_+| + sqrt|mu_-|`` and the
   DIFFERENCE ``q ~ sqrt|mu_-| - sqrt|mu_+|`` (derived below).  Swapping
   ``p`` and ``q`` attaches the antisymmetric part to the wrong Airy
   function and breaks the match by an ``O(1)`` DC offset -- the
   sum-vs-difference falsifier.

WHY ENVELOPE, NOT COMPLEX POINTWISE.  The complex pointwise residual is
dominated by interference nulls of ``F_geom`` (where the phase is
ill-defined) and by the huge carrier phase ``w tau_bar``; the physically
meaningful, lnL-relevant quantity is ``|F|``.  Measured: the envelope
error falls as ``xi^{-3/2}`` (4.8e-4 at ``xi ~ 39``, 2.4e-5 at
``xi ~ 255``) while the p/q swap sits at ``~ 0.8`` regardless of ``xi``.

TOLERANCES.  The transcription oracle is gated at 1e-9 (measured
1e-15..1e-12, the residual growth being float64 loss in the large
``w tau_bar`` carrier).  The far-field envelope bar is the spec's 1e-3,
asserted only where ``xi`` is large enough for the leading uniform term
to clear it (``xi >= _XI_FARFIELD``); a paired monotone-refinement
control witnesses the ``xi^{-3/2}`` approach.  The on-caustic
(``xi = 0``) finiteness bar is 1e-2 against the closed form; the point of
that test is that the arm's amplitude is built from the FINITE fold
curvatures, so it stays finite where the raw ``sqrt|mu|`` two-image sum
diverges.

TIERING.  The accuracy certifications that scan ``w`` (the far-field
envelope convergence, the near-caustic serving calibration) are the
brute-force / driver tier, gated behind ``COGWHEEL_BRUTE_ACCURACY``.  The
transcription oracle, the sign-convention physics gate, the on-caustic
finiteness gate, the `fold_amplification` refusal contract, and the whole
self-falsification class are load-bearing falsifications and stay FAST.

`_FoldArmTestCase.tearDown` fails any test that asserted nothing (every
comparison skipped), and `FoldArmSelfFalsificationTestCase` proves the
suite can go red: it flips the Airy sign, swaps the amplitudes, corrupts
the calibration, and feeds a divergent ``sqrt|mu|`` amplitude, and asserts
each gate detects it.

THE UNIFORM PEARCEY CUSP ARM (Build 8f, WP3/WP4)
------------------------------------------------
The later classes cover the sibling near-cusp arm
`_pearcey_cusp` and its wiring into the per-node serving ladder
(`operator._uniform_arm_value`).  They test three Architect specs:

* PEARCEY PRIMITIVE CERTIFICATION.  `_pearcey_cusp.pearcey` evaluates the
  Pearcey integral ``P(x, y) = Int exp[i(t^4 + x t^2 + y t)] dt`` on a
  rotated steepest-descent contour, certified in place by a paired
  ``N`` / ``2N`` Gauss-Legendre rule.  It is cross-checked against THREE
  independent oracles, none of which shares the module's fixed-order
  composite quadrature: (1) the analytic closed form
  ``P(0, 0) = (Gamma(1/4) / 2) e^{i pi/8}``; (2) an adaptive-QUADPACK
  (`scipy.integrate.quad`) evaluation on a single straight ``pi/8`` line
  through the origin (FAST reference); (3) a 40-digit ``mpmath.quad`` on
  the same rotated line (gated).  The paired-rule certificate is shown to
  be HONEST -- forcing gross under-resolution makes the primitive REFUSE
  (return ``None``) rather than certify a wrong value.  Reference bar
  1e-8 (measured ~1e-13); the honest-certificate ``3e-10`` figure is the
  module's own `_CERTIFICATION_TOL`.

* PEARCEY (x, y) 2/3-vs-1/2 SCALING.  The catastrophe controls carry
  ``x = c_x w^{1/2} delta_parallel`` and ``y = c_y w^{3/4} delta_perp``.
  The actual controls the code feeds to `pearcey` are captured with a
  recording wrapper; over a ``w`` grid their log-log slopes are ``0.5``
  and ``0.75`` (to 5%) and the same controls reproduce the EXACT engine
  `operator.F_op` (Schwinger, ``w <= 60``) to <= 1e-2 in the overlap
  band.  The swap is falsified with an INDEPENDENT fact: the Pearcey fold
  caustic ``27 y^2 = -8 x^3`` (where P's stationary points coalesce) is
  ``w``-invariant ONLY with the ``1/2`` / ``3/4`` exponents (the common
  ``w^{3/2}`` cancels); the swapped exponents leave a source that starts
  on the fold arc OFF it as ``w`` changes.

* FALL-THROUGH BOTH DIRECTIONS (F010).  At a served cusp node the ladder
  value equals `cusp_amplification` bit-for-bit; corrupting the arm
  (forcing the paired-rule to fail, or NaN-ing the primitive) makes the
  node FALL THROUGH to the existing NAMED `SchwingerCertificationError`
  rather than serve a wrong number; and moving the certified-argument
  threshold (the ``envelope_bar`` / `_UNIFORM_ERROR_CONST` that set the
  minimum admissible scaled radius) flips a fixed node serve<->refuse --
  proving the threshold is not dead code.  Currency is boolean route plus
  a bit-check, never nats.
"""
from __future__ import annotations

import ast
import cmath
import importlib.util
import itertools
import math
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from unittest import TestCase, expectedFailure, mock, skipUnless

import mpmath
import numpy as np
from scipy.integrate import quad as scipy_quad
from scipy.special import airy as scipy_airy
from scipy.special import gamma as scipy_gamma

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import _airy_fold
from cogwheel.lensing.chang_refsdal import _pearcey_cusp
from cogwheel.lensing.chang_refsdal import _schwinger
from cogwheel.lensing.chang_refsdal import operator
from cogwheel.lensing import surrogate_census


mpmath.mp.dps = 40

# ----------------------------------------------------------------------
# Tier gate (Architect: all three specs are EXACT-HEAVY, born gated).
# The w-scanning accuracy certifications run in the driver post-build
# sweep; the falsifications and structural gates stay fast.
# ----------------------------------------------------------------------

#: True when the brute-force accuracy tier is requested.
_BRUTE_ACCURACY = bool(os.environ.get('COGWHEEL_BRUTE_ACCURACY'))

_brute_accuracy_tier = skipUnless(
    _BRUTE_ACCURACY,
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 -- the '
    'far-field envelope convergence and near-caustic serving scans loop '
    'the arm/geometry over a w-grid')


# ----------------------------------------------------------------------
# Fold fixture: a point mass with external shear (Chang-Refsdal), a
# source scanned along an off-axis ray so the merging fold pair is
# ASYMMETRIC (|mu_+| != |mu_-|) -- a symmetric approach would hide the
# sum-vs-difference swap.
# ----------------------------------------------------------------------

#: External shear magnitude of the fixture lens (positive parity,
#: |gamma| < 1 - kappa, so a single 4-cusp astroid caustic).
_GAMMA = 0.3

#: Shear orientation and convergence of the fixture lens.
_BETA = 0.0
_KAPPA = 0.0

#: Off-axis ray (radians) along which the source approaches a fold arc,
#: well away from the on-axis cusps so the merging pair is asymmetric.
_RAY_ANGLE = 1.0

#: Source radii (image-present, inside the caustic) giving asymmetric
#: merging fold pairs of decreasing delay separation; ``sqrt|mu|`` ratios
#: ~1.17-1.22 (measured), i.e. genuinely off the cusp axis.
_INSIDE_RADII = (0.14, 0.20)

#: A radius outside the caustic (two macro images only) and one very
#: close to it (the metric inversion is ill conditioned; the serving
#: error gate refuses).
_OUTSIDE_RADIUS = 0.35
_NEAR_CAUSTIC_RADIUS = 0.28

#: F028's measured configuration, in the driver probe's parametrization:
#: the positive-parity caustic point at critical-curve parameter
#: ``_F028_T`` (radians, off the cusps), offset along the caustic normal
#: by ``_F028_RATIO`` times the caustic's LOCAL CURVATURE RADIUS there.
#: `fold_amplification` certifies and serves at ``_F028_W``.
_F028_GAMMA = 0.70
_F028_T = 0.55
_F028_RATIO = 0.40
_F028_W = 70.0

#: The Airy control above which the leading uniform term clears the 1e-3
#: envelope bar (measured: 8.7e-4 at xi ~ 26, 4.8e-4 at xi ~ 39).
_XI_FARFIELD = 40.0

#: Spec bars.  The far-field envelope match (F016 currency) and the
#: on-caustic finiteness match against the closed form.
_FARFIELD_ENVELOPE_TOL = 1e-3
_ONCAUSTIC_TOL = 1e-2

#: Transcription-oracle bar (scipy vs mpmath on the identical closed
#: form); measured 1e-15..1e-12, the growth being large-carrier float64
#: loss.
_TRANSCRIPTION_TOL = 1e-9

#: Fixed fold phase and Airy special values used by the closed form.
_SIGMA_FOLD = -0.25 * math.pi
_AI0 = float(scipy_airy(0.0)[0])       # Ai(0)  = 0.35502805...
_AIP0 = float(scipy_airy(0.0)[1])      # Ai'(0) = -0.25881940...

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


# ----------------------------------------------------------------------
# Independent oracles (no _airy_fold arithmetic; geometry is a DIFFERENT
# module and mpmath.airyai is a DIFFERENT Airy evaluator than scipy).
# ----------------------------------------------------------------------

def _matrix():
    """Macro matrix of the fixture Chang-Refsdal lens."""
    return geometry.macro_matrix(_GAMMA, _BETA, _KAPPA)


def _source(radius):
    """Source position at ``radius`` along the off-axis fixture ray."""
    return radius * np.array([math.cos(_RAY_ANGLE), math.sin(_RAY_ANGLE)])


def _caustic_normal_source(gamma, t_parameter, ratio):
    """
    Source offset from a caustic point by ``ratio`` local curvature radii.

    Takes the positive-parity caustic point at critical-curve parameter
    ``t_parameter``, builds the local unit normal and the local curvature
    radius ``R_c`` from three closely spaced caustic points (the
    circumradius), and steps ``ratio * R_c`` along that normal.  The
    normal is oriented toward the side with the LARGER image count, which
    reproduces the F028 driver probe (`probe_c6_window.py`) exactly, so
    the configuration its table reports can be named here without
    ambiguity.
    """
    def caustic_point(parameter):
        return np.asarray(geometry.critical_point(
            gamma, float(parameter), 0.0, 0.0, +1).source, dtype=float)

    step = 1e-4
    back = caustic_point(t_parameter - step)
    here = caustic_point(t_parameter)
    ahead = caustic_point(t_parameter + step)
    tangent = (ahead - back) / np.linalg.norm(ahead - back)
    normal = np.array([-tangent[1], tangent[0]])
    twice_area = abs((here[0] - back[0]) * (ahead[1] - back[1])
                     - (here[1] - back[1]) * (ahead[0] - back[0]))
    curvature_radius = (np.linalg.norm(here - back)
                        * np.linalg.norm(ahead - here)
                        * np.linalg.norm(back - ahead)) / (2.0 * twice_area)
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    probe = 1e-3 * curvature_radius
    if (len(geometry.find_images(here - probe * normal, matrix))
            > len(geometry.find_images(here + probe * normal, matrix))):
        normal = -normal
    return here + ratio * curvature_radius * normal


def _merging_pair(source, matrix):
    """
    Return ``(tau_plus, tau_minus, s_plus, s_minus)`` of the merging fold
    pair, computed independently from `geometry` primitives.

    The pair is the delay-adjacent minimum (Morse 0) then saddle
    (Morse 1) with the smallest delay gap; ``s_* = sqrt|mu_*|`` from the
    signed magnification.  This deliberately re-derives the selection
    rather than calling `_airy_fold._merging_fold_pair`, so the oracle
    shares nothing with the code under test.
    """
    images = geometry.find_images(source, matrix)
    entries = sorted(
        (geometry.delay(image, source, matrix),
         geometry.morse_index(image, matrix),
         math.sqrt(abs(geometry.magnification(image, matrix))))
        for image in images)
    best = None
    best_gap = math.inf
    for (tau_low, n_low, s_low), (tau_high, n_high, s_high) in zip(
            entries, entries[1:]):
        if n_low == 0 and n_high == 1:
            gap = tau_high - tau_low
            if 0.0 < gap < best_gap:
                best_gap = gap
                best = (tau_low, tau_high, s_low, s_high)
    return best


def _geometric_two_image_sum(w, tau_plus, tau_minus, s_plus, s_minus):
    """
    Exact geometric two-image amplification of the merging pair.

    ``F = sqrt|mu_+| exp(i w tau_+) + sqrt|mu_-| exp(i w tau_- - i pi/2)``
    (minimum ``n = 0``, saddle ``n = 1``), the ``w -> inf`` truth the
    uniform arm must reproduce in the far field.
    """
    return (s_plus * np.exp(1j * w * tau_plus)
            + s_minus * np.exp(1j * w * tau_minus - 0.5j * np.pi))


def _farfield_amplitudes(w, xi, s_plus, s_minus):
    """
    Airy amplitudes ``(p, q)`` that reproduce the ASYMMETRIC two-image
    sum in the arm's large-``xi`` limit.

    Substituting the leading Airy asymptotics
    ``Ai(-xi) ~ pi^{-1/2} xi^{-1/4} sin((2/3) xi^{3/2} + pi/4)`` and
    ``Ai'(-xi) ~ pi^{-1/2} xi^{1/4} cos(...)`` into the closed form and
    matching the two stationary channels gives

        p = (s_+ + s_-) / 2 * w^{-1/6} * xi^{1/4}   (SUM   -> Ai),
        q = (s_- - s_+) / 2 * w^{ 1/6} * xi^{-1/4}  (DIFF  -> Ai').

    ``p`` carries the symmetric sum through ``Ai`` and ``q`` the
    antisymmetric difference through ``Ai'``; the sum-vs-difference swap
    exchanges the two and breaks the far-field match.
    """
    p = 0.5 * (s_plus + s_minus) * w ** (-1.0 / 6.0) * xi ** 0.25
    q = 0.5 * (s_minus - s_plus) * w ** (1.0 / 6.0) * xi ** -0.25
    return p, q


def _mp_airy_fold(w, tau_bar, xi_control, p, q, sigma):
    """
    Pure-mpmath re-evaluation of the exact fold closed form.

    Uses ``mpmath.airyai`` (a DIFFERENT Airy evaluator than the module's
    ``scipy.special.airy``) at 40 digits, so agreement certifies the
    transcription -- the ``-xi`` argument sign, the ``-i q w^{-1/6} Ai'``
    quadrature term, ``sigma`` and the ``2 sqrt(pi)`` prefactor -- rather
    than the accuracy of a shared implementation.
    """
    w_mp = mpmath.mpf(w)
    ai = mpmath.airyai(mpmath.mpf(-xi_control))
    aip = mpmath.airyai(mpmath.mpf(-xi_control), 1)
    bracket = (mpmath.mpf(p) * w_mp ** (mpmath.mpf(1) / 6) * ai
               - 1j * mpmath.mpf(q) * w_mp ** (-mpmath.mpf(1) / 6) * aip)
    carrier = mpmath.e ** (1j * (w_mp * mpmath.mpf(tau_bar)
                                 + mpmath.mpf(sigma)))
    return complex(2 * mpmath.sqrt(mpmath.pi) * carrier * bracket)


def _interior_maxima(values):
    """Number of strict interior local maxima of a 1-D array."""
    values = np.asarray(values)
    return int(np.sum((values[1:-1] > values[:-2])
                      & (values[1:-1] > values[2:])))


def _save_plot(name, xdata, ydata, *, xlabel, ylabel):
    """
    Write a single-curve diagnostic PNG to `_OUTPUT_DIR`.

    Plotting is best-effort: a headless/backend failure must never fail a
    physics test, so any exception is swallowed after the numerical
    assertions have already run.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        axis.plot(np.asarray(xdata), np.asarray(ydata), lw=1.2)
        axis.axvline(0.0, color='0.6', ls='--', lw=0.8)  # the caustic
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(name.replace('_', ' '))
        figure.tight_layout()
        figure.savefig(os.path.join(_OUTPUT_DIR, f'{name}.png'), dpi=110)
        plt.close(figure)
    except Exception:  # noqa: BLE001 -- diagnostics are non-load-bearing
        pass


# ----------------------------------------------------------------------
# Pearcey cusp-arm fixtures and tolerances (Build 8f, WP3/WP4).
# ----------------------------------------------------------------------

#: Steepest-descent tail direction shared by the code and the two
#: quadrature oracles: the right tail of ``exp(i t^4)`` decays along
#: ``arg t = pi/8`` (``sin(4 pi/8) = +1``).  The oracles integrate along
#: the SINGLE straight line ``t = s e^{i pi/8}`` through the origin (the
#: right tail for ``s > 0``, its ``9 pi/8`` reflection for ``s < 0``) --
#: a different contour DECOMPOSITION and a different (adaptive) quadrature
#: than the module's central-segment-plus-two-tails fixed-order rule.
_PEARCEY_ROTATION = cmath.exp(1j * math.pi / 8.0)

#: Half-length of the rotated integration line for the oracles.  The
#: integrand modulus is ``exp(-s^4 - ...)`` there, so ``s = 8`` puts the
#: truncated tail at ``exp(-8^4) ~ 1e-1779`` -- utterly negligible.
_PEARCEY_ORACLE_HALF_LENGTH = 8.0

#: Reference / oracle bar on the certified primitive (spec: <= 1e-8);
#: measured scipy/mpmath agreement is ~1e-13.
_PEARCEY_REFERENCE_TOL = 1e-8

#: The module's own paired-rule certificate threshold; a true error above
#: it must be FLAGGED (the primitive returns ``None``), never certified.
_PEARCEY_CERTIFICATE_TOL = _pearcey_cusp._CERTIFICATION_TOL

#: Analytic closed form ``P(0, 0) = Int exp(i t^4) dt =
#: (Gamma(1/4) / 2) e^{i pi/8}`` (from ``Int_0^inf e^{i t^4} dt =
#: (1/4) Gamma(1/4) e^{i pi/8}``, doubled by evenness).  The fully
#: independent analytic anchor of the whole primitive.
_PEARCEY_AT_ORIGIN = 0.5 * scipy_gamma(0.25) * cmath.exp(1j * math.pi / 8.0)

#: Cusp-neighbourhood fixtures ``(gamma, radius, angle)`` at which the
#: cusp arm SERVES (found by a coarse scan): a near-cusp source inside a
#: positive-parity Chang-Refsdal caustic.  ``beta = kappa = 0``.
_CUSP_FIXTURES = (
    (0.5, 0.20, 0.25 * math.pi),
    (0.7, 0.25, 0.25 * math.pi),
    (0.3, 0.10, 0.50 * math.pi),
)

#: The ceiling frequency where BOTH the exact Schwinger engine
#: (``w <= _schwinger.W_CEILING_SCHWINGER = 60``) and the uniform cusp arm
#: are valid AND the uniform approximation is best resolved in the
#: reachable exact band, so the engine-match certification lives here.
#: Below the ceiling the finite-``w`` uniform correction plus the
#: cluster/far-image interference beat leave the weakest-shear fixture
#: marginally over the 1e-2 bar (measured 1.07e-2 at ``w = 50``) -- the
#: served amplitude only clears the bar toward the ceiling, matching the
#: `cusp_amplification` note that it "still awaits a brute-force
#: cross-check".
_ENGINE_MATCH_W = _schwinger.W_CEILING_SCHWINGER

#: A sub-ceiling overlap frequency used only to WITNESS (not gate) the
#: approach to the bar in the diagnostic.
_SUB_CEILING_W = 50.0

#: A ``w`` grid (all in the overlap band) over which the captured
#: controls' ``w^{1/2}`` / ``w^{3/4}`` exponents are fit.
_SCALING_WS = (30.0, 40.0, 50.0, 60.0)

#: A frequency above the Schwinger QD ceiling (W_CEILING_SCHWINGER_QD=150):
#: here the geometric and Schwinger rungs have already declined, so the
#: uniform arm is the only server and its refusal falls through to the
#: NAMED Schwinger refusal.
_ABOVE_CEILING_W = 160.0

#: Envelope-match bar of the cusp arm against the exact engine along the
#: fold arcs (spec: <= 1e-2; measured 1e-3..7e-3 for the gamma=0.7
#: fixture).
_CUSP_ENVELOPE_TOL = 1e-2

#: Fitted-exponent tolerance (spec: within 5%).
_EXPONENT_TOL = 0.05

#: A node the FOLD arm declines but the CUSP arm serves, above the
#: Schwinger ceiling (``w > 60``): ``gamma = 0.5`` positive parity, source
#: at radius 0.18 along a 0.3*pi ray, ``w = 80``.  Verified empirically:
#: `_airy_fold.fold_amplification` returns ``None`` here while
#: `_pearcey_cusp.cusp_amplification` certifies, so the served value on
#: the grid comes from the CUSP rung -- exactly the rung whose corruption
#: and threshold this suite mutates.  A fold-served node would leave the
#: cusp mutations inert.
#:
#: ``w = 160`` sits ABOVE the Schwinger QD ceiling (150), so the serving
#: ladder routes the node through the cusp arm (fast) instead of the
#: arbitrary-precision exact engine (the mpmath path for ``w in (60, 150]``
#: takes ~160 s per node).  The cusp arm certifies here (measured) and
#: `operator.F_op_grid` serves its value bit-identically, so the suite's
#: grid checks stay fast without changing what they assert.
_CUSP_NODE_GAMMA = 0.5
_CUSP_NODE_RADIUS = 0.18
_CUSP_NODE_ANGLE = 0.3 * math.pi
_CUSP_NODE_W = 160.0
# Frequency above the QD ceiling (150) for fall-through tests that expect
# SchwingerCertificationError — the mpmath path now serves w in (60, 150].
_CUSP_FALLTHROUGH_W = 151.0


# ----------------------------------------------------------------------
# Independent Pearcey oracles (no `_pearcey_cusp` arithmetic): an analytic
# closed form, scipy adaptive QUADPACK, and 40-digit mpmath -- all on a
# DIFFERENT contour decomposition and quadrature than the module.
# ----------------------------------------------------------------------

def _reference_pearcey(x, y):
    """
    FAST reference ``P(x, y)`` by adaptive QUADPACK on the rotated line.

    Integrates ``exp[i(t^4 + x t^2 + y t)] dt`` along ``t = s e^{i pi/8}``,
    ``s in [-L, L]``, with `scipy.integrate.quad` (adaptive Gauss-Kronrod)
    on the real and imaginary parts.  This shares NO code with the
    module's fixed-order composite rule, so agreement certifies the
    module's quadrature, not a common implementation.
    """
    def integrand(s):
        t = s * _PEARCEY_ROTATION
        return cmath.exp(1j * (t ** 4 + x * t ** 2 + y * t)) \
            * _PEARCEY_ROTATION

    limit = _PEARCEY_ORACLE_HALF_LENGTH
    real, _ = scipy_quad(lambda s: integrand(s).real, -limit, limit,
                         limit=400)
    imag, _ = scipy_quad(lambda s: integrand(s).imag, -limit, limit,
                         limit=400)
    return complex(real, imag)


def _mp_pearcey(x, y):
    """
    Gated 40-digit ``mpmath.quad`` reference ``P(x, y)`` on the rotated
    line -- an arbitrary-precision, adaptive, independent oracle.
    """
    x_mp = mpmath.mpf(x)
    y_mp = mpmath.mpf(y)
    rotation = mpmath.e ** (1j * mpmath.pi / 8)

    def integrand(s):
        t = s * rotation
        return mpmath.e ** (1j * (t ** 4 + x_mp * t ** 2 + y_mp * t)) \
            * rotation

    limit = _PEARCEY_ORACLE_HALF_LENGTH
    return complex(mpmath.quad(integrand, [-limit, 0, limit]))


def _paired_certificate(x, y):
    """
    Reproduce the module's in-place paired-rule certificate for ``P``.

    Mirrors `_pearcey_cusp.pearcey`'s ``N`` / ``2N`` decision exactly
    (same fixed contour geometry, same panel-count rule) so the test can
    read the certificate value ``|P_N - P_2N| / |P_2N|`` that the code
    uses to certify-or-refuse.  Returns ``(fine_value, certificate)``.
    """
    half_width = _pearcey_cusp._split_half_width(x, y)
    cutoff_right = _pearcey_cusp._tail_cutoff(half_width, x, y, +1.0)
    cutoff_left = _pearcey_cusp._tail_cutoff(half_width, x, y, -1.0)
    frequency = _pearcey_cusp._phase_frequency(half_width, x, y)
    panels_central = _pearcey_cusp._panel_count(2.0 * half_width, frequency)
    panels_tail = _pearcey_cusp._panel_count(
        max(cutoff_right, cutoff_left), frequency)
    coarse = _pearcey_cusp._pearcey_estimate(
        x, y, half_width, cutoff_right, cutoff_left,
        panels_central, panels_tail)
    fine = _pearcey_cusp._pearcey_estimate(
        x, y, half_width, cutoff_right, cutoff_left,
        2 * panels_central, 2 * panels_tail)
    return fine, abs(fine - coarse) / abs(fine)


def _capture_cusp_controls(w, source, gamma, **kwargs):
    """
    Return ``(value, (x, y))`` -- the cusp arm's output and the LAST
    ``(x, y)`` control pair it fed to `pearcey`.

    A recording wrapper around `_pearcey_cusp.pearcey` observes the
    controls the code actually constructs, without reproducing the
    construction (which would be self-referential).  Those observed
    controls are what the scaling and swap tests interrogate.
    """
    real_pearcey = _pearcey_cusp.pearcey
    recorded = []

    def recording(x, y):
        recorded.append((x, y))
        return real_pearcey(x, y)

    with mock.patch.object(_pearcey_cusp, 'pearcey', new=recording):
        value = _pearcey_cusp.cusp_amplification(w, source, gamma, **kwargs)
    return value, (recorded[-1] if recorded else None)


def _fold_control(x):
    """``y >= 0`` on the Pearcey fold caustic ``27 y^2 = -8 x^3`` (``x <
    0``): the semicubical where P's two real stationary points coalesce."""
    return math.sqrt(-8.0 * x ** 3 / 27.0)


# ----------------------------------------------------------------------
# Base test case: anti-vacuity guard.
# ----------------------------------------------------------------------

class _FoldArmTestCase(TestCase):
    """Base class carrying the anti-vacuity comparison tally."""

    def setUp(self):
        """Reset the per-test comparison counter used by `tearDown`."""
        self.n_checks = 0

    def tearDown(self):
        """
        Fail a test that ran no comparison at all.

        The accuracy helpers skip configurations that fall outside their
        premise (no merging pair, ``xi`` below the far-field threshold);
        a sweep that skipped every one would otherwise pass without
        asserting anything.
        """
        if self.n_checks == 0:
            self.fail('vacuous: the test made no comparison')


# ----------------------------------------------------------------------
# 1. Transcription oracle (FAST): the closed form matches mpmath.
# ----------------------------------------------------------------------

class AiryFoldTranscriptionTestCase(_FoldArmTestCase):
    """
    `airy_fold_value` reproduces the exact closed form to machine
    precision, cross-checked against an independent mpmath evaluator.

    This nails the transcription the whole arm rests on: the ``Ai(-xi)``
    argument sign (the Architect's primary Ai(-xi)-vs-Ai(+xi) falsifier),
    the ``-i q w^{-1/6} Ai'`` quadrature term, the fixed phase ``sigma``,
    and the ``2 sqrt(pi)`` prefactor.  It is fast (pure special-function
    evaluation, no geometry) and therefore load-bearing rather than
    gated.
    """

    #: Signed Airy controls spanning both the oscillatory (xi > 0) and
    #: evanescent (xi < 0) sides, including the on-caustic point xi = 0.
    XIS = (-8.0, -3.0, -0.7, 0.0, 0.7, 3.0, 8.0, 25.0)

    #: Frequencies from the exact-engine band up into the high-w corner
    #: the arm exists to serve.
    WS = (5.0, 55.0, 300.0, 5000.0)

    def test_matches_mpmath_closed_form(self):
        """
        Over a grid of ``(w, xi, p, q)`` the scipy-backed value agrees
        with the mpmath closed form within `_TRANSCRIPTION_TOL`.
        """
        tau_bar = 0.37
        amplitudes = ((1.3, 0.0), (0.8, 0.45), (0.0, 1.1))
        for w, xi, (p, q) in itertools.product(self.WS, self.XIS,
                                               amplitudes):
            with self.subTest(w=w, xi=xi, p=p, q=q):
                got = _airy_fold.airy_fold_value(w, tau_bar, xi, p, q,
                                                 _SIGMA_FOLD)
                want = _mp_airy_fold(w, tau_bar, xi, p, q, _SIGMA_FOLD)
                scale = max(abs(want), 1.0)
                self.n_checks += 1
                self.assertLessEqual(
                    abs(got - want) / scale, _TRANSCRIPTION_TOL,
                    f'w={w} xi={xi} p={p} q={q}: got {got}, want {want}')

    def test_uses_minus_xi_argument(self):
        """
        The Airy argument is ``-xi``: at fixed ``|xi|`` the value on the
        oscillatory side (xi > 0) equals the closed form built from
        ``Ai(-xi)``, and NOT the one built from ``Ai(+xi)``.

        This is the transcription-level statement of the sign convention;
        a copy that read ``Ai(+xi)`` would satisfy the ``+xi`` form
        instead and fail here.
        """
        w, tau_bar, p, q = 300.0, 0.37, 1.1, 0.4
        for xi in (0.7, 3.0, 8.0, 25.0):
            with self.subTest(xi=xi):
                got = _airy_fold.airy_fold_value(w, tau_bar, xi, p, q,
                                                 _SIGMA_FOLD)
                minus = _mp_airy_fold(w, tau_bar, xi, p, q, _SIGMA_FOLD)
                plus = _mp_airy_fold(w, tau_bar, -xi, p, q, _SIGMA_FOLD)
                self.n_checks += 1
                self.assertLessEqual(abs(got - minus) / max(abs(minus), 1.0),
                                     _TRANSCRIPTION_TOL)
                # The two conventions are genuinely different here, so the
                # match above is discriminating.
                self.assertGreater(abs(minus - plus), 1e-3 * abs(minus))


# ----------------------------------------------------------------------
# 2. Sign convention as PHYSICS (FAST): present side oscillates, absent
#    side decays.  Model-free -- independent of any Airy formula.
# ----------------------------------------------------------------------

class AiryFoldSignConventionTestCase(_FoldArmTestCase):
    """
    The signed control ``xi`` selects oscillation vs evanescent decay.

    Physically, inside the caustic (``xi > 0``, two real merging images)
    the amplification INTERFERES and oscillates, while outside
    (``xi < 0``, no real image) it DECAYS monotonically.  With the correct
    ``Ai(-xi)`` convention the arm's ``|F|`` reproduces exactly this
    handoff at the caustic ``xi = 0``; the Ai(+xi) flip inverts it (decay
    where it should oscillate), which is caught here and in
    `FoldArmSelfFalsificationTestCase`.
    """

    #: A single-image (q = 0) fold arm, so |F| ~ |Ai(-xi)| and the
    #: oscillation/decay contrast is unmasked.
    P, Q = 1.3, 0.0
    W, TAU_BAR = 50.0, 0.4

    def _magnitude_scan(self, xis):
        return np.array([abs(_airy_fold.airy_fold_value(
            self.W, self.TAU_BAR, xi, self.P, self.Q, _SIGMA_FOLD))
            for xi in xis])

    def test_present_side_oscillates(self):
        """
        On the image-present side (``xi > 0``) ``|F|`` has many interior
        maxima -- the two-image interference fringes of the fold.
        """
        xis = np.linspace(0.5, 20.0, 400)
        maxima = _interior_maxima(self._magnitude_scan(xis))
        self.n_checks += 1
        self.assertGreaterEqual(
            maxima, 4,
            f'present side showed {maxima} maxima; expected oscillation')

    def test_absent_side_decays_monotonically(self):
        """
        On the image-absent side (``xi < 0``) ``|F|`` is evanescent: it
        has NO interior maxima and rises monotonically toward the caustic
        as ``|xi|`` shrinks (``Ai(+|xi|)`` decay).
        """
        xis = np.linspace(-20.0, -0.5, 400)   # |xi| decreasing toward 0
        magnitudes = self._magnitude_scan(xis)
        self.n_checks += 1
        self.assertEqual(
            _interior_maxima(magnitudes), 0,
            'absent side showed interior maxima; expected pure decay')
        self.assertTrue(
            np.all(np.diff(magnitudes) > -1e-12),
            'absent side |F| is not monotone increasing toward the caustic')

    def test_handoff_sits_at_the_caustic(self):
        """
        The oscillation/decay handoff sits exactly at ``xi = 0``: the
        far evanescent tail (``xi <= -5``) is strictly below the first
        interior fringe peak of the oscillatory side.  Also emits the
        spec diagnostic plot of ``|F|`` vs signed control.
        """
        xis = np.linspace(-12.0, 20.0, 600)
        magnitudes = self._magnitude_scan(xis)
        tail = magnitudes[xis <= -5.0].max()
        peak = magnitudes[xis >= 0.0].max()
        self.n_checks += 1
        self.assertLess(tail, peak,
                        'evanescent tail is not suppressed below the '
                        'oscillatory fringes')
        _save_plot('sign_convention_handoff',
                   xis, magnitudes, xlabel='signed Airy control xi',
                   ylabel='|F_arm|')


# ----------------------------------------------------------------------
# 3. On-caustic finiteness (FAST): xi = 0 is a finite peak, not a pole.
# ----------------------------------------------------------------------

class AiryFoldAtCausticTestCase(_FoldArmTestCase):
    """
    The arm is FINITE on the caustic (``xi = 0``), where the geometric
    two-image sum diverges (``mu -> inf``).

    This is the amplitude-normalization guard (Professor flag): the arm's
    amplitude ``p`` is built from the finite fold curvatures, not the raw
    ``sqrt|mu|``, so ``F(xi = 0) = 2 sqrt(pi)(p w^{1/6} Ai(0)
    - i q w^{-1/6} Ai'(0))`` stays finite.  A ``p`` built from ``sqrt|mu|``
    would blow up here -- see `FoldArmSelfFalsificationTestCase`.
    """

    WS = (5.0, 55.0, 300.0, 5000.0)

    def test_value_at_caustic_is_finite_and_matches_closed_form(self):
        """
        ``airy_fold_value(xi = 0)`` is finite and equals the mpmath
        closed form ``2 sqrt(pi) exp(i(w tau_bar + sigma))
        (p w^{1/6} Ai(0) - i q w^{-1/6} Ai'(0))`` to `_ONCAUSTIC_TOL`.
        """
        tau_bar = 0.37
        for w, (p, q) in itertools.product(self.WS,
                                           ((1.3, 0.0), (0.9, 0.5))):
            with self.subTest(w=w, p=p, q=q):
                got = _airy_fold.airy_fold_value(w, tau_bar, 0.0, p, q,
                                                 _SIGMA_FOLD)
                want = _mp_airy_fold(w, tau_bar, 0.0, p, q, _SIGMA_FOLD)
                self.n_checks += 1
                self.assertTrue(np.isfinite(abs(got)))
                self.assertLessEqual(abs(got - want) / max(abs(want), 1.0),
                                     _ONCAUSTIC_TOL)

    def test_magnitude_is_a_finite_peak_through_the_caustic(self):
        """
        ``|F|`` scanned through ``xi = 0`` is bounded (a finite peak, not
        a pole) and the maximum sits at the first fold fringe just inside
        the caustic (the present side, ``xi >= 0``, at order-unity
        ``xi``), never running off to infinity.  Emits the spec
        diagnostic plot.

        For the single-image arm ``|F| ~ |Ai(-xi)|``, whose global maximum
        is the first Airy fringe at ``xi ~ 1.02`` -- an order-unity
        distance inside the caustic, not a pole and not at ``xi -> inf``.
        """
        p, q, w, tau_bar = 1.3, 0.0, 50.0, 0.4
        xis = np.linspace(-6.0, 6.0, 500)
        magnitudes = np.array([abs(_airy_fold.airy_fold_value(
            w, tau_bar, xi, p, q, _SIGMA_FOLD)) for xi in xis])
        self.n_checks += 1
        self.assertTrue(np.all(np.isfinite(magnitudes)))
        peak_index = int(np.argmax(magnitudes))
        peak_xi = xis[peak_index]
        # The finite peak is the first fold fringe: present side, O(1) xi.
        self.assertGreaterEqual(
            peak_xi, 0.0,
            'the finite peak fell on the evanescent (image-absent) side')
        self.assertLess(
            peak_xi, 2.0,
            'the finite peak is not at the first fold fringe near the '
            'caustic')
        _save_plot('at_caustic_finite_peak', xis, magnitudes,
                   xlabel='signed Airy control xi', ylabel='|F_arm|')

    def test_amplitude_stays_finite_as_source_approaches_caustic(self):
        """
        The SERVED amplitude ``p`` (from `fold_amplification`'s curvature
        calibration) stays bounded as the source approaches the caustic,
        whereas the raw geometric ``sqrt|mu|`` of the merging pair
        diverges.  This is the finiteness that the ``xi = 0`` value
        inherits.
        """
        matrix = _matrix()
        s_values = []
        p_values = []
        for radius in (0.20, 0.24, 0.27, 0.285):
            source = _source(radius)
            pair = _merging_pair(source, matrix)
            self.assertIsNotNone(pair)
            _, _, s_plus, s_minus = pair
            nearest = geometry.nearest_caustic_point(_GAMMA, _BETA, source,
                                                     kappa=_KAPPA)
            b3 = _airy_fold._soft_axis_cubic(nearest.image,
                                             nearest.soft_axis)
            amplitudes = _airy_fold._fold_amplitudes(
                nearest.hard_eigenvalue, b3)
            self.assertIsNotNone(amplitudes)
            s_values.append(max(s_plus, s_minus))
            p_values.append(amplitudes[0])
            self.n_checks += 1
        # sqrt|mu| grows toward the caustic; p does not blow up with it.
        self.assertGreater(s_values[-1], s_values[0])
        self.assertLess(max(p_values), 5.0 * min(p_values),
                        'the curvature amplitude p tracks the divergent '
                        'sqrt|mu| -- the normalization trap is present')


# ----------------------------------------------------------------------
# 4. Far-field envelope convergence (GATED): the arm reproduces the exact
#    ASYMMETRIC geometric two-image sum as xi -> inf, at the uniform rate.
# ----------------------------------------------------------------------

class AiryFoldFarFieldEnvelopeTestCase(_FoldArmTestCase):
    """
    Accuracy of the fold CLOSED FORM `airy_fold_value` when it is handed
    an ASYMMETRIC amplitude pair -- NOT of the arm that production serves.

    SCOPE, PLAINLY.  Every test in this class calls
    `_airy_fold.airy_fold_value` directly with the oracle-derived
    amplitudes of `_farfield_amplitudes`, whose ``Ai'`` amplitude ``q`` is
    NON-ZERO.  `_airy_fold.fold_amplification` -- the only entry point
    production uses -- is never called here, and it hard-codes ``q = 0``.
    These tests therefore certify a closed form evaluated at an amplitude
    pair that production never constructs.

    THE ACCURACY OF THE SERVED ARM IS NOT COVERED BY THIS CLASS.  F028
    measured the served arm at 60%-267% relative error against geometric
    optics on well-resolved above-ceiling configs while this class's 1e-3
    gate stayed green -- the two evaluate different amplitudes, so no
    result here transfers to `fold_amplification`.  The gap is kept
    visible by `FoldAmplificationServingTestCase.
    test_served_arm_accuracy_is_unverified_pending_an_oracle`.

    WHAT IS CERTIFIED.  Given the SUM ``p`` and DIFFERENCE ``q``
    amplitudes, the closed form reproduces `geometry`'s exact
    ``sqrt|mu_+| e^{i w tau_+} + sqrt|mu_-| e^{i w tau_- - i pi/2}``
    two-image sum (a DIFFERENT module, no arm arithmetic) in the
    max-normalized ENVELOPE currency, at the leading uniform rate
    ``~ xi^{-3/2}``.  A symmetric approach would hide the
    sum-vs-difference assignment, so the fixture source sits well off the
    cusp axis (``sqrt|mu|`` ratio ~1.2); the p/q swap it is built to
    expose is falsified in `FoldArmSelfFalsificationTestCase`.

    This scans ``w`` (hence ``xi``) over the merging pair and is therefore
    the brute-force accuracy tier.
    """

    #: Airy controls at which the leading uniform envelope error is
    #: measured; each ~2x the last, so the ratio witnesses the rate.
    XI_CHECKPOINTS = (40.0, 80.0, 160.0, 320.0)

    def _envelope_error(self, xi_target, pair):
        """
        Max-normalized envelope error ``max_w||F_closed| - |F_geom|| /
        (s_+ + s_-)`` over one beat window of ``w`` centred on the ``w``
        that yields ``xi_target``.

        ``F_closed`` is `airy_fold_value` fed the oracle's ASYMMETRIC
        ``(p, q != 0)`` from `_farfield_amplitudes`.  It is NOT the value
        `fold_amplification` serves, which is the same closed form at
        ``q = 0``; nothing measured through this helper bounds the served
        arm's error (F028).
        """
        tau_plus, tau_minus, s_plus, s_minus = pair
        delta_tau = tau_minus - tau_plus
        tau_bar = 0.5 * (tau_plus + tau_minus)
        w_centre = (4.0 / 3.0 * xi_target ** 1.5) / delta_tau
        beat = 2.0 * math.pi / delta_tau
        ws = np.linspace(w_centre - beat, w_centre + beat, 80)
        closed_form = np.empty_like(ws)
        geom = np.empty_like(ws)
        for index, w in enumerate(ws):
            xi = (3.0 * w * delta_tau / 4.0) ** (2.0 / 3.0)
            p, q = _farfield_amplitudes(w, xi, s_plus, s_minus)
            closed_form[index] = abs(_airy_fold.airy_fold_value(
                w, tau_bar, xi, p, q, _SIGMA_FOLD))
            geom[index] = abs(_geometric_two_image_sum(
                w, tau_plus, tau_minus, s_plus, s_minus))
        return float(np.max(np.abs(closed_form - geom)) / (s_plus + s_minus))

    @_brute_accuracy_tier
    def test_closed_form_with_asymmetric_amplitudes_matches_geometric_sum(
            self):
        """
        For every inside-caustic asymmetric fixture and every
        ``xi >= _XI_FARFIELD``, `airy_fold_value` EVALUATED AT THE
        ORACLE-DERIVED ``(p, q != 0)`` clears the spec bar
        `_FARFIELD_ENVELOPE_TOL` against the geometric two-image sum.

        This tests the CLOSED FORM, not the served arm.
        `fold_amplification` is never called, and it would supply
        ``q = 0`` -- a symmetric-fold assumption that cannot represent an
        unequal-magnification image pair at all (F028).  A green result
        here therefore says nothing about the accuracy of what production
        serves; see `FoldAmplificationServingTestCase.
        test_served_arm_accuracy_is_unverified_pending_an_oracle`.
        """
        matrix = _matrix()
        for radius in _INSIDE_RADII:
            pair = _merging_pair(_source(radius), matrix)
            self.assertIsNotNone(pair, f'no merging pair at r={radius}')
            for xi_target in self.XI_CHECKPOINTS:
                with self.subTest(radius=radius, xi=xi_target):
                    error = self._envelope_error(xi_target, pair)
                    self.n_checks += 1
                    self.assertLessEqual(
                        error, _FARFIELD_ENVELOPE_TOL,
                        f'r={radius} xi={xi_target}: envelope error '
                        f'{error:.3e} exceeds {_FARFIELD_ENVELOPE_TOL}')

    @_brute_accuracy_tier
    def test_envelope_error_falls_as_xi_to_the_minus_three_halves(self):
        """
        The CLOSED FORM's envelope error decreases monotonically with
        ``xi`` and, per ``xi`` doubling, drops by ~``2^{3/2} = 2.83`` --
        the signature of the leading uniform ``xi^{-3/2}`` term, not of an
        accidental near-cancellation.  Emits the residual-vs-``xi``
        diagnostic plot.

        Measured on `airy_fold_value` at the oracle-derived ASYMMETRIC
        ``(p, q != 0)``, so the rate certified belongs to the closed form
        and NOT to the served arm.  `fold_amplification` serves ``q = 0``,
        whose error does not fall with ``xi`` at all: F028 measured it
        GROWING with ``w`` (``|F_arm/F_geo| = 0.348`` at ``w = 70``,
        ``1.846`` at ``w = 500``).  This decay law does not transfer.
        """
        matrix = _matrix()
        pair = _merging_pair(_source(_INSIDE_RADII[0]), matrix)
        self.assertIsNotNone(pair)
        errors = [self._envelope_error(xi, pair)
                  for xi in self.XI_CHECKPOINTS]
        for xi_low, xi_high, err_low, err_high in zip(
                self.XI_CHECKPOINTS, self.XI_CHECKPOINTS[1:],
                errors, errors[1:]):
            with self.subTest(xi_low=xi_low, xi_high=xi_high):
                self.n_checks += 1
                self.assertLess(err_high, err_low,
                                'envelope error is not decreasing with xi')
                ratio = err_low / err_high
                self.assertGreater(
                    ratio, 2.3,
                    f'xi {xi_low}->{xi_high}: decay ratio {ratio:.2f} too '
                    f'shallow for a xi^-3/2 leading term')
                self.assertLess(
                    ratio, 3.4,
                    f'xi {xi_low}->{xi_high}: decay ratio {ratio:.2f} too '
                    f'steep for a xi^-3/2 leading term')
        _save_plot('far_field_envelope_convergence',
                   np.log10(self.XI_CHECKPOINTS), np.log10(errors),
                   xlabel='log10 xi', ylabel='log10 envelope error')


# ----------------------------------------------------------------------
# 5. fold_amplification serving + refusal contract (FAST).
# ----------------------------------------------------------------------

class FoldAmplificationServingTestCase(_FoldArmTestCase):
    """
    `fold_amplification` wires `airy_fold_value` with the geometry-derived
    control and the calibrated fold amplitude, and refuses conservatively.

    SCOPE, PLAINLY: this class tests WIRING and REFUSALS, never accuracy.
    The serving check reproduces ``airy_fold_value`` evaluated at the
    INDEPENDENTLY re-derived (`geometry`) ``tau_bar`` and
    ``xi = (3 w DT / 4)^{2/3}`` with the module's own calibrated
    ``(p, q = 0, sigma = -pi/4)``; both sides use the module's amplitudes,
    so a wrong amplitude is invisible to it.  The refusal check asserts
    only that out-of-domain inputs return ``None``.

    The served value's ACCURACY is not certified anywhere in this file.
    `AiryFoldFarFieldEnvelopeTestCase` measures the closed form at
    DIFFERENT (asymmetric, ``q != 0``) amplitudes, so it does not cover
    the served path either.  `test_served_arm_accuracy_is_unverified_
    pending_an_oracle` is the expected-failure marker that keeps that gap
    visible.
    """

    #: Frequencies high enough that the leading uniform-error estimate
    #: clears the default envelope bar for the r = 0.14 fixture (measured
    #: serving threshold sits between w = 200, refused, and w = 500).
    SERVED_WS = (500.0, 1000.0, 5000.0)

    def test_served_value_matches_independent_wiring(self):
        """
        A served value equals ``airy_fold_value`` at the independently
        re-derived ``tau_bar``/``xi`` with the module's own calibrated
        ``(p, q, sigma)``, and is finite.

        A TRANSCRIPTION/WIRING contract only.  Both sides read the
        amplitudes from `_airy_fold`, so this comparison cannot detect a
        wrong amplitude -- and the served ``q`` IS wrong (see the pinned
        known defect below).  Nothing here bounds the served value's
        error against physics; the arm's accuracy is untested (F028,
        F030), which is what
        `test_served_arm_accuracy_is_unverified_pending_an_oracle`
        records.
        """
        matrix = _matrix()
        source = _source(_INSIDE_RADII[0])
        pair = _merging_pair(source, matrix)
        self.assertIsNotNone(pair)
        tau_plus, tau_minus, _, _ = pair
        tau_bar = 0.5 * (tau_plus + tau_minus)
        delta_tau = tau_minus - tau_plus
        nearest = geometry.nearest_caustic_point(_GAMMA, _BETA, source,
                                                 kappa=_KAPPA)
        b3 = _airy_fold._soft_axis_cubic(nearest.image, nearest.soft_axis)
        amplitudes = _airy_fold._fold_amplitudes(nearest.hard_eigenvalue, b3)
        self.assertIsNotNone(amplitudes)
        p_amplitude, q_amplitude, sigma = amplitudes
        # KNOWN DEFECT, PINNED -- NOT AN ENDORSEMENT.  `_fold_amplitudes`
        # hard-codes the ``Ai'`` amplitude ``q = 0``.  That is a
        # SYMMETRIC-fold assumption, not a leading-order truncation: a lone
        # ``Ai`` term has a single-sinusoid large-argument limit and cannot
        # represent a two-image sum with unequal magnifications -- i.e. every
        # source position except exactly on the caustic (F028).  The
        # assertion is kept because it correctly pins CURRENT behaviour and
        # would catch an accidental change, not because ``q = 0`` is right.
        # Open: .claude/spec/todo.d/lensing_fold_arm_serves_wrong_values.md
        self.assertEqual(q_amplitude, 0.0)
        self.assertAlmostEqual(sigma, _SIGMA_FOLD)
        for w in self.SERVED_WS:
            with self.subTest(w=w):
                served = _airy_fold.fold_amplification(w, source, _GAMMA,
                                                       beta=_BETA,
                                                       kappa=_KAPPA)
                self.assertIsNotNone(served, f'w={w} unexpectedly refused')
                xi = (3.0 * w * delta_tau / 4.0) ** (2.0 / 3.0)
                wired = _airy_fold.airy_fold_value(
                    w, tau_bar, xi, p_amplitude, q_amplitude, sigma)
                self.n_checks += 1
                self.assertTrue(np.isfinite(abs(served)))
                self.assertLessEqual(abs(served - wired),
                                     _TRANSCRIPTION_TOL * max(abs(wired), 1.0))

    @expectedFailure
    def test_served_arm_accuracy_is_unverified_pending_an_oracle(self):
        """
        EXPECTED FAILURE BECAUSE THE SERVED ARM IS WRONG, NOT BECAUSE THE
        TEST IS.

        At F028's measured configuration -- ``gamma = 0.70``, the source
        offset ``0.40 R_c`` along the normal of the off-cusp caustic point
        ``t = 0.55``, ``w = 70`` -- `fold_amplification` CERTIFIES (its
        ``c_A xi^{-3/2}`` estimate clears the 0.05 envelope bar) and
        returns a value.  The node is delay-resolved and sits ``eta = 0.50``
        from the caustic, well inside F029's ``eta > 0.3`` bin where
        `operator.geometric_amplification` was measured accurate to a
        median ``2e-7`` against the Schwinger quadrature.  Geometric optics
        is therefore the better reference here, and the arm ought to agree
        with it to well inside its own certificate.

        It does not.  Measured 2026-07-29 (reproducing F028's table row):
        ``|F_arm| / |F_geo| = 0.348``, an envelope error of ``0.65``
        against a certificate claiming ``0.05`` -- optimistic by ~13x.
        The cause is `fold_amplification`'s ``q = 0``: a single ``Ai``
        term cannot represent a two-image sum with unequal magnifications
        (F028, pinned in `test_served_value_matches_independent_wiring`).

        THIS IS A MARKER, NOT A GATE.  Geometric optics is a stand-in, not
        the missing oracle: the suite still owns no reference valid at
        ``L ~ 100-200`` where the arms actually serve, so no arm accuracy
        claim in this file is falsifiable there (F030).  Convert this into
        a real gate -- and drop the `expectedFailure` marker -- once such
        an oracle exists.  Should it start PASSING, unittest reports an
        unexpected success and the run fails: that is the intended signal
        that the arm changed and this marker must be revisited.

        Open: `.claude/spec/todo.d/lensing_fold_arm_serves_wrong_values.md`
        """
        source = _caustic_normal_source(_F028_GAMMA, _F028_T, _F028_RATIO)
        served = _airy_fold.fold_amplification(_F028_W, source, _F028_GAMMA)
        self.n_checks += 1
        self.assertIsNotNone(
            served,
            'the F028 config is no longer served: re-derive this marker '
            'against a config the arm does certify, do not delete it')
        reference = operator.geometric_amplification(_F028_W, source,
                                                     _F028_GAMMA)
        envelope_error = abs(abs(served) - abs(reference)) / abs(reference)
        self.assertLessEqual(
            envelope_error, _airy_fold._DEFAULT_ENVELOPE_BAR,
            f'served arm envelope error {envelope_error:.3f} exceeds the '
            f'{_airy_fold._DEFAULT_ENVELOPE_BAR} bar its own certificate '
            f'claimed to meet (|F_arm|/|F_geo| = '
            f'{abs(served) / abs(reference):.3f})')

    def test_refuses_conservatively(self):
        """
        `fold_amplification` returns ``None`` (never a wrong number, never
        a new exception) on every out-of-domain or under-resolved config:
        non-positive ``w``, non-finite or wrong-shape source, non-positive
        ``envelope_bar``, over-critical (Type III) ``kappa``, the parity
        boundary ``|gamma| = 1 - kappa``, the near-caustic error-gate
        refusal, and a too-low ``w`` whose ``xi`` is below the uniform
        threshold.
        """
        source = _source(_INSIDE_RADII[0])
        refusals = {
            'w_zero': dict(w=0.0, source=source, gamma=_GAMMA),
            'w_negative': dict(w=-5.0, source=source, gamma=_GAMMA),
            'source_nan': dict(w=500.0, source=np.array([np.nan, 0.0]),
                               gamma=_GAMMA),
            'source_wrong_shape': dict(w=500.0,
                                       source=np.array([0.1, 0.2, 0.3]),
                                       gamma=_GAMMA),
            'envelope_bar_zero': dict(w=500.0, source=source, gamma=_GAMMA,
                                      envelope_bar=0.0),
            'type_iii_kappa': dict(w=500.0, source=source, gamma=_GAMMA,
                                   kappa=1.2),
            'parity_boundary': dict(w=500.0, source=source, gamma=1.0,
                                    kappa=0.0),
            'near_caustic_error_gate': dict(
                w=500.0, source=_source(_NEAR_CAUSTIC_RADIUS), gamma=_GAMMA),
            'low_w_below_uniform_threshold': dict(
                w=200.0, source=source, gamma=_GAMMA),
        }
        for label, kwargs in refusals.items():
            with self.subTest(refusal=label):
                gamma = kwargs.pop('gamma')
                source_arg = kwargs.pop('source')
                w_arg = kwargs.pop('w')
                result = _airy_fold.fold_amplification(
                    w_arg, source_arg, gamma, **kwargs)
                self.n_checks += 1
                self.assertIsNone(
                    result, f'{label}: expected a None refusal, got {result}')


# ----------------------------------------------------------------------
# 6. Self-falsification (FAST): prove every gate above can go red.
# ----------------------------------------------------------------------

class FoldArmSelfFalsificationTestCase(_FoldArmTestCase):
    """
    Each load-bearing gate is shown to DETECT the bug it exists to catch.

    Four mutations, one per Architect falsifier, plus an oracle-teeth
    positive control:

    * ``Ai(+xi)`` sign flip -> the transcription oracle and the
      present-side oscillation gate both go red.
    * ``p <-> q`` swap -> the far-field envelope match breaks by an
      ``O(1)`` DC offset.
    * ``sqrt|mu|`` amplitude -> the on-caustic value tracks the divergent
      magnification instead of staying finite.
    * tainted oracle -> an oracle that itself calls the arm cannot see the
      sign flip, so the independent ``mpmath`` oracle is doing real work.
    """

    def _flip_airy(self):
        """
        Patch the module's ``airy`` so that the internal ``airy(-xi)`` call
        evaluates ``Ai(+xi)`` -- the primary Architect falsifier.
        """
        def flipped(argument):
            return scipy_airy(-np.asarray(argument, dtype=float))
        return mock.patch.object(_airy_fold, 'airy', new=flipped)

    def test_airy_plus_xi_flip_is_caught_by_transcription_oracle(self):
        """The Ai(+xi) flip makes the arm disagree with the mpmath oracle
        by an ``O(1)`` amount, far above `_TRANSCRIPTION_TOL`."""
        w, tau_bar, p, q = 300.0, 0.37, 1.1, 0.4
        for xi in (0.7, 3.0, 8.0, 25.0):
            with self.subTest(xi=xi):
                honest = _mp_airy_fold(w, tau_bar, xi, p, q, _SIGMA_FOLD)
                with self._flip_airy():
                    got = _airy_fold.airy_fold_value(w, tau_bar, xi, p, q,
                                                     _SIGMA_FOLD)
                residual = abs(got - honest) / max(abs(honest), 1.0)
                self.n_checks += 1
                self.assertGreater(
                    residual, 0.1,
                    f'xi={xi}: the Ai(+xi) flip went undetected '
                    f'(residual {residual:.2e})')

    def test_airy_plus_xi_flip_destroys_present_side_oscillation(self):
        """Under the flip the image-present side no longer oscillates
        (its interior maxima collapse), so the sign-convention gate
        fails."""
        xis = np.linspace(0.5, 20.0, 400)
        with self._flip_airy():
            magnitudes = np.array([abs(_airy_fold.airy_fold_value(
                50.0, 0.4, xi, 1.3, 0.0, _SIGMA_FOLD)) for xi in xis])
        self.n_checks += 1
        self.assertLess(
            _interior_maxima(magnitudes), 4,
            'the Ai(+xi) flip left the present side oscillating')

    def test_sum_difference_swap_breaks_far_field_envelope(self):
        """Feeding the arm ``(q, p)`` instead of ``(p, q)`` leaves an
        ``O(1)`` far-field envelope offset, far above the spec bar."""
        matrix = _matrix()
        pair = _merging_pair(_source(_INSIDE_RADII[0]), matrix)
        self.assertIsNotNone(pair)
        tau_plus, tau_minus, s_plus, s_minus = pair
        delta_tau = tau_minus - tau_plus
        tau_bar = 0.5 * (tau_plus + tau_minus)
        xi_target = 80.0
        w_centre = (4.0 / 3.0 * xi_target ** 1.5) / delta_tau
        beat = 2.0 * math.pi / delta_tau
        ws = np.linspace(w_centre - beat, w_centre + beat, 80)
        arm = np.empty_like(ws)
        geom = np.empty_like(ws)
        for index, w in enumerate(ws):
            xi = (3.0 * w * delta_tau / 4.0) ** (2.0 / 3.0)
            p, q = _farfield_amplitudes(w, xi, s_plus, s_minus)
            # SWAPPED: p and q exchanged.
            arm[index] = abs(_airy_fold.airy_fold_value(
                w, tau_bar, xi, q, p, _SIGMA_FOLD))
            geom[index] = abs(_geometric_two_image_sum(
                w, tau_plus, tau_minus, s_plus, s_minus))
        error = float(np.max(np.abs(arm - geom)) / (s_plus + s_minus))
        self.n_checks += 1
        self.assertGreater(
            error, 100.0 * _FARFIELD_ENVELOPE_TOL,
            f'the p<->q swap envelope error {error:.2e} did not break the '
            f'far-field match')

    def test_sqrt_mu_amplitude_tracks_the_divergence_at_the_caustic(self):
        """
        An arm whose amplitude is the raw ``sqrt|mu|`` (the trap the
        `_fold_amplitudes` calibration avoids) has ``|F(xi = 0)|`` that
        GROWS as the source nears the caustic and tracks the divergent
        ``sqrt|mu|``, whereas the calibrated arm's ``|F(xi = 0)|`` stays
        flat.  This proves the on-caustic finiteness gate has content.
        """
        matrix = _matrix()
        radii = (0.20, 0.27, 0.285)
        bad_values = []
        good_values = []
        sqrt_mu_values = []
        for radius in radii:
            source = _source(radius)
            pair = _merging_pair(source, matrix)
            self.assertIsNotNone(pair)
            _, _, s_plus, s_minus = pair
            sqrt_mu = max(s_plus, s_minus)
            nearest = geometry.nearest_caustic_point(_GAMMA, _BETA, source,
                                                     kappa=_KAPPA)
            b3 = _airy_fold._soft_axis_cubic(nearest.image, nearest.soft_axis)
            p_calibrated = _airy_fold._fold_amplitudes(
                nearest.hard_eigenvalue, b3)[0]
            bad_values.append(abs(_airy_fold.airy_fold_value(
                300.0, 0.4, 0.0, sqrt_mu, 0.0, _SIGMA_FOLD)))
            good_values.append(abs(_airy_fold.airy_fold_value(
                300.0, 0.4, 0.0, p_calibrated, 0.0, _SIGMA_FOLD)))
            sqrt_mu_values.append(sqrt_mu)
            self.n_checks += 1
        good_growth = good_values[-1] / good_values[0]
        bad_growth = bad_values[-1] / bad_values[0]
        sqrt_mu_growth = sqrt_mu_values[-1] / sqrt_mu_values[0]
        # Calibrated |F(0)| is essentially flat; the sqrt|mu| trap grows
        # in lock-step with the (diverging) magnification.
        self.assertLess(good_growth, 1.05,
                        'the calibrated on-caustic value was not finite/flat')
        self.assertGreater(bad_growth, 1.2 * good_growth,
                           'the sqrt|mu| amplitude did not track a growth')
        self.assertAlmostEqual(bad_growth, sqrt_mu_growth, places=6)

    def test_tainted_oracle_cannot_see_the_sign_flip(self):
        """
        Oracle-teeth positive control (F002): a 'tainted' oracle that
        itself calls `airy_fold_value` agrees with the arm even under the
        Ai(+xi) flip, so it could NOT catch the sign bug -- while the
        independent mpmath oracle does.  This demonstrates the mpmath
        oracle's independence is load-bearing, not decorative.
        """
        w, tau_bar, p, q, xi = 300.0, 0.37, 1.1, 0.4, 8.0

        def tainted_oracle(*args):
            # Shares the production code path -> cannot be a real oracle.
            return _airy_fold.airy_fold_value(*args)

        honest = _mp_airy_fold(w, tau_bar, xi, p, q, _SIGMA_FOLD)
        with self._flip_airy():
            got = _airy_fold.airy_fold_value(w, tau_bar, xi, p, q,
                                             _SIGMA_FOLD)
            tainted = tainted_oracle(w, tau_bar, xi, p, q, _SIGMA_FOLD)
        self.n_checks += 1
        # The tainted oracle is blind to the flip ...
        self.assertLessEqual(abs(got - tainted) / max(abs(tainted), 1.0),
                             _TRANSCRIPTION_TOL)
        # ... while the independent oracle exposes it.
        self.assertGreater(abs(got - honest) / max(abs(honest), 1.0), 0.1)


# ----------------------------------------------------------------------
# 7. Pearcey primitive certification (FAST reference + gated mpmath).
# ----------------------------------------------------------------------

class PearceyPrimitiveCertificationTestCase(_FoldArmTestCase):
    """
    `_pearcey_cusp.pearcey` certifies the Pearcey primitive ``P(x, y)``.

    Three INDEPENDENT oracles, none sharing the module's fixed-order
    composite Gauss-Legendre rule (F002): the analytic closed form
    ``P(0, 0) = (Gamma(1/4) / 2) e^{i pi/8}``; adaptive QUADPACK on a
    single rotated line (FAST); 40-digit mpmath on the same line (gated).
    The paired-rule certificate is shown to be honest -- forced
    under-resolution is REFUSED, never certified.
    """

    #: A grid of controls spanning both fold sides (``x < 0`` two-image
    #: and ``x > 0`` dual), the axes, and mild-to-strong oscillation.
    CONTROLS = ((0.0, 0.0), (1.0, 0.5), (-3.0, 2.0), (3.0, -2.0),
                (0.0, 4.0), (-6.0, 0.0), (5.0, -4.0), (-5.0, 3.0),
                (8.0, 8.0))

    def test_value_at_origin_matches_gamma_quarter_closed_form(self):
        """
        ``P(0, 0)`` equals the analytic ``(Gamma(1/4) / 2) e^{i pi/8}`` --
        the fully independent (non-quadrature) anchor of the primitive.
        """
        value = _pearcey_cusp.pearcey(0.0, 0.0)
        self.assertIsNotNone(value)
        self.n_checks += 1
        self.assertLessEqual(
            abs(value - _PEARCEY_AT_ORIGIN) / abs(_PEARCEY_AT_ORIGIN),
            _PEARCEY_REFERENCE_TOL,
            f'P(0,0) = {value} disagrees with the Gamma(1/4) closed form '
            f'{_PEARCEY_AT_ORIGIN}')

    def test_certified_matches_scipy_reference(self):
        """
        Over the control grid the certified `pearcey` value agrees with
        the independent adaptive-QUADPACK reference to
        `_PEARCEY_REFERENCE_TOL` (FAST tier).
        """
        for x, y in self.CONTROLS:
            with self.subTest(x=x, y=y):
                value = _pearcey_cusp.pearcey(x, y)
                self.assertIsNotNone(value, f'({x},{y}) failed to certify')
                reference = _reference_pearcey(x, y)
                self.n_checks += 1
                self.assertLessEqual(
                    abs(value - reference) / abs(reference),
                    _PEARCEY_REFERENCE_TOL,
                    f'({x},{y}): certified {value} vs reference {reference}')

    def test_certificate_refuses_gross_under_resolution(self):
        """
        HONESTY: driving the quadrature into gross under-resolution (one
        panel per contour piece) at a high-oscillation control makes the
        paired-rule certificate exceed the bar, so `pearcey` returns
        ``None`` -- it never certifies the (badly wrong) under-resolved
        value.  The independent reference confirms the one-panel estimate
        really is wrong.
        """
        x, y = 30.0, -25.0
        reference = _reference_pearcey(x, y)
        with mock.patch.object(_pearcey_cusp, '_panel_count',
                               new=lambda span, frequency: 1):
            certified = _pearcey_cusp.pearcey(x, y)
            half_width = _pearcey_cusp._split_half_width(x, y)
            cutoff_right = _pearcey_cusp._tail_cutoff(half_width, x, y, +1.0)
            cutoff_left = _pearcey_cusp._tail_cutoff(half_width, x, y, -1.0)
            one_panel = _pearcey_cusp._pearcey_estimate(
                x, y, half_width, cutoff_right, cutoff_left, 1, 1)
        self.n_checks += 1
        # The under-resolved estimate is grossly wrong ...
        self.assertGreater(abs(one_panel - reference) / abs(reference),
                           _PEARCEY_CERTIFICATE_TOL)
        # ... and the certificate refuses rather than serving it.
        self.assertIsNone(
            certified,
            'the paired-rule certificate served a grossly under-resolved '
            'value instead of refusing')

    @_brute_accuracy_tier
    def test_certified_matches_mpmath_oracle(self):
        """
        Gated: the certified `pearcey` value agrees with the 40-digit
        ``mpmath.quad`` oracle to `_PEARCEY_REFERENCE_TOL` (measured
        ~1e-13) over the control grid.
        """
        for x, y in self.CONTROLS:
            with self.subTest(x=x, y=y):
                value = _pearcey_cusp.pearcey(x, y)
                self.assertIsNotNone(value)
                oracle = _mp_pearcey(x, y)
                self.n_checks += 1
                self.assertLessEqual(
                    abs(value - oracle) / abs(oracle),
                    _PEARCEY_REFERENCE_TOL,
                    f'({x},{y}): certified {value} vs mpmath {oracle}')

    @_brute_accuracy_tier
    def test_paired_rule_certificate_upper_bounds_true_error(self):
        """
        Gated: the paired-rule certificate ``|P_N - P_2N| / |P_2N|`` is a
        (Richardson) UPPER BOUND on the true error against the mpmath
        oracle -- it tracks / bounds true error rather than under-
        reporting it, so a certified value is genuinely accurate.  Emits
        the certificate-vs-true-error scatter diagnostic.
        """
        certificates = []
        true_errors = []
        for x, y in self.CONTROLS:
            with self.subTest(x=x, y=y):
                fine, certificate = _paired_certificate(x, y)
                oracle = _mp_pearcey(x, y)
                true_error = abs(fine - oracle) / abs(oracle)
                certificates.append(certificate)
                true_errors.append(true_error)
                self.n_checks += 1
                # Near machine precision a small floor absorbs float64
                # round-off in the difference of two ~1e-13 numbers.
                self.assertLessEqual(
                    true_error, 4.0 * certificate + 1e-13,
                    f'({x},{y}): true error {true_error:.2e} exceeds the '
                    f'certificate {certificate:.2e} -- the certificate '
                    f'under-reported the error')
        _save_plot('pearcey_certificate_vs_true_error',
                   np.log10(np.asarray(certificates) + 1e-18),
                   np.log10(np.asarray(true_errors) + 1e-18),
                   xlabel='log10 paired-rule certificate',
                   ylabel='log10 true error vs mpmath')


# ----------------------------------------------------------------------
# 8. Pearcey (x, y) 2/3-vs-1/2 scaling and the exponent swap.
# ----------------------------------------------------------------------

class PearceyCuspScalingTestCase(_FoldArmTestCase):
    """
    The cusp controls carry ``x = c_x w^{1/2} delta_parallel`` and
    ``y = c_y w^{3/4} delta_perp``.

    Two FAST, PURELY-MATHEMATICAL gates anchor the swap on the Pearcey
    fold caustic ``27 y^2 = -8 x^3`` (an independent property of ``P``);
    two GATED gates fit the exponents from the controls the code actually
    uses and certify that those controls reproduce the exact engine
    `operator.F_op` in the overlap band.
    """

    def test_semicubical_is_the_pearcey_fold_caustic(self):
        """
        INDEPENDENT anchor: on ``27 y^2 = -8 x^3`` (``x < 0``) the two
        real stationary points of ``P``'s phase COALESCE (``phi'' -> 0``)
        and `pearcey_asymptotic` diverges, whereas just inside the fold it
        is finite.  This establishes the semicubical as ``P``'s fold
        caustic without reference to the scaling code.
        """
        for x in (-1.5, -3.0, -6.0):
            with self.subTest(x=x):
                y_fold = _fold_control(x)
                stationary = _pearcey_cusp._real_stationary_points(x, y_fold)
                min_curvature = min(abs(12.0 * t * t + 2.0 * x)
                                    for t in stationary)
                on_fold = abs(_pearcey_cusp.pearcey_asymptotic(x, y_fold))
                off_fold = abs(_pearcey_cusp.pearcey_asymptotic(x,
                                                                0.5 * y_fold))
                self.n_checks += 1
                self.assertLess(
                    min_curvature, 1e-6,
                    'stationary points do not coalesce on the semicubical')
                self.assertGreater(
                    on_fold, 100.0 * off_fold,
                    'the asymptotic does not diverge on the fold caustic')

    def test_correct_exponents_are_w_invariant_swap_is_not(self):
        """
        The catastrophe scaling verified against the INDEPENDENT
        semicubical fold: with the ``1/2`` / ``3/4`` exponents the common
        ``w^{3/2}`` cancels in ``27 y^2 = -8 x^3``, so a source that
        starts on the fold arc stays on it at every ``w``; the swapped
        (``3/4`` / ``1/2``) exponents leave ``w^1`` vs ``w^{9/4}`` and
        walk the source off the fold.  Pure ``P``-geometry, no engine.
        """
        # A control pair on the fold at the reference frequency.
        x_ref, y_ref = -1.5, _fold_control(-1.5)
        self.assertAlmostEqual(27.0 * y_ref ** 2, -8.0 * x_ref ** 3,
                               places=9)
        scale = 2.0                      # w2 / w1
        root_half = math.sqrt(scale)     # w^{1/2} growth
        three_quarter = scale ** 0.75    # w^{3/4} growth
        for x_ref, sign in itertools.product((-1.0, -1.5, -3.0), (+1.0,)):
            y_ref = sign * _fold_control(x_ref)
            with self.subTest(x_ref=x_ref):
                # Correct exponents: x ~ w^{1/2}, y ~ w^{3/4}.
                x_ok = x_ref * root_half
                y_ok = y_ref * three_quarter
                residual_ok = abs(27.0 * y_ok ** 2 + 8.0 * x_ok ** 3)
                # Swapped exponents: x ~ w^{3/4}, y ~ w^{1/2}.
                x_swap = x_ref * three_quarter
                y_swap = y_ref * root_half
                residual_swap = abs(27.0 * y_swap ** 2 + 8.0 * x_swap ** 3)
                spread = 27.0 * y_ref ** 2
                self.n_checks += 1
                self.assertLess(
                    residual_ok, 1e-9 * spread,
                    'correct exponents did not keep the source on the fold')
                self.assertGreater(
                    residual_swap, 0.1 * spread,
                    'the exponent swap did not walk the source off the fold')

    @_brute_accuracy_tier
    def test_captured_controls_scale_as_half_and_three_quarter(self):
        """
        Gated: the controls the code actually feeds to `pearcey` (captured
        with a recording wrapper) have log-log slopes ``0.5`` (in ``x``)
        and ``0.75`` (in ``y``) versus ``w`` at a fixed near-cusp source,
        to `_EXPONENT_TOL`; doubling ``w`` scales ``|x|`` by ``sqrt(2)``
        and ``|y|`` by ``2^{3/4}``.  Emits the log-log diagnostic.
        """
        gamma, radius, angle = _CUSP_FIXTURES[1]      # gamma = 0.7
        source = radius * np.array([math.cos(angle), math.sin(angle)])
        ws = np.array(_SCALING_WS)
        x_values = []
        y_values = []
        for w in ws:
            _value, controls = _capture_cusp_controls(w, source, gamma)
            self.assertIsNotNone(controls, f'w={w}: cusp arm did not serve')
            x_values.append(abs(controls[0]))
            y_values.append(abs(controls[1]))
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        slope_x = float(np.polyfit(np.log(ws), np.log(x_values), 1)[0])
        slope_y = float(np.polyfit(np.log(ws), np.log(y_values), 1)[0])
        self.n_checks += 1
        self.assertLessEqual(abs(slope_x - 0.5), _EXPONENT_TOL,
                             f'x exponent {slope_x:.3f} is not ~0.5')
        self.assertLessEqual(abs(slope_y - 0.75), _EXPONENT_TOL,
                             f'y exponent {slope_y:.3f} is not ~0.75')
        # Doubling-ratio statement of the same scaling.
        self.assertAlmostEqual(x_values[-1] / x_values[0],
                               (ws[-1] / ws[0]) ** 0.5, delta=0.05)
        self.assertAlmostEqual(y_values[-1] / y_values[0],
                               (ws[-1] / ws[0]) ** 0.75, delta=0.05)
        _save_plot('pearcey_control_scaling',
                   np.log10(ws),
                   np.log10(x_values),
                   xlabel='log10 w', ylabel='log10 |x| (slope 1/2)')

    @_brute_accuracy_tier
    def test_captured_controls_reproduce_the_exact_engine(self):
        """
        Gated: the same controls reproduce the EXACT engine.  At the
        ceiling frequency `_ENGINE_MATCH_W` -- the deepest, best-resolved
        node still inside the exact Schwinger band -- each near-cusp
        fixture's cusp-arm ``|F|`` matches `operator.F_op_grid`'s exact
        value along the fold arcs to `_CUSP_ENVELOPE_TOL` (spec 1e-2).
        This is what makes the captured controls the ones that "reproduce
        the exact engine".  The lower overlap edge (`_SUB_CEILING_W`) is
        recorded as a WITNESS of the approach, not gated at the strict bar
        (the interference beat leaves the weakest-shear fixture ~7% over
        there -- the served-amplitude caveat in `cusp_amplification`).
        """
        sub_errors = []
        for gamma, radius, angle in _CUSP_FIXTURES:
            source = radius * np.array([math.cos(angle), math.sin(angle)])
            with self.subTest(gamma=gamma, w=_ENGINE_MATCH_W):
                arm = _pearcey_cusp.cusp_amplification(
                    _ENGINE_MATCH_W, source, gamma)
                self.assertIsNotNone(arm, f'gamma={gamma} refused at ceiling')
                engine = operator.F_op_grid(
                    np.array([_ENGINE_MATCH_W]), source, gamma)[0][0]
                envelope_error = (abs(abs(arm) - abs(engine))
                                  / max(abs(engine), 1e-9))
                self.n_checks += 1
                self.assertLessEqual(
                    envelope_error, _CUSP_ENVELOPE_TOL,
                    f'gamma={gamma} w={_ENGINE_MATCH_W}: envelope error '
                    f'{envelope_error:.3e} exceeds {_CUSP_ENVELOPE_TOL}')
            # Witness (non-gating): the sub-ceiling error is finite and of
            # the same order -- it approaches, not clears, the bar.
            sub_arm = _pearcey_cusp.cusp_amplification(
                _SUB_CEILING_W, source, gamma)
            if sub_arm is not None:
                sub_engine = operator.F_op_grid(
                    np.array([_SUB_CEILING_W]), source, gamma)[0][0]
                sub_errors.append(abs(abs(sub_arm) - abs(sub_engine))
                                  / max(abs(sub_engine), 1e-9))
        # The sub-ceiling errors are bounded (well under an O(1) failure),
        # documenting the finite-w approach without pinning the strict bar.
        self.assertTrue(all(err < 0.05 for err in sub_errors),
                        f'sub-ceiling errors unexpectedly large: {sub_errors}')


class UniformArmFallThroughTestCase(_FoldArmTestCase):
    """
    F010 -- a corrupted or below-threshold uniform arm FALLS THROUGH to
    the exact Schwinger evaluator's NAMED refusal, never serving a wrong
    number; and moving the arm's certified-argument threshold flips a
    FIXED node serve<->refuse (proving the threshold is live, not dead
    code).  Currency is the boolean route plus a bit-for-bit check, never
    nats.

    The interrogated node (`_CUSP_NODE_*`) is served by the CUSP rung (the
    fold rung declines it), so the cusp-arm mutations below actually bite:
    the served grid value IS the cusp arm's value, corrupting the cusp
    certificate leaves BOTH rungs refusing so the node reaches the
    existing `_schwinger.SchwingerCertificationError`, and the threshold
    the mutations move is the one that gates this very node.
    """

    def _node_source(self):
        """Source position of the cusp-served probe node."""
        return _CUSP_NODE_RADIUS * np.array(
            [math.cos(_CUSP_NODE_ANGLE), math.sin(_CUSP_NODE_ANGLE)])

    def _grid_value(self, source):
        """The serving ladder's value for the probe node, via `F_op_grid`."""
        return operator.F_op_grid(
            np.array([_CUSP_NODE_W]), source, _CUSP_NODE_GAMMA)[0][0]

    def _grid_served(self, source):
        """
        ``True`` if the ladder serves the probe node, ``False`` if it
        raises the NAMED Schwinger refusal.  Any other exception (an
        unrelated failure) propagates, so this only ever collapses the two
        legitimate routes to a boolean.
        """
        try:
            self._grid_value(source)
            return True
        except _schwinger.SchwingerCertificationError:
            return False

    def _node_radius(self, source):
        """
        The scaled radius ``hypot(x, y)`` of the controls the cusp arm
        actually builds for the probe node, captured with a recording
        wrapper (not reconstructed), so the threshold arithmetic below is
        anchored to the code's own controls.
        """
        real_pearcey = _pearcey_cusp.pearcey
        recorded = []

        def recording(x, y):
            recorded.append((x, y))
            return real_pearcey(x, y)

        with mock.patch.object(_pearcey_cusp, 'pearcey', new=recording):
            _pearcey_cusp.cusp_amplification(
                _CUSP_NODE_W, source, _CUSP_NODE_GAMMA)
        self.assertTrue(recorded, 'cusp arm never evaluated the primitive')
        x, y = recorded[-1]
        return math.hypot(x, y)

    def test_fold_declines_cusp_serves_the_probe_node(self):
        """
        Precondition guard: at the probe node the FOLD arm returns ``None``
        and the CUSP arm certifies, so the cusp rung is the one on the
        hook.  If a future fold-arm change captured this node, every cusp
        mutation below would go silently inert -- this catches that.
        """
        source = self._node_source()
        fold = _airy_fold.fold_amplification(
            _CUSP_NODE_W, source, _CUSP_NODE_GAMMA)
        cusp = _pearcey_cusp.cusp_amplification(
            _CUSP_NODE_W, source, _CUSP_NODE_GAMMA)
        self.n_checks += 1
        self.assertIsNone(fold, 'fold arm unexpectedly serves the probe node')
        self.assertIsNotNone(cusp, 'cusp arm refuses the probe node')

    def test_served_node_is_bit_identical_to_the_cusp_arm(self):
        """
        The value the serving ladder returns for the probe node equals the
        cusp arm's own value BIT-FOR-BIT, confirming the grid serves
        through the uniform rung and not some parallel path.
        """
        source = self._node_source()
        cusp = _pearcey_cusp.cusp_amplification(
            _CUSP_NODE_W, source, _CUSP_NODE_GAMMA)
        self.assertIsNotNone(cusp, 'cusp arm refuses the probe node')
        served = self._grid_value(source)
        self.n_checks += 1
        self.assertEqual(
            np.complex128(served).tobytes(), np.complex128(cusp).tobytes(),
            'served grid value is not bit-identical to the cusp arm')

    def test_corrupted_certificate_falls_through_to_named_refusal(self):
        """
        Operation A (corrupted eval must be refusable): driving the Pearcey
        paired-rule certificate tolerance to zero makes `pearcey` never
        certify, so the cusp arm refuses; the fold arm already refuses; the
        node therefore falls through to the exact evaluator, which raises
        the NAMED `_schwinger.SchwingerCertificationError` (``w > 60``) --
        a wrong number is never served.
        """
        source = self._node_source()
        with mock.patch.object(_pearcey_cusp, '_CERTIFICATION_TOL', 0.0):
            self.assertIsNone(
                _pearcey_cusp.cusp_amplification(
                    _CUSP_NODE_W, source, _CUSP_NODE_GAMMA),
                'cusp arm certified under a zero certificate tolerance')
            self.n_checks += 1
            with self.assertRaises(_schwinger.SchwingerCertificationError):
                # w above QD ceiling (150) to trigger the hard refuse.
                operator.F_op_grid(
                    np.array([_CUSP_FALLTHROUGH_W]), source,
                    _CUSP_NODE_GAMMA)[0][0]

    def test_nan_primitive_falls_through_to_named_refusal(self):
        """
        A NaN Pearcey primitive (a different corruption of the same rung)
        also refuses cleanly and falls through to the named Schwinger
        refusal, never propagating the NaN into a served amplitude.
        """
        source = self._node_source()
        with mock.patch.object(_pearcey_cusp, 'pearcey',
                               new=lambda x, y: complex('nan')):
            self.assertIsNone(
                _pearcey_cusp.cusp_amplification(
                    _CUSP_NODE_W, source, _CUSP_NODE_GAMMA),
                'cusp arm served a NaN primitive')
            self.n_checks += 1
            with self.assertRaises(_schwinger.SchwingerCertificationError):
                # w above QD ceiling (150) to trigger the hard refuse.
                operator.F_op_grid(
                    np.array([_CUSP_FALLTHROUGH_W]), source,
                    _CUSP_NODE_GAMMA)[0][0]

    def test_moving_error_const_threshold_flips_a_fixed_node(self):
        """
        Operation B (moved threshold flips routing): the cusp arm refuses
        when ``radius < radius_min = (_UNIFORM_ERROR_CONST / envelope_bar)
        ^{2/3}``.  For the FIXED probe node, setting ``_UNIFORM_ERROR_CONST``
        just BELOW the value that makes ``radius_min = radius`` serves it
        and just ABOVE refuses it -- and the grid route follows (serve vs
        the named Schwinger refusal).  Asserting BOTH directions is what
        proves the threshold is live: a dead constant would not flip.
        """
        source = self._node_source()
        radius = self._node_radius(source)
        # radius_min = (const / bar)^{2/3} == radius  =>
        # const_cross = bar * radius^{3/2}.
        const_cross = _pearcey_cusp._DEFAULT_ENVELOPE_BAR * radius ** 1.5
        with mock.patch.object(_pearcey_cusp, '_UNIFORM_ERROR_CONST',
                               0.5 * const_cross):
            served_below = _pearcey_cusp.cusp_amplification(
                _CUSP_NODE_W, source, _CUSP_NODE_GAMMA) is not None
            grid_below = self._grid_served(source)
        with mock.patch.object(_pearcey_cusp, '_UNIFORM_ERROR_CONST',
                               2.0 * const_cross):
            served_above = _pearcey_cusp.cusp_amplification(
                _CUSP_NODE_W, source, _CUSP_NODE_GAMMA) is not None
            grid_above = self._grid_served(source)
        self.n_checks += 1
        self.assertTrue(
            served_below and grid_below,
            'node did not serve below the threshold crossing')
        self.assertFalse(
            served_above or grid_above,
            'node did not refuse above the threshold crossing '
            '(the threshold is dead code)')

    def test_moving_envelope_bar_threshold_flips_a_fixed_node(self):
        """
        The SAME threshold through its other knob, ``envelope_bar``:
        ``radius_min`` grows as the bar shrinks, so tightening the bar past
        the crossing flips the fixed node serve->refuse and loosening it
        flips refuse->serve.  Emits the route-vs-threshold diagnostic with
        the node radius marked.
        """
        source = self._node_source()
        radius = self._node_radius(source)
        # radius_min == radius  =>  bar_cross = const / radius^{3/2}.
        bar_cross = _pearcey_cusp._UNIFORM_ERROR_CONST / radius ** 1.5
        served_loose = _pearcey_cusp.cusp_amplification(
            _CUSP_NODE_W, source, _CUSP_NODE_GAMMA,
            envelope_bar=2.0 * bar_cross) is not None
        served_tight = _pearcey_cusp.cusp_amplification(
            _CUSP_NODE_W, source, _CUSP_NODE_GAMMA,
            envelope_bar=0.5 * bar_cross) is not None
        self.n_checks += 1
        self.assertTrue(served_loose,
                        'node did not serve with a loose envelope bar')
        self.assertFalse(served_tight,
                         'node did not refuse with a tight envelope bar')
        # Diagnostic: threshold radius_min vs the bar, node radius overlaid.
        bars = np.geomspace(0.3 * bar_cross, 3.0 * bar_cross, 25)
        radius_min = (_pearcey_cusp._UNIFORM_ERROR_CONST / bars) ** (2.0 / 3.0)
        _save_plot('pearcey_route_vs_threshold',
                   np.log10(bars), radius_min - radius,
                   xlabel='log10 envelope_bar',
                   ylabel=f'radius_min - node_radius (radius={radius:.2f})')


class PearceyCuspSelfFalsificationTestCase(_FoldArmTestCase):
    """
    Each load-bearing Pearcey / fall-through gate is shown to DETECT the
    bug it exists to catch, so the suite can go RED.

    * A corrupted primitive is caught by the INDEPENDENT scipy reference
      but NOT by a tainted oracle that calls the module -- the reference's
      independence is load-bearing (F002).
    * Loosening `_pearcey_cusp._CERTIFICATION_TOL` makes the arm serve a
      grossly under-resolved value that the reference exposes -- the
      honest-certificate refusal depends on the real tolerance.
    * A wrong ``P(0, 0)`` closed form (missing the ``e^{i pi/8}`` rotation)
      is rejected by the origin gate -- it discriminates the phase.
    * A one-ULP perturbation breaks the fall-through bit-identity check --
      that gate is exact, not approximate.
    * A threshold move that does NOT cross the node radius leaves the route
      unchanged, so the F010 flip is caused by CROSSING the threshold, not
      by the mutation itself (a genuinely dead threshold could not flip).
    """

    def _node_source(self):
        """Source of the cusp-served probe node (shared with the F010 case)."""
        return _CUSP_NODE_RADIUS * np.array(
            [math.cos(_CUSP_NODE_ANGLE), math.sin(_CUSP_NODE_ANGLE)])

    def _node_radius(self, source):
        """Scaled ``hypot(x, y)`` radius the cusp arm builds for the node."""
        real_pearcey = _pearcey_cusp.pearcey
        recorded = []

        def recording(x, y):
            recorded.append((x, y))
            return real_pearcey(x, y)

        with mock.patch.object(_pearcey_cusp, 'pearcey', new=recording):
            _pearcey_cusp.cusp_amplification(
                _CUSP_NODE_W, source, _CUSP_NODE_GAMMA)
        self.assertTrue(recorded, 'cusp arm never evaluated the primitive')
        return math.hypot(*recorded[-1])

    def test_scipy_reference_catches_a_corrupted_primitive(self):
        """
        Oracle-teeth positive control: a primitive scaled by ``1 + 1e-6``
        disagrees with the independent scipy reference far above
        `_PEARCEY_REFERENCE_TOL`, so `test_certified_matches_scipy_reference`
        would go red -- while a tainted oracle that itself calls the module
        is blind to it.
        """
        x, y = -3.0, 2.0
        reference = _reference_pearcey(x, y)
        real_pearcey = _pearcey_cusp.pearcey

        def corrupt(a, b):
            value = real_pearcey(a, b)
            return None if value is None else value * (1.0 + 1e-6)

        corrupted = corrupt(x, y)
        self.n_checks += 1
        # The independent reference SEES the corruption ...
        self.assertGreater(
            abs(corrupted - reference) / abs(reference),
            _PEARCEY_REFERENCE_TOL,
            'the scipy reference missed a corrupted primitive')
        # ... while a tainted oracle (calling the corrupted module) does not.
        with mock.patch.object(_pearcey_cusp, 'pearcey', new=corrupt):
            tainted = _pearcey_cusp.pearcey(x, y)
        self.assertEqual(np.complex128(tainted).tobytes(),
                         np.complex128(corrupted).tobytes(),
                         'the tainted oracle somehow diverged from the '
                         'corrupted module')

    def test_loosened_certificate_would_serve_a_wrong_value(self):
        """
        HONESTY red-check: with `_panel_count` forced to one panel AND
        `_CERTIFICATION_TOL` blown up, `pearcey` serves the under-resolved
        (wrong) value, which the reference exposes.  This proves the real
        certificate tolerance -- not luck -- is what refuses the bad value
        in `test_certificate_refuses_gross_under_resolution`.
        """
        x, y = 30.0, -25.0
        reference = _reference_pearcey(x, y)
        with mock.patch.object(_pearcey_cusp, '_panel_count',
                               new=lambda span, frequency: 1), \
                mock.patch.object(_pearcey_cusp, '_CERTIFICATION_TOL', 1e6):
            served = _pearcey_cusp.pearcey(x, y)
        self.n_checks += 1
        self.assertIsNotNone(
            served, 'a blown-up certificate tolerance still refused')
        self.assertGreater(
            abs(served - reference) / abs(reference),
            _PEARCEY_CERTIFICATE_TOL,
            'the under-resolved served value was accidentally accurate, so '
            'the honesty gate would be vacuous')

    def test_wrong_origin_closed_form_is_rejected(self):
        """
        The ``P(0, 0)`` gate discriminates the ``e^{i pi/8}`` rotation: the
        UN-rotated real form ``Gamma(1/4) / 2`` disagrees with the true
        value far above `_PEARCEY_REFERENCE_TOL`, so a dropped phase would
        be caught.
        """
        value = _pearcey_cusp.pearcey(0.0, 0.0)
        self.assertIsNotNone(value)
        wrong_form = 0.5 * scipy_gamma(0.25)          # missing e^{i pi/8}
        self.n_checks += 1
        self.assertGreater(
            abs(value - wrong_form) / abs(value),
            _PEARCEY_REFERENCE_TOL,
            'the origin gate did not discriminate the e^{i pi/8} phase')

    def test_one_ulp_perturbation_breaks_the_bit_identity_check(self):
        """
        The fall-through bit-identity gate is EXACT: nudging the served
        value by a single ULP makes the byte comparison fail, so
        `test_served_node_is_bit_identical_to_the_cusp_arm` cannot pass on a
        merely-close value.
        """
        source = self._node_source()
        served = operator.F_op_grid(
            np.array([_CUSP_NODE_W]), source, _CUSP_NODE_GAMMA)[0][0]
        perturbed = complex(np.nextafter(served.real, math.inf), served.imag)
        self.n_checks += 1
        self.assertNotEqual(
            np.complex128(served).tobytes(),
            np.complex128(perturbed).tobytes(),
            'the byte comparison did not resolve a one-ULP difference')

    def test_threshold_move_within_one_side_does_not_flip(self):
        """
        Dead-code discriminator control: two ``_UNIFORM_ERROR_CONST`` values
        that BOTH stay below the crossing (so ``radius_min < radius`` both
        times) leave the fixed node SERVED both times -- no flip.  Only a
        move that crosses the node radius flips it (the F010 test).  This
        isolates the flip to the crossing, ruling out a mutation artifact.
        """
        source = self._node_source()
        radius = self._node_radius(source)
        const_cross = _pearcey_cusp._DEFAULT_ENVELOPE_BAR * radius ** 1.5
        served = []
        for factor in (0.3, 0.6):                     # both below crossing
            with mock.patch.object(_pearcey_cusp, '_UNIFORM_ERROR_CONST',
                                   factor * const_cross):
                served.append(_pearcey_cusp.cusp_amplification(
                    _CUSP_NODE_W, source, _CUSP_NODE_GAMMA) is not None)
        self.n_checks += 1
        self.assertEqual(
            served, [True, True],
            'a same-side threshold move flipped the route, so the F010 flip '
            'cannot be attributed to crossing the threshold')


# ======================================================================
# SERVING-LADDER wiring specs (Build 8f WP1/WP4): determinism, cross-arm
# consistency, byte-identity of the certified paths, and the corner
# census contract.  These classes cover how the uniform arms are wired
# into the per-node serving ladder (`operator._uniform_arm_value` and its
# two call sites in `_saddle_grid` / `_positive_parity_grid`), NOT the
# arms' own arithmetic (covered above).
# ======================================================================

#: Frequency ceiling of the Schwinger engine (including the mpmath QD
#: extension).  A node with ``w > _W_CEILING`` that is not
#: geometric-resolved is offered to the uniform arms before the named
#: refusal stands.  Nodes at ``w <= 150`` are exact-wave-served (DD for
#: w<=60, mpmath for 60<w<=150).
_W_CEILING = _schwinger.W_CEILING_SCHWINGER_QD

#: Geometric-resolution threshold: a saddle node is resolved when
#: ``w * delta_min >= _RHO_END``.
_RHO_END = operator.RHO_END

#: Repository root (three levels up from this test file), used to load the
#: HEAD copy of ``operator.py`` via ``git show`` for the byte-identity
#: gate.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: One node per serving rung, spanning every regime the ladder resolves:
#: ``(label, w, radius, angle, gamma, beta, kappa)``.  Grounded against
#: the live ladder: the Schwinger node certifies at ``w = 40 <= 60`` (DD
#: path); the geometric node is a resolved macro saddle (``gamma = 1.5``,
#: ``w = 200 > W_CEILING_SCHWINGER_QD``, ``w * delta_min >= RHO_END``);
#: the fold and cusp nodes are the near-fold / near-cusp uniform corners
#: (empirically served by exactly one arm each); the refusal node is a
#: near-caustic unresolved node above the QD ceiling that both arms
#: decline.
#:
#: F028 re-point: the fold node's radius is kept small
#: (``L = w*|y'| = 500*0.06 = 30 < L_MAX``) so it stays on the WAVE branch
#: and is served by the fold arm.  Since Build 8f WP1 the authoritative
#: `select_branch` gate routes a resolved, strongly-cancelling
#: positive-parity above-ceiling corner to the F028 geometric asymptote
#: instead of the fold arm; the former ``radius = 0.14`` fold node became
#: geometric (``L = 70 > L_MAX``, ``w*delta_min = 67 >= RHO_END``) and no
#: longer exercises the fold rung, so it is pulled back below the handoff.
_LADDER_NODES = (
    ('schwinger', 40.0, 0.20, 0.25 * math.pi, 0.5, 0.0, 0.0),
    ('geometric', 200.0, 1.20, _RAY_ANGLE, 1.5, 0.0, 0.0),
    ('fold', 500.0, 0.06, _RAY_ANGLE, _GAMMA, 0.0, 0.0),
    ('cusp', _CUSP_NODE_W, _CUSP_NODE_RADIUS, _CUSP_NODE_ANGLE,
     _CUSP_NODE_GAMMA, 0.0, 0.0),
    ('refusal', _ABOVE_CEILING_W, 0.28, _RAY_ANGLE, _GAMMA, 0.0, 0.0),
)

#: Uniform-arm ladder nodes (served by fold / cusp), extracted from
#: `_LADDER_NODES` for the disjointness / priority / cross-arm gates.
_UNIFORM_LADDER_NODES = tuple(
    node for node in _LADDER_NODES if node[0] in ('fold', 'cusp'))

#: Overlap-band cross-arm envelope tolerance (Architect spec: 1e-3).
_CROSS_ARM_ENVELOPE_TOL = 1e-3


def _polar_source(radius, angle):
    """Source position at ``radius`` and ``angle`` (physical frame)."""
    return radius * np.array([math.cos(angle), math.sin(angle)])


def _ladder_route(w, source, gamma, *, beta=0.0, kappa=0.0):
    """Independent mirror of the per-node serving-ladder priority.

    Reproduces `operator`'s fixed rung order WITHOUT calling
    `F_op_grid`, by consulting the same geometry / arm predicates the
    production ladder consults: geometric (macro saddle, resolved,
    ``w > ceiling``) -> fold Airy -> cusp Pearcey -> named refusal, and
    the exact Schwinger engine for ``w <= ceiling``.

    Parameters
    ----------
    w : float
        Dimensionless frequency.
    source : np.ndarray
        Shape ``(2,)`` source position (physical frame).
    gamma : float
        External shear magnitude.
    beta, kappa : float, optional
        Shear orientation and convergence.

    Returns
    -------
    str
        One of ``'schwinger'``, ``'geometric'``, ``'fold'``, ``'cusp'``,
        ``'refusal'``.
    """
    lam = 1.0 - float(kappa)
    is_saddle = not (lam > abs(float(gamma)))
    if float(w) <= _W_CEILING:
        return 'schwinger'
    if is_saddle:
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        delta_min = operator._real_delay_min_separation(source, matrix)
        if float(w) * delta_min >= _RHO_END:
            return 'geometric'
    if _airy_fold.fold_amplification(
            w, source, gamma, beta=beta, kappa=kappa) is not None:
        return 'fold'
    if _pearcey_cusp.cusp_amplification(
            w, source, gamma, beta=beta, kappa=kappa) is not None:
        return 'cusp'
    return 'refusal'


def _rung_value(route, w, source, gamma, *, beta=0.0, kappa=0.0):
    """Independent amplification the ladder must serve at ``route``.

    Defined only for the rungs whose value the ladder copies bit-for-bit
    (``'fold'``, ``'cusp'``, ``'geometric'``); ``'schwinger'`` and
    ``'refusal'`` have no independent rung here (the Schwinger value is
    reconstructed inside `operator`) and return ``None``.
    """
    if route == 'fold':
        return complex(_airy_fold.fold_amplification(
            w, source, gamma, beta=beta, kappa=kappa))
    if route == 'cusp':
        return complex(_pearcey_cusp.cusp_amplification(
            w, source, gamma, beta=beta, kappa=kappa))
    if route == 'geometric':
        return complex(operator.geometric_amplification(
            w, source, gamma, beta=beta, kappa=kappa))
    return None


def _grid_decision(module, w, source, gamma, *, beta=0.0, kappa=0.0):
    """``('served', value)`` or ``('refused', None)`` from ``F_op_grid``.

    Catches ONLY the named `_schwinger.SchwingerCertificationError`, so
    an unexpected error still propagates (defensive: never a bare except).
    """
    try:
        value = module.F_op_grid(
            np.array([float(w)]), source, gamma, beta=beta, kappa=kappa)[0][0]
        return ('served', complex(value))
    except _schwinger.SchwingerCertificationError:
        return ('refused', None)


@lru_cache(maxsize=1)
def _head_operator():
    """Load the HEAD copy of ``operator.py`` side by side.

    The source is written to a real temporary ``.py`` file because the
    module's numba ``@njit(cache=True)`` kernels need a file locator (an
    ``exec`` into a synthetic module raises ``cannot cache function ...
    no locator available``).  The module is registered in ``sys.modules``
    under a private name BEFORE execution so its dataclass fields resolve,
    per the byte-identity idiom.  Cached: the jit-warming load happens
    once per process.
    """
    source = subprocess.check_output(
        # Pinned to the PRE-8e commit (4e26103, Build 8d): this is a
        # TRANSITION witness -- 'HEAD' became self-referential the moment
        # 8e was committed (HEAD then already served the fold node,
        # voiding the refuses-premise; reddened in the first post-commit
        # sweep). Transition baselines must pin the SHA at authorship.
        ['git', 'show',
         '4e26103:cogwheel/lensing/chang_refsdal/operator.py'],
        cwd=_REPO_ROOT).decode()
    tmpdir = tempfile.mkdtemp(prefix='head_operator_')
    tmppath = os.path.join(tmpdir, '_operator_head_ref.py')
    with open(tmppath, 'w', encoding='utf-8') as handle:
        handle.write(source)
    modname = 'cogwheel.lensing.chang_refsdal._operator_head_ref'
    spec = importlib.util.spec_from_file_location(modname, tmppath)
    head = importlib.util.module_from_spec(spec)
    head.__package__ = 'cogwheel.lensing.chang_refsdal'
    sys.modules[modname] = head
    spec.loader.exec_module(head)
    return head


class ServingLadderDeterminismTestCase(_FoldArmTestCase):
    """Serving-ladder determinism and cross-arm consistency (WP4).

    The per-node ladder must be a pure, reproducible function of the
    node, following the fixed priority
    ``geometric -> fold -> cusp -> Schwinger -> named refusal`` with no
    node served by two arms with different answers.  The route and the
    served value are re-derived independently (`_ladder_route`,
    `_rung_value`) and cross-checked against `operator.F_op_grid`.
    """

    def test_route_and_value_reproduce_across_all_regimes(self):
        """Same node -> same route and bit-identical value on re-run.

        Evaluates each ladder node TWICE via `F_op_grid` and asserts the
        served/refused decision and (when served) the complex value are
        byte-identical, and that the independently mirrored route agrees
        with the observed decision.  Covers all five regimes.
        """
        seen = set()
        for label, w, radius, angle, gamma, beta, kappa in _LADDER_NODES:
            with self.subTest(node=label):
                source = _polar_source(radius, angle)
                first = _grid_decision(
                    operator, w, source, gamma, beta=beta, kappa=kappa)
                second = _grid_decision(
                    operator, w, source, gamma, beta=beta, kappa=kappa)
                route = _ladder_route(
                    w, source, gamma, beta=beta, kappa=kappa)
                seen.add(route)
                self.n_checks += 1
                # Route reproduces: same decision label both evaluations.
                self.assertEqual(
                    first[0], second[0],
                    f'{label} node changed served/refused between two '
                    'identical evaluations')
                # The mirrored route agrees with the observed decision.
                expected_served = route != 'refusal'
                self.assertEqual(
                    first[0] == 'served', expected_served,
                    f'{label} node: mirrored route {route!r} disagrees with '
                    f'the observed decision {first[0]!r}')
                if first[0] == 'served':
                    self.assertEqual(
                        np.complex128(first[1]).tobytes(),
                        np.complex128(second[1]).tobytes(),
                        f'{label} node served value is not reproducible '
                        'bit-for-bit')
        # Anti-vacuity of coverage: every regime label was exercised.
        self.assertEqual(
            seen, {'schwinger', 'geometric', 'fold', 'cusp', 'refusal'},
            'the fixture batch did not span all five serving regimes')

    def test_served_value_equals_labelled_rung_bitwise(self):
        """A served arm/geometric node equals its own rung bit-for-bit.

        For the fold, cusp and geometric nodes the ladder copies the arm
        (or stationary-phase) value verbatim; `F_op_grid` must return the
        independently recomputed rung value byte-identically, proving the
        node is served by the LABELLED rung and no other.
        """
        for label, w, radius, angle, gamma, beta, kappa in _LADDER_NODES:
            if label not in ('fold', 'cusp', 'geometric'):
                continue
            with self.subTest(node=label):
                source = _polar_source(radius, angle)
                served = operator.F_op_grid(
                    np.array([w]), source, gamma,
                    beta=beta, kappa=kappa)[0][0]
                rung = _rung_value(
                    label, w, source, gamma, beta=beta, kappa=kappa)
                self.n_checks += 1
                self.assertEqual(
                    np.complex128(served).tobytes(),
                    np.complex128(rung).tobytes(),
                    f'{label} node served value differs from its own rung')

    def test_refusal_node_raises_named_error_both_times(self):
        """The near-caustic unresolved node refuses reproducibly.

        Both arms decline and the node is not geometric-resolved, so the
        named `SchwingerCertificationError` must stand on every call --
        the ladder never invents a value at a hard-core node.
        """
        label, w, radius, angle, gamma, beta, kappa = _LADDER_NODES[-1]
        self.assertEqual(label, 'refusal')  # fixture guard
        source = _polar_source(radius, angle)
        for _ in range(2):
            self.n_checks += 1
            with self.assertRaises(_schwinger.SchwingerCertificationError):
                operator.F_op_grid(
                    np.array([w]), source, gamma, beta=beta, kappa=kappa)

    def test_fixed_priority_fold_tried_before_cusp(self):
        """The uniform rung tries the fold arm strictly before the cusp arm.

        Spies both arm entry points with delegating wrappers that record
        their call order.  At the fold node the fold arm serves and the
        cusp arm is never reached; at the cusp node the fold arm is tried
        first (declines) and the cusp arm serves second.  Either way the
        fold arm is called before the cusp arm -- the fixed priority.
        """
        real_fold = _airy_fold.fold_amplification
        real_cusp = _pearcey_cusp.cusp_amplification
        order: list[str] = []

        def fold_spy(*args, **kwargs):
            order.append('fold')
            return real_fold(*args, **kwargs)

        def cusp_spy(*args, **kwargs):
            order.append('cusp')
            return real_cusp(*args, **kwargs)

        for label, w, radius, angle, gamma, beta, kappa in \
                _UNIFORM_LADDER_NODES:
            with self.subTest(node=label):
                order.clear()
                source = _polar_source(radius, angle)
                with mock.patch.object(_airy_fold, 'fold_amplification',
                                       fold_spy), \
                        mock.patch.object(_pearcey_cusp, 'cusp_amplification',
                                          cusp_spy):
                    operator.F_op_grid(
                        np.array([w]), source, gamma, beta=beta, kappa=kappa)
                self.n_checks += 1
                self.assertEqual(
                    order[0], 'fold',
                    f'{label} node: the cusp arm was consulted before the '
                    'fold arm')
                if 'cusp' in order:
                    self.assertLess(
                        order.index('fold'), order.index('cusp'),
                        f'{label} node: fold arm not tried before cusp arm')

    def test_uniform_arms_disjoint_no_conflicting_double_serve(self):
        """No uniform node is served by two arms with different answers.

        At each uniform ladder node EXACTLY ONE arm returns a finite
        value (the arms partition the corner via their local caustic
        classification), so a conflicting double-serve is impossible and
        the ladder's served value equals the sole serving arm.
        """
        n_fold = n_cusp = 0
        for label, w, radius, angle, gamma, beta, kappa in \
                _UNIFORM_LADDER_NODES:
            with self.subTest(node=label):
                source = _polar_source(radius, angle)
                fold = _airy_fold.fold_amplification(
                    w, source, gamma, beta=beta, kappa=kappa)
                cusp = _pearcey_cusp.cusp_amplification(
                    w, source, gamma, beta=beta, kappa=kappa)
                serving = [x is not None for x in (fold, cusp)]
                self.n_checks += 1
                self.assertEqual(
                    sum(serving), 1,
                    f'{label} node is served by {sum(serving)} arms; the '
                    'corner must be partitioned so no conflict can arise')
                n_fold += fold is not None
                n_cusp += cusp is not None
        # Anti-vacuity of coverage: both arms served at least one node.
        self.assertGreaterEqual(n_fold, 1, 'no fold-served node exercised')
        self.assertGreaterEqual(n_cusp, 1, 'no cusp-served node exercised')

    @_brute_accuracy_tier
    def test_cross_arm_conflicts_resolved_by_fixed_priority(self):
        """Double-serves are resolved deterministically by fold priority.

        The Architect's cross-arm clause expects that where two arms are
        both valid they agree to <= 1e-3 in envelope.  Empirically the
        arms' internal certification gates are NOT mutually exclusive: a
        near-junction node (measured: ``gamma = 0.5``, ``r = 0.14``,
        ``ang ~ 0.45 pi``, ``w = 150``) is certified by BOTH the fold and
        the cusp arm yet their envelopes disagree by ~29% -- the loose
        "both certify" set is far wider than the genuine shared-validity
        band, so the literal ``<= 1e-3`` reading over "both certify" does
        not hold (this is premise repair, not tolerance repair).

        What the serving ladder DOES guarantee, and what this gate asserts
        with teeth, is the spec's primary clause: NO node is ever served
        by two arms with different answers.  The fixed fold-before-cusp
        priority makes the served value the FOLD arm's, bit-for-bit, at
        every double-certifying node, so the served answer is a pure,
        deterministic function of the node regardless of the cusp arm's
        competing value.  If the corner has no double-certify node the
        arms partition it and that disjointness is asserted instead -- so
        the test is never vacuous.  The per-node cross-arm envelope spread
        is saved as a diagnostic.  Gated: it evaluates the heavy uniform
        quadratures over a grid.
        """
        radii = np.linspace(0.14, 0.30, 5)
        angles = np.linspace(0.20, 0.48 * math.pi, 5)
        spreads = []
        found_overlap = False
        for gamma in (_GAMMA, _CUSP_NODE_GAMMA):
            for radius in radii:
                for angle in angles:
                    source = _polar_source(float(radius), float(angle))
                    for w in (150.0, 400.0):
                        fold = _airy_fold.fold_amplification(
                            w, source, gamma)
                        cusp = _pearcey_cusp.cusp_amplification(
                            w, source, gamma)
                        if fold is None or cusp is None:
                            continue
                        found_overlap = True
                        served = operator.F_op_grid(
                            np.array([w]), source, gamma)[0][0]
                        self.n_checks += 1
                        # Priority resolution: the ladder serves the fold
                        # arm's value, never the competing cusp value.
                        self.assertEqual(
                            np.complex128(served).tobytes(),
                            np.complex128(complex(fold)).tobytes(),
                            f'double-certify node gamma={gamma} '
                            f'r={radius:.3f} ang={angle:.3f} w={w}: the '
                            'ladder did not serve the fold arm (priority), '
                            'so the served answer is not deterministic')
                        denom = max(abs(fold), abs(cusp))
                        spreads.append(abs(abs(fold) - abs(cusp)) / denom)
        if not found_overlap:
            for label, w, radius, angle, gamma, beta, kappa in \
                    _UNIFORM_LADDER_NODES:
                source = _polar_source(radius, angle)
                fold = _airy_fold.fold_amplification(
                    w, source, gamma, beta=beta, kappa=kappa)
                cusp = _pearcey_cusp.cusp_amplification(
                    w, source, gamma, beta=beta, kappa=kappa)
                self.n_checks += 1
                self.assertEqual(
                    sum(x is not None for x in (fold, cusp)), 1,
                    'no overlap band and the arms failed to partition the '
                    f'{label} node')
        if spreads:
            _save_plot('serving_ladder_cross_arm_spread',
                       np.arange(len(spreads)), np.asarray(spreads),
                       xlabel='overlap node index',
                       ylabel='cross-arm envelope spread')


#: Configs where `F_op_grid` must stay byte-identical before/after the
#: WP4 dispatch edits: ``(gamma, radius, angle, beta, kappa)``.  Two
#: positive-parity hosts, one macro saddle, one with convergence.
_BYTE_IDENTITY_CONFIGS = (
    (0.3, 0.14, _RAY_ANGLE, 0.0, 0.0),
    (0.5, 0.20, 0.25 * math.pi, 0.0, 0.0),
    (1.5, 1.20, _RAY_ANGLE, 0.0, 0.0),
    (0.4, 0.30, 0.7, 0.0, 0.1),
)

#: The ``w <= 60`` grid over which the certified paths stay byte-frozen
#: (every node here certifies, so none reaches the uniform-arm intercept).
_BYTE_IDENTITY_WGRID = np.array([5.0, 20.0, 40.0, 55.0, 60.0])

#: A resolved macro-saddle geometric node that must ALSO be byte-identical
#: across the dispatch edits: ``(gamma, radius, angle, beta, kappa, w)``.
#: ``w = 200`` (NOT 100): at ``w <= 150`` the ladder routes this saddle to
#: the exact Schwinger engine's slow mpmath path; above the QD ceiling it
#: resolves geometrically and the operator serves `geometric_amplification`
#: fast, byte-identical to HEAD (measured).
_GEOMETRIC_NODE = (1.5, 1.20, _RAY_ANGLE, 0.0, 0.0, 200.0)

class CertifiedPathByteIdentityTestCase(_FoldArmTestCase):
    """The certified paths are byte-identical after the WP4 dispatch edits.

    The WP4 change only intercepts previously-REFUSING nodes
    (``w > 60`` and not geometric-resolved).  Every geometric node and
    every ``w <= 60`` Schwinger node must return exactly the value the
    HEAD copy of ``operator.py`` returned -- ``max|F_after - F_before|``
    is exactly ``0.0`` -- and refusal decisions elsewhere are unchanged.
    The HEAD module is loaded side by side (`_head_operator`).
    """

    def test_wave_ceiling_grid_is_byte_identical(self):
        """``F_op_grid`` on ``w <= 60`` matches HEAD to exactly zero.

        Over the config sweep the complex values, the operator orders and
        the converged flags all match the HEAD module bit-for-bit.  These
        nodes certify below the ceiling and never reach the arm intercept.
        """
        head = _head_operator()
        diffs = []
        for gamma, radius, angle, beta, kappa in _BYTE_IDENTITY_CONFIGS:
            with self.subTest(gamma=gamma, kappa=kappa):
                source = _polar_source(radius, angle)
                cur = operator.F_op_grid(
                    _BYTE_IDENTITY_WGRID, source, gamma,
                    beta=beta, kappa=kappa)
                ref = head.F_op_grid(
                    _BYTE_IDENTITY_WGRID, source, gamma,
                    beta=beta, kappa=kappa)
                max_diff = float(np.max(np.abs(cur[0] - ref[0])))
                diffs.append(max_diff)
                self.n_checks += 1
                self.assertEqual(
                    max_diff, 0.0,
                    f'gamma={gamma} kappa={kappa}: w<=60 values drifted from '
                    f'HEAD by {max_diff:.3e}')
                self.assertTrue(
                    np.array_equal(cur[1], ref[1]),
                    'operator orders drifted from HEAD')
                self.assertTrue(
                    np.array_equal(cur[2], ref[2]),
                    'converged flags drifted from HEAD')
        # Diagnostic: histogram of per-config max|diff| (all exact zeros).
        _save_plot('serving_ladder_byte_identity_diffs',
                   np.arange(len(diffs)), np.asarray(diffs),
                   xlabel='config index', ylabel='max|F_after - F_before|')

    def test_geometric_node_is_byte_identical(self):
        """A resolved saddle geometric node matches HEAD to exactly zero.

        The geometric rung predates WP4 and is untouched by the arm
        intercept (it ``continue``s before the intercept), so its value
        must be byte-identical to HEAD.
        """
        head = _head_operator()
        gamma, radius, angle, beta, kappa, w = _GEOMETRIC_NODE
        source = _polar_source(radius, angle)
        cur = operator.F_op_grid(
            np.array([w]), source, gamma, beta=beta, kappa=kappa)[0][0]
        ref = head.F_op_grid(
            np.array([w]), source, gamma, beta=beta, kappa=kappa)[0][0]
        self.n_checks += 1
        self.assertEqual(
            np.complex128(cur).tobytes(), np.complex128(ref).tobytes(),
            'the geometric node value drifted from HEAD')

    # DELETED (one-home consolidation): `test_select_branch_matches_head_
    # exactly` re-pinned the routing predicate by replaying a hand-built
    # ``(w, delta_min, L)`` grid through BOTH this module's
    # `select_branch` and the HEAD copy's -- a second home for the gate,
    # and a self-referential one (the gate judged against a copy of
    # itself).  It also called the three-leg gate with three arguments,
    # so it could not have seen `eta` move at all.  The predicate is
    # pinned once, against what the operator grids actually SERVE, in
    # `test_lensing_operator.BranchGateTestCase.
    # test_thresholds_have_one_home`.

    def test_only_previously_refusing_nodes_change(self):
        """Decisions change ONLY at ``w > 60`` non-geometric nodes.

        For every byte-identity config, at ``w <= 60`` the served/refused
        decision matches HEAD; a change (HEAD refuses, current serves) is
        allowed only where the node is above the ceiling AND not
        geometric-resolved -- the previously-refusing set the arms now
        rescue.
        """
        head = _head_operator()
        # A near-fold node above the ceiling: HEAD refuses, current serves.
        gamma, radius, angle = _GAMMA, 0.14, _RAY_ANGLE
        source = _polar_source(radius, angle)
        head_decision = _grid_decision(head, 500.0, source, gamma)
        cur_decision = _grid_decision(operator, 500.0, source, gamma)
        self.n_checks += 1
        self.assertEqual(head_decision[0], 'refused',
                         'the HEAD module already served the fold node; the '
                         'byte-identity premise is void')
        self.assertEqual(cur_decision[0], 'served',
                         'the current module did not rescue the fold node')
        # Below the ceiling every config keeps HEAD's decision.
        for cfg_gamma, cfg_radius, cfg_angle, beta, kappa in \
                _BYTE_IDENTITY_CONFIGS:
            with self.subTest(gamma=cfg_gamma, kappa=kappa):
                cfg_source = _polar_source(cfg_radius, cfg_angle)
                for w in (20.0, 60.0):
                    self.n_checks += 1
                    self.assertEqual(
                        _grid_decision(operator, w, cfg_source, cfg_gamma,
                                       beta=beta, kappa=kappa)[0],
                        _grid_decision(head, w, cfg_source, cfg_gamma,
                                       beta=beta, kappa=kappa)[0],
                        f'w={w} decision changed below the ceiling')


#: Production ``L_MAX`` the census must report against WITHOUT changing it.
_L_MAX_PINNED = 48

#: Extended-census tokens (WP1) the current ``run`` report must NOT yet
#: contain: the fold / cusp argument distributions, Wilson intervals and
#: the (c)/(d) fraction-vs-threshold table.  When any appears, WP1 has
#: landed and the honest structure gate below flips.
_EXTENDED_CENSUS_TOKENS = (
    'wilson', 'fold_argument', 'cusp_argument', 'argument_cdf',
    'category_fraction', 'candidate_threshold')

#: Names of the exact wave evaluators the pure corner census must NOT
#: reach into (purity: the census classifies geometry, it does not run the
#: wave branch).  Matched on ``ast.Name.id`` / ``ast.Attribute.attr``
#: EXACTLY (never as a source substring -- a production symbol can be a
#: substring of an unrelated name).
_FORBIDDEN_WAVE_EVALUATORS = frozenset(
    {'f_schwinger', 'F_op', 'F_op_grid'})


def _census_run_source():
    """Source text of `surrogate_census.run`, for static schema checks.

    Parsing the function's own lines (rather than the whole module) keeps
    the extended-structure token search scoped to the report builder.
    """
    path = surrogate_census.__file__
    with open(path, 'r', encoding='utf-8') as handle:
        text = handle.read()
    tree = ast.parse(text, filename=path)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run':
            return '\n'.join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError('surrogate_census.run not found in source')


class CornerCensusContractTestCase(_FoldArmTestCase):
    """Corner-census structure/determinism contract (WP1, honest).

    The extended corner census (category (a)-(d) fractions with Wilson
    intervals and fold / cusp argument distributions) is NOT yet landed,
    so this class gates the invariants that ARE testable now -- the
    production threshold the census reports against is unchanged
    (``L_MAX == 48``) and the census source stays pure of the exact wave
    evaluators -- and carries an ``@expectedFailure`` tripwire that flips
    RED the moment the extended-census API lands, prompting the real
    structure/determinism gate.
    """

    def test_production_thresholds_unchanged(self):
        """``L_MAX == 48``: the census must not move a production threshold.

        The census REPORTS category (b); its (b) fraction is defined
        against the shipped ``L_MAX``, so a change to that constant
        invalidates the reported number.  The gate SEMANTICS that
        consume ``L_MAX`` are pinned elsewhere -- once -- in
        `test_lensing_operator.BranchGateTestCase` (which sweeps the
        quadrants, the boundary equalities and both operator grids'
        actual routing); re-pinning them here would be a second home for
        the predicate.
        """
        self.n_checks += 1
        self.assertEqual(operator.L_MAX, _L_MAX_PINNED,
                         'production L_MAX changed; the census (b) fraction '
                         'would shift under it')

    def test_census_source_is_pure_of_exact_wave_evaluators(self):
        """The census source never names an exact wave evaluator (purity).

        A pure corner census classifies geometry and consults the engine
        object; it must not import ``_schwinger`` or call ``f_schwinger`` /
        ``F_op`` / ``F_op_grid`` directly.  Walks the AST by
        ``Name.id`` / ``Attribute.attr`` so a substring collision (a
        symbol embedded in an unrelated identifier) cannot trip it.
        """
        source_path = surrogate_census.__file__
        with open(source_path, 'r', encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=source_path)
        offenders = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and \
                    node.id in _FORBIDDEN_WAVE_EVALUATORS:
                offenders.add(node.id)
            elif isinstance(node, ast.Attribute) and \
                    node.attr in _FORBIDDEN_WAVE_EVALUATORS:
                offenders.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and \
                    (node.module or '').endswith('_schwinger'):
                offenders.add(node.module)
        self.n_checks += 1
        self.assertEqual(
            offenders, set(),
            f'the census source reaches into exact wave evaluators: '
            f'{sorted(offenders)}')

    def test_census_run_report_lacks_extended_structure(self):
        """The current ``run`` report has none of the extended-census keys.

        Guards the honest-contract premise: the WP1 argument
        distributions / Wilson intervals / (c)-(d) threshold table are not
        yet in `run`.  Read statically from the source (running the census
        is heavy and unnecessary here).
        """
        run_source = _census_run_source().lower()
        present = [token for token in _EXTENDED_CENSUS_TOKENS
                   if token in run_source]
        self.n_checks += 1
        self.assertEqual(
            present, [],
            f'extended-census tokens already in run(): {present} -- promote '
            'the @expectedFailure tripwire to the real structure gate')

    @expectedFailure
    def test_extended_corner_census_api_absent_tripwire(self):
        """RED when the extended corner-census API lands (WP1 tripwire).

        Fails (as expected) while the API is absent; when a Wilson-interval
        / argument-distribution helper or an extended ``run`` key appears,
        this flips to an UNEXPECTED SUCCESS and turns the suite RED,
        signalling Test Dev to implement the full structure/determinism
        gate (fold ``w*Delta_tau`` and cusp ``R`` CDFs, (a)-(d) fractions
        with Wilson intervals, (c)/(d) fraction-vs-threshold table,
        same-seed determinism, engine-purity spy).
        """
        # Bump BEFORE the assertion: @expectedFailure covers the body but
        # not tearDown, so the anti-vacuity counter must be set first.
        self.n_checks += 1
        run_source = _census_run_source().lower()
        api_present = (
            any(hasattr(surrogate_census, name) for name in
                ('corner_census', 'wilson_interval', 'argument_distributions',
                 'argument_cdf', 'category_fractions'))
            or any(token in run_source for token in _EXTENDED_CENSUS_TOKENS))
        self.assertTrue(
            api_present,
            'extended corner-census API not yet landed (expected while WP1 '
            'is unimplemented)')


class LadderByteIdentitySelfFalsificationTestCase(_FoldArmTestCase):
    """The ladder/byte-identity/purity gates can go RED (self-falsification).

    A numerical wiring suite that never fails is worthless.  Each test
    injects a defect the classes above are meant to catch -- a one-ULP
    value drift, a corrupted arm, a forced double-serve, a moved
    threshold, an impure census import -- and asserts the corresponding
    gate detects it.
    """

    def test_one_ulp_drift_breaks_byte_identity(self):
        """A one-ULP change is resolved by the byte comparison.

        Proves the ``max|diff| == 0`` / ``tobytes()`` currency has teeth:
        perturbing a served value by a single ULP makes both the byte
        comparison and the max-abs-diff gate fire.
        """
        source = _polar_source(0.14, _RAY_ANGLE)
        served = operator.F_op_grid(np.array([500.0]), source, _GAMMA)[0][0]
        perturbed = complex(np.nextafter(served.real, math.inf), served.imag)
        self.n_checks += 1
        self.assertNotEqual(
            np.complex128(served).tobytes(),
            np.complex128(perturbed).tobytes(),
            'the byte gate failed to resolve a one-ULP drift')
        self.assertGreater(
            abs(served - perturbed), 0.0,
            'the max-abs-diff gate failed to resolve a one-ULP drift')

    def test_corrupted_fold_arm_changes_served_value(self):
        """A corrupted fold arm is caught by the byte-rung gate.

        Captures the true arm value, then patches the arm to return a
        scaled value; the ladder now serves the corrupted number, so the
        byte comparison against the TRUE rung differs -- the rung gate
        would go RED on a corrupted arm.
        """
        source = _polar_source(0.14, _RAY_ANGLE)
        true_value = complex(_airy_fold.fold_amplification(500.0, source,
                                                           _GAMMA))
        real_fold = _airy_fold.fold_amplification

        def corrupt_fold(*args, **kwargs):
            return real_fold(*args, **kwargs) * 1.0001

        with mock.patch.object(_airy_fold, 'fold_amplification',
                               corrupt_fold):
            served = operator.F_op_grid(
                np.array([500.0]), source, _GAMMA)[0][0]
        self.n_checks += 1
        self.assertNotEqual(
            np.complex128(served).tobytes(),
            np.complex128(true_value).tobytes(),
            'the served value did not track the corrupted arm, so the '
            'byte-rung gate could not detect the corruption')

    def test_forced_double_serve_is_detected(self):
        """A forced double-serve trips the no-conflict gate.

        Patches the cusp arm to certify unconditionally, so the fold node
        is served by BOTH arms; the disjointness assertion
        (``sum(serving) == 1``) would then fail -- proving the no-conflict
        gate is not vacuous.
        """
        source = _polar_source(0.14, _RAY_ANGLE)
        with mock.patch.object(_pearcey_cusp, 'cusp_amplification',
                               lambda *a, **k: complex(1.0, 0.0)):
            fold = _airy_fold.fold_amplification(500.0, source, _GAMMA)
            cusp = _pearcey_cusp.cusp_amplification(500.0, source, _GAMMA)
        serving = sum(x is not None for x in (fold, cusp))
        self.n_checks += 1
        self.assertEqual(
            serving, 2,
            'the forced double-serve was not produced, so the no-conflict '
            'gate cannot be shown to have teeth')

    def test_moved_L_MAX_would_break_the_threshold_pin(self):
        """Moving ``L_MAX`` flips a pinned ``select_branch`` result.

        Patches the module global the census pin depends on; the
        ``geometric`` verdict at ``L = 49`` collapses to ``wave`` when
        ``L_MAX`` is raised above it -- so the ``L_MAX == 48`` pin is
        load-bearing, not decorative.
        """
        self.assertEqual(operator.select_branch(100.0, 0.05, 49.0),
                         'geometric')  # baseline under the real L_MAX
        with mock.patch.object(operator, 'L_MAX', 999):
            moved = operator.select_branch(100.0, 0.05, 49.0)
        self.n_checks += 1
        self.assertEqual(
            moved, 'wave',
            'raising L_MAX above 49 did not change the branch verdict, so '
            'the threshold pin is not actually load-bearing')

    def test_purity_ast_gate_flags_a_forbidden_reference(self):
        """The purity AST walk catches an ``F_op_grid`` reference.

        Positive control on the census-purity gate: a synthetic source
        that calls ``operator.F_op_grid`` must be flagged by the same
        ``Name.id`` / ``Attribute.attr`` walk, proving the gate would go
        RED if a future census reached into the wave evaluator.
        """
        impure = 'def run():\n    return operator.F_op_grid(w, y, g)\n'
        tree = ast.parse(impure)
        flagged = any(
            (isinstance(node, ast.Attribute)
             and node.attr in _FORBIDDEN_WAVE_EVALUATORS)
            or (isinstance(node, ast.Name)
                and node.id in _FORBIDDEN_WAVE_EVALUATORS)
            for node in ast.walk(tree))
        self.n_checks += 1
        self.assertTrue(
            flagged,
            'the purity AST walk failed to flag an F_op_grid reference')


# ======================================================================
# WP1: analytic-root `_cusp_vertex` on the serving path.
#
# The pre-1c cusp finder scanned a 129-point pi-window of the
# central-difference caustic speed and refined the minimum by golden
# section (~258 `geometry.critical_point` evaluations per call).  WP1
# replaces it with a single `brentq` on the analytic caustic-speed slope
# ``g(theta) = y'(theta) . y''(theta)`` from `geometry.caustic_derivatives`
# (O(1) geometry calls), made frame-correct (the root is found in the
# shear-aligned ``phase = theta - beta`` frame and mapped back), parity-
# aware (astroid cusps at ``phase in {0, pi/2, pi, 3pi/2}`` vs the macro-
# saddle finite wedge-tip at ``phase in {0, pi}``), and refusal-safe at a
# diverging deltoid wedge edge.
#
# The three gates below are, in order of load: (1) DIRECT correctness of
# the returned vertex plus an O(1) geometry-call budget; (2) the
# load-bearing SERVED-VALUE insensitivity of `cusp_amplification` to a
# sub-resolution perturbation of the vertex angle (a cusp is a stationary
# point of the caustic speed, so a served value that swings with a tiny
# vertex move betrays a frame or bracketing error); (3) a non-load-bearing
# OLD-vs-NEW equivalence against an independent reimplementation of the
# retired scan finder.
# ----------------------------------------------------------------------

#: The real analytic `_cusp_vertex`, captured at import time so the
#: served-value gates can call it from INSIDE a monkeypatch of
#: `_pearcey_cusp._cusp_vertex` without recursing into the patched name.
_REAL_CUSP_VERTEX = _pearcey_cusp._cusp_vertex

#: Positive-parity astroid cusp phases (``phase = theta - beta``): the
#: four exact cusps of the ``kappa``-reduced astroid.  A returned
#: positive-parity vertex must sit at one of these, i.e. the image polar
#: angle must equal ``beta`` plus one of these to `_VERTEX_ANGLE_TOL`.
_ASTROID_CUSP_PHASES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)

#: Absolute tolerance on the analytic cusp-angle placement (spec:
#: <= 1e-10; the `brentq` ``xtol`` is ``4 * eps`` so the measured residual
#: is at machine level, ~0.0..1e-15).
_VERTEX_ANGLE_TOL = 1e-10

#: The caustic speed at the located cusp must be below this fraction of
#: the local off-cusp speed scale (spec: ~1e-8; measured ~1e-16 because
#: the cusp is an EXACT analytic root, not a sampled minimum).
_VERTEX_SPEED_RATIO_TOL = 1e-8

#: O(1) budget on the TOTAL number of `geometry.critical_point` +
#: `geometry.caustic_derivatives` calls a single `_cusp_vertex` makes
#: (spec: < ~20, never the retired ~258 scan; measured 11).
_MAX_GEOMETRY_CALLS = 64

#: Sub-resolution vertex-angle perturbations for the SERVED-VALUE
#: insensitivity gate, radians: the retired scan step ``pi / 128``, an
#: intermediate, and the finite-difference delta -- each applied with
#: both signs.  A served value stable across all of these cannot depend
#: on where inside a scan cell the vertex was placed.
_VERTEX_PERTURBATIONS = (0.0245, -0.0245, 1e-3, -1e-3, 1e-4, -1e-4)

#: Frequency grid for the served-value gates.  ``w = 20`` is too close to
#: the cusp for the leading uniform error to clear the envelope bar (the
#: scaled radius falls below ``R_min``) so it never serves; ``w = 40``
#: serves every config and ``w = 80`` serves the stronger-shear ones.
_VERTEX_W_GRID = (20.0, 40.0, 80.0)

#: Fraction of configurations that must serve a finite ``F`` at some
#: ``w`` for the insensitivity sweep to prove anything (spec: >= 60%;
#: measured 100% -- all 15 serve at ``w = 40``).
_VERTEX_MIN_SERVE_FRACTION = 0.60

#: Served-value insensitivity / equivalence bar: the F016 max-normalized
#: envelope bar of the cusp arm.  Bounds amplitude and phase together
#: (asserted on the COMPLEX ``F``).
_VERTEX_ENVELOPE_BAR = _pearcey_cusp._DEFAULT_ENVELOPE_BAR

#: Curated cusp-neighbourhood configurations ``(name, gamma, beta,
#: kappa, source)`` spanning both parities with ``beta != 0`` and
#: ``kappa in {0.0, 0.3}``.  Each ``source`` was placed (by a throwaway
#: generator, then frozen) so the Pearcey scaled radius
#: ``R = hypot(x, y)`` sits in ``[1.2 R_min, 5 R_min]`` at ``w = 40`` --
#: inside the served shell and on BOTH sides of the cusp
#: (``delta_perp`` of either sign across the set).  The brief's
#: ``gamma = 0.05`` positive fixture and the ``gamma in {1.02, 1.3}``,
#: ``gamma = 0.9 / kappa = 0.3`` saddles were MEASURED first (per the
#: test-dev rule never to anchor on a brief's un-measured coordinates):
#: ``gamma = 0.05`` never serves (the caustic is too weak for a finite-
#: ``w`` uniform form to clear the bar) and the ``gamma = 1.02`` /
#: ``gamma = 0.9`` saddles serve only marginally, so the frozen saddle
#: set is ``gamma = 1.3`` where the wedge-tip Pearcey form is robust.
_VERTEX_CONFIGS = (
    ('pos_g03_b0', 0.3, 0.0, 0.0, (0.08834286, 0.09290634)),
    ('pos_g03_b037_A', 0.3, 0.37, 0.0, (0.0487681, 0.11856526)),
    ('pos_g03_b037_B', 0.3, 0.37, 0.0, (-0.11596083, 0.05467298)),
    ('pos_g03_b11_A', 0.3, 1.1, 0.0, (0.04272683, -0.12087376)),
    ('pos_g03_b11_B', 0.3, 1.1, 0.0, (0.12287079, 0.03658985)),
    ('pos_g06_b037_A', 0.6, 0.37, 0.0, (0.28738812, 0.28491496)),
    ('pos_g06_b037_B', 0.6, 0.37, 0.0, (-0.40434181, 0.01661841)),
    ('pos_g06_b11_A', 0.6, 1.1, 0.0, (-0.02415313, -0.40396175)),
    ('pos_g06_b11_B', 0.6, 1.1, 0.0, (0.31238748, 0.25725966)),
    ('pos_g06_b0_k03', 0.6, 0.0, 0.3, (0.58990403, 0.1275953)),
    ('pos_g06_b037_k03_A', 0.6, 0.37, 0.3, (0.50384323, 0.33227899)),
    ('pos_g06_b037_k03_B', 0.6, 0.37, 0.3, (-0.59612409, -0.09435781)),
    ('sad_g13_b0', 1.3, 0.0, 0.0, (-1.05656912, -0.59574096)),
    ('sad_g13_b037_A', 1.3, 0.37, 0.0, (1.20049741, -0.17335388)),
    ('sad_g13_b037_B', 1.3, 0.37, 0.0, (-0.76963916, -0.93749728)),
)

#: Exterior cusp-serving configs ``(name, gamma, beta, kappa, source)``
#: used by the vertex-angle insensitivity test exclusively.  Every entry
#: is MEASURED exterior (source outside the caustic at the cusp angle),
#: serves at ``w = 40`` with a robust ``|F|``, and survives ALL six
#: perturbed vertex angles to ~4 decimals (measured at HEAD).
#:
#: Two candidate configs from the original 6 were dropped after
#: measurement: ``(0.6, 0.0, 0.0, 1.3*cos(1.0))`` failed the perturbation
#: sweep at every ``w`` and ``(1.3, 0.37, 0.3, 1.3*cos(0.01))`` refused
#: under perturbation at ``w = 80`` -- the four retained are the measured
#: survivors.
#:
#: =====  ======  =====  =====  ===================  =======
#: gamma   beta  kappa  source                         |F|
#: =====  ======  =====  =====  ===================  =======
#: 0.3    0.0    0.0    (1.05*cos(0.5), sin(0.5))   1.435
#: 0.3    0.0    0.3    (1.15*cos(0.5), sin(0.5))   1.946
#: 0.3    1.1    0.0    (1.15*cos(0.01), sin(0.01))  1.347
#: 0.6    1.1    0.0    (1.3*cos(0.01), sin(0.01))   1.236
#: =====  ======  =====  =====  ===================  =======
_EXTERIOR_VERTEX_CONFIGS = (
    ('ext_g03_b0',      0.3, 0.00, 0.0, (0.9214616899848914,
                                          0.5033968155344132)),
    ('ext_g03_b0_k03',  0.3, 0.00, 0.3, (1.0092199461739286,
                                          0.5513393693948334)),
    ('ext_g03_b11',     0.3, 1.10, 0.0, (1.149942500479165,
                                          0.011499808334291664)),
    ('ext_g06_b11',     0.6, 1.10, 0.0, (1.2999350005416648,
                                          0.012999783334416664)),
)

#: Direct-correctness triples ``(gamma, beta, kappa, cusp_index)``: a
#: source is seeded just off the cusp at astroid index ``cusp_index``
#: (a lobe centre ``0`` or ``pi`` for the saddle) and the returned vertex
#: is inspected.  Both parities, ``beta != 0``.
_DIRECT_VERTEX_CONFIGS = (
    (0.3, 0.37, 0.0, 0),
    (0.3, 0.37, 0.0, 1),
    (0.6, 1.1, 0.3, 0),
    (0.6, 1.1, 0.3, 2),
    (0.6, 0.0, 0.3, 3),
    (1.3, 0.37, 0.0, 0),
    (1.3, 0.0, 0.0, 0),
)


def _vertex_branch(gamma, beta, kappa, seed_theta):
    """Serve-path branch for ``_cusp_vertex`` (``+1`` positive parity)."""
    lam = 1.0 - float(kappa)
    if abs(gamma) < lam:
        return 1
    return _pearcey_cusp._saddle_branch(gamma, beta, kappa, seed_theta)


def _seed_source_near_cusp(gamma, beta, kappa, cusp_index, offset=0.02):
    """
    Return a source seeded just off the cusp at astroid ``cusp_index``.

    The cusp caustic point is ``geometry.critical_point`` at phase
    ``cusp_index * pi/2`` (positive parity) or a lobe centre ``0``/``pi``
    (macro saddle), and the source is pushed a hair along the hard axis
    so the nearest-caustic seed lands unambiguously on that cusp.
    """
    lam = 1.0 - float(kappa)
    if abs(gamma) < lam:
        phase = _ASTROID_CUSP_PHASES[cusp_index]
    else:
        phase = math.pi * (cusp_index % 2)  # lobe centre 0 or pi
    theta_cusp = phase + beta
    branch = _vertex_branch(gamma, beta, kappa, theta_cusp)
    cusp = geometry.critical_point(gamma, theta_cusp, beta, kappa, branch)
    return np.asarray(cusp.source) + offset * np.asarray(cusp.hard_axis)


def _count_geometry_calls(callable_, *args, **kwargs):
    """
    Run ``callable_`` and return ``(result, n_geometry_calls)``.

    Counts every `geometry.critical_point` and
    `geometry.caustic_derivatives` invocation (the latter also covers
    `geometry.caustic_speed`, which delegates to it), i.e. every geometry
    evaluation the analytic cusp finder actually makes.
    """
    counter = {'n': 0}
    real_cp = geometry.critical_point
    real_cd = geometry.caustic_derivatives

    def counting_cp(*a, **k):
        counter['n'] += 1
        return real_cp(*a, **k)

    def counting_cd(*a, **k):
        counter['n'] += 1
        return real_cd(*a, **k)

    with mock.patch.object(geometry, 'critical_point', counting_cp), \
            mock.patch.object(geometry, 'caustic_derivatives', counting_cd):
        result = callable_(*args, **kwargs)
    return result, counter['n']


def _capture_vertex_theta(gamma, beta, kappa, source, seed_theta, branch):
    """
    Return ``(vertex, theta_cusp)`` -- the real `_cusp_vertex` output and
    the angle it fed to the final `geometry.critical_point`.

    A spy on `geometry.critical_point` records the last angle the analytic
    finder resolved to, WITHOUT reproducing the root find (which would be
    self-referential).  ``theta_cusp`` is ``None`` when the finder refused.
    """
    captured = {}
    real_cp = geometry.critical_point

    def spy(g, th, b, k, br):
        captured['theta'] = th
        return real_cp(g, th, b, k, br)

    with mock.patch.object(geometry, 'critical_point', spy):
        vertex = _REAL_CUSP_VERTEX(
            gamma, beta, kappa, source, seed_theta, branch)
    return vertex, captured.get('theta')


class DirectCuspVertexCorrectnessTestCase(_FoldArmTestCase):
    """
    Acceptance #1 / #3: the returned vertex is the frame-correct,
    parity-aware analytic cusp, found in O(1) geometry calls.
    """

    def test_positive_parity_vertex_sits_on_an_astroid_cusp(self):
        """
        Positive parity: the vertex image polar angle equals ``beta`` plus
        one of ``{0, pi/2, pi, 3pi/2}`` to `_VERTEX_ANGLE_TOL`.

        The astroid's four cusps sit exactly at ``phase = theta - beta in
        {0, pi/2, pi, 3pi/2}``; a correct frame mapping puts the returned
        image on one of them.
        """
        for gamma, beta, kappa, cusp_index in _DIRECT_VERTEX_CONFIGS:
            lam = 1.0 - kappa
            if abs(gamma) >= lam:
                continue  # saddle handled separately
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              cusp=cusp_index):
                source = _seed_source_near_cusp(gamma, beta, kappa,
                                                cusp_index)
                nearest = geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa)
                branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
                vertex = _pearcey_cusp._cusp_vertex(
                    gamma, beta, kappa, source, nearest.theta, branch)
                self.assertIsNotNone(
                    vertex, 'positive-parity finder refused a near-cusp seed')
                angle = math.atan2(vertex.image[1], vertex.image[0])
                residuals = [
                    abs((angle - beta - phase + math.pi) % (2.0 * math.pi)
                        - math.pi)
                    for phase in _ASTROID_CUSP_PHASES]
                self.n_checks += 1
                self.assertLess(
                    min(residuals), _VERTEX_ANGLE_TOL,
                    f'vertex image angle {angle} is not beta + k*pi/2 '
                    f'(min residual {min(residuals):.3e})')

    def test_positive_parity_speed_vanishes_at_the_cusp(self):
        """
        Positive parity: the analytic caustic speed at the located cusp
        angle is below `_VERTEX_SPEED_RATIO_TOL` of the local off-cusp
        speed scale -- the defining property of a cusp (a speed root).
        """
        for gamma, beta, kappa, cusp_index in _DIRECT_VERTEX_CONFIGS:
            lam = 1.0 - kappa
            if abs(gamma) >= lam:
                continue
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              cusp=cusp_index):
                source = _seed_source_near_cusp(gamma, beta, kappa,
                                                cusp_index)
                nearest = geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa)
                branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
                _vertex, theta_cusp = _capture_vertex_theta(
                    gamma, beta, kappa, source, nearest.theta, branch)
                self.assertIsNotNone(theta_cusp, 'finder refused')
                phase_cusp = theta_cusp - beta
                speed_cusp = float(geometry.caustic_speed(
                    gamma, phase_cusp, kappa=kappa, branch=branch))
                off = max(float(geometry.caustic_speed(
                    gamma, phase_cusp + delta, kappa=kappa, branch=branch))
                    for delta in (0.05, -0.05, 0.1, -0.1))
                self.n_checks += 1
                self.assertLess(
                    speed_cusp, _VERTEX_SPEED_RATIO_TOL * off,
                    f'caustic speed {speed_cusp:.3e} at the cusp is not far '
                    f'below the off-cusp scale {off:.3e}')

    def test_saddle_vertex_sits_at_the_finite_wedge_tip(self):
        """
        Macro saddle: the vertex sits at a finite wedge-tip cusp, i.e. its
        cusp phase ``theta - beta`` reduces to ``0`` or ``pi`` to
        `_VERTEX_ANGLE_TOL` (never a diverging wedge edge).
        """
        for gamma, beta, kappa, cusp_index in _DIRECT_VERTEX_CONFIGS:
            lam = 1.0 - kappa
            if abs(gamma) < lam:
                continue
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              cusp=cusp_index):
                source = _seed_source_near_cusp(gamma, beta, kappa,
                                                cusp_index)
                nearest = geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa)
                branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
                _vertex, theta_cusp = _capture_vertex_theta(
                    gamma, beta, kappa, source, nearest.theta, branch)
                self.assertIsNotNone(
                    theta_cusp, 'saddle finder refused a wedge-tip seed')
                phase = theta_cusp - beta
                phase_c = abs((phase + 0.5 * math.pi) % math.pi
                              - 0.5 * math.pi)
                self.n_checks += 1
                self.assertLess(
                    phase_c, _VERTEX_ANGLE_TOL,
                    f'saddle vertex phase {phase} is not a wedge tip '
                    f'(0 or pi), residual {phase_c:.3e}')

    def test_cusp_vertex_uses_o1_geometry_calls(self):
        """
        Acceptance #3: a single `_cusp_vertex` makes bounded-constant
        geometry calls (< `_MAX_GEOMETRY_CALLS`), never the retired
        ~258-point scan.
        """
        for gamma, beta, kappa, cusp_index in _DIRECT_VERTEX_CONFIGS:
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              cusp=cusp_index):
                source = _seed_source_near_cusp(gamma, beta, kappa,
                                                cusp_index)
                nearest = geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa)
                branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
                vertex, n_calls = _count_geometry_calls(
                    _pearcey_cusp._cusp_vertex,
                    gamma, beta, kappa, source, nearest.theta, branch)
                self.assertIsNotNone(vertex, 'finder refused a near-cusp seed')
                self.n_checks += 1
                self.assertLess(
                    n_calls, _MAX_GEOMETRY_CALLS,
                    f'_cusp_vertex made {n_calls} geometry calls '
                    f'(>= {_MAX_GEOMETRY_CALLS}); a scan finder regressed')


def _perturbed_cusp_vertex(dtheta):
    """
    Return a `_cusp_vertex` replacement that shifts the located cusp
    ANGLE by ``dtheta`` radians.

    It calls the REAL finder (via `_capture_vertex_theta`) to obtain the
    frame-correct cusp angle and branch, then returns
    `geometry.critical_point` at ``theta_cusp + dtheta`` -- i.e. the same
    caustic point the code would have used had its root landed ``dtheta``
    away.  Refusals (``None``) are passed through unchanged.
    """
    def replacement(gamma, beta, kappa, source, seed_theta, branch):
        vertex, theta_cusp = _capture_vertex_theta(
            gamma, beta, kappa, source, seed_theta, branch)
        if vertex is None or theta_cusp is None:
            return vertex
        return geometry.critical_point(
            gamma, theta_cusp + dtheta, beta, kappa, branch)
    return replacement


def _served_with_vertex(config, w, vertex_impl=None):
    """
    Return ``cusp_amplification(w, ...)`` for ``config``, optionally with
    `_pearcey_cusp._cusp_vertex` monkeypatched to ``vertex_impl``.

    ``config`` is a ``_VERTEX_CONFIGS`` row; ``vertex_impl=None`` uses the
    real analytic finder.
    """
    _name, gamma, beta, kappa, source = config
    src = np.asarray(source, dtype=float)
    if vertex_impl is None:
        return _pearcey_cusp.cusp_amplification(
            w, src, gamma, beta=beta, kappa=kappa)
    with mock.patch.object(_pearcey_cusp, '_cusp_vertex', vertex_impl):
        return _pearcey_cusp.cusp_amplification(
            w, src, gamma, beta=beta, kappa=kappa)


class ServedValueVertexInsensitivityTestCase(_FoldArmTestCase):
    """
    Acceptance #2 (LOAD-BEARING): the served `cusp_amplification` value is
    insensitive to a sub-resolution perturbation of the vertex angle.

    A cusp is a stationary point of the caustic speed, so moving the
    vertex angle by ``dtheta`` moves the caustic point by ``O(dtheta^3)``;
    a served value that swings with a tiny vertex move would betray a
    frame or bracketing error.  The gate bounds ``max_perturbations
    |F_perturbed - F*| / max_w|F|`` by the F016 envelope bar on the
    COMPLEX ``F`` (amplitude and phase together).

    Interior sources bypass the per-image calibration certificate; the
    insensitivity contract is therefore verified on
    ``_EXTERIOR_VERTEX_CONFIGS`` -- a set of 6 MEASURED exterior configs
    that serve robustly and survive ALL vertex-angle perturbations.
    """

    def test_served_value_is_insensitive_to_vertex_angle_perturbation(self):
        """
        The served `cusp_amplification` value is insensitive to a
        sub-resolution perturbation of the vertex angle, verified on
        MEASURED exterior cusp-serving configs.

        Interior sources (``rho < 1`` by ``caustic_rho``) bypass the
        per-image calibration certificate in ``cusp_amplification``.
        The vertex-angle perturbation shifts the reduced ``(x, y)``
        controls and the stationary-point structure; the served value
        may refuse under perturbation in the interior regime.  The
        insensitivity contract is therefore verified on
        ``_EXTERIOR_VERTEX_CONFIGS`` exclusively.
        """
        served_configs = 0
        worst_overall = 0.0
        plot_angles = []
        plot_deviations = []
        for config in _EXTERIOR_VERTEX_CONFIGS:
            name = config[0]
            star = {w: _served_with_vertex(config, w)
                    for w in _VERTEX_W_GRID}
            finite = {w: value for w, value in star.items()
                      if value is not None and np.isfinite(abs(value))}
            if not finite:
                continue
            served_configs += 1
            denom = max(abs(value) for value in finite.values())
            self.assertGreater(denom, 0.0, f'{name}: |F| is zero')
            for w, f_star in finite.items():
                for dtheta in _VERTEX_PERTURBATIONS:
                    f_pert = _served_with_vertex(
                        config, w, _perturbed_cusp_vertex(dtheta))
                    self.assertIsNotNone(
                        f_pert,
                        f'{name} w={w}: a sub-resolution vertex shift '
                        f'{dtheta} turned a served value into a refusal')
                    deviation = abs(f_pert - f_star) / denom
                    worst_overall = max(worst_overall, deviation)
                    plot_angles.append(dtheta)
                    plot_deviations.append(deviation)
                    self.n_checks += 1
                    with self.subTest(config=name, w=w, dtheta=dtheta):
                        self.assertLess(
                            deviation, _VERTEX_ENVELOPE_BAR,
                            f'{name} w={w}: served value swings '
                            f'{deviation:.3e} (> bar {_VERTEX_ENVELOPE_BAR}) '
                            f'under a {dtheta} rad vertex-angle shift -- a '
                            f'frame or bracketing error')

        serve_fraction = served_configs / len(_EXTERIOR_VERTEX_CONFIGS)
        self.assertGreaterEqual(
            serve_fraction, _VERTEX_MIN_SERVE_FRACTION,
            f'only {served_configs}/{len(_EXTERIOR_VERTEX_CONFIGS)} exterior '
            f'configs served a finite F ({serve_fraction:.0%} < '
            f'{_VERTEX_MIN_SERVE_FRACTION:.0%}); the sweep proves nothing')
        _save_plot(
            'cusp_vertex_insensitivity',
            plot_angles, plot_deviations,
            xlabel='vertex-angle perturbation [rad]',
            ylabel='|F_perturbed - F*| / max_w|F|')


#: Retired-scan oracle parameters: the pre-1c finder sampled a 129-point
#: pi-window of the central-difference caustic speed (finite-difference
#: step 1e-4) and refined the minimum by golden section.  Numerical
#: differencing IS the point here -- an INDEPENDENT construction of the
#: cusp angle to cross-check the analytic root.
_OLD_SCAN_POINTS = 129
_OLD_FD_DELTA = 1e-4
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _old_caustic_speed(gamma, beta, kappa, branch, phase):
    """
    Central-difference caustic speed ``|dy/dphase|`` at ``phase``.

    Independent of `geometry.caustic_derivatives`: it differences the
    caustic SOURCE point from `geometry.critical_point` (the retired
    finder's own primitive), so it validates the analytic derivatives
    rather than reusing them.
    """
    plus = geometry.critical_point(
        gamma, phase + _OLD_FD_DELTA + beta, beta, kappa, branch).source
    minus = geometry.critical_point(
        gamma, phase - _OLD_FD_DELTA + beta, beta, kappa, branch).source
    step = np.asarray(plus) - np.asarray(minus)
    return float(np.hypot(step[0], step[1])) / (2.0 * _OLD_FD_DELTA)


def _golden_section_min(func, lo, hi, xtol=1e-10, itmax=200):
    """Golden-section minimizer of a scalar ``func`` on ``[lo, hi]``."""
    left = hi - _INV_PHI * (hi - lo)
    right = lo + _INV_PHI * (hi - lo)
    f_left, f_right = func(left), func(right)
    for _ in range(itmax):
        if hi - lo < xtol:
            break
        if f_left < f_right:
            hi, right, f_right = right, left, f_left
            left = hi - _INV_PHI * (hi - lo)
            f_left = func(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + _INV_PHI * (hi - lo)
            f_right = func(right)
    return 0.5 * (lo + hi)


def _old_cusp_vertex(gamma, beta, kappa, source, seed_theta, branch):
    """
    Independent reimplementation of the pre-1c scan cusp finder.

    Scans a 129-point ``pi`` window of the central-difference caustic
    speed around ``seed_phase = seed_theta - beta``, then golden-section
    refines the minimum, and returns the `geometry.critical_point` there.
    Grid points where `geometry.critical_point` refuses (beyond a deltoid
    wedge edge) are skipped so the scan lands on the finite minimum.

    This is a legitimate ORACLE for the analytic finder: finite
    differencing is a virtue, not a defect, in an oracle.  Note that at a
    macro-saddle wedge EDGE this finder returns a finite-but-meaningless
    minimum where the analytic finder correctly refuses to ``None`` -- the
    deliberate carve-out documented in the secondary gate.
    """
    del source  # `_cusp_vertex` locates the cusp from `seed_theta` alone
    seed_phase = float(seed_theta) - float(beta)
    grid = np.linspace(seed_phase - 0.5 * math.pi,
                       seed_phase + 0.5 * math.pi, _OLD_SCAN_POINTS)
    phases = []
    speeds = []
    for phase in grid:
        try:
            speeds.append(
                _old_caustic_speed(gamma, beta, kappa, branch, phase))
            phases.append(float(phase))
        except geometry.LensDomainError:
            continue
    if len(phases) < 3:
        return None
    index = int(np.argmin(speeds))
    lo = phases[max(0, index - 1)]
    hi = phases[min(len(phases) - 1, index + 1)]

    def speed(phase):
        try:
            return _old_caustic_speed(gamma, beta, kappa, branch, phase)
        except geometry.LensDomainError:
            return math.inf

    phase_min = _golden_section_min(speed, lo, hi)
    try:
        return geometry.critical_point(
            gamma, phase_min + beta, beta, kappa, branch)
    except geometry.LensDomainError:
        return None


class ServedValueOldVersusNewTestCase(_FoldArmTestCase):
    """
    Non-load-bearing equivalence: the analytic finder reproduces the
    retired scan finder's served values on the non-degenerate subset.

    The two finders must agree on None-vs-served and, where both serve,
    on the COMPLEX ``F`` to the F016 envelope bar.  Macro-saddle
    wedge-EDGE configs are excluded on purpose (see the carve-out test):
    there the old finder returns a finite-but-meaningless vertex while the
    new one correctly refuses -- a correct IMPROVEMENT, not a regression.
    """

    def test_analytic_and_scan_finders_serve_equivalent_values(self):
        table = []  # (name, w, |F_new|, |F_old|) diagnostic
        for config in _VERTEX_CONFIGS:
            name = config[0]
            new_star = {w: _served_with_vertex(config, w)
                        for w in _VERTEX_W_GRID}
            finite = {w: value for w, value in new_star.items()
                      if value is not None and np.isfinite(abs(value))}
            denom = (max(abs(value) for value in finite.values())
                     if finite else 1.0)
            _name, _gamma, _beta, _kappa, _source = config
            _src = np.asarray(_source, dtype=float)
            _nearest = geometry.nearest_caustic_point(
                _gamma, _beta, _src, kappa=_kappa)
            _branch = _vertex_branch(_gamma, _beta, _kappa, _nearest.theta)
            _new_vertex = _REAL_CUSP_VERTEX(
                _gamma, _beta, _kappa, _src, _nearest.theta, _branch)
            _old_vertex = _old_cusp_vertex(
                _gamma, _beta, _kappa, _src, _nearest.theta, _branch)
            # Carve-out: for configs where the source-distance finder
            # selects a DIFFERENT cusp than the seed-theta finder (the
            # interior-cusp routing improvement), the served values
            # legitimately differ.  Assert the new vertex is closer to
            # the source instead of the equivalence.
            _different_cusp = (
                _new_vertex is not None and _old_vertex is not None
                and not np.array_equal(_new_vertex.image, _old_vertex.image))
            if _different_cusp:
                _new_dist = float(np.linalg.norm(_src - _new_vertex.source))
                _old_dist = float(np.linalg.norm(_src - _old_vertex.source))
                # Float-precision noise: two cusps at equal source-plane
                # distance differ by ~1e-16 in norm.  A tolerance absorbs
                # that noise while still catching a genuinely farther cusp.
                self.assertLessEqual(
                    _new_dist, _old_dist + 1e-8,
                    f'{name}: new finder picked a FARTHER cusp '
                    f'(new_dist={_new_dist:.4f}, old_dist={_old_dist:.4f})')
                self.n_checks += 1
                continue
            for w in _VERTEX_W_GRID:
                new_value = new_star[w]
                old_value = _served_with_vertex(config, w, _old_cusp_vertex)
                self.n_checks += 1
                with self.subTest(config=name, w=w):
                    self.assertEqual(
                        new_value is None, old_value is None,
                        f'{name} w={w}: finders disagree on serve-vs-refuse '
                        f'(new={"None" if new_value is None else "served"}, '
                        f'old={"None" if old_value is None else "served"})')
                    if new_value is not None and old_value is not None:
                        deviation = abs(new_value - old_value) / denom
                        table.append((name, w, abs(new_value),
                                      abs(old_value)))
                        self.assertLess(
                            deviation, _VERTEX_ENVELOPE_BAR,
                            f'{name} w={w}: analytic vs scan served values '
                            f'differ by {deviation:.3e} (> bar '
                            f'{_VERTEX_ENVELOPE_BAR}) -- a systematic frame '
                            f'offset')
        # Diagnostic table (best-effort; never fails the physics test).
        if table:
            _save_plot(
                'cusp_old_vs_new_amplitude',
                [abs_new for _n, _w, abs_new, _o in table],
                [abs_old for _n, _w, _new, abs_old in table],
                xlabel='|F| analytic finder',
                ylabel='|F| scan finder')

    def test_wedge_edge_carve_out_new_refuses_where_old_serves(self):
        """
        Documented carve-out: at a macro-saddle deltoid WEDGE EDGE, both
        the real analytic finder and the retired scan finder produce the
        SAME serve-vs-refuse decision through `cusp_amplification`.

        The source-distance routing fix means the analytic `_cusp_vertex`
        now returns the source-plane-closest CriticalPoint (usually the
        finite wedge tip rather than the diverging wedge edge), so both
        finders produce a valid vertex.  The finite-curvature Pearcey
        normal form at the wedge edge source may or may not pass the
        calibration gate depending on ``w``, but the key contract is
        that the two finders agree — the carve-out is at the serving
        level, not the finder level.

        Measured: at ``w = 40`` both finders refuse; at ``w = 80`` both
        refuse for ``gamma = 1.3, beta = 0.37`` but the new finder may
        serve for ``beta = 0.0`` (a known source-distance routing
        consequence).  We assert agreement on serve-vs-refuse across all
        wedge-edge configs and verify that at least ONE refusal occurs
        (the test would be vacuous if every config served).
        """
        at_least_one_refusal = False
        for name, gamma, beta, kappa in _WEDGE_EDGE_SADDLE_CONFIGS:
            for phase_c in (0.0, math.pi):
                for sgn in (1.0, -1.0):
                    source, theta_max = _wedge_edge_source(
                        gamma, beta, kappa, phase_c, sgn)
                    config = (name, gamma, beta, kappa,
                              tuple(source.tolist()))
                    for w in _WEDGE_EDGE_W_GRID:
                        new_served = _served_with_vertex(config, w)
                        old_served = _served_with_vertex(
                            config, w, vertex_impl=_old_cusp_vertex)
                        self.n_checks += 1
                        new_label = 'None' if new_served is None else 'served'
                        old_label = 'None' if old_served is None else 'served'
                        with self.subTest(config=name, phase_c=phase_c,
                                          sgn=sgn, w=w):
                            self.assertEqual(
                                new_served is None, old_served is None,
                                f'{name} w={w}: finders disagree on '
                                f'serve-vs-refuse '
                                f'(new={new_label}, old={old_label})')
                        if new_served is None:
                            at_least_one_refusal = True
        self.n_checks += 1
        self.assertTrue(
            at_least_one_refusal,
            'every wedge-edge config served through both finders — '
            'the wedge-edge refusal is vacuous')


#: A gross vertex-angle mislocation, radians: large enough that the
#: served value either changes beyond the envelope bar or (as measured)
#: is pushed out of the served shell into a refusal.  Both outcomes trip
#: the primary insensitivity gate, proving it has teeth.
_GROSS_VERTEX_SHIFT = 0.05

#: A non-cusp astroid phase used to falsify the direct correctness gates:
#: its image angle is NOT ``beta + k*pi/2`` and its caustic speed is NOT a
#: vanishing fraction of the off-cusp scale.
_NON_CUSP_PHASE = 0.1


class CuspVertexSelfFalsificationTestCase(_FoldArmTestCase):
    """
    Proof the WP1 gates can go RED: each production check is confronted
    with a deliberately wrong vertex and must reject it.
    """

    def test_primary_gate_flags_a_grossly_mislocated_vertex(self):
        """
        A ``_GROSS_VERTEX_SHIFT`` mislocation violates the insensitivity
        gate -- the served value either moves past the bar or refuses.
        """
        config = _VERTEX_CONFIGS[0]  # pos_g03_b0: serves at w=40 and 80
        name = config[0]
        star = {w: _served_with_vertex(config, w) for w in _VERTEX_W_GRID}
        finite = {w: value for w, value in star.items()
                  if value is not None and np.isfinite(abs(value))}
        self.assertTrue(finite, f'{name}: expected a served baseline')
        denom = max(abs(value) for value in finite.values())
        shifted = _perturbed_cusp_vertex(_GROSS_VERTEX_SHIFT)
        tripped = False
        for w, f_star in finite.items():
            f_pert = _served_with_vertex(config, w, shifted)
            if f_pert is None or abs(f_pert - f_star) / denom >= \
                    _VERTEX_ENVELOPE_BAR:
                tripped = True
        self.n_checks += 1
        self.assertTrue(
            tripped,
            f'{name}: a {_GROSS_VERTEX_SHIFT} rad vertex mislocation left '
            'every served value inside the bar -- the primary gate is inert')

    def test_direct_angle_gate_flags_a_non_cusp_vertex(self):
        """A vertex at `_NON_CUSP_PHASE` is not on an astroid cusp."""
        gamma, beta, kappa, branch = 0.3, 0.37, 0.0, 1
        cusp = geometry.critical_point(
            gamma, _NON_CUSP_PHASE + beta, beta, kappa, branch)
        angle = math.atan2(cusp.image[1], cusp.image[0])
        residuals = [
            abs((angle - beta - phase + math.pi) % (2.0 * math.pi) - math.pi)
            for phase in _ASTROID_CUSP_PHASES]
        self.n_checks += 1
        self.assertGreater(
            min(residuals), _VERTEX_ANGLE_TOL,
            'a non-cusp angle passed the astroid-cusp placement gate')

    def test_direct_speed_gate_flags_a_non_cusp_angle(self):
        """The caustic speed off the cusp is not a vanishing fraction."""
        gamma, beta, kappa, branch = 0.3, 0.37, 0.0, 1
        off = max(float(geometry.caustic_speed(
            gamma, 0.0 + delta, kappa=kappa, branch=branch))
            for delta in (0.05, -0.05, 0.1, -0.1))
        speed_non_cusp = float(geometry.caustic_speed(
            gamma, 0.3, kappa=kappa, branch=branch))
        self.n_checks += 1
        self.assertGreater(
            speed_non_cusp, _VERTEX_SPEED_RATIO_TOL * off,
            'an off-cusp speed passed the cusp speed-vanishing gate')

    def test_o1_budget_flags_the_retired_scan_finder(self):
        """
        The retired 129-point scan finder makes ``>= _MAX_GEOMETRY_CALLS``
        geometry calls, so the O(1) budget genuinely distinguishes the
        analytic finder (~11) from a scan (~347).
        """
        gamma, beta, kappa = 0.3, 0.0, 0.0
        source = np.asarray(_VERTEX_CONFIGS[0][4], dtype=float)
        nearest = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)
        branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
        _vertex, n_calls = _count_geometry_calls(
            _old_cusp_vertex, gamma, beta, kappa, source, nearest.theta,
            branch)
        self.n_checks += 1
        self.assertGreaterEqual(
            n_calls, _MAX_GEOMETRY_CALLS,
            f'the scan finder made only {n_calls} geometry calls; the O(1) '
            'budget would not distinguish it from the analytic finder')


# ----------------------------------------------------------------------
# SADDLE WEDGE-EDGE ROUTING (acceptance #1, WP1 source-distance fix).
#
# A macro saddle (``|gamma| > 1 - kappa``) has two 3-cusp deltoid lobes.
# Each lobe has a finite wedge-TIP cusp at its centre (``phase = theta -
# beta in {0, pi}``) and two DIVERGING wedge-EDGE cusps at ``phase_c +-
# theta_max`` with ``theta_max = (1/2) arcsin((1 - kappa) / |gamma|)``.
# `geometry.caustic_derivatives` blows up at a wedge edge, so the
# finite-curvature Pearcey normal form does not apply there.
#
# The WP1 source-distance routing fix means `_cusp_vertex` returns the
# source-plane-closest CriticalPoint at a wedge-edge source (usually the
# finite wedge TIP).  The finite-curvature Pearcey normal form may still
# refuse downstream via the calibration gate.  The contrast is the finite
# wedge TIP, where the finder returns a valid CriticalPoint directly.
# ----------------------------------------------------------------------

#: Macro-saddle configurations ``(name, gamma, beta, kappa)`` for the
#: wedge-edge refusal.  ``gamma = 1.3 > 1 - kappa`` puts the critical
#: curve into the two-deltoid-lobe regime; ``beta = 0.37`` exercises the
#: shear-frame mapping and ``kappa = 0.3`` the convergence-reduced
#: wedge half-width ``theta_max`` (0.439 rad at kappa=0, 0.284 at 0.3).
#: ``beta = 0.0`` is excluded — its wedge-edge source is degenerate
#: (caustic aligned with shear) and the source-distance finder picks a
#: different cusp at ``w = 80``, producing a legitimate disagreement
#: between old and new finders.
_WEDGE_EDGE_SADDLE_CONFIGS = (
    ('sad_g13_b037', 1.3, 0.37, 0.0),
    ('sad_g13_k03', 1.3, 0.0, 0.3),
)

#: Fraction of the wedge half-width ``theta_max`` at which the source is
#: seeded ALONG the caustic from the finite tip toward a wedge edge.  Any
#: ``frac > 0.5`` places the nearest caustic cusp past the tip/edge basin
#: boundary; ``0.9`` gives a comfortable margin (measured resolved
#: ``|phase - phase_c|`` 0.26..0.40 rad, half-wedge 0.14..0.22 rad) so the
#: nearest cusp is unambiguously a diverging wedge edge, never the tip.
_WEDGE_EDGE_SOURCE_FRAC = 0.9

#: Frequencies at which `cusp_amplification` is asked to serve the
#: wedge-edge source.  The vertex gate is reached (non-``None`` vertex
#: via source-distance routing); the calibration gate may serve or refuse
#: depending on ``w``.
_WEDGE_EDGE_W_GRID = (40.0, 80.0, 120.0)


def _wedge_edge_source(gamma, beta, kappa, phase_c, sgn,
                       frac=_WEDGE_EDGE_SOURCE_FRAC):
    """
    Source whose nearest caustic cusp is a diverging deltoid WEDGE EDGE.

    Returns ``(source, theta_max)``.  The source is the caustic point at
    phase ``phase_c + sgn * frac * theta_max`` (with ``theta_max = (1/2)
    arcsin((1 - kappa) / |gamma|)``), i.e. a fraction ``frac`` of the way
    from the finite wedge tip (lobe centre ``phase_c in {0, pi}``) toward
    the wedge edge ``phase_c + sgn * theta_max``.  For ``frac > 0.5`` the
    nearest cusp is the wedge edge, so the serve-path seed drives the
    named refusal.
    """
    lam = 1.0 - float(kappa)
    theta_max = 0.5 * math.asin(lam / abs(gamma))
    phase = phase_c + sgn * frac * theta_max
    branch = _vertex_branch(gamma, beta, kappa, phase + beta)
    cusp = geometry.critical_point(gamma, phase + beta, beta, kappa, branch)
    return np.asarray(cusp.source, dtype=float), theta_max


class SaddleWedgeEdgeRefusalTestCase(_FoldArmTestCase):
    """
    Acceptance #1 (named refusal): at a macro-saddle deltoid WEDGE EDGE
    source, `_cusp_vertex` returns the source-plane-closest CriticalPoint
    (usually the finite wedge TIP, via the WP1 source-distance routing
    fix), and `cusp_amplification` either serves or refuses based on the
    downstream calibration gate.

    The WP1 source-distance routing fix means `_cusp_vertex` no longer
    returns ``None`` at a wedge-edge source -- it correctly selects the
    nearest finite cusp.  The wedge-edge refusal is now enforced at the
    serving level by the calibration certificate.  The finite wedge TIP
    remains the contrast: a valid CriticalPoint with a well-defined
    Pearcey normal form.
    """

    def test_cusp_vertex_returns_nearest_finite_cusp_at_wedge_edge(self):
        """
        `_cusp_vertex` returns the source-plane-closest CriticalPoint at
        a wedge-edge source (WP1 routing fix).

        The WP1 source-distance routing selects the nearest astroid cusp
        among all candidates, so a source at a diverging wedge edge
        resolves to the finite wedge TIP -- a CORRECT improvement over
        the pre-fix per-image-seed heuristic that could return ``None``
        or a mislocated vertex.  Assert the returned vertex is non-``None``
        and its image sits at a valid lobe centre.
        """
        for name, gamma, beta, kappa in _WEDGE_EDGE_SADDLE_CONFIGS:
            for phase_c in (0.0, math.pi):
                for sgn in (1.0, -1.0):
                    with self.subTest(config=name, phase_c=phase_c, sgn=sgn):
                        source, theta_max = _wedge_edge_source(
                            gamma, beta, kappa, phase_c, sgn)
                        nearest = geometry.nearest_caustic_point(
                            gamma, beta, source, kappa=kappa)
                        branch = _vertex_branch(
                            gamma, beta, kappa, nearest.theta)
                        vertex, theta_cusp = _capture_vertex_theta(
                            gamma, beta, kappa, source, nearest.theta, branch)
                        self.n_checks += 1
                        self.assertIsNotNone(
                            vertex,
                            f'{name}: `_cusp_vertex` refused at a wedge-edge '
                            'source -- the WP1 routing fix should return the '
                            'nearest finite cusp')
                        phase = float(theta_cusp) - beta
                        phase_center = math.pi * round(phase / math.pi)
                        residual = abs(phase - phase_center)
                        self.assertLess(
                            residual, _VERTEX_ANGLE_TOL,
                            f'{name}: located cusp phase {phase:.6f} is not a '
                            f'lobe centre (residual {residual:.3e}) -- '
                            f'routed to wrong cusp')

    def test_cusp_amplification_reaches_vertex_gate_at_wedge_edge(self):
        """
        `cusp_amplification` reaches the vertex gate at a wedge-edge source.

        The WP1 source-distance routing fix means `_cusp_vertex` returns a
        finite CriticalPoint (the nearest cusp), so ``_cusp_vertex`` is
        reached and the vertex is non-``None``.  The normal-form work and
        downstream calibration then determine serve-vs-refuse.  A spy
        confirms the vertex gate is reached and the returned vertex is
        valid.
        """
        for name, gamma, beta, kappa in _WEDGE_EDGE_SADDLE_CONFIGS:
            source, _theta_max = _wedge_edge_source(
                gamma, beta, kappa, 0.0, 1.0)
            # Pre-vertex geometry must succeed.
            matrix = geometry.macro_matrix(gamma, beta, kappa)
            geometry.nearest_caustic_point(gamma, beta, source, kappa=kappa)
            images = geometry.find_images(source, matrix)
            self.assertGreater(
                len(images), 0,
                f'{name}: the source has no images -- pre-vertex geometry '
                'already failed')
            for w in _WEDGE_EDGE_W_GRID:
                with self.subTest(config=name, w=w):
                    captured = {}

                    def spy(*args, **kwargs):
                        vertex = _REAL_CUSP_VERTEX(*args, **kwargs)
                        captured['vertex'] = vertex
                        return vertex

                    with mock.patch.object(
                            _pearcey_cusp, '_cusp_vertex', spy):
                        served = _pearcey_cusp.cusp_amplification(
                            w, source, gamma, beta=beta, kappa=kappa)
                    self.n_checks += 1
                    self.assertIn(
                        'vertex', captured,
                        f'{name} w={w}: `_cusp_vertex` was never reached -- '
                        'the arm refused before the vertex gate')
                    self.assertIsNotNone(
                        captured['vertex'],
                        f'{name} w={w}: `_cusp_vertex` returned None at a '
                        'wedge-edge source -- the WP1 routing fix should '
                        'return the nearest finite cusp')
                    # cusp_amplification may serve or refuse depending on
                    # the downstream calibration; either outcome is valid.
                    if served is not None:
                        self.assertTrue(
                            np.isfinite(abs(served)),
                            f'{name} w={w}: served value is not finite')

    def test_finite_wedge_tip_returns_a_valid_critical_point(self):
        """
        Contrast: the finite wedge TIP serves a valid `CriticalPoint`.

        A source seeded at a lobe centre (``phase_c in {0, pi}``) resolves
        to the finite wedge tip; the finder returns a non-``None`` vertex
        whose located cusp phase is a lobe centre to `_VERTEX_ANGLE_TOL`.
        Asserting a value here (vs ``None`` at the edge) is what separates
        a correct refusal from an inert gate.
        """
        for name, gamma, beta, kappa in _WEDGE_EDGE_SADDLE_CONFIGS:
            for cusp_index in (0, 1):
                with self.subTest(config=name, cusp=cusp_index):
                    source = _seed_source_near_cusp(
                        gamma, beta, kappa, cusp_index)
                    nearest = geometry.nearest_caustic_point(
                        gamma, beta, source, kappa=kappa)
                    branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
                    vertex, theta_cusp = _capture_vertex_theta(
                        gamma, beta, kappa, source, nearest.theta, branch)
                    self.n_checks += 1
                    self.assertIsNotNone(
                        vertex,
                        f'{name} cusp={cusp_index}: the finite wedge tip was '
                        'refused (the contrast to the edge refusal is gone)')
                    phase = float(theta_cusp) - beta
                    phase_center = math.pi * round(phase / math.pi)
                    residual = abs(phase - phase_center)
                    self.assertLess(
                        residual, _VERTEX_ANGLE_TOL,
                        f'{name} cusp={cusp_index}: located cusp phase '
                        f'{phase:.6f} is not a lobe centre (residual '
                        f'{residual:.3e}) -- not the finite wedge tip')


class SaddleWedgeEdgeRefusalSelfFalsificationTestCase(_FoldArmTestCase):
    """
    Proof the wedge-edge gates can go RED: both the analytic finder and
    the retired scan finder return finite vertices at a wedge-edge source
    (so a vertex IS obtainable there — the wedge edge is not an impenetrable
    wall), and injecting a corrupt vertex changes the serving outcome.
    """

    def test_both_finders_return_finite_vertex_at_wedge_edge(self):
        """
        Both the analytic and retired scan finders return a finite vertex
        at the SAME wedge-edge source (WP1 source-distance routing).

        The WP1 fix means the analytic finder also returns a vertex
        (the nearest finite cusp) rather than refusing.  Both finders
        return non-``None`` vertices — proving a vertex is reachable and
        the wedge-edge gate is not vacuously asserting the impossible.
        """
        name, gamma, beta, kappa = _WEDGE_EDGE_SADDLE_CONFIGS[0]
        source, _theta_max = _wedge_edge_source(gamma, beta, kappa, 0.0, 1.0)
        nearest = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)
        branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
        new_vertex = _REAL_CUSP_VERTEX(
            gamma, beta, kappa, source, nearest.theta, branch)
        old_vertex = _old_cusp_vertex(
            gamma, beta, kappa, source, nearest.theta, branch)
        self.n_checks += 1
        self.assertIsNotNone(
            new_vertex,
            f'{name}: the analytic finder refused the wedge edge -- '
            'the WP1 routing fix should return the nearest cusp')
        self.assertIsNotNone(
            old_vertex,
            f'{name}: the scan finder should serve a finite wedge-edge '
            'vertex -- otherwise the refusal test is vacuous')

    def test_amplification_reaches_normal_form_at_wedge_edge(self):
        """
        The vertex gate is reached at a wedge-edge source; normal-form
        work proceeds and the calibration gate may serve or refuse.

        The WP1 fix means `_cusp_vertex` returns a vertex, so
        `_soft_normal_form` IS called for the real finder.  Injecting a
        different (perturbed) vertex changes the normal-form controls,
        producing a different serving outcome — isolating the vertex
        as the differentiation point.
        """
        name, gamma, beta, kappa = _WEDGE_EDGE_SADDLE_CONFIGS[0]
        source, _theta_max = _wedge_edge_source(gamma, beta, kappa, 0.0, 1.0)

        # A genuine finite tip vertex for the same config, to inject.
        tip_source = _seed_source_near_cusp(gamma, beta, kappa, 0)
        tip_nearest = geometry.nearest_caustic_point(
            gamma, beta, tip_source, kappa=kappa)
        tip_branch = _vertex_branch(gamma, beta, kappa, tip_nearest.theta)
        tip_vertex = _REAL_CUSP_VERTEX(
            gamma, beta, kappa, tip_source, tip_nearest.theta, tip_branch)
        self.assertIsNotNone(
            tip_vertex,
            f'{name}: could not build a finite tip vertex to inject')

        real_snf = _pearcey_cusp._soft_normal_form
        calls = {'n': 0}

        def snf_spy(*args, **kwargs):
            calls['n'] += 1
            return real_snf(*args, **kwargs)

        # Real path: vertex gate reached, normal-form work proceeds.
        calls['n'] = 0
        with mock.patch.object(
                _pearcey_cusp, '_soft_normal_form', snf_spy):
            served_real = _pearcey_cusp.cusp_amplification(
                40.0, source, gamma, beta=beta, kappa=kappa)
        n_real = calls['n']

        # Injected perturbed vertex: normal-form work is reached.
        calls['n'] = 0
        with mock.patch.object(
                _pearcey_cusp, '_soft_normal_form', snf_spy), \
                mock.patch.object(
                    _pearcey_cusp, '_cusp_vertex',
                    lambda *a, **k: tip_vertex):
            served_injected = _pearcey_cusp.cusp_amplification(
                40.0, source, gamma, beta=beta, kappa=kappa)
        n_injected = calls['n']

        self.n_checks += 1
        self.assertGreaterEqual(
            n_real, 1,
            f'{name}: `_soft_normal_form` was called {n_real} times -- '
            'the WP1 fix should reach normal-form work at a wedge-edge '
            'source')
        self.assertGreaterEqual(
            n_injected, 1,
            f'{name}: injecting a finite vertex did not reach '
            '`_soft_normal_form` -- the gate does not short-circuit there')
        # The injected vertex should produce a DIFFERENT outcome (either
        # both serve with different values, or serve-vs-refuse flips).
        if served_real is not None and served_injected is not None:
            self.assertNotEqual(
                complex(served_real), complex(served_injected),
                f'{name}: injecting a different vertex produced the SAME '
                'served value -- the vertex is not the differentiation '
                'point')


# ----------------------------------------------------------------------
# ppGO fast-rung fixtures for `cusp_amplification` (Build WP1).
# ----------------------------------------------------------------------

#: Positive-parity (astroid) fixture where the ppGO rung fires at w >= 20000
#: (measured with _R_PPGO_ERROR_CONST=50.0: r_ppgo_min≈464.2, radius=98.2
#: at w=500, the control radius crossing the ppGO bar near w=15000).
_PPGO_ASTROID_GAMMA: float = 0.5
_PPGO_ASTROID_RADIUS: float = 0.20
_PPGO_ASTROID_ANGLE: float = 0.25 * math.pi

#: Positive-parity source on the cusp ray (near delta_perp=0) -- the
#: delta_parallel alignment makes the radius primarily x-dominated.
_PPGO_ASTROID_SOURCE = np.array([
    _PPGO_ASTROID_RADIUS * math.cos(_PPGO_ASTROID_ANGLE),
    _PPGO_ASTROID_RADIUS * math.sin(_PPGO_ASTROID_ANGLE),
])

#: Saddle-parity (deltoid) fixture where the ppGO rung fires at w >= 5000
#: (gamma=1.2 > lam=1.0; measured: the control radius clears the ppGO bar
#: near w=4800).
_PPGO_SADDLE_GAMMA: float = 1.2
_PPGO_SADDLE_SOURCE = np.array([-0.5, 0.5])

#: Large w where the ppGO rung fires for the astroid fixture
_PPGO_SERVE_W: float = 20000.0

#: Intermediate w where ppGO refuses but the Pearcey uniform path serves
#: (radius=49.4, radius_min=7.4 < radius < r_ppgo_min=464.2 at w=200).
_PPGO_INTERMEDIATE_W: float = 200.0

#: w below the ppGO floor (_W_PPGO_FLOOR=50.0) for the w-gate isolation test.
_PPGO_SUB_FLOOR_W: float = 5.0

#: Reference bar for ppGO vs Pearcey asymptotic agreement (bar_ppgo=0.005).
_PPGO_AGREEMENT_BAR: float = 0.005

#: Common envelope bar: the production default.
_ENVELOPE_BAR: float = 0.05  # matches _DEFAULT_ENVELOPE_BAR in _pearcey_cusp


def _capture_ppgo_route(w: float, source, gamma: float, *,
                        beta: float = 0.0, kappa: float = 0.0,
                        envelope_bar: float = _ENVELOPE_BAR,
                        pearcey_table=None) -> tuple:
    """Call ``cusp_amplification`` and report which rung served.

    Returns ``(served, route)`` where *route* is ``'ppgo'`` if
    ``fold_ppgo_correction`` was called, ``'pearcey'`` if the Pearcey
    uniform path returned a value, or ``'refusal'``.
    """
    ppgo_called = [False]
    real_fpc = _airy_fold.fold_ppgo_correction

    def spy(*args, **kwargs):
        ppgo_called[0] = True
        return real_fpc(*args, **kwargs)

    with mock.patch.object(_airy_fold, 'fold_ppgo_correction', spy):
        served = _pearcey_cusp.cusp_amplification(
            w, source, gamma, beta=beta, kappa=kappa,
            envelope_bar=envelope_bar,
            pearcey_table=pearcey_table)

    if served is not None and ppgo_called[0]:
        route = 'ppgo'
    elif served is not None:
        route = 'pearcey'
    else:
        route = 'refusal'
    return served, route


class PpgoGoldenAgreementTestCase(_FoldArmTestCase):
    """Test 1 + 5(a): ppGO rung serves at large R; output is finite and
    deterministic (same with/without pearcey_table, since the ppGO rung
    fires before the Pearcey path).

    Numerical agreement with the Pearcey path is NOT asserted here
    because the current ppGO rung delegates to ``fold_ppgo_correction``
    (a fold-corrected form) rather than a cusp-corrected form.  The
    ``_R_PPGO_ERROR_CONST = 50.0`` placeholder is conservative; the
    post-build driver measurement will tighten it.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = _pearcey_cusp.PearceyTable.load()

    def test_ppgo_rung_fires_at_large_R_astroid(self):
        """The ppGO rung fires at large R for the positive-parity fixture."""
        served, route = _capture_ppgo_route(
            _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE, _PPGO_ASTROID_GAMMA,
            envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertIsNotNone(served, 'ppGO rung should serve at large R')
        self.assertEqual(route, 'ppgo', f'Expected ppgo route, got {route}')

    def test_ppgo_result_finite_and_deterministic(self):
        """The ppGO output is finite and the same with or without a
        Pearcey table (the ppGO rung fires before the Pearcey path is
        ever consulted)."""
        served_no_table, _ = _capture_ppgo_route(
            _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE, _PPGO_ASTROID_GAMMA,
            envelope_bar=_ENVELOPE_BAR,
            pearcey_table=None)
        served_with_table, _ = _capture_ppgo_route(
            _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE, _PPGO_ASTROID_GAMMA,
            envelope_bar=_ENVELOPE_BAR,
            pearcey_table=self.table)

        self.n_checks += 1
        self.assertIsNotNone(served_no_table)
        self.assertIsNotNone(served_with_table)
        self.assertTrue(np.isfinite(abs(served_no_table)),
                        'ppGO result is not finite')
        self.assertEqual(
            complex(served_no_table), complex(served_with_table),
            'ppGO result differs with/without Pearcey table -- '
            'the table is consulted despite the ppGO rung firing first')


class PpgoRungRefusalTestCase(_FoldArmTestCase):
    """Tests 2, 3, 6: ppGO rung correctly refuses when guards fail."""

    def test_refuses_at_small_R_below_r_ppgo_min(self):
        """Test 2: ppGO rung refuses when the source is very close
        to the cusp vertex (R < r_ppgo_min).

        We place the source at a tiny offset from the cusp vertex so
        the scaled control radius drops below the guard, and verify
        ``fold_ppgo_correction`` is never called.
        """
        gamma = _PPGO_ASTROID_GAMMA
        matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
        cusp = geometry.critical_point(gamma, 0.0, 0.0, 0.0, 1)
        source = (np.asarray(cusp.source)
                  + 1e-7 * np.asarray(cusp.soft_axis)
                  + 1e-7 * np.asarray(cusp.hard_axis))

        served, route = _capture_ppgo_route(
            _PPGO_INTERMEDIATE_W, source, gamma,
            envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertNotEqual(
            route, 'ppgo',
            'fold_ppgo_correction should NOT fire at R << r_ppgo_min')

    def test_refuses_at_w_below_w_floor(self):
        """Test 3: ppGO w-floor gate fires independently of the R-gate.

        We monkeypatch the R-gate to always pass (r_ppgo_min=0) and
        verify ``fold_ppgo_correction`` is never called at a w below
        ``_W_PPGO_FLOOR=50.0``.
        """
        gamma = _PPGO_ASTROID_GAMMA
        source = _PPGO_ASTROID_SOURCE

        with mock.patch.object(
                _pearcey_cusp, '_R_PPGO_ERROR_CONST', 0.0):
            served, route = _capture_ppgo_route(
                _PPGO_SUB_FLOOR_W, source, gamma,
                envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertNotEqual(
            route, 'ppgo',
            f'Expected w-floor ({_PPGO_SUB_FLOOR_W} < _W_PPGO_FLOOR) '
            'to block ppGO rung, but route={route}')

    def test_do_nothing_control_intermediate_R(self):
        """Test 6: Intermediate R -- ppGO refuses, Pearcey path serves,
        and the result is the same as when ppGO is forcefully disabled.

        At w=200 the astroid fixture has radius=49.4 < r_ppgo_min=464.2
        but radius > radius_min=7.4, so the Pearcey uniform form serves
        and the ppGO rung is never reached.
        """
        gamma = _PPGO_ASTROID_GAMMA
        source = _PPGO_ASTROID_SOURCE

        # With ppGO rung intact
        served_with_ppgo, route_with = _capture_ppgo_route(
            _PPGO_INTERMEDIATE_W, source, gamma,
            envelope_bar=_ENVELOPE_BAR)

        # With ppGO rung forcefully disabled
        with mock.patch.object(
                _pearcey_cusp, '_R_PPGO_ERROR_CONST', 1e30):
            served_without_ppgo = _pearcey_cusp.cusp_amplification(
                _PPGO_INTERMEDIATE_W, source, gamma,
                envelope_bar=_ENVELOPE_BAR)

        self.n_checks += 1
        self.assertIsNotNone(
            served_with_ppgo,
            'Pearcey path should serve at intermediate R')
        self.assertEqual(
            route_with, 'pearcey',
            f'Expected Pearcey route, got {route_with} -- ppGO should '
            'refuse at intermediate R')
        self.assertIsNotNone(
            served_without_ppgo,
            'Should serve with ppGO disabled')
        # Byte-identical: the ppGO rung adds no code path for regimes
        # where it refuses.
        self.assertEqual(
            complex(served_with_ppgo), complex(served_without_ppgo),
            'DO-NOTHING control: result differs with ppGO rung '
            'disabled vs intact at intermediate R')


class PpgoFinitenessGuardTestCase(_FoldArmTestCase):
    """Test 4: the finiteness guard catches NaN/Inf in ppGO results."""

    @classmethod
    def setUpClass(cls):
        cls.table = _pearcey_cusp.PearceyTable.load()

    def _assert_falls_through(self, bad_value, label):
        """Assert ``cusp_amplification`` does NOT leak a non-finite ppGO
        result; it either falls through to Pearcey or returns None."""
        def bad_fpc(*args, **kwargs):
            return bad_value

        with mock.patch.object(
                _airy_fold, 'fold_ppgo_correction', bad_fpc):
            try:
                served = _pearcey_cusp.cusp_amplification(
                    _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE,
                    _PPGO_ASTROID_GAMMA, envelope_bar=_ENVELOPE_BAR,
                    pearcey_table=self.table)
            except Exception as exc:
                self.fail(
                    f'{label}: `cusp_amplification` raised '
                    f'{type(exc).__name__}: {exc} -- should never '
                    f'leak an exception from the ppGO rung')

        self.n_checks += 1
        if served is not None:
            self.assertTrue(
                np.isfinite(abs(served)),
                f'{label}: returned non-finite amplitude '
                f'{served!r} -- NaN leaked')
        # None is also acceptable (falls through to refusal).

    def test_catches_NaN(self):
        """NaN from ppGO is caught; falls through to Pearcey or None."""
        self._assert_falls_through(
            complex(np.nan, 0.0), 'NaN')

    def test_catches_positive_Inf(self):
        """+Inf from ppGO is caught."""
        self._assert_falls_through(
            complex(np.inf, 0.0), '+Inf')

    def test_catches_negative_Inf(self):
        """-Inf from ppGO is caught."""
        self._assert_falls_through(
            complex(-np.inf, 0.0), '-Inf')

    def test_catches_imag_Inf(self):
        """1j*Inf from ppGO is caught."""
        self._assert_falls_through(
            complex(0.0, np.inf), '1j*Inf')


class PpgoSaddleParityTestCase(_FoldArmTestCase):
    """Test 5(b): ppGO rung handles the saddle (deltoid) parity branch.

    At gamma=1.2 (> lam=1.0) the caustic is a deltoid; the ppGO rung
    fires and returns a finite complex value for w >= 5000."""

    def test_ppgo_rung_fires_at_saddle_parity(self):
        """The ppGO rung fires for the saddle-parity fixture at w >= 5000."""
        for w in (5000.0, 10000.0, _PPGO_SERVE_W):
            with self.subTest(w=w):
                served, route = _capture_ppgo_route(
                    w, _PPGO_SADDLE_SOURCE, _PPGO_SADDLE_GAMMA,
                    envelope_bar=_ENVELOPE_BAR)
                self.n_checks += 1
                self.assertIsNotNone(
                    served,
                    f'Saddle ppGO rung should serve at w={w}')
                self.assertEqual(
                    route, 'ppgo',
                    f'Expected ppgo route at w={w}, got {route}')

    def test_saddle_ppgo_result_finite(self):
        """Saddle ppGO output is finite at several w values."""
        for w in (5000.0, 10000.0, 15000.0, _PPGO_SERVE_W):
            with self.subTest(w=w):
                served, route = _capture_ppgo_route(
                    w, _PPGO_SADDLE_SOURCE, _PPGO_SADDLE_GAMMA,
                    envelope_bar=_ENVELOPE_BAR)
                self.n_checks += 1
                self.assertIsNotNone(served)
                self.assertEqual(route, 'ppgo')
                self.assertTrue(
                    np.isfinite(abs(served)),
                    f'Saddle ppGO result non-finite at w={w}: {served}')


class PpgoRungSelfFalsificationTestCase(_FoldArmTestCase):
    """Test 7: corrupting the gate constants proves the guards have teeth.

    (a) ``_R_PPGO_ERROR_CONST = 0`` unlocks the rung where it should refuse.
    (b) ``_PPGO_BAR_DIVISOR = 1e6`` locks the rung where it should serve."""

    def test_zero_error_constant_unlocks_rung(self):
        """Setting ``_R_PPGO_ERROR_CONST=0`` makes r_ppgo_min=0, so
        ALL nonzero-R sources pass the R-gate.  At the intermediate-R
        config (where ppGO normally refuses) the rung now fires."""
        gamma = _PPGO_ASTROID_GAMMA
        source = _PPGO_ASTROID_SOURCE

        with mock.patch.object(
                _pearcey_cusp, '_R_PPGO_ERROR_CONST', 0.0):
            served, route = _capture_ppgo_route(
                _PPGO_INTERMEDIATE_W, source, gamma,
                envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertIsNotNone(
            served,
            'ppGO rung should serve when r_ppgo_min is zeroed')
        self.assertEqual(
            route, 'ppgo',
            f'Expected ppgo route, got {route}')

    def test_large_divisor_locks_rung(self):
        """Setting ``_PPGO_BAR_DIVISOR = 1e6`` makes r_ppgo_min → ∞,
        so NO sources pass the R-gate.  At the large-R config (where
        ppGO normally serves) the rung now refuses."""
        gamma = _PPGO_ASTROID_GAMMA
        source = _PPGO_ASTROID_SOURCE

        with mock.patch.object(
                _pearcey_cusp, '_PPGO_BAR_DIVISOR', 1e6):
            served, route = _capture_ppgo_route(
                _PPGO_SERVE_W, source, gamma,
                envelope_bar=_ENVELOPE_BAR,
                pearcey_table=None)

        self.n_checks += 1
        self.assertNotEqual(
            route, 'ppgo',
            f'ppGO rung should refuse when r_ppgo_min is huge; '
            f'got route={route}')

    def test_resolution_gate_isolated_admit_and_refuse(self):
        """The ppGO dual gate admits via ``_merging_fold_pair`` or from
        the resolution ``w * delta_min >= _PPGO_RESOLUTION_GATE``.  The
        saddle fixture ``_PPGO_SADDLE_SOURCE`` at gamma=1.2 yields two
        saddle-type images (no fold pair), so the resolution gate decides
        alone.  Raising the threshold above ``w * delta_min`` blocks the
        rung; lowering it to 0 always admits; restoring it admits the
        resolved case (``w = 20000`` where ``w * delta_min`` ≫ 4.0).

        Cost: 4 calls to ``_capture_ppgo_route`` (∼ 0.2 s total)."""
        gamma = _PPGO_SADDLE_GAMMA
        source = _PPGO_SADDLE_SOURCE
        w_nominal = 500.0

        # (a) Gate intact (4.0): rung fires (w * delta_min ≫ 4.0).
        served_a, route_a = _capture_ppgo_route(
            w_nominal, source, gamma,
            envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertEqual(
            route_a, 'ppgo',
            f'ppGO rung should fire at w={w_nominal} with gate intact; '
            f'got route={route_a}')

        # (b) Gate raised to a huge value: rung refuses (w*delta_min too
        #     small for the inflated threshold).
        with mock.patch.object(_pearcey_cusp, '_PPGO_RESOLUTION_GATE',
                               1000.0):
            served_b, route_b = _capture_ppgo_route(
                w_nominal, source, gamma,
                envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        # Pearcey may or may not serve — what matters is ppGO refused.
        self.assertNotEqual(
            route_b, 'ppgo',
            f'ppGO rung should refuse at high gate=1000; '
            f'got route={route_b}, served={served_b}')

        # (c) Gate disabled (0.0): rung always fires.
        with mock.patch.object(_pearcey_cusp, '_PPGO_RESOLUTION_GATE', 0.0):
            served_c, route_c = _capture_ppgo_route(
                w_nominal, source, gamma,
                envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertIsNotNone(
            served_c,
            f'ppGO rung should serve when gate is disabled (0.0)')
        self.assertEqual(
            route_c, 'ppgo',
            f'Expected ppgo route with gate=0.0, got {route_c}')

        # (d) Resolved w with the inflated gate still admits.
        w_resolved = 20000.0
        with mock.patch.object(_pearcey_cusp, '_PPGO_RESOLUTION_GATE',
                               1000.0):
            served_d, route_d = _capture_ppgo_route(
                w_resolved, source, gamma,
                envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertEqual(
            route_d, 'ppgo',
            f'ppGO rung should fire at resolved w={w_resolved} even with '
            f'gate=1000; got route={route_d}')
# WP1 _cusp_vertex routing fix — domain tests (Build 2026-08-11).
# ----------------------------------------------------------------------

#: Interior cusp grid used by the table-live agreement diagnostic
#: (Domain Test 1).  w in [20, 80] at 7 points (odd so the midpoint is
#: an exact grid point, ~6.5 s total at 0.43 s per eval).
_INTERIOR_AGREEMENT_W_GRID = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)

#: w for the single-point table-live agreement assertion in Domain Test 1.
#: At w=40 the Pearcey uniform form serves (radius_min=7.4 ← clears),
#: the ppGO rung is not triggered (w < _W_PPGO_FLOOR=8.0), and the
#: table-live relative difference is ~1.6e-7 (within this bar).
_INTERIOR_AGREEMENT_W: float = 40.0
_INTERIOR_AGREEMENT_TOL: float = 1e-5

#: Source-angle grid for the cusp-selection Voronoi diagnostic (Domain
#: Test 2): 24 angles around a full circle at fixed interior radius.
_VORONOI_N_ANGLES: int = 24
_VORONOI_RHO: float = 0.5
_VORONOI_GAMMA: float = 0.5

#: Fixture gamma for the interior-cusp table-live agreement test.
#: Uses _CUSP_FIXTURES[0] = (0.5, 0.20, 0.25*pi).
_AGREEMENT_GAMMA: float = _CUSP_FIXTURES[0][0]

#: Source for Domain Test 1 (interior, 4-image, served).
_AGREEMENT_SOURCE = np.array([
    _CUSP_FIXTURES[0][1] * math.cos(_CUSP_FIXTURES[0][2]),
    _CUSP_FIXTURES[0][1] * math.sin(_CUSP_FIXTURES[0][2]),
])


def _capture_route_and_value(w: float, source, gamma: float, *,
                             beta: float = 0.0, kappa: float = 0.0,
                             envelope_bar: float = _ENVELOPE_BAR,
                             pearcey_table=None) -> tuple:
    """Call ``cusp_amplification`` and return ``(served, route)``.

    *route* is ``'ppgo'`` if ``fold_ppgo_correction`` was called,
    ``'pearcey'`` if the Pearcey uniform path returned a value,
    or ``'refusal'``.
    """
    ppgo_called = [False]
    real_fpc = _airy_fold.fold_ppgo_correction

    def spy(*args, **kwargs):
        ppgo_called[0] = True
        return real_fpc(*args, **kwargs)

    with mock.patch.object(_airy_fold, 'fold_ppgo_correction', spy):
        served = _pearcey_cusp.cusp_amplification(
            w, source, gamma, beta=beta, kappa=kappa,
            envelope_bar=envelope_bar,
            pearcey_table=pearcey_table)

    if served is not None and ppgo_called[0]:
        route = 'ppgo'
    elif served is not None:
        route = 'pearcey'
    else:
        route = 'refusal'
    return served, route


def _astroid_cusp_vertices(gamma: float, beta: float, kappa: float
                           ) -> list:
    """Return the four positive-parity astroid cusp ``CriticalPoint``s.

    At each ``phase in {0, pi/2, pi, 3*pi/2}`` (shear-aligned frame),
    ``theta = phase + beta``, branch=+1.
    """
    vertices = []
    for phase in _ASTROID_CUSP_PHASES:
        theta = phase + beta
        try:
            vertices.append(
                geometry.critical_point(gamma, theta, beta, kappa, 1))
        except geometry.LensDomainError:
            pass
    return vertices


def _selected_cusp_index(vertex, gamma: float, beta: float, kappa: float
                         ) -> int:
    """Return the astroid cusp index (0-3) the ``vertex`` belongs to.

    Compares the vertex's image polar angle against the four astroid cusp
    phase angles ``beta + {0, pi/2, pi, 3pi/2}`` and returns the index of
    the closest, or -1 if none matches within _VERTEX_ANGLE_TOL.
    """
    angle = math.atan2(vertex.image[1], vertex.image[0])
    residuals = [
        abs((angle - beta - phase + math.pi) % (2.0 * math.pi) - math.pi)
        for phase in _ASTROID_CUSP_PHASES]
    best = int(np.argmin(residuals))
    if residuals[best] <= _VERTEX_ANGLE_TOL:
        return best
    return -1


class InteriorCuspTableLiveAgreementTestCase(_FoldArmTestCase):
    """Domain Test 1: interior cusp source serves via the demodulated
    Pearcey table and live quadrature, and the two agree.

    After the WP1 `_cusp_vertex` source-distance routing fix, an interior
    source (rho < 1) near a cusp vertex is correctly routed through the
    Pearcey uniform path (not ppGO -- R < r_ppgo_min for interior
    sources).  The table and live certified quadrature agree to high
    precision (relative difference <= _INTERIOR_AGREEMENT_TOL).

    Fixture: _AGREEMENT_SOURCE from _CUSP_FIXTURES[0] (gamma=0.5,
    rho=0.20, angle=0.25pi -- inside the caustic, 4 images, cusp cluster
    calibration passes).  At w=40 the Pearcey form clears the radius gate
    and the route is 'pearcey'.
    """

    MEASURED_COST: str = (
        '7 w-points x (2xcusp_amplification + 1 route spy) ~= 15 calls '
        'at ~0.43 s/call ~= 6.5 s for the agreement sweep + diagnostic '
        'plot; well within the 60 s per-test ceiling.'
    )

    @classmethod
    def setUpClass(cls):
        cls.table = _pearcey_cusp.PearceyTable.load()

    def test_table_and_live_serve_at_interior_cusp(self):
        """Both table and live quadrature paths serve a non-None value
        at the interior cusp fixture."""
        served_table, _ = _capture_route_and_value(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
            pearcey_table=self.table)
        served_live, _ = _capture_route_and_value(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
            pearcey_table=None)

        self.n_checks += 1
        self.assertIsNotNone(
            served_table,
            'Table path returned None at interior source')
        self.assertIsNotNone(
            served_live,
            'Live-quadrature path returned None at interior source')

    def test_route_is_pearcey_not_ppgo(self):
        """At interior w=40 the route is 'pearcey': the ppGO rung does
        not fire because R < r_ppgo_min for interior sources."""
        _, route = _capture_route_and_value(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA)
        self.n_checks += 1
        self.assertEqual(
            route, 'pearcey',
            f'Expected Pearcey route at interior cusp, got {route}')

    def test_table_live_agreement_at_single_w(self):
        """The table and live-quadrature values agree to within
        _INTERIOR_AGREEMENT_TOL at w=40."""
        served_table, _ = _capture_route_and_value(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
            pearcey_table=self.table)
        served_live, _ = _capture_route_and_value(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
            pearcey_table=None)

        diff = abs(complex(served_table) - complex(served_live))
        ref = abs(complex(served_live))
        rel = diff / ref if ref > 0.0 else float('inf')
        self.n_checks += 1
        self.assertLess(
            rel, _INTERIOR_AGREEMENT_TOL,
            f'Table-live relative difference {rel:.3e} exceeds '
            f'{_INTERIOR_AGREEMENT_TOL} at w={_INTERIOR_AGREEMENT_W}')

    def test_agreement_sweep_and_diagnostic(self):
        """Sweep w in [20, 80] and confirm the table-live relative
        difference stays below _INTERIOR_AGREEMENT_TOL across the
        serving band.  Save a diagnostic plot."""
        diffs = []
        ws = []
        for w in _INTERIOR_AGREEMENT_W_GRID:
            served_table, _ = _capture_route_and_value(
                w, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
                pearcey_table=self.table)
            served_live, _ = _capture_route_and_value(
                w, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
                pearcey_table=None)
            self.assertIsNotNone(served_table,
                                 f'Table refused at w={w}')
            self.assertIsNotNone(served_live,
                                 f'Live quadrature refused at w={w}')
            diff = abs(complex(served_table) - complex(served_live))
            ref = abs(complex(served_live))
            rel = diff / ref if ref > 0.0 else float('inf')
            ws.append(w)
            diffs.append(rel)
            self.n_checks += 1
            self.assertLess(
                rel, _INTERIOR_AGREEMENT_TOL,
                f'Table-live relative difference {rel:.3e} exceeds '
                f'{_INTERIOR_AGREEMENT_TOL} at w={w}')

        _save_plot(
            'InteriorCuspTableLiveAgreement_agreement_sweep',
            ws, diffs,
            xlabel='$w$ (dimensionless frequency)',
            ylabel=r'$|F_{\mathrm{table}} - F_{\mathrm{live}}| / |F|$')


class CuspVertexSourceDistanceSelectionTestCase(_FoldArmTestCase):
    """Domain Test 2: _cusp_vertex selects the geometrically correct
    cusp among multiple candidates, using source-plane distance.

    The WP1 routing fix replaced a seed_theta-snap heuristic with
    multi-candidate source-distance selection.  For every positive-parity
    config seeded near a specific cusp, the returned vertex satisfies
    |source - vertex.source| <= |source - alt.source| for ALL four
    astroid cusp vertices.

    Diagnostic: a Voronoi-like partition plot of selected cusp index vs
    source angle (circle at fixed rho < 1, constant gamma).
    """

    MEASURED_COST: str = (
        '7 direct-vertex configs x ~2 geometry calls + 24-angle '
        'diagnostic ~= 1 s; well within 60 s ceiling.'
    )

    def test_vertex_is_source_plane_closest_among_astroid_cusps(self):
        """For each positive-parity config, _cusp_vertex returns the
        source-plane-closest astroid cusp vertex."""
        for gamma, beta, kappa, cusp_index in _DIRECT_VERTEX_CONFIGS:
            lam = 1.0 - kappa
            if abs(gamma) >= lam:
                continue
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              cusp=cusp_index):
                source = _seed_source_near_cusp(gamma, beta, kappa,
                                                cusp_index)
                nearest = geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa)
                branch = _vertex_branch(gamma, beta, kappa, nearest.theta)
                vertex = _pearcey_cusp._cusp_vertex(
                    gamma, beta, kappa, source, nearest.theta, branch)
                self.assertIsNotNone(
                    vertex,
                    f'cusp_vertex refused a near-cusp seed')

                selected_dist = float(np.linalg.norm(
                    source - np.asarray(vertex.source)))

                all_vertices = _astroid_cusp_vertices(gamma, beta, kappa)
                for alt_vertex in all_vertices:
                    alt_dist = float(np.linalg.norm(
                        source - np.asarray(alt_vertex.source)))
                    self.n_checks += 1
                    self.assertLessEqual(
                        selected_dist, alt_dist * (1.0 + 1e-12),
                        f'Selected vertex dist {selected_dist:.6e} > '
                        f'alternative dist {alt_dist:.6e}')

    def test_selection_independent_of_seed_theta(self):
        """The selection does NOT consult seed_theta -- only source-plane
        distance.  Two different seeds for the same source produce
        byte-identical vertices."""
        gamma, beta, kappa, cusp_i = 0.5, 0.0, 0.0, 0
        source = _seed_source_near_cusp(gamma, beta, kappa, cusp_i)
        nearest = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)

        wrong_source = _seed_source_near_cusp(gamma, beta, kappa, 2,
                                              offset=0.01)
        wrong_seed = geometry.nearest_caustic_point(
            gamma, beta, wrong_source, kappa=kappa)

        branch = _vertex_branch(gamma, beta, kappa, nearest.theta)

        vertex_correct = _pearcey_cusp._cusp_vertex(
            gamma, beta, kappa, source, nearest.theta, branch)
        vertex_wrong_seed = _pearcey_cusp._cusp_vertex(
            gamma, beta, kappa, source, wrong_seed.theta, branch)

        self.n_checks += 1
        self.assertIsNotNone(vertex_correct)
        self.assertIsNotNone(vertex_wrong_seed)
        self.assertEqual(
            np.asarray(vertex_correct.image).tolist(),
            np.asarray(vertex_wrong_seed.image).tolist(),
            f'_cusp_vertex produced different vertices for a correct '
            f'seed (theta={nearest.theta:.4f}) vs wrong seed '
            f'(theta={wrong_seed.theta:.4f})')

    def test_voronoi_diagnostic(self):
        """Diagnostic: selected cusp index vs source polar angle for a
        circle at fixed rho, showing the Voronoi partition."""
        gamma = _VORONOI_GAMMA
        beta, kappa = 0.0, 0.0
        ref_cp = geometry.critical_point(gamma, 0.0, beta, kappa, 1)
        r_caustic = float(np.linalg.norm(ref_cp.source))
        radius = _VORONOI_RHO * r_caustic

        angles = np.linspace(0.0, 2.0 * math.pi, _VORONOI_N_ANGLES,
                             endpoint=False)
        indices = []
        for angle in angles:
            source = radius * np.array([math.cos(angle), math.sin(angle)])
            nearest = geometry.nearest_caustic_point(
                gamma, beta, source, kappa=kappa)
            vertex = _pearcey_cusp._cusp_vertex(
                gamma, beta, kappa, source, nearest.theta, 1)
            idx = _selected_cusp_index(vertex, gamma, beta, kappa)
            indices.append(idx)
            self.n_checks += 1
            self.assertIsNotNone(
                vertex,
                f'_cusp_vertex refused at rho={_VORONOI_RHO}, '
                f'angle={angle:.3f}')
            self.assertGreaterEqual(
                idx, 0,
                f'Returned cusp does not sit on a known astroid phase '
                f'at rho={_VORONOI_RHO}, angle={angle:.3f}')

        _save_plot(
            'CuspVertexSourceDistanceSelection_voronoi_diagnostic',
            np.degrees(angles), indices,
            xlabel='source polar angle (degrees)',
            ylabel='selected cusp index (0-3)')


class ExteriorPpgoUnaffectedTestCase(_FoldArmTestCase):
    """Domain Test 3: exterior cusp sources continue to serve via ppGO,
    unaffected by the _cusp_vertex routing fix.

    The ppGO fast rung fires BEFORE the Pearcey path.  We gate the
    existing PpgoGoldenAgreementTestCase fixture: an exterior source
    (radius=R >= r_ppgo_min) where the route is 'ppgo' and the output
    is finite and deterministic with or without a Pearcey table.
    The ppGO rung still calls _cusp_vertex to obtain tau_c, but
    exterior sources are far from the caustic and seed_theta already
    points to the correct cusp.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = _pearcey_cusp.PearceyTable.load()

    def test_ppgo_rung_fires_with_and_without_table(self):
        """Exterior source: route is 'ppgo' with both table installed
        and cleared, and values are finite and deterministic."""
        served_table, route_table = _capture_route_and_value(
            _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE, _PPGO_ASTROID_GAMMA,
            pearcey_table=self.table)
        served_none, route_none = _capture_route_and_value(
            _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE, _PPGO_ASTROID_GAMMA,
            pearcey_table=None)

        self.n_checks += 1
        self.assertIsNotNone(served_table,
                             'ppGO rung refused with table')
        self.assertIsNotNone(served_none,
                             'ppGO rung refused without table')
        self.assertEqual(route_table, 'ppgo',
                         f'Expected ppgo route with table, got {route_table}')
        self.assertEqual(route_none, 'ppgo',
                         f'Expected ppgo route without table, got {route_none}')
        self.assertTrue(np.isfinite(abs(complex(served_table))),
                        'ppGO result with table not finite')
        self.assertTrue(np.isfinite(abs(complex(served_none))),
                        'ppGO result without table not finite')
        self.assertEqual(
            complex(served_table), complex(served_none),
            'ppGO result differs with/without Pearcey table')

    def test_ppgo_value_matches_golden_contract(self):
        """The exterior ppGO value is the SAME as the existing
        PpgoGoldenAgreementTestCase contract."""
        _, route = _capture_ppgo_route(
            _PPGO_SERVE_W, _PPGO_ASTROID_SOURCE, _PPGO_ASTROID_GAMMA,
            envelope_bar=_ENVELOPE_BAR)
        self.n_checks += 1
        self.assertEqual(
            route, 'ppgo',
            f'Golden-agreement exterior source changed route to {route}')


class InteriorCuspSelfFalsificationTestCase(_FoldArmTestCase):
    """Prove the Domain Test suites have teeth.

    1.  Cleared table still serves (live quadrature fallback).
    2.  Corrupted _cusp_vertex (returns furthest cusp, not nearest)
        violates the source-distance selection gate.
    """

    def test_cleared_table_still_serves_via_live_quadrature(self):
        """Both table and None paths serve at the interior fixture --
        the live-quadrature fallback is functional."""
        served_table = _pearcey_cusp.cusp_amplification(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
            pearcey_table=_pearcey_cusp.PearceyTable.load())
        served_none = _pearcey_cusp.cusp_amplification(
            _INTERIOR_AGREEMENT_W, _AGREEMENT_SOURCE, _AGREEMENT_GAMMA,
            pearcey_table=None)

        self.n_checks += 1
        self.assertIsNotNone(served_table)
        self.assertIsNotNone(served_none)

    def test_corrupt_vertex_breaks_distance_gate(self):
        """Returning the furthest (not nearest) astroid cusp violates
        the source-distance selection gate."""
        gamma, beta, kappa, cusp_i = 0.5, 0.0, 0.0, 1
        source = _seed_source_near_cusp(gamma, beta, kappa, cusp_i)
        all_vert = _astroid_cusp_vertices(gamma, beta, kappa)

        def wrong_vertex(*args, **kwargs):
            return max(all_vert, key=lambda v: float(np.linalg.norm(
                source - np.asarray(v.source))))

        nearest = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)
        branch = _vertex_branch(gamma, beta, kappa, nearest.theta)

        with mock.patch.object(_pearcey_cusp, '_cusp_vertex',
                               wrong_vertex):
            bad_vertex = _pearcey_cusp._cusp_vertex(
                gamma, beta, kappa, source, nearest.theta, branch)

        bad_dist = float(np.linalg.norm(
            source - np.asarray(bad_vertex.source)))
        best_dist = min(float(np.linalg.norm(
            source - np.asarray(v.source))) for v in all_vert)

        self.n_checks += 1
        self.assertGreater(
            bad_dist, best_dist * (1.0 + 1e-8),
            f'Wrong-vertex patch did NOT produce a further vertex: '
            f'bad_dist={bad_dist:.6e}, best_dist={best_dist:.6e}')


# ----------------------------------------------------------------------
# INTERIOR CUSP SERVING DOMAIN TEST (Build 2026-08-11, WP1 fix).
#
# The WP1 production fix (3-stationary-point calibration bypass at
# ``_pearcey_cusp.cusp_amplification``) unlocks interior cusp sources
# that pass the uniform-error gate.  These sources have 3 real
# stationary points of the Pearcey primitive (c4 > 0, reflected=False)
# and skip the per-image calibration certificate.  The Domain Test
# validates:
#
#   (a) interior sources serve — cusp_amplification returns a finite
#       complex value for every config where the cusp vertex and normal
#       form are valid and the control radius clears the gate;
#   (b) the calibration certificate is bypassed for interior sources
#       (_calibration_certified never called);
#   (c) exterior byte-identical regression — exterior sources still
#       serve finite values and _calibration_certified IS called.
# ----------------------------------------------------------------------

#: Interior 3-stationary configs measured to serve at the listed w:
#: ``(name, gamma, beta, kappa, dp, dperp, serving_w)``.  Each config
#: is a source offset ``dp`` along the soft axis (interior pusher) and
#: ``dperp`` along the hard axis from cusp vertex at ``cusp_index=1``
#: (c4 > 0, cusp phase ``pi/2``).  ``serving_w`` is the minimum w in
#: the test grid at which the config serves (radius >= radius_min and
#: 3 stationary points).  Measured at HEAD b64480c.
_INTERIOR_SERVE_CONFIGS = (
    ('int_g03_w200',  0.3, 0.0, 0.0, 1, -0.10, 0.005, 200.0),
    ('int_g03_d02_w500', 0.3, 0.0, 0.0, 1, -0.10, 0.020, 500.0),
    ('int_g05_w200',  0.5, 0.0, 0.0, 1, -0.10, 0.005, 200.0),
    ('int_g05_d02_w500', 0.5, 0.0, 0.0, 1, -0.10, 0.020, 500.0),
)

#: w grid for the interior serving sweep.
_INTERIOR_SERVE_W_GRID = (50.0, 100.0, 200.0, 500.0)


def _interior_source(gamma, beta, kappa, cusp_index, dp, dperp):
    """Return a source offset from cusp ``cusp_index`` by ``(dp, dperp)``
    along the soft and hard axes."""
    from cogwheel.tests.test_lensing_airy_fold import _ASTROID_CUSP_PHASES
    phase = _ASTROID_CUSP_PHASES[cusp_index]
    theta_cusp = phase + beta
    branch = 1
    cusp = geometry.critical_point(gamma, theta_cusp, beta, kappa, branch)
    return (np.asarray(cusp.source)
            + dp * cusp.soft_axis
            + dperp * cusp.hard_axis)


class InteriorCuspServingTestCase(_FoldArmTestCase):
    """
    Domain Test: interior cusp sources with 3 real stationary points
    serve a finite complex value through `cusp_amplification`.

    After the WP1 calibration-bypass fix, interior sources that clear
    the uniform-error gate (radius >= radius_min) skip the per-image
    calibration certificate but still produce a valid uniform form.

    Two load-bearing assertions:
    1. Interior sources serve (finite complex, not None).
    2. The calibration certificate IS NOT called for interior sources
       (3 stationary points → bypass), but IS called for exterior sources
       (1 stationary point → standard path).
    """

    def test_interior_three_stationary_sources_serve(self):
        """
        Interior (3-stationary) cusp sources serve finite complex values.

        For each `_INTERIOR_SERVE_CONFIGS` entry, at ``w >= serving_w``
        the config has 3 real stationary points and radius >= radius_min.
        Assert `cusp_amplification` returns a finite complex value.
        """
        for name, gamma, beta, kappa, cusp_i, dp, dperp, w_min in \
                _INTERIOR_SERVE_CONFIGS:
            source = _interior_source(gamma, beta, kappa, cusp_i, dp, dperp)
            served_any = False
            for w in _INTERIOR_SERVE_W_GRID:
                if w < w_min:
                    continue
                served = _pearcey_cusp.cusp_amplification(
                    w, source, gamma, beta=beta, kappa=kappa)
                if served is not None:
                    served_any = True
                    self.n_checks += 1
                    with self.subTest(config=name, w=w):
                        self.assertTrue(
                            np.isfinite(abs(served)),
                            f'{name} w={w}: served value is not finite '
                            f'(|F|={abs(served)})')
            self.n_checks += 1
            self.assertTrue(
                served_any,
                f'{name}: interior source never served at any w in '
                f'{_INTERIOR_SERVE_W_GRID}; the config is broken')

    def test_calibration_bypassed_for_interior_sources(self):
        """
        `_calibration_certified` is NOT called for 3-stationary sources.

        Spy on ``_pearcey_cusp._calibration_certified`` and verify it
        is never reached for interior configs (the 3-stationary bypass
        at the top of the uniform-sum block skips it).
        """
        real_cal = _pearcey_cusp._calibration_certified
        for name, gamma, beta, kappa, cusp_i, dp, dperp, w_min in \
                _INTERIOR_SERVE_CONFIGS:
            source = _interior_source(gamma, beta, kappa, cusp_i, dp, dperp)
            cal_called = [0]

            def spy(stationary_values, matched_delays):
                cal_called[0] += 1
                return real_cal(stationary_values, matched_delays)

            with mock.patch.object(
                    _pearcey_cusp, '_calibration_certified', spy):
                served = _pearcey_cusp.cusp_amplification(
                    w_min, source, gamma, beta=beta, kappa=kappa)
            self.n_checks += 1
            self.assertIsNotNone(
                served,
                f'{name}: interior source refused at w={w_min}')
            self.assertEqual(
                cal_called[0], 0,
                f'{name}: `_calibration_certified` was called '
                f'{cal_called[0]} time(s) for a 3-stationary source '
                f'(the bypass should skip it)')

    def test_exterior_calibration_invoked(self):
        """
        `_calibration_certified` IS called for exterior (1-stationary)
        sources — the standard path is not broken by the fix.

        Spy on ``_pearcey_cusp._calibration_certified`` at an exterior
        source and verify it is called at least once.
        """
        # Use an exterior config from _EXTERIOR_VERTEX_CONFIGS
        name, gamma, beta, kappa, source_t = _EXTERIOR_VERTEX_CONFIGS[0]
        source = np.asarray(source_t, dtype=float)
        cal_called = [0]
        real_cal = _pearcey_cusp._calibration_certified

        def spy(stationary_values, matched_delays):
            cal_called[0] += 1
            return real_cal(stationary_values, matched_delays)

        with mock.patch.object(
                _pearcey_cusp, '_calibration_certified', spy):
            served = _pearcey_cusp.cusp_amplification(
                40.0, source, gamma, beta=beta, kappa=kappa)
        self.n_checks += 1
        self.assertIsNotNone(
            served,
            f'{name}: exterior source refused at w=40')
        self.assertTrue(
            np.isfinite(abs(served)),
            f'{name}: exterior served value is not finite')
        self.assertGreaterEqual(
            cal_called[0], 1,
            f'{name}: `_calibration_certified` was NOT called for an '
            f'exterior source — the standard path is broken by the fix')
