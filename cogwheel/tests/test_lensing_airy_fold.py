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
from unittest import TestCase, expectedFailure, main, mock, skipUnless

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

#: A frequency above the Schwinger ceiling: here the geometric and
#: Schwinger rungs have already declined, so the uniform arm is the only
#: server and its refusal falls through to the NAMED Schwinger refusal.
_ABOVE_CEILING_W = 100.0

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
_CUSP_NODE_GAMMA = 0.5
_CUSP_NODE_RADIUS = 0.18
_CUSP_NODE_ANGLE = 0.3 * math.pi
_CUSP_NODE_W = 80.0


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
    The large-``xi`` arm reproduces the exact geometric two-image sum of
    an ASYMMETRIC merging fold pair, in the max-normalized ENVELOPE
    currency, at the leading uniform rate ``~ xi^{-3/2}``.

    The oracle is `geometry`'s exact ``sqrt|mu_+| e^{i w tau_+} +
    sqrt|mu_-| e^{i w tau_- - i pi/2}`` (a DIFFERENT module, no arm
    arithmetic).  The Airy amplitudes fed to `airy_fold_value` are the SUM
    ``p`` and DIFFERENCE ``q`` derived in `_farfield_amplitudes`; a
    symmetric approach would hide the sum-vs-difference assignment, so the
    fixture source sits well off the cusp axis (``sqrt|mu|`` ratio ~1.2).

    This scans ``w`` (hence ``xi``) over the merging pair and is therefore
    the brute-force accuracy tier.  The p/q swap that this fixture is
    built to expose is falsified in `FoldArmSelfFalsificationTestCase`.
    """

    #: Airy controls at which the leading uniform envelope error is
    #: measured; each ~2x the last, so the ratio witnesses the rate.
    XI_CHECKPOINTS = (40.0, 80.0, 160.0, 320.0)

    def _envelope_error(self, xi_target, pair):
        """
        Max-normalized envelope error ``max_w||F_arm| - |F_geom|| /
        (s_+ + s_-)`` over one beat window of ``w`` centred on the ``w``
        that yields ``xi_target``.
        """
        tau_plus, tau_minus, s_plus, s_minus = pair
        delta_tau = tau_minus - tau_plus
        tau_bar = 0.5 * (tau_plus + tau_minus)
        w_centre = (4.0 / 3.0 * xi_target ** 1.5) / delta_tau
        beat = 2.0 * math.pi / delta_tau
        ws = np.linspace(w_centre - beat, w_centre + beat, 80)
        arm = np.empty_like(ws)
        geom = np.empty_like(ws)
        for index, w in enumerate(ws):
            xi = (3.0 * w * delta_tau / 4.0) ** (2.0 / 3.0)
            p, q = _farfield_amplitudes(w, xi, s_plus, s_minus)
            arm[index] = abs(_airy_fold.airy_fold_value(
                w, tau_bar, xi, p, q, _SIGMA_FOLD))
            geom[index] = abs(_geometric_two_image_sum(
                w, tau_plus, tau_minus, s_plus, s_minus))
        return float(np.max(np.abs(arm - geom)) / (s_plus + s_minus))

    @_brute_accuracy_tier
    def test_far_field_envelope_matches_geometric_sum(self):
        """
        For every inside-caustic asymmetric fixture and every
        ``xi >= _XI_FARFIELD``, the envelope error clears the spec bar
        `_FARFIELD_ENVELOPE_TOL`.
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
        The envelope error decreases monotonically with ``xi`` and, per
        ``xi`` doubling, drops by ~``2^{3/2} = 2.83`` -- the signature of
        the leading uniform ``xi^{-3/2}`` term, not of an accidental
        near-cancellation.  Emits the residual-vs-``xi`` diagnostic plot.
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

    The serving check is a WIRING contract: for a served config it
    reproduces ``airy_fold_value`` evaluated at the INDEPENDENTLY
    re-derived (`geometry`) ``tau_bar`` and ``xi = (3 w DT / 4)^{2/3}``,
    with the module's calibrated leading amplitude ``(p, q = 0,
    sigma = -pi/4)``.  It never returns anything but ``None`` or a finite
    complex.  (The served amplitude is leading order in the asymmetric
    fold -- see the module docstring -- so its accuracy against the
    geometric sum is NOT asserted here; that certification lives at the
    `airy_fold_value` level with the full SUM/DIFFERENCE amplitudes in
    `AiryFoldFarFieldEnvelopeTestCase`.)
    """

    #: Frequencies high enough that the leading uniform-error estimate
    #: clears the default envelope bar for the r = 0.14 fixture (measured
    #: serving threshold sits between w = 200, refused, and w = 500).
    SERVED_WS = (500.0, 1000.0, 5000.0)

    def test_served_value_matches_independent_wiring(self):
        """
        A served value equals ``airy_fold_value`` at the geometry-derived
        ``tau_bar``/``xi`` with the module's calibrated ``(p, 0, sigma)``,
        and is finite.
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
                self._grid_value(source)

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
                self._grid_value(source)

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

#: Frequency ceiling of the exact Schwinger engine.  A node with
#: ``w > _W_CEILING`` that is not geometric-resolved is offered to the
#: uniform arms before the named refusal stands.
_W_CEILING = _schwinger.W_CEILING_SCHWINGER

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
#: the live ladder: the Schwinger node certifies at ``w = 40 <= 60``; the
#: geometric node is a resolved macro saddle (``gamma = 1.5``,
#: ``w * delta_min >= RHO_END``); the fold and cusp nodes are the
#: near-fold / near-cusp uniform corners (empirically served by exactly
#: one arm each); the refusal node is a near-caustic unresolved node both
#: arms decline.
_LADDER_NODES = (
    ('schwinger', 40.0, 0.20, 0.25 * math.pi, 0.5, 0.0, 0.0),
    ('geometric', 100.0, 1.20, _RAY_ANGLE, 1.5, 0.0, 0.0),
    ('fold', 500.0, 0.14, _RAY_ANGLE, _GAMMA, 0.0, 0.0),
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
_GEOMETRIC_NODE = (1.5, 1.20, _RAY_ANGLE, 0.0, 0.0, 100.0)

#: ``select_branch`` grid over which the frozen wave/geometric gate must
#: match HEAD exactly: ``(w, delta_min, cancellation_exp)``.
_SELECT_BRANCH_GRID = tuple(
    itertools.product((10.0, 61.0, 100.0), (0.01, 0.05, 0.2),
                      (10.0, 48.0, 49.0, 60.0)))


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

    def test_select_branch_matches_head_exactly(self):
        """The frozen wave/geometric gate is unchanged from HEAD.

        `select_branch` is the byte-frozen positive-parity gate; WP4 must
        not perturb it.  Every point of the ``(w, delta_min, L)`` grid
        must return the identical label from both modules.
        """
        head = _head_operator()
        for w, delta_min, cancellation_exp in _SELECT_BRANCH_GRID:
            with self.subTest(w=w, delta_min=delta_min, L=cancellation_exp):
                self.n_checks += 1
                self.assertEqual(
                    operator.select_branch(w, delta_min, cancellation_exp),
                    head.select_branch(w, delta_min, cancellation_exp),
                    'select_branch diverged from HEAD')

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
    production thresholds are unchanged (``L_MAX == 48``, ``select_branch``
    byte-frozen) and the census source stays pure of the exact wave
    evaluators -- and carries an ``@expectedFailure`` tripwire that flips
    RED the moment the extended-census API lands, prompting the real
    structure/determinism gate.
    """

    def test_production_thresholds_unchanged(self):
        """``L_MAX == 48`` and ``select_branch`` keeps its two-condition gate.

        The census must REPORT category (b) without moving any production
        threshold.  ``geometric`` requires BOTH resolution
        (``w*delta_min >= RHO_END``) and strong cancellation
        (``L > L_MAX``); neither alone licenses it.
        """
        self.n_checks += 1
        self.assertEqual(operator.L_MAX, _L_MAX_PINNED,
                         'production L_MAX changed; the census (b) fraction '
                         'would shift under it')
        pinned = {
            (100.0, 0.05, 49.0): 'geometric',   # resolved AND L>48
            (100.0, 0.05, 48.0): 'wave',         # L == L_MAX, not > it
            (100.0, 0.01, 49.0): 'wave',         # w*dmin = 1.0 < RHO_END
            (61.0, 0.20, 49.0): 'geometric',     # 12.2 >= 4 and 49 > 48
            (10.0, 0.20, 60.0): 'wave',          # 2.0 < RHO_END
        }
        for (w, delta_min, cancellation_exp), expected in pinned.items():
            with self.subTest(w=w, delta_min=delta_min, L=cancellation_exp):
                self.n_checks += 1
                self.assertEqual(
                    operator.select_branch(w, delta_min, cancellation_exp),
                    expected,
                    'select_branch gate moved from its frozen semantics')

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
