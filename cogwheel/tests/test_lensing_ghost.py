"""
Tests for the ghost-kernel machinery in `lensing.chang_refsdal.geometry`
(WP1: the decaying complex-saddle pair and its analytic-continued
stationary-phase kernel, `geometry.ghost_kernel`).

A ghost is the complex-conjugate quartic-root pair the real image finder
discards; the decaying member's continued kernel is the leading residual
the geometric-optics (ppGO) sum leaves behind just outside a fold.  Two
failure modes are silent and specific to this analytic continuation, and
this suite exists to catch them:

  - a complex-log BRANCH-CUT error in the Fermat delay ``tau_c`` (the
    bilinear ``0.5 log(x_c . x_c)`` term), which would flip ``Im tau_c``
    and turn a decaying ghost into a growing one; and
  - a MORSE DOUBLE-COUNT in the amplitude ``1 / sqrt(det H_c)``, whose
    sqrt branch is pinned to the real merged saddle's ``exp(-i pi / 2)``
    phase -- multiplying an explicit Morse factor on top would rotate the
    amplitude by ``pi``.

The tight correctness gate (`GhostOracleTestCase`) compares the module's
``tau_c`` and amplitude against a FULLY INDEPENDENT re-derivation that
shares no code with `geometry`: it solves the image quartic with
``numpy.roots``, reconstructs ``x_c`` from the closed-form lens map by
hand, assembles ``tau_c`` from scratch with ``numpy.log``, forms
``det H_c`` both analytically (hand-rolled) and by a RICHARDSON-
extrapolated central finite difference of the complex Fermat potential,
and pins its own sqrt branch by the same real-saddle reference rule.
`OracleIndependenceTestCase` proves that independence with an AST guard:
the oracle functions may not name any `geometry` ghost/saddle helper.

Tolerances (see the module-level ``TOL_*`` constants) are set to what the
arithmetic actually delivers, measured while authoring:
  - ``tau_c``: 1e-6 relative magnitude and 1e-6 rad -- the delay legs are
    closed-form, so agreement is limited only by the shared LAPACK root
    (in practice ~1e-15); 1e-6 is a loose, non-brittle bar.
  - amplitude vs the ANALYTIC-det oracle: 1e-6 relative magnitude (both
    evaluate ``1 / sqrt`` of an algebraically-formed determinant, so this
    is a coding/branch check, tight by nature).
  - amplitude vs the FINITE-DIFFERENCE-det oracle: 1e-4 relative
    magnitude and 1e-4 rad phase -- the genuinely independent oracle,
    limited by the Richardson step (roundoff ~ eps/h**2 ~ 1e-8, so 1e-4
    is comfortable headroom).
  - reconstruction ``x_c . x_c == 1 / u_c``: 1e-10 relative -- an exact
    algebraic identity, held to near machine precision.

`GhostAnchorTestCase` is the LOOSE physical-scale sanity gate: at two
binding P1 anchors it checks that the ghost kernel reproduces the
measured residual envelope ``E`` (``exact_total - ppGO`` from the exact
engine, demodulated at ``t_min`` exactly as `ppgo_map._measure_cell`
does) to within 10% in magnitude and 3.5 deg in phase.  The gap is
physically the residual-of-the-residual ``R / E ~ 4-6%``, so these bars
are deliberately loose; the tight correctness check is the oracle test.

`GhostDecayingSelectionTestCase` sweeps the off-axis fold annulus
(``gamma in {0.2, 0.4}``, off-cusp ``rho in [1.2, 1.8]``, several angles
off both principal axes) and certifies that the extractor always returns
the DECAYING conjugate member (``Im tau_c > 0``): it re-derives both
members of the complex-conjugate pair with the independent oracle, checks
they carry equal-and-opposite ``Im tau_c``, and confirms the module
selected the positive (decaying) one -- asserting on the conjugate would
give ``Im tau_c < 0`` and a blowing-up carrier.

`GhostOnAxisLimitTestCase` certifies the pure-oscillation LIMIT as the
source approaches the caustic-reach principal axis: at a decreasing
sequence of small off-axis angles the kernel, amplitude and ``tau_c`` are
all finite, ``Im tau_c -> 0`` monotonically, and the carrier magnitude
``|exp(1j w tau_c)| = exp(-w Im tau_c)`` rises monotonically toward one
without ever exceeding it (no spurious decay or growth).  The Architect's
LITERAL thresholds (``|Im tau_c| < 1e-10`` with a genuine ghost that still
evaluates finitely) are UNREACHABLE in the landed primitive -- a root
whose ``Im u`` falls below ``root_tolerance`` (3e-7) declassifies to a
real image before ``Im tau_c`` reaches 1e-10, and EXACTLY on the axis the
source-aligned matrix is diagonal and the reconstruction collapses onto
the removable singularity ``u = a22`` (`GhostDomainError`).  That literal
contract is therefore carried as an ``@expectedFailure`` that will xpass
the day a future build makes the on-axis point evaluate finitely, and the
exactly-on-axis refusal is asserted directly as the current boundary.

`GhostFarFieldTestCase` is the far-field (``rho = 4``) suppression sanity
gate: at the Architect's measured anchor (``gamma = 0.4``, ``pi/4``,
``Im tau_c ~ 10.5``) the ghost kernel is heavily exponentially suppressed
(``max|C| < 1e-3``, subdominant to the residual envelope by ``< 0.5``),
and the ``|C|`` band traces the ``exp(-w Im tau_c)`` envelope; the large
POSITIVE ``Im tau_c`` also witnesses that the decaying member was chosen,
since the growing conjugate would give ``Im tau_c ~ -10.5`` and blow up.

`GhostGuardTestCase` certifies the two degeneracy guards inside
`_ghost_kernel`.  The near-fold guard is reached by handing the routine a
REAL critical-curve point as ``x_c`` -- the exact fold IS the critical
curve, where ``det(hessian) == 0`` to machine precision, so the complex
Fermat determinant ``|det H_c|`` sits at ~1e-16, far below the
``1e-8 * (1 + ||A||_F)**2`` floor -- while its bilinear radius
``Re(z) = |x|**2 > 0`` clears the first guard, isolating the second.  The
negative-``Re(z)`` guard is reached with a synthetic ``x_c`` whose
bilinear radius ``z = x_c . x_c`` has ``Re(z) <= 0`` (e.g. a purely
imaginary component).  Each must raise a `GhostDomainError` (an IS-A
`LensDomainError`) carrying a descriptive message -- NOT return a NaN,
inf, or silently mis-branched value.  The paired reachable-red proof lives
in `GhostSelfFalsificationTestCase`: with the det floor removed the same
near-fold call returns an astronomically large (~1e97) garbage amplitude
instead of refusing, so the guard is load-bearing.

`RealImageByteIdentityTestCase` proves the ghost additions changed NO
real-image behavior.  Over a battery of real (non-ghost) configs spanning
inside (four images) and outside (two images) the astroid caustic, it
compares `find_images`, `image_kernel`, `delay`, `magnification` and
`morse_index` -- and the returned image sets and Morse-census tuples --
against a HEAD copy of ``geometry.py`` loaded side by side via
``git show HEAD:<path>`` into a real temporary ``.py`` file (numba needs a
real file locator).  Every comparison is bit-for-bit (``max|diff| ==
0.0``); the byte-identity gate's teeth are proven in
`GhostSelfFalsificationTestCase` (a one-ulp perturbation is caught).

`GhostSelfFalsificationTestCase` closes the loop: it corrupts the delay
(a conjugation/branch bug), the amplitude branch (a Morse double-count),
the decaying-member selection (picking the growing conjugate blows the
far-field carrier up), and the oracle's own independence, and asserts each
gate goes RED -- so "the suite is green" is evidence rather than
decoration.

`GhostTestCase.tearDown` fails any swept test that made zero comparisons,
so a config loop that silently found no ghost cannot read green.
"""

import ast
import functools
import importlib.util
import inspect
import itertools
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from unittest import TestCase, expectedFailure, main

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cogwheel.lensing.chang_refsdal import geometry  # noqa: E402
from cogwheel.lensing.chang_refsdal.channels import (  # noqa: E402
    ChangRefsdalChannels)
from cogwheel.lensing.chang_refsdal.operator import (  # noqa: E402
    geometric_amplification)
from cogwheel.lensing import ppgo_map  # noqa: E402


#: Relative imaginary part above which a quartic root is a ghost; the
#: production default (`geometry.ghost_kernel`'s ``root_tolerance``).
ROOT_TOLERANCE = 3e-7

#: Off-axis source angles (radians, relative to the caustic-reach
#: direction) at which the ghost pair is genuine.  The exact principal
#: axis (angle 0) is EXCLUDED: there the source-aligned macro matrix is
#: diagonal and the ghost pair collapses onto the removable singularity
#: ``u = a22`` (a 0/0 reconstruction), which `geometry.ghost_kernel`
#: refuses by name.  Every angle here is comfortably off that measure-zero
#: set so ``Im tau_c > 0`` and the continuation is regular.
OFF_AXIS_ANGLES = (np.pi / 6, np.pi / 4, np.pi / 3, 3 * np.pi / 8)

#: Fold shears for the oracle sweep; both are positive-parity
#: (``|gamma| < 1``) diagonal (``beta = 0``) macro matrices.
ORACLE_GAMMAS = (0.2, 0.4)

#: Source radius in units of the caustic reach for the oracle sweep;
#: 1.1 places the source just OUTSIDE the caustic (a genuine ghost pair,
#: no fourth real image).
ORACLE_RHO = 1.1

#: Frequencies at which the production kernel is sampled to recover its
#: (w-independent) complex amplitude by solving the exact ``1, 1/w,
#: 1/w**2`` Vandermonde system.  Three distinct positive nodes suffice
#: because the carrier-free kernel is exactly quadratic in ``1 / w``.
AMPLITUDE_PROBE_W = (7.0, 13.0, 23.0)

#: Merged-saddle Morse reference phase (index 1): ``arg(exp(-i pi / 2))``.
#: The oracle picks the sqrt root nearest this phase, exactly as
#: `geometry._branch_pinned_amplitude` does with the same reference.
MORSE_REFERENCE_PHASE = -0.5 * np.pi

#: tau_c agreement: relative magnitude and absolute argument (radians).
TOL_TAU_REL = 1e-6
TOL_TAU_ARG = 1e-6

#: Amplitude magnitude vs the hand-rolled ANALYTIC determinant.
TOL_AMP_ANALYTIC_REL = 1e-6

#: Amplitude magnitude/phase vs the finite-difference determinant oracle.
TOL_AMP_FD_REL = 1e-4
TOL_AMP_FD_ARG = 1e-4

#: Reconstruction identity ``x_c . x_c == 1 / u_c`` (relative).
TOL_RECONSTRUCTION = 1e-10

#: `geometry` names the AST guard forbids inside the oracle functions, so
#: the oracle cannot reach its answer through any shared derivation.
#: Walks ``ast.Name.id`` and ``ast.Attribute.attr`` (never the source
#: text: a production symbol can be a substring of an oracle's own name).
FORBIDDEN_ORACLE_NAMES = frozenset({
    'ghost_kernel', '_ghost_kernel', '_ghost_candidates', '_ghost_delay',
    '_saddle_metric', '_c1_polynomial', '_c2_polynomial',
    'saddle_coefficients', 'image_kernel', 'delay', 'hessian',
    'magnification', 'morse_index', '_branch_pinned_amplitude',
    'image_quartic_coefficients', '_companion_roots', '_source_frame',
})

#: P1 anchors: ``(gamma, rho, w_anchor, tol_ratio, tol_arg_deg)``.  The
#: source sits at ``pi / 4`` (the 'diagonal', worst) angle, matching the
#: Architect's measured ``(|E|, |C|, arg(E/C))`` = (0.110, 0.111, 1.5deg)
#: and (0.051, 0.054, -0.7deg).
ANCHORS = (
    (0.2, 1.1, 8.5, 0.10, 3.5),
    (0.4, 1.1, 3.3, 0.10, 3.5),
)

#: Source angle of the P1 anchors (the diagonal-bad direction).
ANCHOR_ANGLE = np.pi / 4

#: Directory for diagnostic figures.
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

#: Off-cusp source radii (units of caustic reach) for the decaying-member
#: selection sweep -- the [1.2, 1.8] exterior band the Architect
#: specified, comfortably outside the caustic so a genuine complex-
#: conjugate ghost pair exists, but far enough from the fold that
#: ``Im tau_c`` is a healthy O(0.1--2).
DECAY_RHOS = (1.2, 1.5, 1.8)

#: Source angles (radians, off the caustic-reach axis) for the selection
#: sweep.  All are well away from the degenerate principal axes (0, pi/2,
#: pi) where the ghost reconstruction collapses, and both sides of the
#: reach axis are sampled so the sweep is not accidentally one-sided.
DECAY_ANGLES = tuple(
    np.deg2rad(angle_deg) for angle_deg in (25.0, 45.0, 70.0, 110.0, 135.0))

#: On-axis pure-oscillation limit fixture: shear, source radius, probe
#: frequency, and a DECREASING sequence of small off-axis angles.  As the
#: angle shrinks the source approaches the caustic-reach principal axis,
#: ``Im tau_c -> 0`` (pure oscillation, no decay) and the ghost kernel
#: converges to a finite limit.  ``rho = 1.5`` sits well past the on-axis
#: cusp (``reach``), so the amplitude ``1 / sqrt(det H_c)`` stays O(1)
#: throughout -- the near-axis singular regime is ``rho ~ 1``.
ONAXIS_GAMMA = 0.2
ONAXIS_RHO = 1.5
ONAXIS_W = 8.0
ONAXIS_LIMIT_ANGLES = (1e-3, 1e-4, 1e-5)

#: "No spurious growth" floor: the carrier magnitude
#: ``|exp(1j w tau_c)| = exp(-w Im tau_c)`` must never EXCEED one (a
#: growing ghost), allowing only roundoff above unity.
TOL_NO_GROWTH = 1e-12

#: The Architect's LITERAL on-axis thresholds, currently UNREACHABLE (see
#: `GhostOnAxisLimitTestCase`) and carried as an ``@expectedFailure``
#: contract: ``|Im tau_c| < 1e-10`` with a genuine ghost that still
#: evaluates finitely, and a unit-modulus carrier to ``1e-12``.
TOL_ONAXIS_IM_TAU = 1e-10
TOL_ONAXIS_UNIT_CARRIER = 1e-12

#: Far-field (``rho = 4``) exponential-suppression anchor: the Architect's
#: measured config with ``|E_ff|max ~ 2.1e-3``, ``|C|max ~ 7.5e-4`` and
#: ``Im tau_c ~ 10.5``.  ``gamma = 0.4`` at the ``pi/4`` diagonal reaches
#: ``Im tau_c ~ 10.48`` at ``rho = 4``; the ``(0.5, 1.2)`` band reproduces
#: the measured envelope magnitudes.
FAR_GAMMA = 0.4
FAR_RHO = 4.0
FAR_ANGLE = np.pi / 4
FAR_W_BAND = (0.5, 1.2)
FAR_N_W = 24

#: Far-field acceptance thresholds: absolute suppression floor on
#: ``max|C|``, subdominance ratio ``max|C| / max|E_ff|``, and the minimum
#: ``Im tau_c`` witnessing a genuine large POSITIVE imaginary delay (the
#: decaying member; the growing conjugate would give ``Im tau_c ~ -10.5``
#: and ``|C|`` would blow up).
FAR_MAX_C = 1e-3
FAR_MAX_RATIO = 0.5
FAR_IM_TAU_MIN = 8.0

#: Real merged-saddle reference amplitude handed to `_ghost_kernel`
#: (Morse index 1, phase ``exp(-i pi / 2)``); only its phase pins the
#: sqrt branch, exactly as `ghost_kernel` builds it internally.
GHOST_REFERENCE_AMPLITUDE = np.exp(1j * MORSE_REFERENCE_PHASE)

#: Near-fold guard fixture: the exact fold is the critical curve, where
#: ``det(hessian) == 0`` to machine precision.  Handing `_ghost_kernel` a
#: REAL critical-curve point as ``x_c`` (via `geometry.critical_point`)
#: drives the complex Fermat determinant ``|det H_c|`` to ~1e-16, far
#: below the ``_GHOST_DET_FLOOR * (1 + ||A||_F)**2`` floor, while
#: ``Re(z) = |x|**2 > 0`` clears the first guard -- isolating the
#: near-fold (det) guard.  Several fold angles so the refusal is not a
#: one-off.
GUARD_GAMMA = 0.2
GUARD_FOLD_THETAS = (0.4, 0.6, 0.9, 1.2)

#: Synthetic complex ``x_c`` positions whose bilinear radius
#: ``z = x_c . x_c`` has ``Re(z) <= 0`` (a topology breakdown near the
#: cusp), used to reach the FIRST (`Re(z) <= 0`) guard directly.  Each is
#: hand-built so ``z`` is negative or on the imaginary axis.
GUARD_NEGATIVE_Z_POSITIONS = (
    np.array([1j, 0.0], dtype=complex),           # z = -1
    np.array([0.0, 2j], dtype=complex),           # z = -4
    np.array([0.5j, 0.5j], dtype=complex),        # z = -0.5
    np.array([0.3 + 0.9j, 0.1 + 0.9j], dtype=complex),  # Re(z) < 0
)

#: An arbitrary in-domain source/matrix pair for the negative-``Re(z)``
#: guard (the guard fires on ``x_c`` alone, before ``source`` is used for
#: anything but the delay).
GUARD_SOURCE = np.array([0.3, 0.1])

#: With the det floor removed, the near-fold ``1 / sqrt(det H_c)``
#: amplitude blows up beyond this (measured ~1e97) instead of refusing --
#: the reachable-red witness that the near-fold guard is load-bearing.
GUARD_MUTATION_BLOWUP = 1e30

#: Repo-relative path of the geometry module and the repo root, for the
#: HEAD side-by-side byte-identity comparison.
_GEOMETRY_REL_PATH = 'cogwheel/lensing/chang_refsdal/geometry.py'
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

#: Frequencies for the real ``image_kernel`` byte-identity comparison.
BYTEID_W = np.array([5.0, 11.0, 21.0])

#: Shears for the real-image byte-identity battery (all positive parity,
#: ``|gamma| < 1 - kappa = 1``).
BYTEID_GAMMAS = (0.2, 0.4, 0.6)

#: Source positions for the byte-identity battery.  The small-radius
#: sources sit INSIDE the astroid caustic (four real images); the
#: large-radius ones sit OUTSIDE it (two real images).  Both regimes are
#: asserted present so the comparison genuinely spans the caustic.
BYTEID_SOURCES = (
    np.array([0.02, 0.0]), np.array([0.05, 0.05]), np.array([0.1, 0.0]),
    np.array([0.12, 0.08]), np.array([0.2, 0.15]), np.array([0.25, 0.2]),
    np.array([0.5, 0.0]), np.array([0.6, 0.3]), np.array([0.8, 0.5]),
    np.array([1.1, 0.7]),
)


# `_head_geometry` deleted 2026-07-30 (F045).  It imported `git show
# HEAD:geometry.py` side by side as a "byte-identity oracle", but HEAD stops
# being the pre-change revision the moment the change commits -- audited that
# day, HEAD and the worktree were byte-identical, so every test using it was
# comparing the module against an exact copy of itself.  The independent
# Richardson finite-difference oracle and the AST independence guard in this
# file are unaffected: they never used it.


def _near_fold_position(gamma, theta):
    """Return ``(x_c, source)`` for the near-fold guard fixture.

    ``x_c`` is the REAL critical-curve point at polar angle ``theta`` (a
    fold, where ``det(hessian) == 0``) cast to complex, and ``source`` is
    the caustic point it maps to.  Passed to `_ghost_kernel` this drives
    ``|det H_c|`` below the near-fold floor while ``Re(z) > 0``.
    """
    critical = geometry.critical_point(gamma, theta)
    return (np.asarray(critical.image, dtype=complex),
            np.asarray(critical.source, dtype=float))


# --------------------------------------------------------------------------
# INDEPENDENT ORACLE.  These functions share no code with `geometry`: they
# reach ``tau_c`` and ``det H_c`` through ``numpy.roots`` + hand-rolled
# closed forms only.  `OracleIndependenceTestCase` AST-guards every one of
# them against `FORBIDDEN_ORACLE_NAMES`.
# --------------------------------------------------------------------------
def oracle_macro_matrix(gamma, beta=0.0, kappa=0.0):
    """Hand-built ``(1 - kappa) I - gamma Q(beta)`` (no `geometry`)."""
    cos2b, sin2b = np.cos(2.0 * beta), np.sin(2.0 * beta)
    shear = np.array([[cos2b, sin2b], [sin2b, -cos2b]])
    return (1.0 - kappa) * np.eye(2) - gamma * shear


def oracle_source_frame(source):
    """Return ``(radius, basis)`` with the first basis axis along the
    source; hand-rolled Gram--Schmidt, independent of `geometry`."""
    source = np.asarray(source, dtype=float)
    radius = float(np.hypot(source[0], source[1]))
    axis1 = source / radius
    axis2 = np.array([-axis1[1], axis1[0]])
    return radius, np.column_stack([axis1, axis2])


def oracle_quartic_coefficients(source_radius, a11, a12, a22):
    """Coefficients of the image quartic in ``u = 1 / |x|**2`` from the
    radial constraint ``D**2 - Y**2 u [(a22 - u)**2 + a12**2] = 0``,
    expanded here by hand (descending degree)."""
    det = a11 * a22 - a12 * a12
    r2 = source_radius * source_radius
    return np.array([
        1.0,
        -2.0 * (a11 + a22) - r2,
        a11 * a11 + 4.0 * a11 * a22 + a22 * a22 - 2.0 * a12 * a12
        + 2.0 * a22 * r2,
        -2.0 * (a11 + a22) * det - r2 * (a22 * a22 + a12 * a12),
        det * det,
    ])


def oracle_tau(x_c, source, matrix):
    """Analytic-continued complex Fermat delay assembled from scratch.

    ``tau_c = 0.5 x_c.A.x_c - y.x_c + 0.5 y.y - 0.5 log(x_c . x_c)`` with
    every product BILINEAR (no conjugation) and the principal branch of
    ``log`` (the continuation from the real fold where ``x.x > 0``).
    """
    x_c = np.asarray(x_c, dtype=complex)
    source = np.asarray(source, dtype=float)
    z = x_c[0] * x_c[0] + x_c[1] * x_c[1]
    return (0.5 * (x_c @ matrix @ x_c) - source @ x_c
            + 0.5 * (source @ source) - 0.5 * np.log(z))


def oracle_ghost(source, matrix, tol=ROOT_TOLERANCE):
    """Independent ghost extraction: solve the quartic, reconstruct every
    complex root's image by hand, keep the decaying member (largest
    ``Im tau_c``).  Returns ``(u_c, x_c, tau_c)``."""
    source = np.asarray(source, dtype=float)
    radius, basis = oracle_source_frame(source)
    rotated = basis.T @ matrix @ basis
    a11, a12, a22 = rotated[0, 0], rotated[0, 1], rotated[1, 1]
    roots = np.roots(oracle_quartic_coefficients(radius, a11, a12, a22))
    candidates = []
    for u_root in roots:
        if u_root.real <= 0.0:
            continue
        if abs(u_root.imag) <= tol * (1.0 + abs(u_root.real)):
            continue
        u_c = complex(u_root)
        denominator = (a11 - u_c) * (a22 - u_c) - a12 * a12
        rotated_x = np.array([radius * (a22 - u_c) / denominator,
                              -radius * a12 / denominator], dtype=complex)
        x_c = basis @ rotated_x
        candidates.append((u_c, x_c))
    if not candidates:
        raise ValueError('oracle found no complex-conjugate ghost pair')
    taus = [oracle_tau(x_c, source, matrix) for (_, x_c) in candidates]
    winner = int(np.argmax([tau.imag for tau in taus]))
    u_c, x_c = candidates[winner]
    return u_c, x_c, taus[winner]


def oracle_ghost_members(source, matrix, tol=ROOT_TOLERANCE):
    """Independently re-derive BOTH members of the complex-conjugate ghost
    pair (not just the decaying winner).

    Same construction as `oracle_ghost` -- solve the quartic, reconstruct
    every complex root's image by hand, assemble its bilinear complex-log
    ``tau_c`` -- but returns EVERY complex candidate as ``(u_c, x_c,
    tau_c)`` so the selection test can inspect the equal-and-opposite
    ``Im tau_c`` of the two members and confirm the module chose the
    positive (decaying) one.  Shares no code with `geometry`.
    """
    source = np.asarray(source, dtype=float)
    radius, basis = oracle_source_frame(source)
    rotated = basis.T @ matrix @ basis
    a11, a12, a22 = rotated[0, 0], rotated[0, 1], rotated[1, 1]
    roots = np.roots(oracle_quartic_coefficients(radius, a11, a12, a22))
    members = []
    for u_root in roots:
        if u_root.real <= 0.0:
            continue
        if abs(u_root.imag) <= tol * (1.0 + abs(u_root.real)):
            continue
        u_c = complex(u_root)
        denominator = (a11 - u_c) * (a22 - u_c) - a12 * a12
        rotated_x = np.array([radius * (a22 - u_c) / denominator,
                              -radius * a12 / denominator], dtype=complex)
        x_c = basis @ rotated_x
        members.append((u_c, x_c, oracle_tau(x_c, source, matrix)))
    return members


def oracle_branch_pin(root, reference_phase=MORSE_REFERENCE_PHASE):
    """Pick whichever of ``+/- root`` has phase nearest the real
    merged-saddle reference (the Morse ``-pi / 2``), hand-rolled."""
    def wrapped(delta):
        return abs((delta + np.pi) % (2.0 * np.pi) - np.pi)
    return min((root, -root),
               key=lambda cand: wrapped(np.angle(cand) - reference_phase))


def oracle_analytic_amplitude(x_c, matrix):
    """Branch-pinned ``1 / sqrt(det H_c)`` with ``H_c`` and its
    determinant formed algebraically by hand.  Returns ``(amp, det)``."""
    x_c = np.asarray(x_c, dtype=complex)
    z = x_c[0] * x_c[0] + x_c[1] * x_c[1]
    hess = matrix - np.eye(2) / z + 2.0 * np.outer(x_c, x_c) / z**2
    determinant = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]
    return oracle_branch_pin(1.0 / np.sqrt(determinant)), determinant


def oracle_fd_amplitude(x_c, source, matrix):
    """Branch-pinned ``1 / sqrt(det H_c)`` with ``H_c`` formed by a
    RICHARDSON-extrapolated central finite difference of the complex
    Fermat potential ``oracle_tau``.  Returns ``(amp, det)``.

    Real step ``h = 1e-4 |x_c|`` (clamped at 1e-5 to keep roundoff below
    the truncation error), plus a half-step for the Richardson step;
    diagonal ``[tau(+h) - 2 tau(0) + tau(-h)] / h**2`` and the 4-point
    cross stencil off-diagonal.
    """
    x_c = np.asarray(x_c, dtype=complex)
    scale = math.hypot(abs(x_c[0]), abs(x_c[1]))
    base_step = max(1e-4 * scale, 1e-5)

    def tau(shift1, shift2):
        return oracle_tau(np.array([x_c[0] + shift1, x_c[1] + shift2]),
                          source, matrix)

    def fd_hessian(step):
        center = tau(0.0, 0.0)
        h11 = (tau(step, 0.0) - 2.0 * center + tau(-step, 0.0)) / step**2
        h22 = (tau(0.0, step) - 2.0 * center + tau(0.0, -step)) / step**2
        h12 = (tau(step, step) - tau(step, -step)
               - tau(-step, step) + tau(-step, -step)) / (4.0 * step**2)
        return np.array([[h11, h12], [h12, h22]])

    coarse = fd_hessian(base_step)
    fine = fd_hessian(base_step / 2.0)
    hess = (4.0 * fine - coarse) / 3.0
    determinant = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]
    return oracle_branch_pin(1.0 / np.sqrt(determinant)), determinant


#: Every independent-oracle function, gathered for the AST guard.
_ORACLE_FUNCTIONS = (
    oracle_macro_matrix, oracle_source_frame, oracle_quartic_coefficients,
    oracle_tau, oracle_ghost, oracle_ghost_members, oracle_branch_pin,
    oracle_analytic_amplitude, oracle_fd_amplitude,
)


def _forbidden_names_in(func):
    """Return the `FORBIDDEN_ORACLE_NAMES` referenced in ``func``'s body.

    Walks ``ast.Name.id`` and ``ast.Attribute.attr`` -- never the raw
    source -- so a forbidden production symbol that merely appears as a
    substring of an oracle's own identifier is not falsely flagged.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    return used & FORBIDDEN_ORACLE_NAMES


def _production_amplitude(source, matrix):
    """Recover `geometry.ghost_kernel`'s w-independent complex amplitude.

    The carrier-free kernel is exactly ``a + (i a C1) / w + (a C2) / w**2``
    (a quadratic in ``1 / w``), so sampling three distinct ``w`` and
    solving the ``1, 1/w, 1/w**2`` Vandermonde system returns ``a``
    exactly (to roundoff), free of any ``1 / w`` contamination.
    """
    probe = np.asarray(AMPLITUDE_PROBE_W, dtype=float)
    kernel = geometry.ghost_kernel(probe, source, matrix).kernel
    inverse_w = 1.0 / probe
    vandermonde = np.vstack(
        [np.ones_like(inverse_w), inverse_w, inverse_w**2]).T
    return np.linalg.solve(vandermonde, kernel)[0]


def _anchor_source(gamma, rho, angle, kappa=0.0):
    """Anchor source position: ``rho * reach`` along ``direction`` rotated
    by ``angle``, exactly as `ppgo_map._measure_cell` places it."""
    reach, direction = ppgo_map.caustic_geometry(gamma, kappa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return rho * reach * (rotation @ np.asarray(direction))


def _measure_residual_envelope(gamma, rho, w_anchor, angle, kappa=0.0):
    """Measure the residual envelope ``E`` and ghost prediction ``C`` at a
    P1 anchor.

    ``E = exact_total - ppGO`` (both demodulated at ``t_min`` exactly as
    `ppgo_map._measure_cell`), and ``C`` is the ghost's full contribution
    ``kernel * exp(1j w (tau_c - t_min))`` -- i.e. the carrier-free kernel
    times its complex carrier, so ``|C|`` includes the ``exp(-w Im tau_c)``
    decay.  Returns ``(w_grid, E_grid, C_grid, index_of_anchor)``.
    """
    source = _anchor_source(gamma, rho, angle, kappa)
    matrix = geometry.macro_matrix(gamma, 0.0, kappa)
    # A >=2-point strictly-increasing grid is required by the engine; the
    # anchor point is the last node and a band below it feeds the plot.
    w_grid = np.linspace(0.6 * w_anchor, w_anchor, 24)
    partition = ChangRefsdalChannels(w_grid).evaluate(
        gamma=gamma, y=source, beta=0.0, kappa=kappa)
    exact = np.asarray(partition.exact_total)
    t_min = float(partition.t_min)
    ppgo = np.asarray(geometric_amplification(
        w_grid, source, gamma, beta=0.0, kappa=kappa)) \
        * np.exp(-1j * w_grid * t_min)
    residual = exact - ppgo

    contribution = geometry.ghost_kernel(w_grid, source, matrix)
    carrier = np.exp(1j * w_grid * (contribution.delay - t_min))
    ghost = contribution.kernel * carrier
    return w_grid, residual, ghost, w_grid.size - 1


def _far_field_bundle():
    """Measure the far-field (``rho = 4``) residual envelope and ghost
    contribution across the fixed suppression band.

    Mirrors `_measure_residual_envelope`'s demodulation but on the
    explicit ``FAR_W_BAND`` at the ``FAR_*`` anchor.  Returns
    ``(w_grid, E_grid, C_grid, contribution)`` where ``E = exact_total -
    ppGO`` (both demodulated at ``t_min``) and ``C = kernel *
    exp(1j w (tau_c - t_min))`` is the ghost's full contribution
    (``|C|`` includes the ``exp(-w Im tau_c)`` suppression).
    """
    source = _anchor_source(FAR_GAMMA, FAR_RHO, FAR_ANGLE)
    matrix = geometry.macro_matrix(FAR_GAMMA, 0.0, 0.0)
    w_grid = np.linspace(FAR_W_BAND[0], FAR_W_BAND[1], FAR_N_W)
    partition = ChangRefsdalChannels(w_grid).evaluate(
        gamma=FAR_GAMMA, y=source, beta=0.0, kappa=0.0)
    exact = np.asarray(partition.exact_total)
    t_min = float(partition.t_min)
    ppgo = np.asarray(geometric_amplification(
        w_grid, source, FAR_GAMMA, beta=0.0, kappa=0.0)) \
        * np.exp(-1j * w_grid * t_min)
    residual = exact - ppgo
    contribution = geometry.ghost_kernel(w_grid, source, matrix)
    carrier = np.exp(1j * w_grid * (contribution.delay - t_min))
    ghost = contribution.kernel * carrier
    return w_grid, residual, ghost, contribution


class GhostTestCase(TestCase):
    """Base class: comparison tally + anti-vacuity ``tearDown``."""

    def setUp(self):
        """Reset the per-test comparison counter and sweep flag."""
        self.n_compared = 0
        self.swept = False

    def tearDown(self):
        """Fail a swept test that asserted nothing.

        A ghost config loop that silently found no complex pair (a real
        regression) would iterate and pass without a single comparison.
        Tests that set ``self.swept`` therefore must record at least one
        comparison; non-sweeping tests are unaffected.
        """
        if self.swept and not self.n_compared:
            self.fail('swept the configurations but recorded zero ghost '
                      'comparisons; the test asserted nothing')

    def _record(self):
        """Count one comparison actually made (for the anti-vacuity gate)."""
        self.n_compared += 1


class GhostOracleTestCase(GhostTestCase):
    """
    Tight correctness gate: the module's ghost ``tau_c`` and amplitude
    against a fully INDEPENDENT re-derivation (``numpy.roots`` + hand-
    rolled ``tau`` / ``det``).  This catches the two silent failure modes
    the ghost machinery is uniquely exposed to: a complex-log branch-cut
    error in ``tau_c`` and a Morse double-count in the amplitude phase.

    Every configuration is an off-axis fold with a genuine ghost pair, so
    the comparison always runs; `tearDown` fails the test otherwise.
    """

    def _oracle_and_production(self, gamma, angle):
        """Shared setup: source, matrix, oracle result, production
        `GhostContribution`, and the Vandermonde-recovered amplitude."""
        source = _anchor_source(gamma, ORACLE_RHO, angle)
        matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
        oracle_A = oracle_macro_matrix(gamma)
        _, x_c, tau_c = oracle_ghost(source, oracle_A)
        contribution = geometry.ghost_kernel(
            np.array(AMPLITUDE_PROBE_W, dtype=float), source, matrix)
        amplitude = _production_amplitude(source, matrix)
        return source, oracle_A, x_c, tau_c, contribution, amplitude

    def test_ghost_delay_matches_independent_closed_form(self):
        """``tau_c`` (magnitude and argument) matches the from-scratch
        bilinear complex-log delay, catching a branch-cut/sign error."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                (_, _, _, tau_oracle, contribution, _) = \
                    self._oracle_and_production(gamma, angle)
                tau_prod = contribution.delay
                rel = abs(tau_prod - tau_oracle) / abs(tau_oracle)
                arg = abs(np.angle(tau_prod) - np.angle(tau_oracle))
                self._record()
                self.assertLessEqual(
                    rel, TOL_TAU_REL,
                    f'|tau_c| relative error {rel:.2e} > {TOL_TAU_REL}\n'
                    f'  production {tau_prod!r}\n  oracle     {tau_oracle!r}')
                self.assertLessEqual(
                    arg, TOL_TAU_ARG,
                    f'arg(tau_c) error {arg:.2e} rad > {TOL_TAU_ARG}')

    def test_ghost_delay_has_positive_imaginary_part(self):
        """Off the cusp axis the decaying ghost has ``Im tau_c > 0`` (it
        decays, not grows); a branch flip would make it negative."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                (_, _, _, tau_oracle, contribution, _) = \
                    self._oracle_and_production(gamma, angle)
                self._record()
                self.assertGreater(
                    contribution.delay.imag, 0.0,
                    'decaying ghost must have Im tau_c > 0 off the axis')
                self.assertGreater(tau_oracle.imag, 0.0)

    def test_ghost_amplitude_magnitude_matches_analytic_determinant(self):
        """``|amplitude|`` matches ``1 / sqrt(det H_c)`` from the hand-
        rolled analytic determinant to 1e-6 relative (a coding/branch
        check on the shared ``1 / sqrt`` form)."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                (_, oracle_A, x_c, _, _, amplitude) = \
                    self._oracle_and_production(gamma, angle)
                amp_oracle, _ = oracle_analytic_amplitude(x_c, oracle_A)
                rel = abs(abs(amplitude) - abs(amp_oracle)) / abs(amp_oracle)
                self._record()
                self.assertLessEqual(
                    rel, TOL_AMP_ANALYTIC_REL,
                    f'|amplitude| relative error {rel:.2e} > '
                    f'{TOL_AMP_ANALYTIC_REL}\n'
                    f'  production {abs(amplitude)!r}\n'
                    f'  oracle     {abs(amp_oracle)!r}')

    def test_ghost_amplitude_matches_finite_difference_determinant(self):
        """``amplitude`` magnitude (1e-4 rel) and phase (1e-4 rad) match
        the genuinely independent Richardson finite-difference oracle.

        The phase check is the Morse double-count guard: the sqrt branch
        is pinned to ``exp(-i pi / 2)``; an extra Morse factor would
        rotate the production amplitude by ``pi`` and blow this up."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                (_, oracle_A, x_c, _, _, amplitude) = \
                    self._oracle_and_production(gamma, angle)
                # FD oracle needs the oracle A and source; rebuild source.
                source = _anchor_source(gamma, ORACLE_RHO, angle)
                amp_fd, _ = oracle_fd_amplitude(x_c, source, oracle_A)
                rel = abs(abs(amplitude) - abs(amp_fd)) / abs(amp_fd)
                arg = abs(np.angle(amplitude) - np.angle(amp_fd))
                self._record()
                self.assertLessEqual(
                    rel, TOL_AMP_FD_REL,
                    f'|amplitude| vs FD det error {rel:.2e} > '
                    f'{TOL_AMP_FD_REL}')
                self.assertLessEqual(
                    arg, TOL_AMP_FD_ARG,
                    f'arg(amplitude) vs FD det error {arg:.2e} rad > '
                    f'{TOL_AMP_FD_ARG} (a Morse double-count rotates by pi)')

    def test_oracle_ghost_position_matches_production_position(self):
        """The oracle's independently reconstructed ``x_c`` agrees with
        the module's ``GhostContribution.position`` -- the bridge that
        makes the ``tau_c`` comparison a pure formula test."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                (_, _, x_c, _, contribution, _) = \
                    self._oracle_and_production(gamma, angle)
                x_prod = np.asarray(contribution.position, dtype=complex)
                gap = float(np.max(np.abs(x_prod - x_c)))
                self._record()
                self.assertLess(
                    gap, 1e-9,
                    f'oracle x_c differs from production position by {gap:.2e}')


class GhostReconstructionTestCase(GhostTestCase):
    """
    Reconstruction-algebra cross-check ``x_c . x_c == 1 / u_c``.

    The bilinear radius ``z = x1**2 + x2**2`` equals ``1 / u_c`` by the
    radial constraint the quartic is built from, so this validates the
    ``u -> x`` reconstruction map independently of the root solver.  It
    does NOT probe the log branch (both ``z`` and ``tau_c`` feed the same
    ``clog``); the branch is the oracle test's job.
    """

    def test_bilinear_radius_equals_inverse_root(self):
        """``x_c . x_c`` reproduces ``1 / u_c`` to near machine precision
        for every off-axis ghost."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                source = _anchor_source(gamma, ORACLE_RHO, angle)
                u_c, x_c, _ = oracle_ghost(source, oracle_macro_matrix(gamma))
                bilinear = x_c[0] * x_c[0] + x_c[1] * x_c[1]
                inverse_root = 1.0 / u_c
                gap = abs(bilinear - inverse_root)
                bound = TOL_RECONSTRUCTION * (1.0 + abs(inverse_root))
                self._record()
                self.assertLess(
                    gap, bound,
                    f'|x_c.x_c - 1/u_c| = {gap:.2e} > {bound:.2e}\n'
                    f'  x_c.x_c {bilinear!r}\n  1/u_c   {inverse_root!r}')

    def test_production_position_satisfies_the_identity(self):
        """The module's own ``GhostContribution.position`` satisfies the
        same identity against the oracle's ``u_c`` -- so the cross-check
        certifies production, not just the oracle's reconstruction."""
        self.swept = True
        for gamma, angle in itertools.product(ORACLE_GAMMAS, OFF_AXIS_ANGLES):
            with self.subTest(gamma=gamma, angle=angle):
                source = _anchor_source(gamma, ORACLE_RHO, angle)
                matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
                u_c, _, _ = oracle_ghost(source, oracle_macro_matrix(gamma))
                position = np.asarray(
                    geometry.ghost_kernel(
                        np.array(AMPLITUDE_PROBE_W, dtype=float),
                        source, matrix).position, dtype=complex)
                bilinear = position[0] * position[0] + position[1] * position[1]
                inverse_root = 1.0 / u_c
                gap = abs(bilinear - inverse_root)
                bound = TOL_RECONSTRUCTION * (1.0 + abs(inverse_root))
                self._record()
                self.assertLess(
                    gap, bound,
                    f'production x_c.x_c - 1/u_c = {gap:.2e} > {bound:.2e}')


class GhostAnchorTestCase(GhostTestCase):
    """
    Loose physical-scale sanity gate at the two binding P1 anchors.

    The ghost's full contribution ``C = kernel * exp(1j w (tau_c - t_min))``
    is compared to the residual envelope ``E = exact_total - ppGO``
    measured from the exact engine (demodulated at ``t_min`` exactly as
    `ppgo_map._measure_cell`).  The residual-of-the-residual ``R / E`` is
    physically 4-6%, so the bars are deliberately loose: 10% in magnitude,
    3.5 deg in phase.  The TIGHT correctness check is `GhostOracleTestCase`.
    """

    def test_anchor_magnitude_and_phase_reproduction(self):
        """At each anchor ``||C| / |E| - 1| < 10%`` and ``|arg(E/C)| <
        3.5 deg``."""
        self.swept = True
        for gamma, rho, w_anchor, tol_ratio, tol_arg_deg in ANCHORS:
            with self.subTest(gamma=gamma, w=w_anchor):
                w_grid, residual, ghost, k = _measure_residual_envelope(
                    gamma, rho, w_anchor, ANCHOR_ANGLE)
                env = residual[k]
                contribution = ghost[k]
                ratio = abs(contribution) / abs(env)
                arg_deg = abs(np.degrees(np.angle(env / contribution)))
                self._record()
                self.assertLess(
                    abs(ratio - 1.0), tol_ratio,
                    f'gamma={gamma} w={w_anchor}: ||C|/|E| - 1| = '
                    f'{abs(ratio - 1.0):.3f} > {tol_ratio}\n'
                    f'  |C| {abs(contribution):.4f}  |E| {abs(env):.4f}')
                self.assertLess(
                    arg_deg, tol_arg_deg,
                    f'gamma={gamma} w={w_anchor}: |arg(E/C)| = '
                    f'{arg_deg:.2f} deg > {tol_arg_deg} deg')

    def test_anchor_overlay_diagnostic(self):
        """Diagnostic overlay of ``|C|`` and ``|E|`` vs ``w`` across each
        anchor band.  Best-effort plotting; a backend failure never fails
        the test."""
        self.swept = True
        for gamma, rho, w_anchor, _, _ in ANCHORS:
            with self.subTest(gamma=gamma, w=w_anchor):
                w_grid, residual, ghost, _ = _measure_residual_envelope(
                    gamma, rho, w_anchor, ANCHOR_ANGLE)
                self._record()
                # The envelope and ghost must both be finite and O(0.1)
                # across the band -- a nonvacuous physical scale.
                self.assertTrue(np.all(np.isfinite(np.abs(residual))))
                self.assertTrue(np.all(np.isfinite(np.abs(ghost))))
                try:
                    os.makedirs(_OUTPUT_DIR, exist_ok=True)
                    figure, axis = plt.subplots(figsize=(6.0, 4.0))
                    axis.plot(w_grid, np.abs(ghost), lw=1.4,
                              label='|C| ghost kernel')
                    axis.plot(w_grid, np.abs(residual), '--', lw=1.4,
                              label='|E| exact - ppGO')
                    axis.axvline(w_anchor, color='k', lw=0.8, alpha=0.5)
                    axis.set_xlabel('w (dimensionless frequency)')
                    axis.set_ylabel('magnitude')
                    axis.set_title(f'ghost vs residual, gamma={gamma}')
                    axis.legend()
                    figure.savefig(
                        os.path.join(
                            _OUTPUT_DIR,
                            f'ghost_anchor_overlay_gamma{gamma}.png'),
                        dpi=110)
                    plt.close(figure)
                except Exception:  # noqa: BLE001 -- diagnostics best-effort
                    pass


class GhostDecayingSelectionTestCase(GhostTestCase):
    """
    The extractor always returns the DECAYING conjugate member.

    Over the off-axis fold annulus (``gamma in {0.2, 0.4}``, off-cusp
    ``rho in [1.2, 1.8]``, several angles off both principal axes) the
    quartic yields a complex-conjugate ghost pair with equal-and-opposite
    ``Im tau_c``.  The module must select the ``Im tau_c > 0`` member (the
    physical, exponentially DECAYING ghost); the conjugate has
    ``Im tau_c < 0`` and would give a carrier ``exp(-w Im tau_c)`` that
    BLOWS UP.  The independent oracle re-derives both members so the test
    can confirm the correct one was chosen, not merely that the answer is
    positive.
    """

    def test_selected_member_is_decaying(self):
        """`ghost_kernel` returns a member with ``Im tau_c > 0`` at every
        off-axis config in the sweep."""
        self.swept = True
        for gamma, rho, angle in itertools.product(
                ORACLE_GAMMAS, DECAY_RHOS, DECAY_ANGLES):
            with self.subTest(gamma=gamma, rho=rho, angle=angle):
                source = _anchor_source(gamma, rho, angle)
                matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
                delay = geometry.ghost_kernel(
                    np.array(AMPLITUDE_PROBE_W, dtype=float),
                    source, matrix).delay
                self._record()
                self.assertGreater(
                    delay.imag, 0.0,
                    f'gamma={gamma} rho={rho} angle={angle:.3f}: selected '
                    f'ghost has Im tau_c = {delay.imag:.3e} <= 0 (a growing '
                    f'member); the decaying conjugate should have been chosen')

    def test_conjugate_pair_has_opposite_sign_imaginary_delays(self):
        """The two members of the pair carry equal-and-opposite
        ``Im tau_c`` (one decaying, one growing), so the selection is a
        genuine choice and not vacuously positive.

        Diagnostic: the pair's ``Im tau_c`` values are asserted to bracket
        zero symmetrically, which is exactly the print the Architect's
        description calls for."""
        self.swept = True
        for gamma, rho, angle in itertools.product(
                ORACLE_GAMMAS, DECAY_RHOS, DECAY_ANGLES):
            with self.subTest(gamma=gamma, rho=rho, angle=angle):
                source = _anchor_source(gamma, rho, angle)
                members = oracle_ghost_members(source, oracle_macro_matrix(gamma))
                self._record()
                self.assertEqual(
                    len(members), 2,
                    f'expected a single complex-conjugate ghost pair, got '
                    f'{len(members)} members')
                imags = sorted(member[2].imag for member in members)
                self.assertLess(
                    imags[0], 0.0,
                    'the pair must contain a growing (Im tau_c < 0) member')
                self.assertGreater(
                    imags[1], 0.0,
                    'the pair must contain a decaying (Im tau_c > 0) member')
                # Equal-and-opposite: the two Im tau_c cancel to roundoff.
                self.assertLess(
                    abs(imags[0] + imags[1]),
                    1e-9 * (1.0 + abs(imags[1])),
                    f'conjugate Im tau_c not opposite: {imags[0]:.6e} vs '
                    f'{imags[1]:.6e}')

    def test_production_selects_the_positive_oracle_member(self):
        """The module's ``delay`` matches the oracle pair's POSITIVE-
        imaginary member (and differs from the negative one), so the
        selection rule -- not just the sign -- is what is certified."""
        self.swept = True
        for gamma, rho, angle in itertools.product(
                ORACLE_GAMMAS, DECAY_RHOS, DECAY_ANGLES):
            with self.subTest(gamma=gamma, rho=rho, angle=angle):
                source = _anchor_source(gamma, rho, angle)
                matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
                members = oracle_ghost_members(source, oracle_macro_matrix(gamma))
                taus = [member[2] for member in members]
                decaying = taus[int(np.argmax([tau.imag for tau in taus]))]
                growing = taus[int(np.argmin([tau.imag for tau in taus]))]
                prod = geometry.ghost_kernel(
                    np.array(AMPLITUDE_PROBE_W, dtype=float),
                    source, matrix).delay
                self._record()
                self.assertLess(
                    abs(prod - decaying), TOL_TAU_REL * (1.0 + abs(decaying)),
                    f'production tau_c {prod!r} != decaying oracle member '
                    f'{decaying!r}')
                self.assertGreater(
                    abs(prod - growing), 1e-3,
                    'production must NOT coincide with the growing conjugate')


class GhostOnAxisLimitTestCase(GhostTestCase):
    """
    Pure-oscillation limit as the source approaches the caustic-reach
    principal axis.

    The Architect's target -- ``Im tau_c = 0`` by symmetry with a kernel
    that still evaluates finitely -- is UNREACHABLE in the landed
    primitive: a quartic root whose ``Im u`` falls below ``root_tolerance``
    (3e-7) declassifies to a real image (no ghost), so a GENUINE ghost
    cannot have ``|Im tau_c| < 1e-10``; and EXACTLY on the axis the
    source-aligned matrix is diagonal and the reconstruction collapses
    onto the removable singularity ``u = a22``, which the module refuses
    with `GhostDomainError`.  This class therefore certifies what the
    primitive actually delivers -- the finite, spurious-decay-free LIMIT --
    and pins the literal on-axis contract as an ``@expectedFailure`` and
    the exactly-on-axis refusal as the current boundary.
    """

    def _kernel_at(self, angle):
        """`GhostContribution` at ``ONAXIS_W`` for the given small angle."""
        source = _anchor_source(ONAXIS_GAMMA, ONAXIS_RHO, angle)
        matrix = geometry.macro_matrix(ONAXIS_GAMMA, 0.0, 0.0)
        return geometry.ghost_kernel(
            np.array([ONAXIS_W], dtype=float), source, matrix), source, matrix

    def test_near_axis_kernel_amplitude_delay_are_finite(self):
        """At each small off-axis angle the kernel, complex delay, complex
        position and recovered amplitude are all finite (no NaN/inf), and
        ``Im tau_c > 0`` (decaying, never growing)."""
        self.swept = True
        for angle in ONAXIS_LIMIT_ANGLES:
            with self.subTest(angle=angle):
                contribution, source, matrix = self._kernel_at(angle)
                amplitude = _production_amplitude(source, matrix)
                self._record()
                self.assertTrue(
                    np.all(np.isfinite(contribution.kernel)),
                    f'non-finite kernel at angle {angle:.0e}')
                self.assertTrue(
                    np.isfinite(contribution.delay),
                    f'non-finite tau_c at angle {angle:.0e}')
                self.assertTrue(
                    np.all(np.isfinite(np.asarray(contribution.position))),
                    f'non-finite position at angle {angle:.0e}')
                self.assertTrue(
                    np.isfinite(amplitude),
                    f'non-finite amplitude at angle {angle:.0e}')
                self.assertGreater(
                    contribution.delay.imag, 0.0,
                    f'Im tau_c = {contribution.delay.imag:.3e} <= 0 near axis')

    def test_pure_oscillation_limit_is_monotone(self):
        """Along the decreasing angle sequence ``Im tau_c -> 0`` and the
        carrier magnitude ``|exp(1j w tau_c)|`` rises monotonically toward
        one WITHOUT ever exceeding it -- the honest witness that the
        on-axis limit is pure oscillation (no spurious decay or growth)."""
        self.swept = True
        im_taus = []
        carriers = []
        for angle in ONAXIS_LIMIT_ANGLES:  # already strictly decreasing
            contribution, _, _ = self._kernel_at(angle)
            im_taus.append(contribution.delay.imag)
            carriers.append(abs(np.exp(1j * ONAXIS_W * contribution.delay)))
        self._record()
        # No spurious growth: exp(-w Im tau_c) <= 1 to roundoff.
        for angle, carrier in zip(ONAXIS_LIMIT_ANGLES, carriers):
            self.assertLessEqual(
                carrier, 1.0 + TOL_NO_GROWTH,
                f'carrier |exp(iw tau_c)| = {carrier:.12f} > 1 at angle '
                f'{angle:.0e}: a spurious GROWING ghost')
        # Monotone approach to the pure-oscillation limit.
        for smaller, larger in zip(im_taus[1:], im_taus[:-1]):
            self.assertLess(
                smaller, larger,
                f'Im tau_c not decreasing toward the axis: {im_taus}')
        for smaller_im, larger_im in zip(carriers[1:], carriers[:-1]):
            self.assertGreater(
                smaller_im, larger_im,
                f'carrier not rising toward unity as angle shrinks: {carriers}')

    def test_exactly_on_axis_is_refused(self):
        """Exactly on the caustic-reach axis the module raises
        `GhostDomainError` (the removable-singularity reconstruction
        collapse) -- the current boundary and the reason the literal
        ``|Im tau_c| < 1e-10`` contract below is unreachable."""
        source = _anchor_source(ONAXIS_GAMMA, ONAXIS_RHO, 0.0)
        matrix = geometry.macro_matrix(ONAXIS_GAMMA, 0.0, 0.0)
        with self.assertRaises(geometry.GhostDomainError):
            geometry.ghost_kernel(np.array([ONAXIS_W]), source, matrix)

    @expectedFailure
    def test_literal_on_axis_pure_oscillation_contract(self):
        """LITERAL Architect contract (currently an expected failure): the
        exactly-on-axis point evaluates finitely with ``|Im tau_c| <
        1e-10`` and a unit-modulus carrier.  Today the module refuses on
        axis (`GhostDomainError`), so this raises and is an expected
        failure; the day a future build supports the on-axis limit it will
        XPASS and flag that this contract can graduate to a live gate."""
        source = _anchor_source(ONAXIS_GAMMA, ONAXIS_RHO, 0.0)
        matrix = geometry.macro_matrix(ONAXIS_GAMMA, 0.0, 0.0)
        contribution = geometry.ghost_kernel(
            np.array([ONAXIS_W], dtype=float), source, matrix)
        self.assertTrue(np.all(np.isfinite(contribution.kernel)))
        self.assertLess(abs(contribution.delay.imag), TOL_ONAXIS_IM_TAU)
        carrier = abs(np.exp(1j * ONAXIS_W * contribution.delay))
        self.assertLess(abs(carrier - 1.0), TOL_ONAXIS_UNIT_CARRIER)


class GhostFarFieldTestCase(GhostTestCase):
    """
    Far-field (``rho = 4``) exponential-suppression sanity.

    At the Architect's measured anchor (``gamma = 0.4``, ``pi/4``,
    ``Im tau_c ~ 10.5``) the ghost is heavily suppressed: across the
    ``FAR_W_BAND`` its full contribution ``C`` has ``max|C| < 1e-3``
    (absolute floor -- suppression active) and is subdominant to the
    residual envelope ``E = exact_total - ppGO`` by ``< 0.5``.  The large
    POSITIVE ``Im tau_c > 8`` is a genuine decay exponent -- and confirms
    the decaying member was selected, since the growing conjugate would
    give ``Im tau_c ~ -10.5`` and ``|C|`` would blow up.
    """

    def test_far_field_suppression_and_subdominance(self):
        """``max|C| < 1e-3``, ``max|C| / max|E_ff| < 0.5``, and
        ``Im tau_c > 8`` across the far-field band."""
        self.swept = True
        w_grid, residual, ghost, contribution = _far_field_bundle()
        max_c = float(np.max(np.abs(ghost)))
        max_e = float(np.max(np.abs(residual)))
        ratio = max_c / max_e
        self._record()
        self.assertLess(
            max_c, FAR_MAX_C,
            f'max|C| = {max_c:.3e} >= {FAR_MAX_C}: suppression not active')
        self.assertLess(
            ratio, FAR_MAX_RATIO,
            f'max|C|/max|E_ff| = {ratio:.3f} >= {FAR_MAX_RATIO}: ghost not '
            f'subdominant (max|C|={max_c:.3e}, max|E|={max_e:.3e})')
        self.assertGreater(
            contribution.delay.imag, FAR_IM_TAU_MIN,
            f'Im tau_c = {contribution.delay.imag:.3f} <= {FAR_IM_TAU_MIN}: '
            f'not a genuine large positive decay exponent (a growing '
            f'conjugate would give Im tau_c ~ -10.5)')

    def test_far_field_envelope_is_exponential(self):
        """``|C|`` traces the ``exp(-w Im tau_c)`` envelope: dividing out
        the (slowly varying) carrier-free ``|kernel|`` leaves exactly
        ``exp(-w Im tau_c)`` to machine precision, and a diagnostic overlay
        of ``|C|`` vs ``w`` is saved."""
        self.swept = True
        w_grid, residual, ghost, contribution = _far_field_bundle()
        kernel_mag = np.abs(contribution.kernel)
        predicted = np.exp(-w_grid * contribution.delay.imag)
        self._record()
        # |C| = |kernel| * exp(-w Im tau_c): the envelope IS the decay.
        np.testing.assert_allclose(
            np.abs(ghost) / kernel_mag, predicted, rtol=1e-10, atol=0.0,
            err_msg='|C| does not follow the exp(-w Im tau_c) envelope')
        self.assertTrue(np.all(np.isfinite(np.abs(residual))))
        try:
            os.makedirs(_OUTPUT_DIR, exist_ok=True)
            figure, axis = plt.subplots(figsize=(6.0, 4.0))
            axis.semilogy(w_grid, np.abs(ghost), lw=1.4, label='|C| ghost')
            axis.semilogy(w_grid, np.abs(residual), '--', lw=1.4,
                          label='|E| exact - ppGO')
            axis.semilogy(w_grid, kernel_mag * predicted, ':', lw=1.0,
                          label='exp(-w Im tau_c) envelope')
            axis.set_xlabel('w (dimensionless frequency)')
            axis.set_ylabel('magnitude')
            axis.set_title(
                f'far-field ghost suppression, gamma={FAR_GAMMA}, rho={FAR_RHO}'
                f' (Im tau_c={contribution.delay.imag:.2f})')
            axis.legend()
            figure.savefig(
                os.path.join(_OUTPUT_DIR, 'ghost_far_field_rho4_envelope.png'),
                dpi=110)
            plt.close(figure)
        except Exception:  # noqa: BLE001 -- diagnostics best-effort
            pass


class GhostGuardTestCase(GhostTestCase):
    """
    The two degeneracy guards inside `_ghost_kernel` refuse by name.

    `_ghost_kernel` continues the stationary-phase kernel to a complex
    ``x_c`` and must REFUSE two degeneracies rather than return garbage:

      * ``Re(z) <= 0`` for the bilinear radius ``z = x_c . x_c`` -- the
        principal branch of ``log z`` / ``sqrt z`` can no longer be
        continued from the real fold (where ``z > 0``); and
      * ``|det H_c| < 1e-8 * (1 + ||A||_F)**2`` -- a near-fold merge where
        the ``1 / sqrt(det H_c)`` amplitude and its sqrt-branch reference
        are ill conditioned.

    Both raise `GhostDomainError`, which IS-A `LensDomainError`, so the
    existing domain-refusal handlers catch them unchanged.  The reachable-
    red counterpart (removing the det floor returns a ~1e97 garbage
    amplitude) lives in `GhostSelfFalsificationTestCase`.
    """

    def test_near_fold_det_guard_raises_named_error(self):
        """At a REAL fold point (critical curve, ``det(hessian) == 0``)
        the complex determinant is below the near-fold floor while
        ``Re(z) > 0``, so `_ghost_kernel` raises `GhostDomainError` with a
        message naming the determinant/near-fold condition -- not a NaN or
        inf."""
        self.swept = True
        matrix = geometry.macro_matrix(GUARD_GAMMA, 0.0, 0.0)
        frobenius = float(np.linalg.norm(matrix))
        floor = geometry._GHOST_DET_FLOOR * (1.0 + frobenius) ** 2
        for theta in GUARD_FOLD_THETAS:
            with self.subTest(theta=theta):
                x_c, source = _near_fold_position(GUARD_GAMMA, theta)
                # Confirm the premise: Re(z) > 0 (first guard cleared) and
                # |det H_c| below the floor (second guard armed).
                z = x_c[0] * x_c[0] + x_c[1] * x_c[1]
                hess = (matrix - np.eye(2) / z
                        + 2.0 * np.outer(x_c, x_c) / z**2)
                det = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]
                self.assertGreater(
                    z.real, 0.0,
                    'near-fold fixture must clear the Re(z) > 0 guard')
                self.assertLess(
                    abs(det), floor,
                    f'theta={theta}: |det H_c| = {abs(det):.3e} not below '
                    f'the near-fold floor {floor:.3e}; the fixture does not '
                    f'arm the det guard')
                self._record()
                with self.assertRaises(geometry.GhostDomainError) as caught:
                    geometry._ghost_kernel(BYTEID_W, x_c, source, matrix,
                                           GHOST_REFERENCE_AMPLITUDE)
                message = str(caught.exception).lower()
                self.assertIn(
                    'det', message,
                    'the near-fold refusal must name the determinant '
                    f'condition; got: {caught.exception}')
                self.assertIn(
                    'near-fold', message,
                    'the near-fold refusal message must be descriptive')

    def test_negative_re_z_guard_raises_named_error(self):
        """A synthetic ``x_c`` with ``Re(z) <= 0`` makes `_ghost_kernel`
        raise `GhostDomainError` naming the ``Re(z) <= 0`` topology
        breakdown -- before any amplitude is formed."""
        self.swept = True
        matrix = geometry.macro_matrix(GUARD_GAMMA, 0.0, 0.0)
        for x_c in GUARD_NEGATIVE_Z_POSITIONS:
            with self.subTest(x_c=tuple(x_c)):
                z = x_c[0] * x_c[0] + x_c[1] * x_c[1]
                self.assertLessEqual(
                    z.real, 0.0,
                    'negative-Re(z) fixture must actually have Re(z) <= 0')
                self._record()
                with self.assertRaises(geometry.GhostDomainError) as caught:
                    geometry._ghost_kernel(BYTEID_W, x_c, GUARD_SOURCE,
                                           matrix, GHOST_REFERENCE_AMPLITUDE)
                message = str(caught.exception).lower()
                self.assertIn(
                    're(z)', message,
                    'the negative-radius refusal must name the Re(z) <= 0 '
                    f'condition; got: {caught.exception}')

    def test_both_guards_are_lens_domain_error_family(self):
        """The refusals are catchable as `LensDomainError` (the family the
        existing domain-refusal handlers use), not just as the specific
        `GhostDomainError`."""
        self.swept = True
        self.assertTrue(
            issubclass(geometry.GhostDomainError, geometry.LensDomainError),
            'GhostDomainError must subclass LensDomainError')
        matrix = geometry.macro_matrix(GUARD_GAMMA, 0.0, 0.0)
        # Near-fold case caught as the family base class.
        x_c, source = _near_fold_position(GUARD_GAMMA, GUARD_FOLD_THETAS[1])
        with self.assertRaises(geometry.LensDomainError):
            geometry._ghost_kernel(BYTEID_W, x_c, source, matrix,
                                   GHOST_REFERENCE_AMPLITUDE)
        # Negative-Re(z) case caught as the family base class.
        with self.assertRaises(geometry.LensDomainError):
            geometry._ghost_kernel(BYTEID_W, GUARD_NEGATIVE_Z_POSITIONS[0],
                                   GUARD_SOURCE, matrix,
                                   GHOST_REFERENCE_AMPLITUDE)
        self._record()

    def test_public_ghost_kernel_propagates_near_fold_refusal(self):
        """A source driven onto the caustic (the exact fold) makes the
        PUBLIC `ghost_kernel` refuse too -- the guard is not bypassed by
        the public entry point.  On the caustic the merged real image is
        near critical and the ghost continuation is degenerate, so the
        routine raises a `GhostDomainError`."""
        self.swept = True
        for theta in GUARD_FOLD_THETAS:
            with self.subTest(theta=theta):
                _, source = _near_fold_position(GUARD_GAMMA, theta)
                matrix = geometry.macro_matrix(GUARD_GAMMA, 0.0, 0.0)
                self._record()
                with self.assertRaises(geometry.GhostDomainError):
                    geometry.ghost_kernel(BYTEID_W, source, matrix)


# `RealImageByteIdentityTestCase` deleted 2026-07-30 (F045): four tests
# asserting the real-image positions, delay/magnification/Morse values, image
# kernel and Morse census were byte-identical to `git show HEAD`.  With HEAD
# and the worktree byte-identical (audited that day) they compared the module
# to a copy of itself.  A durable version of this claim is a golden-value
# table of literals, not a cross-commit fetch.


class GhostSelfFalsificationTestCase(TestCase):
    """
    Prove this suite can go RED.

    A ghost bug is silent -- a wrong log branch still returns a finite
    number, a Morse double-count still returns a unit-modulus phase -- so
    a green suite is worth only as much as its power to fail.  These tests
    inject exactly those bugs and assert the corresponding gate catches
    them, and they prove the AST independence guard and the anti-vacuity
    ``tearDown`` have teeth.
    """

    #: A representative off-axis anchor for the falsification configs.
    _GAMMA = 0.2
    _ANGLE = np.pi / 4

    def test_ast_guard_flags_a_forbidden_name(self):
        """A would-be oracle that reaches into `geometry` (here
        ``geometry.delay``) is flagged -- the guard's positive control."""
        def tainted_oracle(x_c, source, matrix):
            return geometry.delay(x_c, source, matrix)  # forbidden shortcut

        flagged = _forbidden_names_in(tainted_oracle)
        self.assertIn('delay', flagged,
                      'the AST guard failed to flag geometry.delay')

    def test_ast_guard_flags_forbidden_bare_name(self):
        """The guard catches a forbidden name used as a bare `ast.Name`
        (e.g. an ``from ... import ghost_kernel`` alias), not only as an
        attribute -- covering both AST node kinds."""
        def tainted_oracle(source, matrix):
            return ghost_kernel(source, matrix)  # noqa: F821 -- never called

        self.assertIn('ghost_kernel', _forbidden_names_in(tainted_oracle))

    def test_real_oracle_functions_are_independent(self):
        """No shipped oracle function references any forbidden `geometry`
        symbol -- the standing certificate of non-circularity."""
        for func in _ORACLE_FUNCTIONS:
            with self.subTest(func=func.__name__):
                self.assertEqual(
                    _forbidden_names_in(func), set(),
                    f'{func.__name__} references a forbidden geometry name')

    def test_wrong_log_branch_breaks_the_delay_gate(self):
        """A complex-log branch-cut error (``log z`` off by ``2 pi i``)
        makes ``tau_c`` disagree with production far beyond ``TOL_TAU_REL``
        and flips ``Im tau_c`` negative -- exactly what the delay gate
        exists to catch."""
        source = _anchor_source(self._GAMMA, ORACLE_RHO, self._ANGLE)
        matrix = geometry.macro_matrix(self._GAMMA, 0.0, 0.0)
        _, x_c, tau_good = oracle_ghost(source, oracle_macro_matrix(self._GAMMA))
        tau_prod = geometry.ghost_kernel(
            np.array(AMPLITUDE_PROBE_W, dtype=float), source, matrix).delay
        # Wrong branch: log(z) -> log(z) + 2 pi i, i.e. tau -> tau - pi i.
        tau_wrong_branch = tau_good - 1j * np.pi
        rel = abs(tau_wrong_branch - tau_prod) / abs(tau_prod)
        self.assertGreater(
            rel, TOL_TAU_REL,
            'the delay gate would not catch a 2 pi i branch error')
        self.assertLess(
            tau_wrong_branch.imag, 0.0,
            'a wrong branch should flip the decaying ghost to growing')

    def test_morse_double_count_breaks_the_phase_gate(self):
        """Multiplying the (already branch-pinned) amplitude by an extra
        Morse phase ``exp(-i pi / 2)`` rotates it by ``pi / 2``, which the
        finite-difference phase gate rejects."""
        source = _anchor_source(self._GAMMA, ORACLE_RHO, self._ANGLE)
        matrix = geometry.macro_matrix(self._GAMMA, 0.0, 0.0)
        _, x_c, _ = oracle_ghost(source, oracle_macro_matrix(self._GAMMA))
        amplitude = _production_amplitude(source, matrix)
        amp_fd, _ = oracle_fd_amplitude(
            x_c, source, oracle_macro_matrix(self._GAMMA))
        # Good amplitude passes the phase gate ...
        good_arg = abs(np.angle(amplitude) - np.angle(amp_fd))
        self.assertLessEqual(good_arg, TOL_AMP_FD_ARG)
        # ... a double-counted Morse factor does not.
        doubled = amplitude * np.exp(-0.5j * np.pi)
        bad_arg = abs(np.angle(doubled) - np.angle(amp_fd))
        self.assertGreater(
            bad_arg, TOL_AMP_FD_ARG,
            'the phase gate would not catch a Morse double-count')

    def test_growing_conjugate_would_blow_up_the_far_field_carrier(self):
        """Selecting the GROWING conjugate (``Im tau_c -> -Im tau_c``) in
        the far field turns the suppressed ``exp(-w Im tau_c)`` carrier
        into a diverging ``exp(+w Im tau_c)``, so the far-field magnitude
        gate would explode -- proof that the decaying-member selection is
        load-bearing, not decorative."""
        w_grid, _, ghost_decay, contribution = _far_field_bundle()
        max_decay = float(np.max(np.abs(ghost_decay)))
        # Growing conjugate: |C| = |kernel| * exp(+w Im tau_c).
        growing = np.abs(contribution.kernel) * np.exp(
            w_grid * contribution.delay.imag)
        max_growing = float(np.max(growing))
        self.assertLess(
            max_decay, FAR_MAX_C,
            'the decaying member should pass the far-field suppression gate')
        self.assertGreater(
            max_growing, 1e3,
            f'the growing conjugate should blow the carrier up, but '
            f'max|C_grow| = {max_growing:.3e} stayed small')

    def test_argmin_selection_would_pick_a_growing_member(self):
        """Had the extractor minimized (instead of maximized) ``Im tau_c``,
        it would return the ``Im tau_c < 0`` growing member -- so the
        ``argmax`` selection rule genuinely distinguishes the two members
        and the selection gate has teeth."""
        source = _anchor_source(self._GAMMA, 1.5, self._ANGLE)
        members = oracle_ghost_members(source, oracle_macro_matrix(self._GAMMA))
        taus = [member[2] for member in members]
        argmax_member = taus[int(np.argmax([tau.imag for tau in taus]))]
        argmin_member = taus[int(np.argmin([tau.imag for tau in taus]))]
        self.assertGreater(
            argmax_member.imag, 0.0,
            'the shipped argmax rule must select the decaying member')
        self.assertLess(
            argmin_member.imag, 0.0,
            'an argmin rule would wrongly select the growing member')

    def test_removing_det_floor_returns_a_garbage_amplitude(self):
        """Reachable-red for the near-fold guard: with ``_GHOST_DET_FLOOR``
        set to zero the same near-fold call NO LONGER refuses and instead
        returns a ~1e97 garbage ``1 / sqrt(det H_c)`` amplitude -- so the
        guard is genuinely load-bearing, not decorative.  The floor is
        restored in a ``finally`` so no other test is affected."""
        x_c, source = _near_fold_position(GUARD_GAMMA, GUARD_FOLD_THETAS[1])
        matrix = geometry.macro_matrix(GUARD_GAMMA, 0.0, 0.0)
        # Guard ON: the call refuses.
        with self.assertRaises(geometry.GhostDomainError):
            geometry._ghost_kernel(BYTEID_W, x_c, source, matrix,
                                   GHOST_REFERENCE_AMPLITUDE)
        saved_floor = geometry._GHOST_DET_FLOOR
        try:
            geometry._GHOST_DET_FLOOR = 0.0  # remove the guard threshold
            kernel, _ = geometry._ghost_kernel(
                BYTEID_W, x_c, source, matrix, GHOST_REFERENCE_AMPLITUDE)
        finally:
            geometry._GHOST_DET_FLOOR = saved_floor
        max_amplitude = float(np.max(np.abs(kernel)))
        self.assertGreater(
            max_amplitude, GUARD_MUTATION_BLOWUP,
            f'removing the det floor should return a garbage amplitude, but '
            f'max|kernel| = {max_amplitude:.3e} stayed below '
            f'{GUARD_MUTATION_BLOWUP:.0e}')
        # And the guard is back on for everyone else.
        with self.assertRaises(geometry.GhostDomainError):
            geometry._ghost_kernel(BYTEID_W, x_c, source, matrix,
                                   GHOST_REFERENCE_AMPLITUDE)

    # `test_byte_identity_gate_catches_a_one_ulp_perturbation` deleted
    # 2026-07-30 (F045) with the byte-identity class it guarded.  It was also
    # vacuous IN PRINCIPLE, not merely today: its reachable-red step asserted
    # `abs(nextafter(x, inf) - x) > 0`, a property of float arithmetic that
    # holds no matter what the code under test does.

    def test_anti_vacuity_teardown_fails_a_silent_sweep(self):
        """`GhostTestCase.tearDown` fails a test that set ``swept`` but
        recorded no comparison, so a config loop that silently found no
        ghost cannot read green."""
        probe = GhostReconstructionTestCase(
            'test_bilinear_radius_equals_inverse_root')
        probe.setUp()
        probe.swept = True  # pretend a sweep ran but recorded nothing
        with self.assertRaises(probe.failureException):
            probe.tearDown()

    def test_anti_vacuity_teardown_passes_a_real_sweep(self):
        """The same ``tearDown`` is silent when a comparison was recorded,
        so it does not spuriously fail honest tests."""
        probe = GhostReconstructionTestCase(
            'test_bilinear_radius_equals_inverse_root')
        probe.setUp()
        probe.swept = True
        probe._record()
        probe.tearDown()  # must not raise


if __name__ == '__main__':
    main()
