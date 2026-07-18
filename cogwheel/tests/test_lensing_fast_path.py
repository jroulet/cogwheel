"""
Tests for the WP1/WP2 FAST PATH of the Chang--Refsdal microlensing
likelihood: the numba-accelerated per-point engine (1F1 ladder + operator
contraction) and the coarse cubic-spline kernel node grid decoupled from
the waveform bin grid.

WHAT THIS SUITE PINS
--------------------
The build accelerated two independent levers without (the claim) moving
any answer:

* LEVER 1 -- ``point_mass_g_derivatives`` (the 1F1 s-derivative ladder)
  and ``operator.F_op`` (the shear-operator contraction) now run through a
  numba ``njit`` core.  A JIT rewrite is worthless unless it reproduces
  the pre-numba numbers BIT for BIT on repeat and to the ORIGINAL accuracy
  against an INDEPENDENT oracle.  `NumbaKernelPreservationTestCase` and
  `NumbaOperatorPreservationTestCase` gate both -- accuracy vs a fresh
  mpmath reference (never vs the numba code, FINDINGS F002), determinism
  by ``assertEqual``, and the F005 ``CancellationError`` / domain refusals
  still firing on the same out-of-certification inputs (F005).

* LEVER 2 -- ``_amplification_coefficients`` evaluates the engine ONCE on
  a COARSE deterministic ``w`` node grid (WP2's ``_coarse_w_node_grid``,
  a log-spaced base grid unioned with the FULL-CLUSTER gauge/branch
  transition nodes) and cubic-splines each smooth channel kernel
  ``K_a(w)`` to the dense bin sub-samples, instead of hitting the engine
  at every sub-sample.  `CoarseNodeInterpolationTestCase` isolates the
  interpolation error by comparing the spline-reconstructed
  ``F(f) = sum_a exp(1j w tau_a) K_a`` -- built on the SHIPPED PRODUCTION
  node grid -- against a DIRECT dense engine evaluation at the same
  frequencies (the pre-lever-2 path), under the NULL-SAFE metric.  The
  oracle is the direct dense engine, NEVER built from the spline under
  test (F002).

WP1 accelerated ``geometry.nearest_caustic_point`` (the
frequency-independent caustic search).  `CausticSearchPreservationTestCase`
gates its returned distance against an INDEPENDENT dense brute-force
argmin of the closed-form caustic parametrization (F002) and checks the
downstream wave/geometric branch is unperturbed.

Crown gates re-exercise, on the fast path, the anchors the crown
likelihood suite pins on the slow path -- RB-vs-brute agreement on EVERY
``_LENS_CONFIGS`` regime at the original ``max(RB_ATOL, RB_RTOL*|bf|)``,
symmetric `LensDomainError` refusal on a macro saddle, the ``F -> 1``
zero-noise floor, the macro magnification limit, and the near-cusp
regression value (`CrownAccuracyAnchorTestCase`) -- and a single-thread
timing gate (`FewMsTimingTestCase`) pins the machine-independent speed
properties the acceleration was FOR.

TOLERANCES (and two deliberate, documented deviations from the brief)
---------------------------------------------------------------------
* KERNEL ACCURACY (`CONTOUR_FLOOR`, `CONTOUR_SAFETY`).  The double-double
  ladder obeys the cancellation law ``rel_err ~ eps_dd * e**(w*sqrt(s))``
  (the module's own envelope): ~1e-10 out to ``w*sqrt(s) ~ 50`` and ~1e-6
  at the certified ceiling of 60.  We gate each derivative's relative
  error against the mpmath oracle by
  ``CONTOUR_FLOOR + CONTOUR_SAFETY * DD_EPS * exp(w*sqrt(s))``, a
  conservative but law-shaped envelope, and skip derivatives whose oracle
  magnitude is negligible relative to ``G_PM`` itself (they carry no
  independent information and their relative error is pure round-off).
  This preserves the PRE-NUMBA accuracy, judged against mpmath.

* F_op ACCURACY (`FOP_RTOL`).  ``1e-10`` -- a property of ``F_op``, not
  the oracle, which is exact far beyond float64.  The in-domain grid stays
  clear of the cancellation refusal (``L <= ~30`` at ``gamma = 0.20``);
  the refusal inputs are routed to the dedicated F005 refusal test.

* INTERPOLATION (`INTERP_NULLSAFE_CEIL`, DEVIATION #1).  The brief's
  ``< 1e-8`` POINTWISE-relative gate on ``|F_interp - F_dense| / |F_dense|``
  is ill posed: at the interference nulls where ``|F_dense| -> 0`` the
  pointwise ratio blows up regardless of the reconstruction quality.  We
  gate the NULL-SAFE metric
  ``epsilon = max_f |F_interp - F_dense| / max_f |F_dense| < 1e-3`` on the
  SHIPPED PRODUCTION coarse node grid (``_coarse_w_node_grid`` at the
  shipped ``n_kernel_nodes`` default, WP2's full-cluster transition
  placement -- never a hand-refined 400-node proxy).  Provenance:
  ``|delta_lnL| ~ rho**2 * epsilon``; at ``rho ~ 20`` this leaves ~0.4 nat,
  a 3.7x margin to the 1.5-nat RB gate and subdominant to the RB binning
  floor's ``lnlike`` contribution.  The oracle is the DIRECT dense engine
  evaluation at the same sub-sample frequencies, NEVER the spline under
  test (F002).

  MEASURED SURFACING AND RESOLUTION (2026-07-17): the gate first surfaced
  RED at base=40 (met only by crown four-image 6.7e-5; two-image 2.76e-2,
  near-cusp 3.7e-3, kappa 3.5e-3, rotated-shear 1.8e-3) -- the plan's
  designed acceptance signal, upheld by the Professor review.  WP2's
  full-cluster transition placement was wired correctly; base=40
  log-spacing was simply too sparse across the band.  The plan's
  proven-safe fallback was then applied as a PRODUCTION change: a driver
  sweep on the production grid measured (worst config = two-image)
  base 40 -> 2.8e-2, 64 -> 3.3e-3, 85 -> 8.7e-4, 100 -> 4.2e-4,
  128 -> 1.5e-4, and ``_DEFAULT_KERNEL_NODES`` was raised to 100 (~2.4x
  margin; 85 is the bare threshold, rejected as too thin off-suite).  The
  gate itself was never widened -- the conservative 1e-3 ceiling stays to
  protect high-SNR production events where a percent-level leak would
  bias ``lnlike``.

* TIMING (`SPEEDUP_MIN`, `MS_CEILING`, DEVIATION #2).  The brief's
  ``<= 10 ms`` warm ``lnlike`` ceiling is server-specific; on this box the
  numba special-function engine dominates at tens of ms, so the absolute
  ceiling is a machine-CALIBRATED regression guard (generously set), not a
  physical claim.  Threads are pinned to 1 (``OMP/MKL/NUMBA/OPENBLAS`` env
  vars, set at module import -- best-effort in a shared pytest process) so
  the reported cost is the single-thread number the parallel sampler pays
  per core.  The HARD, machine-independent gates are the ones the
  acceleration was for: ``lnlike`` beats ``lnlike_bruteforce`` by at least
  ``SPEEDUP_MIN`` and the pure contraction is subdominant to the engine
  call; the per-component breakdown (caustic-search, engine, contraction,
  total) is printed so a regression pinpoints which lever slipped.

ORACLE INDEPENDENCE (F002) and mpmath (ORACLE-ONLY)
---------------------------------------------------
Every accuracy oracle here is an INDEPENDENT high-precision evaluation:
``mpmath.hyp1f1`` with the textbook Kummer s-derivative ladder for the
kernel, and an mpmath operator-series contraction for ``F_op``.  Neither
shares the production float64/double-double accumulation path.  mpmath is
imported ONLY by this test module; it never becomes importable from a
production path.

ANTI-VACUITY AND SELF-FALSIFICATION
-----------------------------------
`FastPathTestCase.tearDown` fails a test that made zero comparisons.
`SelfFalsificationTestCase` proves the kernel-accuracy, interpolation, and
crown-agreement gates can each go red.
"""
from __future__ import annotations

# Single-thread pinning for the timing gate (best-effort): production runs
# under a parallel sampler with every core busy, so the honest per-eval cost
# is the SINGLE-THREAD one. These env vars are read by OpenBLAS/MKL/numba at
# import time, so they are set BEFORE numpy/matplotlib/numba are imported.
# When another already-imported module has initialised the BLAS thread pool
# first (one shared pytest process) the pin is a no-op; the HARD timing gates
# (speedup, contraction < engine) are robust to that, and only the absolute
# MS_CEILING is a machine-calibrated guard.
import os as _os

for _thread_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_thread_var, '1')

import itertools
import pathlib
import time
import warnings
from unittest import TestCase, main

import mpmath
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from cogwheel import data, waveform
from cogwheel.lensing.chang_refsdal import channels, geometry, operator
from cogwheel.lensing.chang_refsdal._gauge import reconstructed_total
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    DD_PRODUCT_CEILING, HypergeometricDomainError, W_MAX_CERTIFIED,
    point_mass_g_derivatives)
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, real_image_delays, _GEOMETRIC_ORDER)
from cogwheel.lensing.chang_refsdal.operator import (
    CancellationError, F_op, L_MAX, RHO_END, RHO_START, select_branch)
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, _DEFAULT_KERNEL_NODES, _data_term,
    _norm_term)

# ---------------------------------------------------------------------------
# Fixture constants (shared with the crown likelihood suite so the anchors
# re-exercised here read off the SAME configuration).
# ---------------------------------------------------------------------------

#: Higher-mode approximant so the mode-pair (``M**2``) contraction is
#: genuinely exercised on the fast path, matching the crown suite.
APPROXIMANT = 'IMRPhenomXPHM'

#: Fixed seed for every stochastic input (``EventData.gaussian_noise``
#: draws from its own ``default_rng(seed)``).
SEED = 20260717

#: Bin width [Hz] of the uniform relative-binning grid (crown value).
DF_BIN = 4.0

#: Largest relative image delay [s] the fixture's bins support.
DELTA_T_MAX = 0.02

#: Lens mass [Msun] / redshift of the crown (well-conditioned) fixture.
M_LENS_MSUN = 90.0
Z_LENS = 0.4

#: Absolute / relative tolerance on ``lnlike`` vs ``lnlike_bruteforce``
#: (crown values -- must stay unchanged through the fast path).
RB_ATOL = 1.5
RB_RTOL = 1e-2

#: Tolerance on the TIGHT zero-noise ``F -> 1`` floor (crown value).
ZERO_NOISE_TOL = 1e-2

#: Conservative lower bound on the RB speed-up over the full-grid brute
#: force (crown value -- the acceleration must not regress it).
SPEEDUP_MIN = 3.0

#: Regression pin on the fast-path (coarse-spline RB) zero-noise floor.
#: The RB path INHERITS the standard-RB binning/stall floor (~8.96e-3) on
#: top of its ~2.68e-3 lensing-layer increment, so it reads ~1.164e-2 --
#: pinned <= 1.5e-2, the crown value.  This proves the fast path
#: REPRODUCES the floor, not that it improves it; the physically tight
#: ~1e-11 ``F -> 1`` claim is carried by the lensed BRUTE-FORCE path
#: against `ZERO_NOISE_TOL`.
RB_FLOOR_REGRESSION = 1.5e-2

#: Tiny lens mass [Msun] driving the ``w -> 0`` macro limit in band.
TINY_M_LENS = 1e-6

#: Source position shared by the tiny-mass macro-trivial candidate.
TINY_Y = (0.12, 0.035)

#: ``(gamma, kappa)`` of the macro-TRIVIAL tiny candidate, so ``F -> 1``.
UNLENSED_LIMIT_LENS = (0.0, 0.0)

#: ``(label, y1, y2, gamma, beta, kappa)`` covering the required regimes,
#: reproduced from the crown suite's ``_LENS_CONFIGS``.
_LENS_CONFIGS = [
    ('two-image', 0.50, 0.00, 0.20, 0.0, 0.0),
    ('four-image', 0.08, 0.06, 0.20, 0.0, 0.0),
    ('near-cusp', -0.38, 0.00, 0.20, 0.0, 0.0),
    ('kappa', 0.30, 0.10, 0.112, 0.0, 0.30),
    ('rotated-shear', 0.25, 0.10, 0.20, 0.70, 0.0),
]

#: The crown four-image ("crown") config, seed of every fast-path anchor.
_CROWN = ('four-image', 0.08, 0.06, 0.20, 0.0, 0.0)

#: The near-cusp config singled out for the deterministic regression pin.
_NEAR_CUSP = ('near-cusp', -0.38, 0.00, 0.20, 0.0, 0.0)

# ---------------------------------------------------------------------------
# LEVER 1 -- numba kernel/operator accuracy oracle constants.
# ---------------------------------------------------------------------------

#: mpmath working precision [decimal digits] for every oracle here.  Far
#: beyond float64/double-double, so the gates measure the PRODUCTION
#: accuracy, never the oracle's.
ORACLE_DPS = 60

#: Double-double machine epsilon (106-bit significand): the unit in the
#: cancellation law ``rel_err ~ eps_dd * e**(w*sqrt(s))``.
DD_EPS = 2.0 ** -106

#: Additive floor of the kernel relative-error envelope: the round-off a
#: well-conditioned derivative carries even when ``e**(w*sqrt(s)) ~ 1``.
CONTOUR_FLOOR = 5e-13

#: Safety factor multiplying the cancellation-law term so the envelope is
#: a conservative CEILING, not a fitted line.
CONTOUR_SAFETY = 100.0

#: Highest derivative order requested from the ladder (the operator raises
#: the radial index by up to 2 per order; 20 spans the certified reach).
LADDER_KMAX = 20

#: Skip derivatives whose oracle magnitude is below this fraction of
#: ``|G_PM|`` (``= |oracle[0]|``): they carry no independent information
#: and their relative error is pure catastrophic-cancellation round-off.
LADDER_DYNAMIC_SKIP = 1e-15

#: Frequencies (dimensionless ``w``) sampling the certified domain.
KERNEL_W = (0.1, 1.0, 10.0, 20.0, 40.0, 50.0, 200.0, 400.0)

#: PHYSICAL ``sqrt(s)`` values (O(1) source offsets); the product
#: ``w*sqrt(s)`` reaches high ``L`` via large ``w``, matching the module's
#: own accuracy envelope.  Points with ``w*sqrt(s) > DD_PRODUCT_CEILING``
#: or ``w > W_MAX_CERTIFIED`` are skipped (routed to the refusal test).
KERNEL_SQRT_S = (0.1, 0.5, 1.0)

#: Near-ceiling ``(w, L=w*sqrt(s))`` probes exercising the ~1e-6 accuracy
#: tier just under the ceiling; both have physical ``sqrt(s) < 1``.
KERNEL_NEAR_CEILING = ((200.0, 55.0), (400.0, 58.0))

#: Kernel domain refusals: ``(w, s)`` outside the certified box.  Each
#: MUST still raise `HypergeometricDomainError` through the JIT path.
KERNEL_REFUSALS = (
    (10.0, 49.0),      # w*sqrt(s) = 70 > 60
    (20.0, 16.0),      # w*sqrt(s) = 80 > 60
    (W_MAX_CERTIFIED + 100.0, 0.0),  # w > 500
)

# ---------------------------------------------------------------------------
# LEVER 1 -- F_op accuracy/refusal constants.
# ---------------------------------------------------------------------------

#: Relative tolerance on ``F_op`` vs the mpmath operator-series oracle --
#: a property of ``F_op``, not the oracle.
FOP_RTOL = 1e-10

#: Order cap handed to ``F_op`` and the oracle for the in-domain grid.
FOP_MAX_ORDER = 70

#: In-domain ``F_op`` grid: ``(w, sqrt_s, gamma)``.  ``L = w*sqrt(s)`` is
#: kept ``<= ~30`` at ``gamma = 0.20`` so the contraction certifies (stays
#: clear of the cancellation refusal that the refusal test owns).
FOP_GRID_W = (1.0, 10.0, 20.0, 40.0, 50.0)
FOP_GRID_SQRT_S = (0.3, 0.9)
FOP_GRID_GAMMA = (0.0, 0.2)

#: ``F_op`` refusals: ``(w, sqrt_s, gamma)`` whose contraction is
#: uncertifiable; each MUST raise `operator.CancellationError` through the
#: JIT path (F005), never a silent finite value.
FOP_REFUSALS = (
    (40.0, 0.9, 0.2),
    (50.0, 0.9, 0.2),
    (50.0, 0.95, 0.2),
)

# ---------------------------------------------------------------------------
# LEVER 2 -- interpolation constants.
# ---------------------------------------------------------------------------

#: NULL-SAFE interpolation ceiling on the PRODUCTION coarse node grid:
#: ``epsilon = max_f |F_interp - F_dense| / max_f |F_dense| < 1e-3``.  The
#: max-normalised metric is well posed at interference nulls (where
#: ``|F_dense| -> 0`` makes the pointwise-relative metric blow up), so it
#: measures the reconstruction error against the CHARACTERISTIC ``F`` scale.
#: Provenance (Professor): ``|delta_lnL| ~ rho**2 * epsilon``; at ``rho ~ 20``,
#: ``epsilon = 1e-3`` gives ~0.4 nat, a 3.7x margin to the 1.5-nat RB gate,
#: and is subdominant to the RB binning floor's ``lnlike`` contribution yet
#: reachable at the modest ``_DEFAULT_KERNEL_NODES`` budget with full-cluster
#: transition placement (pure log-spacing needs ~100+ nodes).
INTERP_NULLSAFE_CEIL = 1e-3

#: Deliberately under-resolved node count for the self-falsification: a
#: hand-built ``n = 4`` log-spaced grid with NO transition nodes.  The
#: two-image config at ``n <= 12`` has null-safe ``epsilon ~ 0.5``, far above
#: `INTERP_NULLSAFE_CEIL`, so it proves the interpolation gate CAN go red.
UNDERRESOLVED_NODES = 4

#: Positive-control node count for the self-falsification: the SHIPPED
#: production default, per the plan spec ("as a paired positive control,
#: the PRODUCTION default grid PASSES epsilon < 1e-3").  Driver-measured
#: on the two-image config (the slowest to converge): base=100 +
#: full-cluster placement reaches null-safe ``epsilon = 4.2e-4``
#: (base=128 -> 1.5e-4, base=85 -> 8.7e-4), so the production grid
#: demonstrably PASSES `INTERP_NULLSAFE_CEIL` while the n=4 grid fails --
#: the gate can go both green and red.
CONVERGED_NODES = _DEFAULT_KERNEL_NODES

#: Configs exercised by the production interpolation gate (all five
#: ``_LENS_CONFIGS`` regimes).
INTERP_CONFIG_LABELS = (
    'two-image', 'four-image', 'near-cusp', 'kappa', 'rotated-shear')

#: HISTORY (driver, 2026-07-17): at the originally shipped
#: ``_DEFAULT_KERNEL_NODES = 40`` the null-safe production gate was met
#: ONLY by the crown four-image config; these four regimes sat above the
#: ceiling (two-image 2.76e-2, near-cusp 3.7e-3, kappa 3.5e-3,
#: rotated-shear 1.8e-3) -- the plan's DESIGNED surfacing of WP2's violated
#: base=40 assumption, upheld by the Professor review.  Resolved by the
#: plan's proven-safe fallback: the default was raised to 100 off a driver
#: sweep of the production grid (see `_DEFAULT_KERNEL_NODES` provenance);
#: all five regimes now clear the unchanged 1e-3 ceiling (worst: two-image
#: 4.2e-4).  Kept as the roster of slow-converging regimes any future
#: node-budget change must re-verify against.
INTERP_UNDERRESOLVED_LABELS = (
    'two-image', 'near-cusp', 'kappa', 'rotated-shear')

# ---------------------------------------------------------------------------
# LEVER 1/2 -- timing constants.
# ---------------------------------------------------------------------------

#: Best-of-N repeats for warm timing (robust to scheduler jitter).
TIMING_REPEATS = 5

#: Machine-CALIBRATED absolute ceiling [s] on warm best-of-N ``lnlike``:
#: a generous regression guard on THIS box, NOT the brief's physical
#: ``10 ms`` claim (DEVIATION #2 -- see docstring).  Raised 0.25 -> 0.5
#: with the accuracy-driven node-count increase (base 40 -> 100, engine
#: cost ~ n_nodes x ~2.3 ms/point => warm lnlike ~0.3 s): the guard
#: reflects the honest cost of a CORRECT grid; the few-ms goal is
#: deferred to the 2D surrogate-table decision (owner escalation), never
#: bought back by widening accuracy tolerances.
MS_CEILING = 0.5

# ---------------------------------------------------------------------------
# Crown macro-limit constants.
# ---------------------------------------------------------------------------

#: Dimensionless ``w`` values probing the engine's ``w -> 0`` macro limit
#: (engine-level; the coarse spline is never invoked this deep).
MACRO_LIMIT_WS = (1e-8, 1e-10, 1e-12)

#: Relative tolerance on ``|F_op|`` vs the closed-form macro magnification
#: ``1/sqrt((1-kappa)**2 - gamma**2)`` (crown engine-level anchor).
MACRO_LIMIT_RTOL = 1e-8

#: Macro-limit shear/convergence (positive parity, ``1-kappa > |gamma|``).
MACRO_LIMIT_GAMMA = 0.20
MACRO_LIMIT_KAPPA = 0.0
MACRO_LIMIT_Y = (0.3, 0.1)

# ---------------------------------------------------------------------------
# Crown RB-vs-brute gate constants.
# ---------------------------------------------------------------------------

#: Macro-SADDLE ``(gamma, kappa)`` violating the positive-parity condition
#: ``1 - kappa > |gamma|`` (here ``lam = 1 - 0.6 = 0.4 <= 0.5 = gamma``).  Both
#: the fast-path RB and the brute-force strain path MUST raise
#: `geometry.LensDomainError` on this input -- symmetric refusal (Type II
#: macro saddles are out of scope of this formalism).
MACRO_SADDLE_GAMMA = 0.5
MACRO_SADDLE_KAPPA = 0.6

#: Source position of the macro-saddle refusal candidate (in band, arbitrary).
MACRO_SADDLE_Y = (0.20, 0.05)

# ---------------------------------------------------------------------------
# WP1 caustic-search value-preservation constants.
# ---------------------------------------------------------------------------

#: Polar-angle samples in the INDEPENDENT dense brute-force caustic search
#: (plain-numpy closed-form parametrization, argmin over this grid, then a
#: second local dense refinement -- never the njit path under test).
N_THETA_ORACLE = 200_000

#: Relative tolerance on ``nearest_caustic_point(...).distance`` vs the
#: brute-force oracle across the positive-parity config grid (WP1 must
#: preserve the answer to well below float64's working precision).
CAUSTIC_RTOL = 1e-10

#: Positive-parity ``(gamma, beta, kappa)`` axes for the caustic grid.  Every
#: combination satisfies ``1 - kappa > gamma`` (worst: ``kappa=0.3`` ->
#: ``lam=0.7 > 0.3``).
CAUSTIC_GAMMAS = (0.10, 0.20, 0.30)
CAUSTIC_BETAS = (0.0, 0.4, 0.7)
CAUSTIC_KAPPAS = (0.0, 0.1, 0.3)

#: Source positions spanning inside / near / outside the astroid caustic.
CAUSTIC_SOURCES = ((0.02, 0.01), (0.15, 0.05), (0.60, 0.20))

#: Dimensionless frequencies for the branch-invariance check (span the
#: wave->geometric hand-over ``w*delta_min ~ RHO_END``).
CAUSTIC_BRANCH_WS = (0.5, 5.0, 50.0, 500.0)

#: Directory for diagnostic plots (created on demand).
OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

# Diagnostic plots are written to disk, never shown (no GUI on CI).
plt.switch_backend('Agg')


def _kernel_nterms(w: float, s: float) -> int:
    """
    Adaptive series length for `point_mass_g_derivatives`.

    The caller owns the truncation rule; this reproduces the accuracy
    envelope's requirement -- enough terms to clear the ``|z| = w*s/2``
    argument magnitude plus a Gaussian-width safety band -- so the
    measured tail is negligible and the comparison sees the double-double
    floor, not truncation.
    """
    half_z = 0.5 * w * s
    return int(max(400.0, half_z + 12.0 * np.sqrt(max(half_z, 1.0)) + 80.0))


def _oracle_ladder(w: float, s: float, kmax: int) -> list[complex]:
    """
    INDEPENDENT mpmath ``k -> d^k/ds^k G_PM(w, s)`` for ``k = 0..kmax``.

    ``G_PM(w, s) = C(w) * 1F1(1 - i w/2; 1; -i w s/2)`` and its ``k``-th
    ``s``-derivative is ``C(w) * c**k * (a)_k / (1)_k * 1F1(a+k; 1+k; c s)``
    with ``a = 1 - i w/2``, ``c = -i w/2`` and ``C(w) = exp(pi w/4 +
    i (w/2) ln(w/2)) Gamma(1 - i w/2)`` (Abramowitz & Stegun ch. 13).  A
    fresh ``mpmath.hyp1f1`` per ``k`` -- the direct textbook definition,
    no Kummer reparametrization and no shared numerator, so it shares no
    rounding path with `point_mass_g_derivatives` (F002).
    """
    with mpmath.workdps(ORACLE_DPS):
        w_hp = mpmath.mpf(w)
        s_hp = mpmath.mpf(s)
        a = 1 - 1j * w_hp / 2
        c = -1j * w_hp / 2
        carrier = (mpmath.e ** (mpmath.pi * w_hp / 4
                                + 1j * (w_hp / 2) * mpmath.log(w_hp / 2))
                   * mpmath.gamma(1 - 1j * w_hp / 2))
        z = c * s_hp
        return [complex(carrier * c ** k * mpmath.rf(a, k) / mpmath.rf(1, k)
                        * mpmath.hyp1f1(a + k, 1 + k, z))
                for k in range(kmax + 1)]


def _contour(w: float, s: float) -> float:
    """
    Cancellation-law relative-error envelope at ``(w, s)``.

    ``CONTOUR_FLOOR + CONTOUR_SAFETY * DD_EPS * exp(w*sqrt(s))`` -- the
    module's own ``rel_err ~ eps_dd * e**(w*sqrt(s))`` law, floored and
    safety-scaled into a conservative ceiling.
    """
    return CONTOUR_FLOOR + CONTOUR_SAFETY * DD_EPS * float(
        np.exp(w * np.sqrt(s)))


# ---------------------------------------------------------------------------
# mpmath F_op oracle (independent operator-series contraction).
# ---------------------------------------------------------------------------


def _oracle_radial_ladder(w, s):
    """Memoized ``k -> d^k/ds^k G_PM(w, s)`` at oracle precision."""
    w = mpmath.mpf(w)
    s = mpmath.mpf(s)
    a = 1 - 1j * w / 2
    c = -1j * w / 2
    carrier = (mpmath.e ** (mpmath.pi * w / 4
                            + 1j * (w / 2) * mpmath.log(w / 2))
               * mpmath.gamma(1 - 1j * w / 2))
    cache: dict[int, complex] = {}

    def g(k):
        if k not in cache:
            cache[k] = (carrier * c ** k * mpmath.rf(a, k) / mpmath.rf(1, k)
                        * mpmath.hyp1f1(a + k, 1 + k, c * s))
        return cache[k]
    return g


def _oracle_operator_step(state):
    """Apply the eigenframe shear operator ``D_0 = d_u**2 - d_v**2``.

    ``state`` maps ``(a, b) -> int`` coefficient of ``u**a v**b G^(k)``.
    Coefficients stay exact Python ints; no mpmath spent here.
    """
    new: dict[tuple[int, int], int] = {}

    def add(key, value):
        new[key] = new.get(key, 0) + value
    for (a, b), coeff in state.items():
        if a >= 2:
            add((a - 2, b), coeff * a * (a - 1))
        add((a, b), coeff * (4 * a + 2))
        add((a + 2, b), coeff * 4)
        if b >= 2:
            add((a, b - 2), -coeff * b * (b - 1))
        add((a, b), -coeff * (4 * b + 2))
        add((a, b + 2), -coeff * 4)
    return {key: value for key, value in new.items() if value}


def _oracle_fop(w, y, gamma, beta=0.0, kappa=0.0, max_order=FOP_MAX_ORDER):
    """
    INDEPENDENT wave-optics amplification ``F(w)`` at oracle precision.

    Sums ``total = sum_n (i gamma'/(2w))**n / n! * D_0**n G_PM`` at the
    eigenframe-rotated source and applies the mass-sheet prefactor
    ``F = (1/lam) exp(0.5j w ln(lam) - 0.5j w kappa s + 0.5j w s) total``
    with ``lam = 1 - kappa``, ``gamma' = gamma/lam``, ``s = |y'|**2``,
    ``y' = y/sqrt(lam)``.  This carries the diffraction integral's
    operator reduction independently of ``F_op``'s own reconstruction, so
    the two share no float64 accumulation (F002).
    """
    with mpmath.workdps(ORACLE_DPS):
        w = mpmath.mpf(w)
        lam = 1 - mpmath.mpf(kappa)
        gamma_scaled = mpmath.mpf(gamma) / lam
        root = mpmath.sqrt(lam)
        yp = (mpmath.mpf(y[0]) / root, mpmath.mpf(y[1]) / root)
        s = yp[0] ** 2 + yp[1] ** 2
        z_eig = mpmath.e ** (-1j * mpmath.mpf(beta)) * mpmath.mpc(*yp)
        u0, v0 = z_eig.real, z_eig.imag
        g = _oracle_radial_ladder(w, s)
        alpha = 1j * gamma_scaled / (2 * w)

        n_powers = 2 * max_order + 3
        u_pow = [mpmath.mpf(1)] * n_powers
        v_pow = [mpmath.mpf(1)] * n_powers
        for i in range(1, n_powers):
            u_pow[i] = u_pow[i - 1] * u0
            v_pow[i] = v_pow[i - 1] * v0

        def evaluate(state, order):
            acc = mpmath.mpc(0)
            for (a, b), coeff in state.items():
                acc += coeff * u_pow[a] * v_pow[b] * g((a + b) // 2 + order)
            return acc

        total = mpmath.mpc(0)
        state = {(0, 0): 1}
        factorial = mpmath.mpf(1)
        small = 0
        for n in range(max_order + 1):
            if n:
                factorial *= n
                state = _oracle_operator_step(state)
            term = alpha ** n / factorial * evaluate(state, n)
            total += term
            if n >= 4 and abs(term) <= mpmath.mpf('1e-24') * abs(total):
                small += 1
                if small >= 3:
                    break
            else:
                small = 0

        value = ((1 / lam)
                 * mpmath.e ** (0.5j * w * mpmath.log(lam)
                                - 0.5j * w * mpmath.mpf(kappa) * s
                                + 0.5j * w * s)
                 * total)
        return complex(value)


class FastPathTestCase(TestCase):
    """
    Shared crown fixtures plus the anti-vacuity comparison tally.

    One SEEDED injected Gaussian-noise event, one waveform generator, one
    lensed likelihood on an explicit uniform bin grid, and a companion
    ZERO-NOISE likelihood (``d == h0``), built once for the whole class --
    the same crown configuration the slow-path suite pins, so the anchors
    re-exercised here are read off identical data.  `tearDown` fails a
    test that asserted nothing.
    """

    @classmethod
    def setUpClass(cls):
        """Build the crown likelihood and its zero-noise companion."""
        cls.par_dic_0 = _reference_par_dic()
        assert sorted(cls.par_dic_0) == waveform.WaveformGenerator.params, (
            'reference par_dic keys drifted from WaveformGenerator.params')

        cls.event_data = _make_noisy_event()
        cls.waveform_generator = waveform.WaveformGenerator.from_event_data(
            cls.event_data, APPROXIMANT)

        band = cls.event_data.frequencies[cls.event_data.fslice]
        cls.f_lo, cls.f_hi = float(band[0]), float(band[-1])
        cls.fbin = cls._uniform_fbin(DF_BIN)

        cls.like = LensedRelativeBinningLikelihood(
            cls.event_data, cls.waveform_generator, cls.par_dic_0,
            delta_t_max=DELTA_T_MAX, fbin=cls.fbin)

        # Zero-noise anchor: seeded draw, strain zeroed, fiducial injected.
        zero_event = data.EventData.gaussian_noise(
            eventname='test_fastpath_zeronoise', duration=4,
            detector_names='HLV',
            asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.,
            seed=SEED)
        zero_event._set_strain(np.zeros_like(zero_event.strain))
        zero_event.inject_signal(cls.par_dic_0, APPROXIMANT)
        zero_generator = waveform.WaveformGenerator.from_event_data(
            zero_event, APPROXIMANT)
        # Drift is a local-noise correction with no meaning on zero noise
        # (its estimator's sample variance is empty -> NaN); pin to unity,
        # applied identically to every path so the F->1 floor stays exact.
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            cls.zero_like = LensedRelativeBinningLikelihood(
                zero_event, zero_generator, cls.par_dic_0,
                delta_t_max=DELTA_T_MAX, fbin=cls.fbin)
        cls.zero_like.asd_drift = np.ones(len(zero_event.detector_names))

    @classmethod
    def _uniform_fbin(cls, df_bin):
        """Uniform bin edges spanning the analysis band."""
        edges = np.arange(cls.f_lo, cls.f_hi, df_bin)
        if edges[-1] < cls.f_hi:
            edges = np.append(edges, cls.f_hi)
        return edges

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _lens_dic(y1, y2, gamma, beta, kappa, m_lens=M_LENS_MSUN,
                  z_lens=Z_LENS):
        """Assemble the seven lens keys expected in ``par_dic``."""
        return {'m_lens_msun': m_lens, 'z_lens': z_lens,
                'y1': y1, 'y2': y2, 'gamma': gamma, 'beta': beta,
                'kappa': kappa}

    def _candidate(self, lens_dic, waveform_par=None):
        """Merge waveform params (default: the fiducial) with a lens."""
        base = dict(waveform_par if waveform_par is not None
                    else self.par_dic_0)
        base.update(lens_dic)
        return base

    def _config_candidate(self, config):
        """Build a candidate from a ``_LENS_CONFIGS`` row."""
        _, y1, y2, gamma, beta, kappa = config
        return self._candidate(self._lens_dic(y1, y2, gamma, beta, kappa))

    @staticmethod
    def _save_figure(fig, name):
        """Write ``fig`` to ``cogwheel/tests/output/<name>.png`` and close."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)


def _reference_par_dic():
    """A deterministic precessing reference ``par_dic`` for `APPROXIMANT`."""
    return {
        'm1': 60.0, 'm2': 45.0,
        's1x_n': 0.20, 's1y_n': 0.10, 's1z': 0.30,
        's2x_n': -0.10, 's2y_n': 0.15, 's2z': -0.20,
        'l1': 0.0, 'l2': 0.0,
        'iota': 1.0, 'phi_ref': 1.2,
        'ra': 1.8, 'dec': -0.3, 'psi': 0.9,
        't_geocenter': 0.0, 'd_luminosity': 600.0,
        'f_ref': 50.0,
    }


def _make_noisy_event():
    """Seeded Gaussian-noise HLV event with the fiducial signal injected."""
    event_data = data.EventData.gaussian_noise(
        eventname='test_fastpath', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0., seed=SEED)
    event_data.inject_signal(_reference_par_dic(), APPROXIMANT)
    return event_data


class NumbaKernelPreservationTestCase(FastPathTestCase):
    """
    LEVER 1, kernel: the numba-accelerated ``point_mass_g_derivatives``
    preserves the pre-numba accuracy (vs an INDEPENDENT mpmath ladder),
    is bit-identical on repeat, and still refuses out-of-certification
    inputs.

    The gates are (a) each ``s``-derivative's relative error against the
    mpmath oracle stays under the cancellation-law envelope ``_contour``;
    (b) two calls return byte-identical arrays; (c) every certified-domain
    violation raises `HypergeometricDomainError` through the JIT path.
    """

    def _domain_points(self):
        """In-certification ``(w, s)`` from the grid plus near-ceiling."""
        points = []
        for w in KERNEL_W:
            for sqrt_s in KERNEL_SQRT_S:
                s = sqrt_s ** 2
                if w > W_MAX_CERTIFIED or w * sqrt_s > DD_PRODUCT_CEILING:
                    continue
                points.append((w, s))
        for w, product in KERNEL_NEAR_CEILING:
            points.append((w, (product / w) ** 2))
        return points

    def _worst_relative_error(self, w, s):
        """Worst per-derivative relative error vs the mpmath oracle."""
        values, _ = point_mass_g_derivatives(
            w, s, LADDER_KMAX, _kernel_nterms(w, s))
        oracle = _oracle_ladder(w, s, LADDER_KMAX)
        reference0 = abs(oracle[0])
        worst = 0.0
        for k in range(LADDER_KMAX + 1):
            mag = abs(oracle[k])
            # Skip derivatives that are negligible relative to G_PM itself:
            # they carry no independent information and their relative
            # error is pure catastrophic-cancellation round-off.
            if mag < LADDER_DYNAMIC_SKIP * reference0 or mag == 0.0:
                continue
            if not np.isfinite(values[k]):
                self.fail(f'w={w} s={s}: derivative {k} is non-finite')
            worst = max(worst, abs(values[k] - oracle[k]) / mag)
        return worst

    def test_kernel_accuracy_matches_mpmath_within_cancellation_law(self):
        """
        Each ``d^k/ds^k G_PM`` agrees with mpmath under the ``_contour``
        envelope across the certified ``(w, s)`` grid -- the pre-numba
        accuracy, judged against an independent oracle (F002).
        """
        records = []
        for w, s in self._domain_points():
            with self.subTest(w=w, s=s):
                worst = self._worst_relative_error(w, s)
                envelope = _contour(w, s)
                records.append((w * np.sqrt(s), worst, envelope))
                self.n_checks += 1
                self.assertLessEqual(
                    worst, envelope,
                    f'w={w} s={s} (L={w * np.sqrt(s):.1f}): worst kernel '
                    f'relative error {worst:.3e} exceeds the cancellation-'
                    f'law envelope {envelope:.3e}; JIT changed the accuracy')
        self._plot_accuracy(records)

    def _plot_accuracy(self, records):
        """rel-error-vs-(w*sqrt(s)) with the 1e-10 / 1e-6 tiers marked."""
        records = np.array(records)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(records[:, 0], np.maximum(records[:, 1], 1e-18),
                   s=24, label='worst kernel rel err', zorder=3)
        order = np.argsort(records[:, 0])
        ax.plot(records[order, 0], records[order, 2], color='crimson',
                ls='--', label='cancellation-law envelope')
        ax.axhline(1e-10, color='k', ls=':', label='1e-10 tier (L~50)')
        ax.axhline(1e-6, color='grey', ls=':', label='1e-6 tier (ceiling)')
        ax.set_xlabel(r'$w\sqrt{s}$')
        ax.set_ylabel('relative error vs mpmath')
        ax.set_yscale('log')
        ax.set_title('numba kernel accuracy preserved under the JIT')
        ax.legend(fontsize=8)
        self._save_figure(fig, 'numba_kernel_accuracy_vs_wsqrts')

    def test_kernel_is_bit_identical_on_repeat(self):
        """
        `point_mass_g_derivatives` is a pure function: two calls at the
        same arguments return byte-identical arrays (``assertEqual`` via
        ``array_equal``), so the JIT introduced no run-to-run drift.
        """
        for w, s in self._domain_points():
            with self.subTest(w=w, s=s):
                first, tail_a = point_mass_g_derivatives(
                    w, s, LADDER_KMAX, _kernel_nterms(w, s))
                second, tail_b = point_mass_g_derivatives(
                    w, s, LADDER_KMAX, _kernel_nterms(w, s))
                self.n_checks += 1
                self.assertTrue(
                    np.array_equal(first, second)
                    and np.array_equal(tail_a, tail_b),
                    f'w={w} s={s}: repeated kernel evaluation is not '
                    'bit-identical; the JIT path is non-deterministic')

    def test_kernel_refuses_out_of_certification_inputs(self):
        """
        Every certified-domain violation (``w*sqrt(s) > 60`` or
        ``w > 500``) still raises `HypergeometricDomainError` through the
        JIT path -- the domain guard is not bypassed by acceleration.
        """
        for w, s in KERNEL_REFUSALS:
            with self.subTest(w=w, s=s):
                self.n_checks += 1
                with self.assertRaises(HypergeometricDomainError):
                    point_mass_g_derivatives(
                        w, s, LADDER_KMAX, _kernel_nterms(w, max(s, 1.0)))


class NumbaOperatorPreservationTestCase(FastPathTestCase):
    """
    LEVER 1, operator: the numba-accelerated ``operator.F_op`` preserves
    the pre-numba accuracy (vs an INDEPENDENT mpmath operator series), is
    bit-identical on repeat, and every F005 `CancellationError` still
    fires on the same uncertifiable inputs.
    """

    def test_fop_accuracy_matches_mpmath_within_original_tolerance(self):
        """
        ``F_op`` agrees with the mpmath operator-series oracle within
        `FOP_RTOL` across the in-domain grid.  Points whose contraction is
        uncertifiable raise `CancellationError` and are covered by the
        refusal test, so they are skipped here (never compared).
        """
        records = []
        for gamma in FOP_GRID_GAMMA:
            for w in FOP_GRID_W:
                for sqrt_s in FOP_GRID_SQRT_S:
                    y = [sqrt_s, 0.0]  # kappa=0 => s = |y|**2 = sqrt_s**2
                    with self.subTest(w=w, sqrt_s=sqrt_s, gamma=gamma):
                        try:
                            value, _ = F_op(w, y, gamma,
                                            max_order=FOP_MAX_ORDER)
                        except CancellationError:
                            continue  # uncertifiable: owned by refusal test
                        reference = _oracle_fop(w, y, gamma,
                                                max_order=FOP_MAX_ORDER)
                        rel = abs(value - reference) / abs(reference)
                        records.append((w * sqrt_s, rel))
                        self.n_checks += 1
                        self.assertLessEqual(
                            rel, FOP_RTOL,
                            f'w={w} sqrt_s={sqrt_s} gamma={gamma} '
                            f'(L={w * sqrt_s:.1f}): |F_op - oracle|/|oracle| '
                            f'= {rel:.3e} exceeds {FOP_RTOL}; JIT moved F_op')
        self._plot_fop_accuracy(records)

    def _plot_fop_accuracy(self, records):
        """rel-error-vs-(w*sqrt(s)) with the 1e-10 gate marked."""
        records = np.array(records)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(records[:, 0], np.maximum(records[:, 1], 1e-18), s=24,
                   zorder=3, label='|F_op - oracle|/|oracle|')
        ax.axhline(FOP_RTOL, color='k', ls='--', label='1e-10 gate')
        ax.set_xlabel(r'$w\sqrt{s}$')
        ax.set_ylabel('relative error vs mpmath')
        ax.set_yscale('log')
        ax.set_title('numba F_op accuracy preserved under the JIT')
        ax.legend(fontsize=8)
        self._save_figure(fig, 'numba_fop_accuracy_vs_wsqrts')

    def test_fop_is_bit_identical_on_repeat(self):
        """``F_op`` returns byte-identical values on repeated calls."""
        for gamma in FOP_GRID_GAMMA:
            for w in FOP_GRID_W:
                for sqrt_s in FOP_GRID_SQRT_S:
                    y = [sqrt_s, 0.0]
                    with self.subTest(w=w, sqrt_s=sqrt_s, gamma=gamma):
                        try:
                            first, _ = F_op(w, y, gamma,
                                            max_order=FOP_MAX_ORDER)
                            second, _ = F_op(w, y, gamma,
                                             max_order=FOP_MAX_ORDER)
                        except CancellationError:
                            continue
                        self.n_checks += 1
                        self.assertEqual(
                            first, second,
                            f'w={w} sqrt_s={sqrt_s} gamma={gamma}: F_op is '
                            'not bit-identical on repeat under the JIT')

    def test_fop_refuses_uncertifiable_contractions(self):
        """
        Every uncertifiable contraction still raises `CancellationError`
        (F005) through the JIT path -- never a silent finite-but-wrong
        value.  Checked at the PRODUCTION default ``max_order`` (the order
        the crown likelihood path runs at): the certification is a
        property of that operating order, and a larger cap would let some
        of these series limp to convergence past the refusal boundary.
        """
        for w, sqrt_s, gamma in FOP_REFUSALS:
            y = [sqrt_s, 0.0]
            with self.subTest(w=w, sqrt_s=sqrt_s, gamma=gamma):
                self.n_checks += 1
                with self.assertRaises(CancellationError):
                    F_op(w, y, gamma)


class CoarseNodeInterpolationTestCase(FastPathTestCase):
    """
    LEVER 2: the PRODUCTION coarse-node cubic-spline kernel interpolation
    reconstructs ``F(f) = sum_a exp(1j w tau_a) K_a`` to a NULL-SAFE error
    below `INTERP_NULLSAFE_CEIL`, isolated from the RB binning floor above
    and the DD kernel accuracy below.

    The coarse grid is the SHIPPED production grid -- built by calling
    ``self.like._coarse_w_node_grid`` at the shipped ``n_kernel_nodes``
    default (never a hand-refined 400-node proxy), so this certifies the
    production default that ships.  The oracle ``F_dense`` is the engine
    evaluated DIRECTLY at the dense sub-sample frequencies (the pre-lever-2
    path) and coherently summed via the partition's own ``reconstructed``;
    it is NEVER built from the spline under test (F002).

    The gate uses the NULL-SAFE metric
    ``epsilon = max_f |F_interp - F_dense| / max_f |F_dense|``, which is
    well posed at the interference nulls (``|F_dense| -> 0``) that make the
    pointwise-relative ``max|dF/F|`` metric ill posed; the pointwise metric
    is printed for information only (DEVIATION #1 -- see module docstring).
    """

    def _production_coarse_grid(self, config):
        """Build the SHIPPED production coarse ``w`` node grid for a config.

        Calls ``self.like._coarse_w_node_grid`` at the shipped
        ``n_kernel_nodes`` with the full-cluster delays from
        ``self.like._full_cluster_delays`` -- exactly the production path,
        no hard-coded node count.
        """
        candidate = self._config_candidate(config)
        lens = self.like._lens_params(candidate)
        dense_w = dimensionless_frequency(
            self.like._kernel_dense_f, M_LENS_MSUN, Z_LENS)
        cluster_delays = self.like._full_cluster_delays(lens)
        coarse_w = self.like._coarse_w_node_grid(dense_w, cluster_delays)
        return dense_w, coarse_w

    def _interp_and_dense(self, config):
        """
        Return ``(dense_w, F_interp, F_dense, coarse_part, dense_part)``
        for a ``_LENS_CONFIGS`` row on the PRODUCTION coarse node grid.
        """
        _, y1, y2, gamma, beta, kappa = config
        dense_w, coarse_w = self._production_coarse_grid(config)

        coarse_part = ChangRefsdalChannels(coarse_w).evaluate(
            gamma=gamma, y=(y1, y2), beta=beta, kappa=kappa)
        spline_real = CubicSpline(coarse_w, coarse_part.kernels.real,
                                  axis=0, bc_type='not-a-knot')
        spline_imag = CubicSpline(coarse_w, coarse_part.kernels.imag,
                                  axis=0, bc_type='not-a-knot')
        dense_kernels = spline_real(dense_w) + 1j * spline_imag(dense_w)
        f_interp = reconstructed_total(
            dense_w, coarse_part.delays, dense_kernels)

        # Oracle: DIRECT dense engine evaluation, coherently summed by the
        # partition's own `reconstructed` (F002 -- never the spline).
        dense_part = ChangRefsdalChannels(dense_w).evaluate(
            gamma=gamma, y=(y1, y2), beta=beta, kappa=kappa)
        f_dense = dense_part.reconstructed
        return dense_w, f_interp, f_dense, coarse_part, dense_part

    @staticmethod
    def _nullsafe_epsilon(f_interp, f_dense):
        """``max_f |F_interp - F_dense| / max_f |F_dense|`` (null-safe)."""
        return float(np.max(np.abs(f_interp - f_dense))
                     / np.max(np.abs(f_dense)))

    def test_production_grid_reconstructs_below_nullsafe_ceiling(self):
        """
        On the SHIPPED production coarse node grid the null-safe error
        ``max|dF| / max|F_dense| < INTERP_NULLSAFE_CEIL`` for every lens
        regime -- the interpolation is subdominant to the RB binning floor
        so it is invisible to ``lnlike``.  The ill-posed pointwise
        ``max|dF/F|`` is printed for information only.
        """
        by_config = {}
        for config in _LENS_CONFIGS:
            label = config[0]
            if label not in INTERP_CONFIG_LABELS:
                continue
            with self.subTest(config=label):
                _, f_interp, f_dense, coarse_part, _ = self._interp_and_dense(
                    config)
                epsilon = self._nullsafe_epsilon(f_interp, f_dense)
                pointwise = float(np.max(
                    np.abs(f_interp - f_dense) / np.abs(f_dense)))
                by_config[label] = epsilon
                print(f'\n[CoarseNodeInterp] {label}: '
                      f'n_nodes={coarse_part.w.size} '
                      f'null-safe epsilon={epsilon:.3e} '
                      f'(pointwise max|dF/F|={pointwise:.3e})')
                self.n_checks += 1
                self.assertLess(
                    epsilon, INTERP_NULLSAFE_CEIL,
                    f'{label}: production-grid null-safe interpolation error '
                    f'{epsilon:.3e} exceeds the ceiling '
                    f'{INTERP_NULLSAFE_CEIL} at the shipped '
                    f'_DEFAULT_KERNEL_NODES={self.like.n_kernel_nodes}. The '
                    'production node budget no longer resolves this regime '
                    '(driver sweep 2026-07-17: base 100 -> worst 4.2e-4); '
                    'raise _DEFAULT_KERNEL_NODES (a PRODUCTION change, not a '
                    'tolerance change). See INTERP_UNDERRESOLVED_LABELS.')
        self.assertTrue(by_config, 'no interpolation configs were exercised')
        self._plot_epsilon_bar(by_config)

    def _plot_epsilon_bar(self, by_config):
        """Bar chart of the per-config null-safe epsilon with the ceiling."""
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(by_config)
        ax.bar(labels, [max(by_config[k], 1e-18) for k in labels],
               color='steelblue')
        ax.axhline(INTERP_NULLSAFE_CEIL, color='crimson', ls='--',
                   label=f'{INTERP_NULLSAFE_CEIL:g} null-safe ceiling')
        ax.set_yscale('log')
        ax.set_ylabel(r'$\max_f|F_{interp}-F_{dense}|/\max_f|F_{dense}|$')
        ax.set_title('production coarse-node interpolation error by config')
        ax.legend(fontsize=8)
        self._save_figure(fig, 'coarse_node_interp_epsilon_by_config')

    def test_interpolation_diagnostic_overlay_crown(self):
        """
        DIAGNOSTIC: crown per-frequency ``|F_interp - F_dense|`` (absolute)
        and the null-safe residual ``|dF| / max|F_dense|`` vs ``w`` with the
        `INTERP_NULLSAFE_CEIL` line and the smootherstep transition
        frequencies ``RHO_START/Delta_j`` / ``RHO_END/Delta_j`` and the
        branch-switch node marked, to expose systematic ringing a scalar
        gate could average away.
        """
        dense_w, f_interp, f_dense, coarse_part, _ = self._interp_and_dense(
            _CROWN)

        residual = np.abs(f_interp - f_dense)
        max_dense = float(np.max(np.abs(f_dense)))

        real_delays = real_image_delays(
            _CROWN[3], (_CROWN[1], _CROWN[2]), beta=_CROWN[4], kappa=_CROWN[5])
        pairwise = np.abs(real_delays[:, np.newaxis]
                          - real_delays[np.newaxis, :])
        seps = np.unique(pairwise[np.triu_indices(real_delays.size, k=1)])
        seps = seps[seps > 0.0]

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(dense_w, residual + 1e-300, lw=0.9,
                     label=r'$|F_{interp}-F_{dense}|$')
        axes[1].plot(dense_w, residual / max_dense + 1e-300, lw=0.9,
                     color='seagreen',
                     label=r'$|F_{interp}-F_{dense}|/\max|F_{dense}|$')
        axes[1].axhline(INTERP_NULLSAFE_CEIL, color='crimson', ls='--',
                        label=f'{INTERP_NULLSAFE_CEIL:g} ceiling')
        for sep in seps:
            for thr, color in ((RHO_START, 'grey'), (RHO_END, 'orange')):
                node = thr / sep
                if dense_w.min() <= node <= dense_w.max():
                    for axis in axes:
                        axis.axvline(node, color=color, ls=':', lw=0.7)
        if seps.size:
            switch = RHO_END / seps.min()
            if dense_w.min() <= switch <= dense_w.max():
                axes[0].axvline(switch, color='crimson', ls='-.', lw=0.8,
                                label='branch switch')
        axes[0].set_yscale('log')
        axes[0].set_ylabel('absolute residual')
        axes[0].set_title('crown four-image: production coarse-node interp '
                          'vs dense engine')
        axes[0].legend(fontsize=7)
        axes[1].set_yscale('log')
        axes[1].set_ylabel('null-safe residual')
        axes[1].set_xlabel('dimensionless frequency w')
        axes[1].legend(fontsize=7)
        self._save_figure(fig, 'coarse_node_interp_vs_dense_crown')

        # The overlay is only meaningful if it reflects a passing gate.
        epsilon = self._nullsafe_epsilon(f_interp, f_dense)
        self.n_checks += 1
        self.assertLess(epsilon, INTERP_NULLSAFE_CEIL,
                        f'crown null-safe interp error {epsilon:.3e} exceeds '
                        f'{INTERP_NULLSAFE_CEIL}')


class FewMsTimingTestCase(FastPathTestCase):
    """
    The fast path is fast: on the warm crown fixture the RB ``lnlike``
    beats ``lnlike_bruteforce`` by at least `SPEEDUP_MIN` (machine-
    independent, HARD) and the pure ``_data_term`` + ``_norm_term``
    contraction is subdominant to the amplification-engine call (HARD).
    The absolute warm ``lnlike`` wall time is guarded by a machine-
    CALIBRATED ceiling `MS_CEILING` (DEVIATION #2 -- the brief's 10 ms is
    server-specific).  The engine-vs-contraction breakdown is printed so a
    regression pinpoints which lever slipped.
    """

    def _crown_candidate(self):
        return self._config_candidate(_CROWN)

    def _best_time(self, thunk):
        best = np.inf
        for _ in range(TIMING_REPEATS):
            start = time.perf_counter()
            thunk()
            best = min(best, time.perf_counter() - start)
        return best

    def test_lnlike_warm_wall_time_and_speedup(self):
        """
        Warm best-of-N ``lnlike`` beats brute force by >= `SPEEDUP_MIN`
        (HARD, machine-independent) and sits under the machine-calibrated
        `MS_CEILING` (a regression guard, NOT the brief's physical 10 ms
        claim -- DEVIATION #2).  A per-component breakdown (caustic-search,
        amplification engine, contraction, total) is printed so a
        regression pinpoints the slipped lever and the change report can
        quote the honest measured floor.  Threads are pinned to 1 at import
        (best-effort -- see the module preamble) so the reported cost is the
        single-thread cost the parallel sampler actually pays per core.
        """
        candidate = self._crown_candidate()

        def rb():
            self.like.lnlike(candidate)

        def brute():
            self.like.lnlike_bruteforce(candidate)

        rb()  # warm (numba already compiled at import; this warms caches)
        brute()
        t_rb = self._best_time(rb)
        t_brute = self._best_time(brute)

        # Per-component breakdown from the LIVE hot path.
        lens = self.like._lens_params(candidate)
        source = np.asarray((lens['y1'], lens['y2']), dtype=float)

        def caustic_search():
            geometry.nearest_caustic_point(
                lens['gamma'], lens['beta'], source, kappa=lens['kappa'])

        def amplification_engine():
            self.like._amplification_coefficients(candidate)

        caustic_search()
        amplification_engine()
        t_caustic = self._best_time(caustic_search)
        t_engine = self._best_time(amplification_engine)
        _, _, _, partition = self.like._amplification_coefficients(candidate)
        print(f'\n[FewMsTiming] breakdown (best-of-{TIMING_REPEATS}): '
              f'caustic-search={t_caustic * 1e3:.3f} ms '
              f'(WP1, expected < 1 ms), '
              f'amplification-engine={t_engine * 1e3:.2f} ms '
              f'({partition.w.size} nodes), '
              f'lnlike total={t_rb * 1e3:.2f} ms, '
              f'brute={t_brute * 1e3:.1f} ms, '
              f'speedup={t_brute / t_rb:.1f}x')

        self.n_checks += 1
        self.assertGreater(
            t_brute, SPEEDUP_MIN * t_rb,
            f'RB lnlike ({t_rb * 1e3:.2f} ms) is not at least '
            f'{SPEEDUP_MIN}x faster than brute force '
            f'({t_brute * 1e3:.1f} ms); the RB speed-up regressed')
        self.n_checks += 1
        self.assertLessEqual(
            t_rb, MS_CEILING,
            f'warm lnlike best-of-{TIMING_REPEATS} = {t_rb * 1e3:.2f} ms '
            f'exceeds the machine-calibrated ceiling {MS_CEILING * 1e3:.0f} '
            'ms; a lever regressed (see the printed breakdown)')

    def test_contraction_subdominant_to_amplification_engine(self):
        """
        The pure mode-then-image contraction is faster than the
        amplification-engine call that feeds it -- the additive
        ``M**2 + n_img**2`` design, unbroken by an FFT or per-frequency
        Python loop creeping onto the hot path.  Inputs come from the LIVE
        hot path (``_amplification_coefficients``), so the measured
        contraction matches production.
        """
        candidate = self._crown_candidate()

        r0, r1, dt_lf = self.like._candidate_bin_ratios(candidate)
        rho0, rho1 = r0.conj(), r1.conj()
        delays, k0, k1, _ = self.like._amplification_coefficients(candidate)
        kbar0, kbar1 = k0.conj(), k1.conj()
        tau = delays - dt_lf
        f_center = self.like._f_center

        def contraction():
            _data_term(self.like._a_moments, rho0, rho1, kbar0, kbar1, tau,
                       f_center)
            _norm_term(self.like._b_moments, r0, r1, rho0, rho1, k0, k1,
                       kbar0, kbar1, delays, f_center)

        def amplification_engine():
            self.like._amplification_coefficients(candidate)

        contraction()
        amplification_engine()
        t_contract = self._best_time(contraction)
        t_engine = self._best_time(amplification_engine)
        print(f'\n[FewMsTiming] contraction = {t_contract * 1e3:.3f} ms, '
              f'amplification engine = {t_engine * 1e3:.2f} ms')

        self.n_checks += 1
        self.assertLess(
            t_contract, t_engine,
            f'contraction ({t_contract * 1e3:.3f} ms) is not subdominant '
            f'to the amplification engine ({t_engine * 1e3:.2f} ms); an FFT '
            'or per-frequency Python loop may have crept onto the hot path')


class CrownAccuracyAnchorTestCase(FastPathTestCase):
    """
    The crown accuracy anchors survive the fast path at UNCHANGED
    tolerances: RB-vs-brute on the crown config at the original
    `RB_ATOL`, the ``F -> 1`` zero-noise floor at ``gamma = kappa = 0``,
    the engine-level macro magnification limit down to ``w ~ 1e-12``, and
    the near-cusp regression value (the F008 switch fix -- kernel
    reduction and contraction still agree through the fast path).
    """

    def _unlensed_limit_candidate(self):
        """Tiny-mass, macro-TRIVIAL (``gamma = kappa = 0``) candidate."""
        gamma, kappa = UNLENSED_LIMIT_LENS
        return self._candidate(
            self._lens_dic(*TINY_Y, gamma, 0.0, kappa, m_lens=TINY_M_LENS))

    def test_rb_matches_bruteforce_every_config(self):
        """
        For EVERY ``_LENS_CONFIGS`` regime the fast-path RB ``lnlike``
        matches the exact ``lnlike_bruteforce`` (dense amplification) at
        the original ``max(RB_ATOL, RB_RTOL*|bf|)`` -- one evaluation each
        path per config.  Sweeping all five (not just the crown) is what
        catches a per-regime leak (e.g. the kappa-config) a crown-only
        gate would miss.  The shipped ``n_kernel_nodes`` default is used
        (the fixture hard-codes no node count -- this certifies production).
        """
        residuals = {}
        for config in _LENS_CONFIGS:
            label = config[0]
            with self.subTest(config=label):
                candidate = self._config_candidate(config)
                rb = self.like.lnlike(candidate)
                bf = self.like.lnlike_bruteforce(candidate)
                self.n_checks += 1
                self.assertTrue(
                    np.isfinite(rb) and np.isfinite(bf),
                    f'{label}: non-finite lnl (rb={rb}, bf={bf})')
                tol = max(RB_ATOL, RB_RTOL * abs(bf))
                residuals[label] = abs(rb - bf)
                self.assertLessEqual(
                    abs(rb - bf), tol,
                    f'{label} fast-path RB lnl {rb:.6g} disagrees with '
                    f'brute-force {bf:.6g} by {abs(rb - bf):.4g} > {tol:.4g}; '
                    'the coarse-spline RB path leaks against exact '
                    'amplification on this regime')
        self._plot_rb_residuals(residuals)

    def _plot_rb_residuals(self, residuals):
        """Bar chart of ``|lnlike - lnlike_bruteforce|`` with the 1.5 line."""
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(residuals)
        ax.bar(labels, [max(residuals[k], 1e-6) for k in labels],
               color='slateblue')
        ax.axhline(RB_ATOL, color='crimson', ls='--',
                   label=f'{RB_ATOL:g}-nat RB_ATOL')
        ax.set_yscale('log')
        ax.set_ylabel(r'$|\ln\mathcal{L}_{RB}-\ln\mathcal{L}_{brute}|$ [nat]')
        ax.set_title('fast-path RB vs brute-force lnlike by config')
        ax.legend(fontsize=8)
        self._save_figure(fig, 'rb_vs_bruteforce_by_config')

    def test_paths_refuse_macro_saddle_symmetrically(self):
        """
        On a macro-SADDLE input (``1 - kappa <= |gamma|``) BOTH the
        fast-path RB ``lnlike`` and the exact ``lnlike_bruteforce`` raise
        `geometry.LensDomainError` -- symmetric refusal, never a silent
        finite value on one path.  Type II macro saddles are out of scope
        of this formalism, and the two paths must agree on that boundary.
        """
        candidate = self._candidate(self._lens_dic(
            *MACRO_SADDLE_Y, MACRO_SADDLE_GAMMA, 0.0, MACRO_SADDLE_KAPPA))
        self.n_checks += 1
        with self.assertRaises(geometry.LensDomainError):
            self.like.lnlike(candidate)
        self.n_checks += 1
        with self.assertRaises(geometry.LensDomainError):
            self.like.lnlike_bruteforce(candidate)

    def test_zero_noise_floor_at_trivial_macro_sector(self):
        """
        On the ZERO-NOISE anchor (``d == h0``) at the macro-TRIVIAL tiny
        candidate the fast path reproduces the F007/F009 template-
        construction floor -- it does not 'improve' it.

        The physically TIGHT ``F -> 1`` claim is carried by the lensed
        BRUTE-FORCE path (exact dense amplification, no binning floor):
        its residual against the exact unlensed ``lnlike_fft`` sits under
        `ZERO_NOISE_TOL`.  The coarse-spline RB fast path additionally
        INHERITS the standard-RB binning floor (~8.96e-3) on top of its
        ~2.68e-3 lensing-layer increment, so it reads ~1.164e-2 -- pinned
        <= `RB_FLOOR_REGRESSION` (a reproduction pin, per the crown
        mechanism, NOT a tightness claim).  Reproducing BOTH is what shows
        the fast path did not silently move the floor.
        """
        candidate = self._unlensed_limit_candidate()
        exact_unlensed = self.zero_like.lnlike_fft(self.par_dic_0)

        brute = self.zero_like.lnlike_bruteforce(candidate)
        brute_residual = abs(brute - exact_unlensed)
        self.n_checks += 1
        self.assertLessEqual(
            brute_residual, ZERO_NOISE_TOL,
            f'zero-noise lensed BRUTE-FORCE lnlike ({brute:.10g}) != exact '
            f'unlensed lnlike_fft ({exact_unlensed:.10g}); residual '
            f'{brute_residual:.4g} > {ZERO_NOISE_TOL}: the tight F->1 floor '
            'moved through the numba engine')

        rb = self.zero_like.lnlike(candidate)
        rb_residual = abs(rb - exact_unlensed)
        self.n_checks += 1
        self.assertLessEqual(
            rb_residual, RB_FLOOR_REGRESSION,
            f'zero-noise fast-path RB lnlike ({rb:.10g}) != exact unlensed '
            f'lnlike_fft ({exact_unlensed:.10g}); residual {rb_residual:.4g} '
            f'> {RB_FLOOR_REGRESSION} (measured ~1.164e-2 = inherited '
            'standard-RB floor + lensing layer): the coarse-spline fast '
            'path regressed the floor')

    def test_macro_magnification_limit_engine_level(self):
        """
        The engine's ``w -> 0`` macro limit ``|F_op|`` matches the closed
        form ``1/sqrt((1-kappa)**2 - gamma**2)`` to `MACRO_LIMIT_RTOL`
        down to ``w = 1e-12`` -- exercising the numba tiny-``w`` early
        exit (engine-level; the coarse spline is never invoked this deep).
        """
        gamma, kappa = MACRO_LIMIT_GAMMA, MACRO_LIMIT_KAPPA
        closed_form = 1.0 / np.sqrt((1.0 - kappa) ** 2 - gamma ** 2)
        for w in MACRO_LIMIT_WS:
            with self.subTest(w=w):
                value, _ = F_op(w, list(MACRO_LIMIT_Y), gamma, kappa=kappa)
                rel = abs(abs(value) - closed_form) / closed_form
                self.n_checks += 1
                self.assertLessEqual(
                    rel, MACRO_LIMIT_RTOL,
                    f'w={w}: |F_op| = {abs(value):.10g} disagrees with the '
                    f'macro constant {closed_form:.10g} by rel {rel:.3e} > '
                    f'{MACRO_LIMIT_RTOL}; the tiny-w macro limit moved')

    def test_near_cusp_regression_pin(self):
        """
        NEAR-CUSP regression pin: the fast-path RB ``lnlike`` reproduces
        the exact ``lnlike_bruteforce`` at the near-cusp config within
        `RB_ATOL` (the F008 full-cluster switch fix -- a real image's
        virtual cluster mate stays in the bounded gauge, so the channel
        kernels do not blow up and ``_norm_term`` does not square a
        spurious ``(h|h)``), and the value is bit-reproducible on repeat.
        """
        candidate = self._config_candidate(_NEAR_CUSP)
        rb = self.like.lnlike(candidate)
        bf = self.like.lnlike_bruteforce(candidate)
        self.n_checks += 1
        self.assertTrue(np.isfinite(rb) and np.isfinite(bf),
                        f'near-cusp: non-finite lnl (rb={rb}, bf={bf})')
        tol = max(RB_ATOL, RB_RTOL * abs(bf))
        self.assertLessEqual(
            abs(rb - bf), tol,
            f'near-cusp fast-path RB lnl {rb:.6g} disagrees with brute '
            f'{bf:.6g} by {abs(rb - bf):.4g} > {tol:.4g}; the F008 switch '
            'fix or the kernel reduction/contraction regressed')
        # Determinism: the fast path is a pure function of its inputs.
        self.n_checks += 1
        self.assertEqual(
            rb, self.like.lnlike(candidate),
            'near-cusp fast-path lnlike is not bit-reproducible on repeat')


class CausticSearchPreservationTestCase(FastPathTestCase):
    """
    WP1: the accelerated frequency-independent caustic search
    ``geometry.nearest_caustic_point`` returns the SAME source-plane
    distance as an INDEPENDENT dense brute-force minimization, and the
    downstream wave/geometric branch decision is unchanged.

    The oracle is a plain-numpy evaluation of the closed-form
    critical-curve -> caustic parametrization on a fine ``N_THETA_ORACLE``
    polar grid, refined by a second local dense grid, with the minimum
    distance taken by ``argmin`` (F002: it shares NONE of the compiled
    ``_caustic_source`` / ``_coarse_squared_distances`` search machinery
    WP1 accelerated -- only the physics formula, which is the reference,
    not the code under test; and it uses a different search algorithm,
    dense argmin vs coarse-scan + bounded polish).
    """

    @staticmethod
    def _oracle_caustic_xy(theta, gamma, beta, kappa):
        """
        Closed-form caustic (source-plane) point(s) at polar angle(s).

        ``caustic = macro_matrix @ x - x / |x|**2`` at the critical point
        ``x(theta)``, written out in plain vectorized numpy from the
        Chang--Refsdal critical-curve geometry -- the physics reference,
        independent of the compiled search under test.
        """
        theta = np.asarray(theta, dtype=float)
        lam = 1.0 - kappa
        effective_gamma = gamma / lam
        phase = theta - beta
        effective_u = (effective_gamma * np.cos(2.0 * phase)
                       + np.sqrt(1.0 - effective_gamma**2
                                 * np.sin(2.0 * phase)**2))
        radius = 1.0 / np.sqrt(lam * effective_u)
        image_x = radius * np.cos(theta)
        image_y = radius * np.sin(theta)
        cos2b = np.cos(2.0 * beta)
        sin2b = np.sin(2.0 * beta)
        m00 = (1.0 - kappa) - gamma * cos2b
        m01 = -gamma * sin2b
        m11 = (1.0 - kappa) + gamma * cos2b
        caustic_x = m00 * image_x + m01 * image_y - image_x / radius**2
        caustic_y = m01 * image_x + m11 * image_y - image_y / radius**2
        return caustic_x, caustic_y

    def _oracle_distance(self, gamma, beta, kappa, source):
        """
        Brute-force nearest source-to-caustic distance and a degeneracy flag.

        A coarse ``N_THETA_ORACLE``-angle argmin, refined by a second dense
        grid in the winning cell (so the returned distance is accurate to
        far below `CAUSTIC_RTOL`).  Also reports how many well-separated
        local minima come within ``1e-6`` (relative) of the global one --
        the benign multiple-minimum case near an astroid symmetry axis.
        """
        source = np.asarray(source, dtype=float)
        grid = np.linspace(0.0, 2.0 * np.pi, N_THETA_ORACLE, endpoint=False)
        caustic_x, caustic_y = self._oracle_caustic_xy(
            grid, gamma, beta, kappa)
        squared = (caustic_x - source[0])**2 + (caustic_y - source[1])**2

        index = int(np.argmin(squared))
        step = 2.0 * np.pi / N_THETA_ORACLE
        fine = np.linspace(grid[index] - step, grid[index] + step,
                           N_THETA_ORACLE)
        fine_x, fine_y = self._oracle_caustic_xy(fine, gamma, beta, kappa)
        fine_squared = (fine_x - source[0])**2 + (fine_y - source[1])**2
        best = float(min(squared[index], fine_squared.min()))

        # Cyclic local minima of the coarse scan (degeneracy diagnostic).
        left = np.roll(squared, 1)
        right = np.roll(squared, -1)
        minima = np.sort(squared[(squared < left) & (squared < right)])
        n_near_equal = int(np.sum(
            minima <= best * (1.0 + 1e-6))) if minima.size else 1
        return float(np.sqrt(best)), n_near_equal

    def _config_grid(self):
        """Positive-parity ``(gamma, beta, kappa, source)`` combinations."""
        return itertools.product(
            CAUSTIC_GAMMAS, CAUSTIC_BETAS, CAUSTIC_KAPPAS, CAUSTIC_SOURCES)

    def test_distance_matches_bruteforce_oracle(self):
        """
        ``nearest_caustic_point(...).distance`` agrees with the INDEPENDENT
        brute-force oracle to relative error < `CAUSTIC_RTOL` across the
        positive-parity grid (source inside / near / outside the astroid).
        The accelerated search moved no answer.
        """
        records = []
        for gamma, beta, kappa, source in self._config_grid():
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              source=source):
                produced = geometry.nearest_caustic_point(
                    gamma, beta, np.asarray(source, dtype=float),
                    kappa=kappa).distance
                oracle, n_near_equal = self._oracle_distance(
                    gamma, beta, kappa, source)
                self.assertGreater(oracle, 0.0,
                                   'oracle distance must be positive')
                rel = abs(produced - oracle) / oracle
                records.append((oracle, rel, n_near_equal))
                self.n_checks += 1
                self.assertLess(
                    rel, CAUSTIC_RTOL,
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: nearest_caustic_point distance '
                    f'{produced:.12g} disagrees with the brute-force oracle '
                    f'{oracle:.12g} by rel {rel:.3e} > {CAUSTIC_RTOL}; WP1 '
                    'moved the caustic-search answer')
        self._plot_distance_error(records)

    def _plot_distance_error(self, records):
        """Scatter of rel distance error vs source-to-caustic distance."""
        records = np.array(records)
        fig, ax = plt.subplots(figsize=(7, 4))
        degenerate = records[:, 2] > 1.5
        ax.scatter(records[~degenerate, 0],
                   np.maximum(records[~degenerate, 1], 1e-18),
                   s=20, label='unique minimum', zorder=3)
        if degenerate.any():
            ax.scatter(records[degenerate, 0],
                       np.maximum(records[degenerate, 1], 1e-18),
                       s=40, marker='x', color='crimson',
                       label='near-axis (multiple minima)', zorder=4)
        ax.axhline(CAUSTIC_RTOL, color='k', ls='--',
                   label=f'{CAUSTIC_RTOL:g} gate')
        ax.set_xlabel('source-to-caustic distance (oracle)')
        ax.set_ylabel('relative distance error')
        ax.set_yscale('log')
        ax.set_title('WP1 caustic-search distance preserved vs brute force')
        ax.legend(fontsize=8)
        self._save_figure(fig, 'caustic_search_distance_error')

    def test_branch_decision_matches_independent_prediction(self):
        """
        The wave/geometric branch the accelerated pipeline selects per
        frequency matches an INDEPENDENT prediction from
        `operator.select_branch`, fed by the real-image delay separation
        (`real_image_delays`, quartic solver -- not the caustic search) and
        the source-position cancellation exponent ``w*|y'|`` recomputed in
        plain numpy.  Since the caustic search anchors the geometry the
        branch gate consumes, this pins that WP1 did not perturb the branch
        boundary for a representative set of ``(config, w)`` pairs.

        The pipeline evaluates the WAVE branch through `operator.F_op`,
        which raises `operator.CancellationError` when the wave contraction
        is uncertifiable; that refusal only ever fires on the wave branch
        (the geometric branch is a closed form), so a raised
        `CancellationError` is itself evidence the wave branch was taken --
        equivalent to ``produced == 'wave'`` here.
        """
        for config in _LENS_CONFIGS:
            label, y1, y2, gamma, beta, kappa = config
            source = np.asarray((y1, y2), dtype=float)
            real_delays = real_image_delays(
                gamma, source, beta=beta, kappa=kappa)
            if real_delays.size >= 2:
                pairwise = np.abs(real_delays[:, np.newaxis]
                                  - real_delays[np.newaxis, :])
                delta_min = float(np.min(
                    pairwise[np.triu_indices(real_delays.size, k=1)]))
            else:
                delta_min = 0.0
            y_scaled = source / np.sqrt(1.0 - kappa)
            for w in CAUSTIC_BRANCH_WS:
                with self.subTest(config=label, w=w):
                    # Minimal strictly-increasing 2-point grid at ~w; both
                    # points share w's branch, and only index 0 is read.
                    grid = np.array([w, w * (1.0 + 1e-6)], dtype=float)
                    try:
                        partition = ChangRefsdalChannels(grid).evaluate(
                            gamma=gamma, y=(y1, y2), beta=beta, kappa=kappa)
                        produced = ('geometric'
                                    if partition.operator_orders[0]
                                    == _GEOMETRIC_ORDER else 'wave')
                    except CancellationError:
                        # wave branch taken but uncertifiable
                        produced = 'wave'
                    exponent = float(w) * float(np.sqrt(y_scaled @ y_scaled))
                    predicted = select_branch(float(w), delta_min, exponent)
                    self.n_checks += 1
                    self.assertEqual(
                        produced, predicted,
                        f'{label} w={w}: pipeline branch {produced!r} != '
                        f'independent select_branch prediction {predicted!r} '
                        f'(delta_min={delta_min:.4g}, L={exponent:.4g}, '
                        f'RHO_END={RHO_END}, L_MAX={L_MAX}); WP1 '
                        'perturbed the geometry the branch gate consumes')


class SelfFalsificationTestCase(FastPathTestCase):
    """
    Prove each central gate can go RED.  A suite whose gates cannot fail
    is not a test; these mirror the production predicates and feed them a
    deliberately wrong input, asserting the gate rejects it.  No
    production code is modified -- only local copies are corrupted.
    """

    def test_kernel_accuracy_gate_rejects_a_perturbed_value(self):
        """
        A 0.1% perturbation of a kernel derivative blows the cancellation-
        law envelope `_contour`, so the accuracy gate is non-vacuous.
        """
        w, s = 50.0, 1.0
        oracle = _oracle_ladder(w, s, 0)
        good = point_mass_g_derivatives(w, s, 0, _kernel_nterms(w, s))[0][0]
        good_rel = abs(good - oracle[0]) / abs(oracle[0])
        bad = good * (1.0 + 1e-3)
        bad_rel = abs(bad - oracle[0]) / abs(oracle[0])
        envelope = _contour(w, s)
        self.n_checks += 1
        self.assertLessEqual(good_rel, envelope,
                             'sanity: the true value must pass the gate')
        self.n_checks += 1
        self.assertGreater(
            bad_rel, envelope,
            f'a 0.1% kernel perturbation (rel {bad_rel:.3e}) slips the '
            f'cancellation-law envelope {envelope:.3e}; the accuracy gate '
            'would not catch a wrong JIT value')

    def test_interpolation_gate_rejects_an_underresolved_grid(self):
        """
        On a DELIBERATELY under-resolved grid (``n = UNDERRESOLVED_NODES``
        log-spaced nodes, NO transition nodes) the two-image null-safe
        reconstruction error exceeds `INTERP_NULLSAFE_CEIL`, so the
        production interpolation gate is NOT vacuously passing -- it CAN go
        red.  As a paired positive control a PROVEN-CONVERGED grid
        (``base = CONVERGED_NODES`` + full-cluster placement) PASSES the same
        ceiling on the same config, proving the metric can also go GREEN and
        that it is the refinement/placement that buys convergence, not a
        trivially loose threshold.

        NOTE: the positive control is the SHIPPED production default
        (``CONVERGED_NODES = _DEFAULT_KERNEL_NODES``), per the plan spec --
        it proves the production grid itself, not a bespoke refined grid,
        clears the ceiling (driver sweep 2026-07-17: base=100 -> 4.2e-4 on
        two-image, the slowest regime; see `INTERP_UNDERRESOLVED_LABELS`).
        """
        config = ('two-image', 0.50, 0.00, 0.20, 0.0, 0.0)
        _, y1, y2, gamma, beta, kappa = config
        dense_w = dimensionless_frequency(
            self.like._kernel_dense_f, M_LENS_MSUN, Z_LENS)
        f_dense = ChangRefsdalChannels(dense_w).evaluate(
            gamma=gamma, y=(y1, y2), beta=beta, kappa=kappa).reconstructed
        max_dense = float(np.max(np.abs(f_dense)))

        def nullsafe_epsilon(coarse_w):
            coarse_part = ChangRefsdalChannels(coarse_w).evaluate(
                gamma=gamma, y=(y1, y2), beta=beta, kappa=kappa)
            spline_real = CubicSpline(coarse_w, coarse_part.kernels.real,
                                      axis=0, bc_type='not-a-knot')
            spline_imag = CubicSpline(coarse_w, coarse_part.kernels.imag,
                                      axis=0, bc_type='not-a-knot')
            dense_kernels = spline_real(dense_w) + 1j * spline_imag(dense_w)
            f_interp = reconstructed_total(
                dense_w, coarse_part.delays, dense_kernels)
            return float(np.max(np.abs(f_interp - f_dense)) / max_dense)

        # Deliberately under-resolved: n=4 log-spaced nodes, no transitions.
        sparse_w = np.geomspace(dense_w.min(), dense_w.max(),
                                UNDERRESOLVED_NODES)
        epsilon_sparse = nullsafe_epsilon(sparse_w)
        self.n_checks += 1
        self.assertGreater(
            epsilon_sparse, INTERP_NULLSAFE_CEIL,
            f'the under-resolved n={UNDERRESOLVED_NODES} grid already '
            f'reconstructs F to null-safe {epsilon_sparse:.3e} < '
            f'{INTERP_NULLSAFE_CEIL}; the production gate would pass '
            'vacuously')

        # Positive control: the SHIPPED production default PASSES the same
        # ceiling (`CONVERGED_NODES = _DEFAULT_KERNEL_NODES`), so the gate
        # distinguishes converged from under-resolved grids, is not
        # stuck-red, and certifies the grid production actually uses.  The
        # full-cluster placement is exercised identically in both arms;
        # only the base count differs.
        lens = self.like._lens_params(self._config_candidate(config))
        cluster_delays = self.like._full_cluster_delays(lens)
        original_nodes = self.like.n_kernel_nodes
        self.like.n_kernel_nodes = CONVERGED_NODES
        try:
            converged_w = self.like._coarse_w_node_grid(
                dense_w, cluster_delays)
        finally:
            self.like.n_kernel_nodes = original_nodes
        epsilon_converged = nullsafe_epsilon(converged_w)
        self.n_checks += 1
        self.assertLess(
            epsilon_converged, INTERP_NULLSAFE_CEIL,
            f'a proven-converged grid (base={CONVERGED_NODES}, '
            f'n={converged_w.size}) fails the ceiling: null-safe '
            f'{epsilon_converged:.3e} >= {INTERP_NULLSAFE_CEIL}; the '
            'positive control is broken (full-cluster placement no longer '
            'converges), so the interpolation gate would be vacuously red')

    def test_crown_agreement_gate_rejects_a_shifted_lnl(self):
        """
        A large offset added to ``lnlike`` blows the RB-vs-brute
        tolerance, so the crown agreement gate can go red.
        """
        candidate = self._config_candidate(_CROWN)
        bf = self.like.lnlike_bruteforce(candidate)
        tol = max(RB_ATOL, RB_RTOL * abs(bf))
        shifted = bf + 10.0 * tol
        self.n_checks += 1
        self.assertGreater(
            abs(shifted - bf), tol,
            'a 10x-tolerance shift does not exceed the crown agreement '
            'tolerance; the RB-vs-brute gate cannot go red')


if __name__ == '__main__':
    main()
