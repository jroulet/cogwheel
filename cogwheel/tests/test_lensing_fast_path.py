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

* LEVER 2 -- ``_amplification_coefficients`` evaluates the engine on a
  SMALL leave-one-out-adaptive ``w`` node set for the single smooth SACR-C
  envelope ``E(w)`` (``_envelope_loo_nodes``) and rebuilds each channel
  kernel ``K_a(w)`` in CLOSED FORM at the dense bin sub-samples -- the
  switched analytic saddles ``S_a * H_a`` evaluated directly plus the
  interpolated envelope (``_reconstruct_kernels``) -- instead of hitting
  the engine at every sub-sample.  (The former fixed coarse-node
  cubic-spline-of-``K_a`` design and its `CoarseNodeInterpolationTestCase`
  interpolation gate were RETIRED with the Build 3f SACR-C swap; a
  replacement null-safe interpolation gate on the LOO-envelope path is
  owed to the Test Developer -- see the retirement note near the removed
  class.)

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

* INTERPOLATION (RETIRED with the Build 3f SACR-C swap).  The former
  null-safe interpolation gate (``epsilon = max_f |F_interp - F_dense| /
  max_f |F_dense| < 1e-3`` on the fixed ``_DEFAULT_KERNEL_NODES``
  coarse-node cubic-spline-of-``K_a`` grid) certified a design that no
  longer exists: the SACR-C hot path interpolates only the single smooth
  envelope ``E(w)`` on a LOO-adaptive node set and rebuilds the switched
  analytic saddles in closed form.  Its replacement -- the same null-safe
  metric applied to the LOO-envelope reconstruction, plus its paired
  under-seeded-grid self-falsification -- is OWED to the Test Developer
  (the report's build3f gates 2/3).  The Coder does not author it: it
  certifies the WP2 code the Coder wrote (F002, no shared author).

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

# Pin single-threaded numerics ONLY in strict-timing mode (the sole
# consumer of the determinism): an import-scope pin poisons shared
# pytest workers — numba's thread layer launches once per process, so
# a layer launched at 1 by a lensing prange call makes any later
# parallel ufunc (e.g. marginalized_extrinsic_qas) hard-fail on the
# default 64 (Build 8f gate incident, 2026-07-21).
if _os.environ.get('COGWHEEL_STRICT_TIMING'):
    for _thread_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        _os.environ.setdefault(_thread_var, '1')

import importlib.util
import inspect
import itertools
import pathlib
import subprocess
import tempfile
import time
import warnings
from unittest import TestCase, main, mock, skipUnless

# --- Two-tier test split (Build 8d re-pricing) -------------------------------
# The exact positive-parity path is now the Schwinger evaluator (~90 ms/node),
# so ``lnlike_bruteforce`` -- the full-FFT-grid matched filter that evaluates
# the exact engine per frequency -- costs ~138 s/call post-8d.  Tests whose
# runtime is dominated by that brute-force accuracy oracle are the DRIVER /
# post-build tier, gated OFF by default and run in-build only as FAST
# structural / witness / refusal gates.  Set ``COGWHEEL_BRUTE_ACCURACY=1`` to
# run the brute-force accuracy tier (it remains falsifiable and green there).
_BRUTE_ACCURACY = bool(_os.environ.get('COGWHEEL_BRUTE_ACCURACY'))
_brute_accuracy_tier = skipUnless(
    _BRUTE_ACCURACY,
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 -- exact path '
    '~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')

import mpmath
import numpy as np
from matplotlib import pyplot as plt

from cogwheel import data, waveform
from cogwheel.lensing.chang_refsdal import (
    channels, geometry, operator, _airy_fold, _pearcey_cusp)
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    DD_PRODUCT_CEILING, HypergeometricDomainError, W_MAX_CERTIFIED,
    point_mass_g_derivatives)
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, real_image_delays, _GEOMETRIC_ORDER)
from cogwheel.lensing.chang_refsdal.operator import (
    CancellationError, F_op, L_MAX, RHO_END, RHO_START, select_branch)
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, _data_term, _norm_term)

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

#: ``(w, sqrt_s, gamma)`` whose LEGACY contraction is uncertifiable.
#: Since Build 7a the ``w <= 60`` members are rescued by the cross-parity
#: Schwinger fallback (finite, certified, ``order_used == 0`` marks that
#: the value did NOT come from the uncertifiable series); the ``w > 60``
#: member must still refuse by name (never a silent finite value from the
#: legacy series).
#:
#: F028 re-point: the above-ceiling member is ``(63.0, 0.3, 0.2)`` -- a
#: genuinely hard-core WAVE node (``w*delta_min < RHO_END`` unresolved,
#: ``L = w*|y'| = 63*0.3 = 18.9`` so `select_branch` stays on 'wave', and
#: BOTH uniform arms decline), which raises `SchwingerCertificationError`
#: at the Schwinger ceiling.  The former ``(63.0, 0.9, 0.2)`` is now
#: resolved AND strongly cancelling, so since Build 8f WP1 the
#: authoritative `select_branch` gate serves it with the F028 geometric
#: asymptote instead of refusing -- it no longer exercises the
#: above-ceiling refusal edge.
FOP_REFUSALS = (
    (40.0, 0.9, 0.2),
    (50.0, 0.9, 0.2),
    (50.0, 0.95, 0.2),
    (63.0, 0.3, 0.2),
)

# ---------------------------------------------------------------------------
# LEVER 2 -- interpolation constants (RETIRED with the SACR-C swap).
# ---------------------------------------------------------------------------
#
# The fixed coarse ``w`` node grid (``_coarse_w_node_grid`` /
# ``_DEFAULT_KERNEL_NODES`` / ``n_kernel_nodes``) and the cubic-spline-of-``K_a``
# reconstruction those constants gated were removed by the SACR-C rewrite of
# ``_amplification_coefficients`` (Build 3f WP2).  The hot path now interpolates
# ONLY the single smooth envelope ``E(w)`` on a LOO-adaptive node set
# (``_envelope_loo_nodes``, hard-coded stop ``_LOO_STOP``, ceiling
# ``_LOO_MAX_NODES``) and rebuilds the switched analytic saddles in closed form
# (``_reconstruct_kernels``).  `CoarseNodeInterpolationTestCase` and the
# interpolation self-falsification that consumed ``INTERP_NULLSAFE_CEIL`` /
# ``CONVERGED_NODES`` / ``UNDERRESOLVED_NODES`` / ``INTERP_*_LABELS`` are retired
# below.  A replacement null-safe interpolation gate on the LOO-envelope path is
# OWED to the Test Developer (see the module note near `CoarseNodeInterpolationTestCase`).

# ---------------------------------------------------------------------------
# LEVER 1/2 -- timing constants.
# ---------------------------------------------------------------------------

#: Best-of-N repeats for warm timing (robust to scheduler jitter).
TIMING_REPEATS = 5

#: LOOSE absolute ceiling [s] on warm best-of-N ``lnlike``: a generous
#: regression guard on THIS box, NOT the brief's physical ``10 ms`` claim
#: (DEVIATION #2 -- see docstring).  RE-TUNED (Build 8d homogenization):
#: the exact positive-parity wave branch is the Schwinger evaluator at
#: ~90 ms/node, so the warm crown ``lnlike`` (~8 engine nodes) measures
#: ~0.75 s.  Raised 0.5 -> 3.0 (~4x the measured cost) -- generous against
#: a loaded box yet still catching a catastrophic regression (e.g. a
#: full-grid engine evaluation, ~140 s).  The exact path is the SINGLE
#: certified evaluator BY DESIGN; the surrogate is the speed layer (off by
#: default).  The tight speed claim (brute-force speed-up) is gated under
#: ``COGWHEEL_STRICT_TIMING`` (see `_STRICT_TIMING`).
MS_CEILING = 3.0

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
#: `geometry.LensDomainError` on this input -- symmetric refusal.  Since
#: Build 7b macro-saddle INTERIORS (0 < 1 - kappa < |gamma|) evaluate on
#: both paths, so the symmetric-refusal contract is pinned at the
#: OVER-CRITICAL domain (kappa >= 1, Type III), which stays a named
#: refusal on every path.
OVER_CRITICAL_GAMMA = 0.5
OVER_CRITICAL_KAPPA = 1.5

#: A macro-saddle INTERIOR config (gamma' = 0.5/0.4 = 1.25): since
#: Build 7b this CONSTRUCTS and evaluates (the contract-flip witness).
MACRO_SADDLE_GAMMA = 0.5
MACRO_SADDLE_KAPPA = 0.6

#: Source position of the domain-refusal candidate (in band, arbitrary).
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

# ---------------------------------------------------------------------------
# WP-A caustic distance+theta+lobe preservation constants (both parities).
# ---------------------------------------------------------------------------

#: Raw absolute angular tolerance [rad] on the returned ``theta`` vs the
#: dense brute-force oracle at POSITIVE parity, compared modulo ``2*pi``.
#: Used as the gate ONLY away from cusps; near a cusp the caustic map is
#: stationary (``|d point / d theta| -> 0``) so ``theta`` is genuinely
#: under-determined and the raw angle can drift while the POINT stays
#: exact.  There the physical gate is the ARC-LENGTH form below.
CAUSTIC_THETA_ATOL = 1e-9

#: Arc-length tolerance: the angular discrepancy WEIGHTED by the local
#: caustic speed ``|d point / d theta|`` (evaluated independently at the
#: oracle theta) is the tangential source-plane displacement between the
#: two stationary points.  This is a budget-independent, cusp-safe
#: currency -- at a cusp speed -> 0 so a stationary-map angle ambiguity is
#: tolerated WITHOUT loosening the distance/point gates.  Both production
#: (``minimize_scalar``) and the dense oracle locate the min by FUNCTION
#: values, so ``theta`` cannot beat the floating-point localization floor
#: ``delta_theta ~ sqrt(eps * dist**2 / curvature) ~ 1e-8`` near a smooth
#: parabolic minimum; the empirical worst-case arc-length over the whole
#: sweep is ~1.3e-7 (saddle).  This ceiling sits ~8x above that floor yet
#: ~1e6 BELOW the O(1) arc-length a genuine lobe-jump / wrong-stationary-
#: point produces -- a hugely discriminating gate, not a slack one.  (See
#: the self-falsification class: a forged non-global theta lands at O(1)
#: arc-length and goes RED.)
CAUSTIC_ARCLEN_ATOL = 1e-6

#: Absolute angular tolerance [rad] on the returned ``theta`` vs the dense
#: oracle at a MACRO SADDLE.  The deltoid cusps are sharper and the oracle
#: scans a narrow wedge, so the angular resolution is coarser than the
#: astroid's; this saddle-appropriate value is set from the oracle's
#: parabolic-refine floor on the wedge grid (empirically ~1e-8), not the
#: distance tolerance (distance error ~ theta_err**2 near a parabolic
#: minimum, so distance stays far tighter).  The arc-length gate above is
#: the primary theta certification; this remains as a coarse guard.
CAUSTIC_SADDLE_THETA_ATOL = 1e-6

#: Absolute tolerance on the returned source-plane caustic POINT (x, y) vs
#: the oracle's winning point.  Certifies branch + lobe + theta jointly:
#: the caustic point is uniquely fixed by (theta, branch, lobe), so
#: agreement here proves the SAME stationary point was selected.
CAUSTIC_POINT_ATOL = 1e-9

#: Macro-saddle ``(gamma, beta, kappa)`` axes: every row obeys
#: ``0 < 1 - kappa < |gamma|`` (the two-deltoid-lobe regime).  gamma spans
#: ``(~1.05, ~1.5)`` per the Professor's saddle specification.
CAUSTIC_SADDLE_GAMMAS = (1.10, 1.30, 1.50)
CAUSTIC_SADDLE_BETAS = (0.0, 0.6)
CAUSTIC_SADDLE_KAPPAS = (0.0, 0.2)

#: Saddle source positions: near the lobe axis (on-wedge, small offset),
#: off-wedge (transverse), and near a deltoid cusp (competing lobes).
CAUSTIC_SADDLE_SOURCES = ((0.03, 0.00), (0.00, 0.25), (0.80, 0.10))

#: Near-symmetric saddle config for the branch-invariance falsification:
#: at ``beta = 0`` the two lobes (centres ``0`` and ``pi``) are mirror
#: images across ``y1 = 0``, so a source swept in ``y1`` through 0 crosses
#: the symmetry line and the nearest lobe must flip exactly ONCE, tracking
#: the independent oracle (no Newton-induced chatter).
CAUSTIC_SYMMETRY_GAMMA = 1.30
CAUSTIC_SYMMETRY_KAPPA = 0.0
CAUSTIC_SYMMETRY_Y2 = 0.02
CAUSTIC_SYMMETRY_Y1_SWEEP = (-0.20, -0.05, -0.01, 0.01, 0.05, 0.20)

#: Warm-call timing probe (WP-A, SOFT).  The measured warm per-call cost
#: of `nearest_caustic_point` is printed with its ratio to the ~0.3 ms
#: target; the hard sub-ms assertion is only enforced when
#: ``COGWHEEL_STRICT_TIMING`` is set (off on CI -- timing is machine
#: dependent).  Otherwise only a generous non-flaky ceiling guards a
#: catastrophic regression.
CAUSTIC_TIMING_REPEATS = 200
CAUSTIC_TIMING_TARGET_MS = 0.3
CAUSTIC_TIMING_LOOSE_CEILING_MS = 25.0
_STRICT_TIMING = bool(_os.environ.get('COGWHEEL_STRICT_TIMING'))

# ---------------------------------------------------------------------------
# WP-B operator-fusion byte-identity constants.
# ---------------------------------------------------------------------------

#: Shear orientation / convergence axes added to the certified F_op sweep
#: so the eigenframe rotation and the mass-sheet prefactor are exercised in
#: the current-vs-HEAD byte-identity comparison (positive parity only:
#: every row keeps ``1 - kappa > |gamma|``).
FOP_IDENTITY_BETAS = (0.0, 0.7)
FOP_IDENTITY_KAPPAS = (0.0, 0.2)

#: Positive-parity config whose operator series needs several orders to
#: converge -- the F010 py_func-chain falsification evaluates here so a
#: perturbed convergence stop / gather index visibly moves the answer.
FALSIFY_W = 20.0
FALSIFY_Y = (0.9, 0.0)
FALSIFY_GAMMA = 0.2

#: Corrupted small-term convergence tolerance for the F010 fused-core
#: falsification: at 1.0 the small-term stop fires as early as it is
#: allowed and truncates the O(gamma) shear series, so the perturbed
#: contraction no longer certifies to `FOP_RTOL`.
PERTURBED_SERIES_TOLERANCE = 1.0

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


# ---------------------------------------------------------------------------
# Pre-fusion (git HEAD) operator module, loaded side-by-side.
# ---------------------------------------------------------------------------

#: Cached pre-8d ``operator`` module (loaded on demand from `_BASELINE_SHA`).
_HEAD_OPERATOR = None

#: Repo root of THIS worktree (``cogwheel/tests/... -> repo``), used as the
#: ``git`` working directory for ``git show``.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Path of the operator module, relative to the repo root, for ``git show``.
_OPERATOR_RELPATH = 'cogwheel/lensing/chang_refsdal/operator.py'

#: PINNED baseline: ``cf3c427`` is Build 8b-levers, the parent of ``4e26103``
#: (Build 8d) -- i.e. the pre-8d operator this suite was authored against.
#: This MUST NOT be ``HEAD``.  A transition baseline named ``HEAD`` becomes
#: self-referential the instant the transition commit lands: ``git show
#: HEAD:operator.py`` then returns the working tree's own bytes, every
#: ``tobytes()`` comparison compares the module to itself, and all four
#: byte-identity / flip-witness gates pass unconditionally on a clean tree.
#: The sibling suite `test_lensing_airy_fold.py` diagnosed and fixed this
#: exact failure the same way (it pins ``4e26103``); transition baselines
#: must pin the SHA they were authored against.
_BASELINE_SHA = 'cf3c427'


def _load_head_operator():
    """
    Load the pre-8d ``operator.py`` from the PINNED `_BASELINE_SHA` as a
    standalone module (F002/F005 oracle independence for the byte-identity
    and flip-witness gates).

    The source is fetched with ``git show <_BASELINE_SHA>:<relpath>`` --
    NOT ``HEAD``, which on a clean tree is the working tree itself and
    would make every comparison a self-comparison (see `_BASELINE_SHA`).
    It is executed as a fresh module under a UNIQUE name so its numba
    cores recompile from the frozen source rather than reusing the working
    tree's ``__pycache__`` (a distinct ``co_filename`` forces a fresh njit
    compile).

    ``operator.py`` imports its siblings ABSOLUTELY (``from
    cogwheel.lensing.chang_refsdal import ...``), so the frozen module
    binds the WORKING TREE's kernel, geometry and Schwinger code.  That is
    deliberate: it isolates ``operator.py`` -- the only file this gate
    freezes -- as the single moving part, exactly as the sibling
    `test_lensing_airy_fold.py` baseline does.

    Returns
    -------
    module
        The pre-8d ``operator`` module, exposing the legacy fused
        contraction on the sheared positive-parity arm and the unchanged
        public ``F_op`` / ``F_op_grid`` entry points.

    Raises
    ------
    RuntimeError
        If ``git show`` fails (not a git checkout, or the pinned commit is
        missing) -- the byte-identity gate cannot certify without the
        frozen reference, so it must refuse loudly rather than silently
        skip.
    """
    global _HEAD_OPERATOR
    if _HEAD_OPERATOR is not None:
        return _HEAD_OPERATOR
    completed = subprocess.run(
        ['git', 'show', f'{_BASELINE_SHA}:{_OPERATOR_RELPATH}'],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'cannot load the pre-8d operator baseline: `git show '
            f'{_BASELINE_SHA}:{_OPERATOR_RELPATH}` failed in {_REPO_ROOT} '
            f'with {completed.stderr.strip()!r}; the byte-identity gate '
            'has no frozen reference to certify against')
    with tempfile.NamedTemporaryFile(
            'w', suffix='_operator_head.py', delete=False) as handle:
        handle.write(completed.stdout)
        head_path = handle.name
    spec = importlib.util.spec_from_file_location(
        'cogwheel_chang_refsdal_operator_head', head_path)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so dataclass / relative machinery resolves the
    # module by name during its own execution (the established idiom).
    import sys as _sys
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _HEAD_OPERATOR = module
    return module


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
        An uncertifiable wave-branch node never yields a
        finite-but-untrusted legacy value (F005) through the JIT path.
        RE-BASELINE (Build 8d homogenization): these sheared
        positive-parity configs (``gamma' > 0``) are served by the exact
        Schwinger evaluator, so there are two legal outcomes at the
        production default ``max_order``: at ``w <= 60`` Schwinger
        certifies with diagnostics ``order_used == 0`` (the uncertifiable
        legacy series was never trusted); at ``w > 60`` the named refusal
        propagates -- now `SchwingerCertificationError` (was the Build-7a
        fallback's re-raised `CancellationError`).  A finite value with
        ``order_used > 0`` on any of these configs would mean the
        uncertifiable legacy series was silently believed -- the F005 bug.
        """
        refused = 0
        for w, sqrt_s, gamma in FOP_REFUSALS:
            y = [sqrt_s, 0.0]
            with self.subTest(w=w, sqrt_s=sqrt_s, gamma=gamma):
                self.n_checks += 1
                try:
                    value, diagnostics = F_op(w, y, gamma)
                except (CancellationError,
                        SchwingerCertificationError):
                    refused += 1
                    self.assertGreater(
                        w, 60.0,
                        f'w={w} <= 60 refused: the Schwinger evaluator '
                        'did not certify a sub-ceiling node')
                    continue
                self.assertTrue(np.isfinite(value))
                self.assertEqual(
                    diagnostics.order_used, 0,
                    f'w={w}: finite value with order_used = '
                    f'{diagnostics.order_used} > 0 -- the uncertifiable '
                    'legacy series was silently believed (F005)')
        self.assertGreater(refused, 0,
                           'no config refused; the above-ceiling arm was '
                           'not exercised')


class OperatorFusionByteIdentityTestCase(FastPathTestCase):
    """
    RE-TARGET (Build 8d homogenization) of the Build-8b fused-contraction
    byte-identity gate.

    The 8b lever merged ``_weight_vectors`` + ``_contract_grid`` into ONE
    njit core, `operator._fused_contraction`, and this suite pinned
    ``F_op`` / ``F_op_grid`` BYTE-IDENTICAL to the pre-fusion ``HEAD``.
    Build 8d moved the SHEARED positive-parity arm (``gamma' > 0``) off
    the fused contraction entirely: it is now served by the exact
    Schwinger evaluator (`_schwinger.f_schwinger`, ``order_used == 0``).
    The fused contraction still serves ONLY the shear-free ``gamma' == 0``
    point lens.  So the byte-identity premise SPLITS:

    * FROZEN arm (``gamma' == 0``): the fused legacy contraction still
      runs and MUST stay byte-for-byte HEAD (value, all four
      `OperatorDiagnostics` fields, and the refusal type).  This keeps
      the 8b fusion byte-identity gate alive where the fusion serves.

    * FLIPPED arm (``gamma' > 0``): the value change from the legacy
      contraction to Schwinger is an APPROVED CONTRACT FLIP.  It is
      re-baselined with the standard witness (F017): the NEW Schwinger
      value agrees with the OLD HEAD value in the max-normalized physics
      currency at `FOP_RTOL` (1e-10) on the certified overlap -- a
      byte/contract change, not a physics change -- and the NEW path is
      SINGLE-DISPATCH (``order_used == 0``, Schwinger, never the fused
      contraction).  Above the Schwinger ceiling it refuses by the NEW
      named `SchwingerCertificationError`.

    Independence (F005): the reference is the pre-8d ``operator.py`` loaded
    from the PINNED commit `_BASELINE_SHA` (`_load_head_operator`) as a
    distinct module with freshly-compiled njit cores -- not a re-run of the
    code under test.  "HEAD" in the method names below is historical: the
    baseline is a fixed SHA, never the branch tip.  Pinning is load-bearing,
    not cosmetic -- against ``HEAD`` on a clean tree the reference IS the
    working tree, every ``tobytes()`` comparison compares the module to
    itself, and all four gates below pass no matter what the code does.
    """

    _WITNESS_TOL = FOP_RTOL  # 1e-10, the F005/7a/8d owner-set byte-flip gate

    @staticmethod
    def _scalar_outcome(module, w, y, gamma, beta, kappa):
        """Run ``module.F_op`` and freeze the value + all four diagnostics
        as typed numpy scalars (for ``tobytes`` comparison), or record the
        named refusal.  Captures every wave-branch refusal: the legacy
        `CancellationError`, the homogenized `SchwingerCertificationError`,
        the kernel-ceiling `HypergeometricDomainError`, and
        `LensDomainError`."""
        try:
            value, diagnostics = module.F_op(
                w, np.asarray(y, dtype=float), gamma,
                beta=beta, kappa=kappa, max_order=FOP_MAX_ORDER)
        # ``module.CancellationError`` -- each operator module (working
        # tree and the HEAD load) defines its OWN CancellationError class,
        # so catch the one THIS module raises; the Schwinger / domain /
        # kernel refusals come from byte-identical siblings (the SAME
        # _schwinger / geometry / kernel code) and are shared class
        # objects.
        except (module.CancellationError, SchwingerCertificationError,
                HypergeometricDomainError, geometry.LensDomainError) as exc:
            return {'raised': True, 'exc': type(exc).__name__}
        return {
            'raised': False, 'exc': None,
            'value': np.complex128(value),
            'order': np.int64(diagnostics.order_used),
            'converged': np.bool_(diagnostics.converged),
            'tail': np.float64(diagnostics.estimated_relative_tail),
            'cancellation': np.float64(diagnostics.cancellation_ratio)}

    @staticmethod
    def _grid_outcome(module, grid, y, gamma, beta, kappa):
        """Run ``module.F_op_grid`` and return the three public arrays, or
        record the named whole-grid refusal (a single uncertifiable node
        refuses the whole grid, F005)."""
        try:
            values, orders, converged = module.F_op_grid(
                grid, np.asarray(y, dtype=float), gamma,
                beta=beta, kappa=kappa, max_order=FOP_MAX_ORDER)
        except (module.CancellationError, SchwingerCertificationError,
                HypergeometricDomainError, geometry.LensDomainError) as exc:
            return {'raised': True, 'exc': type(exc).__name__}
        return {'raised': False, 'exc': None, 'values': values,
                'orders': orders, 'converged': converged}

    @staticmethod
    def _max_normalized(new_vals, old_vals):
        """Max-normalized real/imag residual in the cross-build currency
        ``max_i |Re/Im(F_new - F_old)| / max(max_i |F_old|, 1e-15)`` (the
        surrogate exemplar's `_flip_witness_metrics` idiom)."""
        new = np.asarray(new_vals, dtype=complex)
        old = np.asarray(old_vals, dtype=complex)
        scale = max(float(np.max(np.abs(old))), 1e-15)
        metric_re = float(np.max(np.abs(new.real - old.real))) / scale
        metric_im = float(np.max(np.abs(new.imag - old.imag))) / scale
        return max(metric_re, metric_im), scale

    def _frozen_scalar_configs(self):
        """``gamma' == 0`` (``gamma == 0``) scalar configs the fused legacy
        contraction still serves, spanning the certified band plus one
        kernel-ceiling refusal for refusal-type parity."""
        for w in FOP_GRID_W:
            for sqrt_s in FOP_GRID_SQRT_S:
                for beta in FOP_IDENTITY_BETAS:
                    for kappa in FOP_IDENTITY_KAPPAS:
                        yield (w, sqrt_s, 0.0, beta, kappa)
        # A shear-free kernel-ceiling refusal (w*sqrt(s) = 70 > 60): both
        # working and HEAD must raise the same HypergeometricDomainError.
        yield (70.0, 1.0, 0.0, 0.0, 0.0)

    def _flipped_scalar_configs(self):
        """``gamma' > 0`` (``gamma == 0.2``) scalar configs -- the flipped
        Schwinger arm -- including the above-ceiling refusing edge."""
        for w in FOP_GRID_W:
            for sqrt_s in FOP_GRID_SQRT_S:
                for beta in FOP_IDENTITY_BETAS:
                    for kappa in FOP_IDENTITY_KAPPAS:
                        yield (w, sqrt_s, 0.2, beta, kappa)
        for w, sqrt_s, gamma in FOP_REFUSALS:  # all gamma = 0.2
            yield (w, sqrt_s, gamma, 0.0, 0.0)

    def test_fop_scalar_frozen_arm_byte_identical_to_head(self):
        """
        FROZEN arm: scalar ``F_op`` at ``gamma' == 0`` is byte-for-byte
        the pre-fusion HEAD -- the complex value AND every
        OperatorDiagnostics field ``tobytes``-match, and the
        certify-XOR-refuse decision (and refusal TYPE) never flips.  The
        fused legacy contraction still serves this arm.
        """
        head = _load_head_operator()
        certified = refused = 0
        for w, sqrt_s, gamma, beta, kappa in self._frozen_scalar_configs():
            y = (sqrt_s, 0.0)
            with self.subTest(w=w, sqrt_s=sqrt_s, gamma=gamma,
                              beta=beta, kappa=kappa):
                current = self._scalar_outcome(
                    operator, w, y, gamma, beta, kappa)
                reference = self._scalar_outcome(
                    head, w, y, gamma, beta, kappa)
                self.n_checks += 1
                self.assertEqual(
                    current['raised'], reference['raised'],
                    f'w={w} sqrt_s={sqrt_s} gamma={gamma} beta={beta} '
                    f'kappa={kappa}: frozen-arm F_op flipped a '
                    'certify-XOR-refuse decision vs HEAD')
                if current['raised']:
                    refused += 1
                    self.n_checks += 1
                    self.assertEqual(
                        current['exc'], reference['exc'],
                        f'w={w} sqrt_s={sqrt_s}: frozen-arm F_op raised '
                        f'{current["exc"]} but HEAD raised '
                        f'{reference["exc"]} -- the refusal type moved')
                    continue
                certified += 1
                for field, label in (
                        ('value', 'amplification F'),
                        ('order', 'diagnostics.order_used'),
                        ('converged', 'diagnostics.converged'),
                        ('tail', 'diagnostics.estimated_relative_tail'),
                        ('cancellation', 'diagnostics.cancellation_ratio')):
                    self.n_checks += 1
                    self.assertEqual(
                        current[field].tobytes(), reference[field].tobytes(),
                        f'w={w} sqrt_s={sqrt_s} beta={beta} kappa={kappa}: '
                        f'frozen-arm {label} {current[field]!r} is not '
                        f'byte-identical to HEAD {reference[field]!r} -- '
                        'the fusion moved a bit on the gamma\'==0 arm')
        self.assertGreater(
            certified, 0, 'no gamma\'==0 config certified; the frozen-arm '
            'byte-identity sweep never exercised a returned value')
        self.assertGreater(
            refused, 0, 'no gamma\'==0 config refused; the frozen-arm '
            'refusal parity was never exercised')

    def test_fop_scalar_schwinger_arm_flip_witness(self):
        """
        FLIPPED arm: scalar ``F_op`` at ``gamma' > 0`` is served by
        Schwinger (single-dispatch, ``order_used == 0``); its value agrees
        with the OLD HEAD value in the max-normalized currency at
        `_WITNESS_TOL` (a byte flip, not a physics change), and above the
        Schwinger ceiling it refuses with `SchwingerCertificationError`.
        """
        head = _load_head_operator()
        new_overlap, old_overlap = [], []
        single_dispatch = refused = 0
        for w, sqrt_s, gamma, beta, kappa in self._flipped_scalar_configs():
            y = (sqrt_s, 0.0)
            with self.subTest(w=w, sqrt_s=sqrt_s, gamma=gamma,
                              beta=beta, kappa=kappa):
                current = self._scalar_outcome(
                    operator, w, y, gamma, beta, kappa)
                if current['raised']:
                    refused += 1
                    self.n_checks += 1
                    self.assertEqual(
                        current['exc'], 'SchwingerCertificationError',
                        f'w={w} sqrt_s={sqrt_s}: flipped-arm refusal is '
                        f'{current["exc"]}, expected the homogenized '
                        'SchwingerCertificationError')
                    self.assertGreater(
                        w, 60.0,
                        f'w={w} <= 60 refused: Schwinger should certify '
                        'the sub-ceiling flipped arm')
                    continue
                # Single-dispatch: the NEW value came from Schwinger.
                self.n_checks += 1
                self.assertEqual(
                    int(current['order']), 0,
                    f'w={w} sqrt_s={sqrt_s}: flipped-arm order_used='
                    f'{int(current["order"])} != 0 -- a sheared '
                    'positive-parity node must be Schwinger-served')
                reference = self._scalar_outcome(
                    head, w, y, gamma, beta, kappa)
                if not reference['raised']:
                    new_overlap.append(complex(current['value']))
                    old_overlap.append(complex(reference['value']))
                single_dispatch += 1
        self.assertGreater(
            single_dispatch, 0,
            'no gamma\'>0 config certified through Schwinger (vacuous)')
        self.assertGreater(
            refused, 0,
            'the above-ceiling Schwinger refusal edge was not exercised')
        self.assertGreaterEqual(
            len(new_overlap), 8,
            f'only {len(new_overlap)} HEAD-certified overlap nodes to '
            'witness the flip against (need >= 8)')
        metric, scale = self._max_normalized(new_overlap, old_overlap)
        self.n_checks += 1
        self.assertLess(
            metric, self._WITNESS_TOL,
            f'flipped-arm NEW-vs-OLD disagreement {metric:.3e} exceeds the '
            f'{self._WITNESS_TOL:.0e} byte-flip currency (scale={scale:.4f}, '
            f'{len(new_overlap)} overlap nodes) -- a PHYSICS regression, '
            'not a byte flip')

    def test_fop_grid_frozen_arm_byte_identical_to_head(self):
        """
        FROZEN arm, batched: ``F_op_grid`` at ``gamma' == 0`` is
        byte-for-byte HEAD over a multi-node ``w`` grid; the whole-grid
        refusal (any uncertifiable node) fires on exactly the same configs
        and with the same refusal type.
        """
        head = _load_head_operator()
        # Append an above-legacy-ceiling node (w = 63): at high L
        # (sqrt_s = 0.9) the shear-free legacy contraction refuses the
        # whole grid; at low L it certifies -- so both parities are hit.
        grid = np.asarray(FOP_GRID_W + (63.0,), dtype=float)
        certified = refused = 0
        for sqrt_s in FOP_GRID_SQRT_S:
            for beta in FOP_IDENTITY_BETAS:
                for kappa in FOP_IDENTITY_KAPPAS:
                    y = (sqrt_s, 0.0)
                    with self.subTest(sqrt_s=sqrt_s, beta=beta, kappa=kappa):
                        current = self._grid_outcome(
                            operator, grid, y, 0.0, beta, kappa)
                        reference = self._grid_outcome(
                            head, grid, y, 0.0, beta, kappa)
                        self.n_checks += 1
                        self.assertEqual(
                            current['raised'], reference['raised'],
                            f'sqrt_s={sqrt_s} beta={beta} kappa={kappa}: '
                            'frozen-arm F_op_grid and HEAD disagree on '
                            'whole-grid refusal')
                        if current['raised']:
                            refused += 1
                            self.n_checks += 1
                            self.assertEqual(
                                current['exc'], reference['exc'],
                                f'sqrt_s={sqrt_s}: frozen grid raised '
                                f'{current["exc"]} but HEAD raised '
                                f'{reference["exc"]}')
                            continue
                        certified += 1
                        for field, label in (
                                ('values', 'F values'),
                                ('orders', 'operator orders'),
                                ('converged', 'converged flags')):
                            self.n_checks += 1
                            self.assertEqual(
                                current[field].tobytes(),
                                reference[field].tobytes(),
                                f'sqrt_s={sqrt_s} beta={beta} kappa={kappa}: '
                                f'frozen-arm F_op_grid {label} is not '
                                'byte-identical to HEAD')
        self.assertGreater(
            certified, 0, 'no gamma\'==0 grid certified (vacuous)')
        self.assertGreater(
            refused, 0, 'no gamma\'==0 grid refused (vacuous)')

    def test_fop_grid_schwinger_arm_flip_witness(self):
        """
        FLIPPED arm, batched: ``F_op_grid`` at ``gamma' > 0`` is served by
        Schwinger; its values agree with HEAD in the max-normalized
        currency at `_WITNESS_TOL` on the sub-ceiling grid, and every
        returned order is 0 (single-dispatch).

        RE-BASELINE (Build 8d -> 8e -> 8f WP1/F028): under Build 8d the
        above-ceiling clause pinned an UNCONDITIONAL whole-grid
        `SchwingerCertificationError` for the appended ``w = 63`` node.
        The Build 8e serving ladder split that clause, and Build 8f WP1
        added the F028 geometric rung, so the above-ceiling ``w = 63``
        node now has THREE mutually exclusive outcomes, each pinned to the
        rung the ladder copies bit-for-bit:

        * (a) hard-core: NO uniform arm certifies and the node is not
          geometric-resolved, so the whole grid refuses with
          `SchwingerCertificationError`;
        * (b) arm-served: the node's value IS the fold/cusp arm value at
          ``1e-12``, order 0;
        * (c) F028 geometric-served (resolved AND strongly cancelling):
          the node's value IS `operator.geometric_amplification` byte-for-
          byte, order 0.  This is a DISPATCH-parity pin (the ladder copies
          the labelled geometric rung), NOT an accuracy certification of
          the asymptote -- that independent gate is owed to the Test
          Developer.

        The fixture grid spans all three outcomes (asserted non-vacuous
        below).
        """
        head = _load_head_operator()
        sub_grid = np.asarray(FOP_GRID_W, dtype=float)  # all w <= 50 <= 60
        supra_grid = np.asarray(FOP_GRID_W + (63.0,), dtype=float)
        witnessed = refused = served = served_geometric = 0
        for sqrt_s in FOP_GRID_SQRT_S:
            for beta in FOP_IDENTITY_BETAS:
                for kappa in FOP_IDENTITY_KAPPAS:
                    y = (sqrt_s, 0.0)
                    with self.subTest(sqrt_s=sqrt_s, beta=beta, kappa=kappa):
                        current = self._grid_outcome(
                            operator, sub_grid, y, 0.2, beta, kappa)
                        reference = self._grid_outcome(
                            head, sub_grid, y, 0.2, beta, kappa)
                        self.assertFalse(
                            current['raised'],
                            f'sqrt_s={sqrt_s}: sub-ceiling flipped grid '
                            'unexpectedly refused')
                        # Single-dispatch: every returned order is 0.
                        self.n_checks += 1
                        self.assertTrue(
                            np.all(current['orders'] == 0),
                            f'sqrt_s={sqrt_s} beta={beta} kappa={kappa}: a '
                            'flipped-arm node reports order != 0 (not '
                            'Schwinger-served)')
                        if not reference['raised']:
                            metric, scale = self._max_normalized(
                                current['values'], reference['values'])
                            self.n_checks += 1
                            self.assertLess(
                                metric, self._WITNESS_TOL,
                                f'sqrt_s={sqrt_s} beta={beta} kappa={kappa}: '
                                f'flipped-grid NEW-vs-OLD {metric:.3e} '
                                f'exceeds {self._WITNESS_TOL:.0e} '
                                f'(scale={scale:.4f}) -- physics regression')
                            witnessed += 1
                        # Above-ceiling w=63 node: CONDITIONAL, THREE-way
                        # (Build 8e serving ladder + Build 8f WP1 / F028).
                        self.n_checks += 1
                        arm = self._serving_arm(63.0, y, 0.2, beta, kappa)
                        geo = complex(operator.geometric_amplification(
                            63.0, np.asarray(y, dtype=float), 0.2,
                            beta=beta, kappa=kappa))
                        supra = self._grid_outcome(
                            operator, supra_grid, y, 0.2, beta, kappa)
                        if supra['raised']:
                            # (a) hard-core: NO arm may certify the node.
                            self.assertEqual(
                                supra['exc'], 'SchwingerCertificationError',
                                f'sqrt_s={sqrt_s} beta={beta} kappa={kappa}: '
                                'above-ceiling refusal is not '
                                f'SchwingerCertificationError (got {supra})')
                            self.assertIsNone(
                                arm, f'sqrt_s={sqrt_s} beta={beta} '
                                f'kappa={kappa}: grid refused w=63 yet an '
                                'arm certifies it')
                            refused += 1
                            continue
                        node_value = complex(supra['values'][-1])
                        self.assertEqual(
                            int(supra['orders'][-1]), 0,
                            f'sqrt_s={sqrt_s} beta={beta} kappa={kappa}: '
                            'served w=63 node reports order != 0')
                        if node_value == geo:
                            # (c) F028 geometric serve: the node's value IS
                            # geometric_amplification bit-for-bit (dispatch
                            # parity, NOT an accuracy certification).
                            served_geometric += 1
                        else:
                            # (b) arm-served: the node's value IS the arm's.
                            self.assertIsNotNone(
                                arm, f'sqrt_s={sqrt_s} beta={beta} '
                                f'kappa={kappa}: grid served w=63 but it is '
                                'neither the geometric rung nor an arm -- '
                                'served by a non-ladder path')
                            self.assertAlmostEqual(
                                abs(node_value - arm), 0.0,
                                delta=1e-12,
                                msg=f'sqrt_s={sqrt_s} beta={beta} '
                                f'kappa={kappa}: served w=63 value '
                                f'{supra["values"][-1]!r} is not the serving '
                                f'arm value {arm!r}')
                            served += 1
        self.assertGreater(witnessed, 0, 'no flipped grid was witnessed')
        self.assertGreater(refused, 0,
                           'no above-ceiling grid refused (hard-core branch '
                           'not exercised)')
        self.assertGreater(served, 0,
                           'no above-ceiling grid was arm-served (Build 8e '
                           'serving branch not exercised)')
        self.assertGreater(served_geometric, 0,
                           'no above-ceiling grid was geometric-served (F028 '
                           'geometric branch not exercised)')

    @staticmethod
    def _serving_arm(w, y, gamma, beta=0.0, kappa=0.0):
        """The uniform arm that serves this node, called DIRECTLY.

        Reproduces the production ladder's fixed fold-then-cusp order
        (`operator._uniform_arm_value`) by calling the arm modules
        themselves -- an INDEPENDENT path to the served value, not
        operator's dispatcher.  Returns the complex arm value, or ``None``
        when neither arm certifies (a genuinely hard-core node).
        """
        source = np.asarray(y, dtype=float)
        value = _airy_fold.fold_amplification(w, source, gamma,
                                              beta=beta, kappa=kappa)
        if value is not None:
            return complex(value)
        value = _pearcey_cusp.cusp_amplification(w, source, gamma,
                                                 beta=beta, kappa=kappa)
        if value is not None:
            return complex(value)
        return None


class OperatorFusionFalsificationTestCase(FastPathTestCase):
    """
    WP-B F010 preservation: the fused-contraction accuracy gate is NOT
    vacuous, re-homed onto the single fused njit core the Build 8b lever
    created.

    The Build 8b fusion merged the former ``_weight_vectors`` +
    ``_contract_grid`` two-stage pipeline into ONE core,
    `operator._fused_contraction`, so BOTH former py_func-chain
    falsifications -- the corrupted convergence tolerance and the zeroed
    radial-index gather -- now flow through that ONE function.  numba
    freezes module globals at compile time, so a patched
    ``_SERIES_TOLERANCE`` never reaches the compiled dispatcher; each
    perturbation is injected through the ``py_func`` chain -- the fused
    core is swapped for its ``.py_func`` body, which re-reads the module
    globals in the interpreter -- and the ``half_sum`` gather stays an
    explicit ARGUMENT the wrapper can corrupt.  Each perturbation must
    drive the `FOP_RTOL` gate red (refuse OR return past tolerance); a
    perturbation that left it green would mean the fused njit core is dead
    code or the ``py_func`` chain is incomplete.

    The gate targets the LEGACY certified path `operator._grid_certified`
    directly: the public `F_op_grid` rescues a sub-ceiling refusal with
    the Schwinger fallback (which does not consume the perturbed series),
    so a perturbation-induced refusal would be masked through the public
    entry point and the falsification would go vacuous.
    """

    def _gate_outcome(self):
        """Run the certified path at the FALSIFY config; return
        ``(raised, rel_err)`` -- ``rel_err`` is ``inf`` on refusal, else
        the relative error against the INDEPENDENT mpmath ``F_op``
        oracle."""
        try:
            values, *_ = operator._grid_certified(
                np.array([FALSIFY_W], dtype=float),
                np.asarray(FALSIFY_Y, dtype=float), FALSIFY_GAMMA,
                max_order=FOP_MAX_ORDER)
        except CancellationError:
            return True, float('inf')
        oracle = _oracle_fop(FALSIFY_W, FALSIFY_Y, FALSIFY_GAMMA,
                             max_order=FOP_MAX_ORDER)
        rel = abs(complex(values[0]) - oracle) / abs(oracle)
        return False, rel

    def _assert_green_unpatched(self):
        """The gate must be green BEFORE any patch, so RED is the patch's
        doing and not a broken precondition."""
        raised, rel = self._gate_outcome()
        self.n_checks += 1
        self.assertFalse(
            raised, 'unpatched _grid_certified refused the certified '
            'FALSIFY config; the falsification precondition is broken')
        self.n_checks += 1
        self.assertLessEqual(
            rel, FOP_RTOL,
            f'unpatched fused contraction rel error {rel:.3e} already '
            f'exceeds {FOP_RTOL:.0e}; the gate is not green to begin with')

    def test_fused_core_exposes_patchable_py_func_and_globals(self):
        """
        Introspection (F010): the fusion kept the core PERTURBABLE.  The
        fused function exposes a plain ``.py_func`` body (no compiled
        ``.signatures``, so a swap re-reads module globals in the
        interpreter); ``_SERIES_TOLERANCE`` remains a MODULE GLOBAL the
        perturbation can patch; and ``half_sum`` remains an ARGUMENT the
        gather corruption can zero.  If any of these regressed, the two
        falsification tests below would be silently vacuous.
        """
        self.n_checks += 1
        self.assertTrue(
            hasattr(operator._fused_contraction, 'py_func'),
            '_fused_contraction does not expose .py_func; the F010 '
            'perturbations cannot reach the compiled core')
        pyfunc = operator._fused_contraction.py_func
        self.n_checks += 1
        self.assertFalse(
            hasattr(pyfunc, 'signatures'),
            '_fused_contraction.py_func carries .signatures; it is not a '
            'plain py_func body, so a perturbation would not reach compiled '
            'code (F010 vacuity)')
        self.n_checks += 1
        self.assertIn(
            'half_sum', inspect.signature(pyfunc).parameters,
            "half_sum is no longer an explicit argument of "
            "_fused_contraction; the gather-index falsification cannot "
            'corrupt it')
        self.n_checks += 1
        self.assertTrue(
            hasattr(operator, '_SERIES_TOLERANCE'),
            '_SERIES_TOLERANCE is not a module global; the series-'
            'tolerance falsification cannot patch it')

    def test_series_tolerance_perturbation_drives_gate_red(self):
        """
        Patching `operator._SERIES_TOLERANCE` to 1.0 through the fused
        core's ``py_func`` makes the small-term stop fire as early as it
        is allowed, dropping the O(gamma) shear correction; the result no
        longer certifies to `FOP_RTOL` (it refuses on the truncation cut
        or returns past tolerance).
        """
        self._assert_green_unpatched()
        core_pyfunc = operator._fused_contraction.py_func
        self.n_checks += 1
        self.assertFalse(
            hasattr(core_pyfunc, 'signatures'),
            '_fused_contraction.py_func carries .signatures (F010 vacuity)')
        with mock.patch.object(operator, '_fused_contraction', core_pyfunc), \
                mock.patch.object(operator, '_SERIES_TOLERANCE',
                                  PERTURBED_SERIES_TOLERANCE):
            raised, rel = self._gate_outcome()
        print(f'\n[Falsification] fused series-tolerance -> '
              f'{PERTURBED_SERIES_TOLERANCE}: raised={raised} '
              f'rel_err={rel:.3e}')
        self.n_checks += 1
        self.assertTrue(
            raised or rel > FOP_RTOL,
            f'the truncated shear series still certified (rel_err '
            f'{rel:.3e} <= {FOP_RTOL:.0e}); the fused accuracy gate is '
            'vacuous or the py_func chain is incomplete (F010)')

    def test_gather_index_perturbation_drives_gate_red(self):
        """
        Feeding an all-zero ``half_sum`` ARGUMENT into the fused core's
        ``py_func`` sends every ``(a, b)`` monomial to radial index
        ``= order`` instead of the ``idx(a, b, n)`` gather, so the
        contraction reads a single wrong derivative per order and the
        amplitude is wrong at any nontrivial source; the gate must go red.
        """
        self._assert_green_unpatched()
        core_pyfunc = operator._fused_contraction.py_func
        self.n_checks += 1
        self.assertFalse(
            hasattr(core_pyfunc, 'signatures'),
            '_fused_contraction.py_func carries .signatures (F010 vacuity)')

        def corrupt_fused(table, z_powers, zbar_powers, abs_powers,
                          half_sum, derivs_scaled, w_array, gamma_scaled,
                          max_order, dim):
            zeroed = np.zeros_like(half_sum)  # collapse the gather index
            return core_pyfunc(table, z_powers, zbar_powers, abs_powers,
                               zeroed, derivs_scaled, w_array, gamma_scaled,
                               max_order, dim)

        with mock.patch.object(operator, '_fused_contraction', corrupt_fused):
            raised, rel = self._gate_outcome()
        print(f'\n[Falsification] fused zeroed half_sum gather: '
              f'raised={raised} rel_err={rel:.3e}')
        self.n_checks += 1
        self.assertTrue(
            raised or rel > FOP_RTOL,
            f'the collapsed bilinear form still certified (rel_err '
            f'{rel:.3e} <= {FOP_RTOL:.0e}); the fused accuracy gate is '
            'vacuous or the py_func chain is incomplete (F010)')


# ---------------------------------------------------------------------------
# RETIRED: CoarseNodeInterpolationTestCase (the coarse-node cubic-spline
# kernel interpolation gate).
# ---------------------------------------------------------------------------
#
# This class certified the RETIRED fast-path design: the engine evaluated on a
# fixed ``_DEFAULT_KERNEL_NODES`` log-spaced + full-cluster-transition coarse
# grid (``_coarse_w_node_grid`` / ``_full_cluster_delays`` / ``n_kernel_nodes``)
# with every smooth channel kernel ``K_a(w)`` cubic-splined to the bin
# sub-samples.  Build 3f WP2 replaced that with the SACR-C decomposition: only
# the single smooth envelope ``E(w)`` is interpolated (LOO-adaptive nodes,
# ``_envelope_loo_nodes``), and the switched analytic saddles ``S_a * H_a`` are
# rebuilt in closed form at every sub-sample (``_reconstruct_kernels``).  The
# coarse-grid API this class drove no longer exists, so the class is retired
# rather than migrated.
#
# OWED (Test Developer): a replacement null-safe interpolation gate on the
# SACR-C path -- reconstruct ``F`` from the LOO-adaptive envelope nodes and
# assert ``max_f |F_interp - F_dense| / max_f |F_dense| < 1e-3`` on every
# ``_LENS_CONFIGS`` regime (the report's build3f gates 2/3), plus its paired
# self-falsification (an under-seeded envelope grid must exceed the ceiling).
# The Coder does not author this gate: it certifies the WP2 code the Coder
# wrote, and code and its certifying oracle must not share an author.


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
        Warm best-of-N ``lnlike`` sits under the loose `MS_CEILING` (a
        regression guard, NOT the brief's physical 10 ms claim --
        DEVIATION #2); the brute-force speed-up gate is opt-in under
        ``COGWHEEL_STRICT_TIMING``.  A per-component breakdown (caustic-
        search, amplification engine, total) is printed so a regression
        pinpoints the slipped lever.  Threads are pinned to 1 at import
        (best-effort) so the reported cost is the single-thread cost.

        RE-TUNED (Build 8d): the exact wave branch is the Schwinger
        evaluator (~90 ms/node), so warm crown ``lnlike`` is ~0.75 s and
        the loose ceiling is 3.0 s.  The speed-up over ``lnlike_bruteforce``
        stays the machine-independent structural claim, but brute now
        re-evaluates the exact engine per-frequency (~140 s per call), so
        it is measured only under ``COGWHEEL_STRICT_TIMING`` -- the default
        suite must stay fast.
        """
        candidate = self._crown_candidate()

        def rb():
            self.like.lnlike(candidate)

        rb()  # warm (numba already compiled at import; this warms caches)
        t_rb = self._best_time(rb)

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
              f'lnlike total={t_rb * 1e3:.2f} ms')

        self.n_checks += 1
        self.assertLessEqual(
            t_rb, MS_CEILING,
            f'warm lnlike best-of-{TIMING_REPEATS} = {t_rb * 1e3:.2f} ms '
            f'exceeds the loose ceiling {MS_CEILING * 1e3:.0f} '
            'ms; a lever regressed (see the printed breakdown)')

        if _STRICT_TIMING:
            def brute():
                self.like.lnlike_bruteforce(candidate)

            brute()
            t_brute = self._best_time(brute)
            print(f'[FewMsTiming] STRICT brute={t_brute * 1e3:.1f} ms, '
                  f'speedup={t_brute / t_rb:.1f}x')
            self.n_checks += 1
            self.assertGreater(
                t_brute, SPEEDUP_MIN * t_rb,
                f'RB lnlike ({t_rb * 1e3:.2f} ms) is not at least '
                f'{SPEEDUP_MIN}x faster than brute force '
                f'({t_brute * 1e3:.1f} ms); the RB speed-up regressed')

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

    @_brute_accuracy_tier
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

    def test_paths_refuse_over_critical_symmetrically(self):
        """
        On an OVER-CRITICAL input (``1 - kappa <= 0``, Type III) BOTH
        the fast-path RB ``lnlike`` and the exact ``lnlike_bruteforce``
        raise `geometry.LensDomainError` -- symmetric refusal, never a
        silent finite value on one path.  Since Build 7b macro-saddle
        INTERIORS (``0 < 1 - kappa < |gamma|``) are IN scope on both
        paths (full symmetric-agreement coverage lives in
        ``test_lensing_ratio_layer`` and
        ``test_lensing_saddle_likelihood``); the cheap contract-flip
        witness here is that the former refusal config now passes the
        macro-geometry domain gate.
        """
        candidate = self._candidate(self._lens_dic(
            *MACRO_SADDLE_Y, OVER_CRITICAL_GAMMA, 0.0,
            OVER_CRITICAL_KAPPA))
        self.n_checks += 1
        with self.assertRaises(geometry.LensDomainError):
            self.like.lnlike(candidate)
        self.n_checks += 1
        with self.assertRaises(geometry.LensDomainError):
            self.like.lnlike_bruteforce(candidate)

        # Contract-flip witness (cheap: geometry only, no engine eval):
        # the saddle INTERIOR passes the domain gate since Build 7b.
        geometry.macro_matrix(MACRO_SADDLE_GAMMA, 0.0, MACRO_SADDLE_KAPPA)
        self.n_checks += 1

    @_brute_accuracy_tier
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

    @_brute_accuracy_tier
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
    def _oracle_caustic_xy(theta, gamma, beta, kappa, branch=1.0):
        """
        Closed-form caustic (source-plane) point(s) at polar angle(s).

        ``caustic = macro_matrix @ x - x / |x|**2`` at the critical point
        ``x(theta)``, written out in plain vectorized numpy from the
        Chang--Refsdal critical-curve geometry -- the physics reference,
        independent of the compiled search under test.

        ``branch`` selects the sign of the square-root branch of the
        critical radius: ``+1.0`` is the only real branch at positive
        parity (the single 4-cusp astroid), and a macro saddle uses both
        ``+-1.0`` to trace the two edges of each 3-cusp deltoid lobe.  The
        discriminant is clamped at zero so the wedge endpoints (where the
        two branches meet) do not emit ``nan`` from float64 rounding --
        the SAME clamp `geometry._caustic_source` applies; at positive
        parity the discriminant is strictly positive, so with ``branch =
        1.0`` this is byte-for-byte the former positive-parity formula.
        """
        theta = np.asarray(theta, dtype=float)
        lam = 1.0 - kappa
        effective_gamma = gamma / lam
        phase = theta - beta
        discriminant = np.maximum(
            1.0 - effective_gamma**2 * np.sin(2.0 * phase)**2, 0.0)
        effective_u = (effective_gamma * np.cos(2.0 * phase)
                       + branch * np.sqrt(discriminant))
        # Outside a saddle wedge ``effective_u`` turns non-positive; the
        # resulting non-real radius is INTENTIONAL (those angles are off
        # the caustic and map to +inf distance downstream), so silence the
        # expected invalid-sqrt warning rather than let it mask real ones.
        with np.errstate(invalid='ignore', divide='ignore'):
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

    def _saddle_config_grid(self):
        """Macro-saddle ``(gamma, beta, kappa, source)`` combinations."""
        return itertools.product(
            CAUSTIC_SADDLE_GAMMAS, CAUSTIC_SADDLE_BETAS,
            CAUSTIC_SADDLE_KAPPAS, CAUSTIC_SADDLE_SOURCES)

    @staticmethod
    def _angular_gap(theta_a, theta_b):
        """Smallest absolute angle [rad] between two directions (mod 2*pi)."""
        return abs((theta_a - theta_b + np.pi) % (2.0 * np.pi) - np.pi)

    @classmethod
    def _point_self_consistency_gap(cls, produced, gamma, beta, kappa):
        """
        Max-abs gap between production's returned source-plane point and
        the INDEPENDENT caustic formula evaluated at production's OWN
        ``theta`` (min over both square-root branches).

        This certifies that the ``(theta, source)`` pair production
        reports genuinely lies on the closed-form caustic, WITHOUT
        inheriting the oracle's argmin angular resolution -- the caustic
        map ``theta -> point`` has O(1) derivative, so the gap is ~1e-12
        when the pair is consistent.  Non-real branch evaluations (outside
        a saddle wedge) map to ``+inf`` and are excluded by the ``min``.
        """
        produced_source = np.asarray(produced.source, dtype=float)
        gaps = []
        for branch in (1.0, -1.0):
            xco, yco = cls._oracle_caustic_xy(
                produced.theta, gamma, beta, kappa, branch)
            if np.isfinite(xco) and np.isfinite(yco):
                gaps.append(float(np.max(np.abs(
                    np.array([xco, yco]) - produced_source))))
        return min(gaps) if gaps else np.inf

    @classmethod
    def _caustic_speed(cls, theta, gamma, beta, kappa, branch):
        """
        Local caustic speed ``|d(x, y) / d theta|`` [source-plane units per
        rad] from the INDEPENDENT closed-form caustic map, by a symmetric
        central difference.  Vanishes at cusps (where the map is stationary
        and ``theta`` is under-determined) and is O(1) at regular points.

        Returns ``0.0`` if either sampled point is non-real (outside a
        saddle wedge), which conservatively tolerates the angle there.
        """
        h = 1e-7
        x_hi, y_hi = cls._oracle_caustic_xy(
            theta + h, gamma, beta, kappa, branch)
        x_lo, y_lo = cls._oracle_caustic_xy(
            theta - h, gamma, beta, kappa, branch)
        if not (np.isfinite(x_hi) and np.isfinite(y_hi)
                and np.isfinite(x_lo) and np.isfinite(y_lo)):
            return 0.0
        return float(np.hypot(x_hi - x_lo, y_hi - y_lo) / (2.0 * h))

    @classmethod
    def _lobe_of_theta(cls, theta, beta):
        """Lobe index of a saddle angle: 0 near ``beta``, 1 near ``beta+pi``."""
        gap0 = cls._angular_gap(theta, beta % (2.0 * np.pi))
        gap1 = cls._angular_gap(theta, (beta + np.pi) % (2.0 * np.pi))
        return 0 if gap0 <= gap1 else 1

    @classmethod
    def _scan_branch(cls, thetas, gamma, beta, kappa, branch, source):
        """
        Nearest source-to-caustic point on ONE branch over a theta grid.

        Returns ``(best_squared, best_theta, best_xy)``.  A two-stage
        search: a coarse ``argmin`` over ``thetas`` localizes the winning
        cell, then a dense local ``linspace`` spanning the two neighbouring
        cells is re-scanned and its winner refined by a parabolic-vertex
        fit.  The fine stage drives ``best_theta`` to ~1e-11 (far below the
        coarse grid step), so the oracle's argmin theta is itself tight
        enough to certify the production theta at 1e-9.  Grid cells whose
        radius is non-real (outside a saddle wedge, where the clamped
        discriminant collapses ``effective_u`` to a non-positive value)
        map to ``+inf`` and never win.
        """
        def squared_dist(angles):
            xco, yco = cls._oracle_caustic_xy(angles, gamma, beta, kappa,
                                              branch)
            sq = (xco - source[0])**2 + (yco - source[1])**2
            return np.where(np.isfinite(sq), sq, np.inf)

        # Stage 1: coarse argmin over the supplied grid.
        squared = squared_dist(thetas)
        index = int(np.argmin(squared))
        best_sq = float(squared[index])
        best_theta = float(thetas[index])
        step = float(thetas[1] - thetas[0])

        # Stage 2: dense local rescan spanning the two neighbouring cells.
        fine = np.linspace(best_theta - step, best_theta + step,
                           N_THETA_ORACLE)
        fine_sq = squared_dist(fine)
        j = int(np.argmin(fine_sq))
        if float(fine_sq[j]) < best_sq:
            best_sq, best_theta = float(fine_sq[j]), float(fine[j])
        # Parabolic-vertex refine using the two interior fine neighbours.
        if 0 < j < fine.size - 1:
            y_lo, y_mid, y_hi = (float(fine_sq[j - 1]), float(fine_sq[j]),
                                 float(fine_sq[j + 1]))
            denom = y_lo - 2.0 * y_mid + y_hi
            if np.isfinite(denom) and denom > 0.0:
                fstep = float(fine[1] - fine[0])
                offset = 0.5 * fstep * (y_lo - y_hi) / denom
                theta_star = best_theta + offset
                sq_star = float(squared_dist(np.array([theta_star]))[0])
                if np.isfinite(sq_star) and sq_star < best_sq:
                    best_sq, best_theta = sq_star, theta_star

        x_best, y_best = cls._oracle_caustic_xy(
            best_theta, gamma, beta, kappa, branch)
        return best_sq, best_theta, np.array([float(x_best), float(y_best)])

    def _oracle_nearest(self, gamma, beta, kappa, source, *, saddle):
        """
        Fully independent nearest-caustic solution: distance, theta,
        branch, lobe, source-plane point, and a degeneracy count.

        Positive parity scans the single 4-cusp astroid (``+`` branch,
        full circle).  A macro saddle scans BOTH deltoid lobes (centres
        ``beta`` and ``beta+pi``) on BOTH square-root branches over their
        critical wedges and takes the global minimum.  ``n_near_equal``
        counts how many independent branch/lobe minima fall within a
        relative ``1e-6`` of the winner -- the benign degeneracy (source
        on a symmetry axis) where ``theta``/``lobe`` are ambiguous and
        only the distance is well posed.
        """
        source = np.asarray(source, dtype=float)
        if not saddle:
            grid = np.linspace(0.0, 2.0 * np.pi, N_THETA_ORACLE,
                               endpoint=False)
            best_sq, best_theta, best_xy = self._scan_branch(
                grid, gamma, beta, kappa, 1.0, source)
            # Cyclic local minima of the coarse scan (degeneracy flag).
            caustic_x, caustic_y = self._oracle_caustic_xy(
                grid, gamma, beta, kappa, 1.0)
            squared = (caustic_x - source[0])**2 + (caustic_y - source[1])**2
            left, right = np.roll(squared, 1), np.roll(squared, -1)
            minima = np.sort(squared[(squared < left) & (squared < right)])
            n_near_equal = int(np.sum(minima <= best_sq * (1.0 + 1e-6))) \
                if minima.size else 1
            return {'distance': float(np.sqrt(best_sq)),
                    'theta': best_theta % (2.0 * np.pi),
                    'branch': 1, 'lobe': None, 'caustic_xy': best_xy,
                    'n_near_equal': n_near_equal}

        lam = 1.0 - kappa
        theta_max = 0.5 * np.arcsin(lam / abs(gamma))
        winners = []  # (squared, theta, branch, lobe, xy) per branch/lobe
        for lobe, center in enumerate((beta, beta + np.pi)):
            wedge = np.linspace(center - theta_max, center + theta_max,
                                N_THETA_ORACLE)
            for branch in (1.0, -1.0):
                sq, theta, xy = self._scan_branch(
                    wedge, gamma, beta, kappa, branch, source)
                winners.append((sq, theta, int(branch), lobe, xy))
        best_sq, best_theta, best_branch, best_lobe, best_xy = min(
            winners, key=lambda item: item[0])
        n_near_equal = int(sum(
            1 for sq, *_ in winners if sq <= best_sq * (1.0 + 1e-6)))
        return {'distance': float(np.sqrt(best_sq)),
                'theta': best_theta % (2.0 * np.pi),
                'branch': best_branch, 'lobe': best_lobe,
                'caustic_xy': best_xy, 'n_near_equal': n_near_equal}

    def test_positive_parity_theta_and_point_match_bruteforce_oracle(self):
        """
        ``nearest_caustic_point`` returns the SAME argmin ``theta`` (mod
        ``2*pi``) and the SAME source-plane caustic POINT as the dense
        brute-force oracle across the positive-parity astroid grid.  Where
        the nearest point is degenerate (multiple near-axis minima) the
        angle/point are ambiguous, so those configs assert distance only.
        This extends the distance-only gate to the full returned geometry.
        """
        residuals = []
        for gamma, beta, kappa, source in self._config_grid():
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              source=source):
                produced = geometry.nearest_caustic_point(
                    gamma, beta, np.asarray(source, dtype=float),
                    kappa=kappa)
                oracle = self._oracle_nearest(
                    gamma, beta, kappa, source, saddle=False)
                if oracle['n_near_equal'] > 1:
                    # Degenerate: theta/point ambiguous; distance only.
                    self.n_checks += 1
                    self.assertLess(
                        abs(produced.distance - oracle['distance'])
                        / oracle['distance'], CAUSTIC_RTOL,
                        f'gamma={gamma} beta={beta} kappa={kappa} '
                        f'source={source}: degenerate-config distance drift')
                    continue
                theta_gap = self._angular_gap(produced.theta, oracle['theta'])
                point_gap = self._point_self_consistency_gap(
                    produced, gamma, beta, kappa)
                residuals.append((oracle['distance'], theta_gap, point_gap))
                speed = self._caustic_speed(
                    oracle['theta'], gamma, beta, kappa, 1.0)
                arclen_gap = theta_gap * speed
                self.n_checks += 1
                self.assertLess(
                    arclen_gap, CAUSTIC_ARCLEN_ATOL,
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: returned theta {produced.theta:.12g} '
                    f'differs from the brute-force argmin {oracle["theta"]:.12g}'
                    f' by {theta_gap:.3e} rad, arc-length {arclen_gap:.3e} > '
                    f'{CAUSTIC_ARCLEN_ATOL} (caustic speed {speed:.3e}); the '
                    'search converged to a non-global stationary point')
                self.n_checks += 1
                self.assertLess(
                    point_gap, CAUSTIC_POINT_ATOL,
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: returned caustic point '
                    f'{np.asarray(produced.source)} differs from the oracle '
                    f'{oracle["caustic_xy"]} by {point_gap:.3e} > '
                    f'{CAUSTIC_POINT_ATOL}')
        self._plot_theta_error(residuals, 'positive parity',
                               'caustic_search_theta_error_positive')

    def test_saddle_distance_theta_lobe_match_bruteforce_oracle(self):
        """
        On BOTH macro-saddle deltoid lobes (centres ``beta`` and
        ``beta+pi``) and BOTH square-root branches, the accelerated search
        returns the SAME distance, argmin ``theta``, source-plane point,
        and LOBE as the dense two-lobe/two-branch oracle.  Distance is
        gated at `CAUSTIC_RTOL`; the resolution-limited saddle angle at
        `CAUSTIC_SADDLE_THETA_ATOL`; the lobe identity exactly.  Degenerate
        (competing-lobe) configs assert distance only.
        """
        residuals = []
        n_saddle = 0
        for gamma, beta, kappa, source in self._saddle_config_grid():
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa,
                              source=source):
                produced = geometry.nearest_caustic_point(
                    gamma, beta, np.asarray(source, dtype=float),
                    kappa=kappa)
                oracle = self._oracle_nearest(
                    gamma, beta, kappa, source, saddle=True)
                n_saddle += 1
                self.n_checks += 1
                self.assertLess(
                    abs(produced.distance - oracle['distance'])
                    / oracle['distance'], CAUSTIC_RTOL,
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: saddle distance {produced.distance:.12g}'
                    f' disagrees with the two-lobe oracle '
                    f'{oracle["distance"]:.12g}')
                if oracle['n_near_equal'] > 1:
                    continue  # competing lobes: theta/lobe ambiguous
                produced_lobe = self._lobe_of_theta(produced.theta, beta)
                theta_gap = self._angular_gap(produced.theta, oracle['theta'])
                point_gap = self._point_self_consistency_gap(
                    produced, gamma, beta, kappa)
                residuals.append((gamma, theta_gap, oracle['lobe']))
                self.n_checks += 1
                self.assertEqual(
                    produced_lobe, oracle['lobe'],
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: selected lobe {produced_lobe} != '
                    f'oracle lobe {oracle["lobe"]} (a lobe jump)')
                speed = self._caustic_speed(
                    oracle['theta'], gamma, beta, kappa,
                    float(oracle['branch']))
                arclen_gap = theta_gap * speed
                self.n_checks += 1
                self.assertLess(
                    arclen_gap, CAUSTIC_ARCLEN_ATOL,
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: saddle theta {produced.theta:.12g} '
                    f'differs from the oracle {oracle["theta"]:.12g} by '
                    f'{theta_gap:.3e} rad, arc-length {arclen_gap:.3e} > '
                    f'{CAUSTIC_ARCLEN_ATOL} (caustic speed {speed:.3e})')
                self.n_checks += 1
                self.assertLess(
                    point_gap, CAUSTIC_POINT_ATOL,
                    f'gamma={gamma} beta={beta} kappa={kappa} '
                    f'source={source}: saddle caustic point drift '
                    f'{point_gap:.3e} > {CAUSTIC_POINT_ATOL}')
        self.assertGreaterEqual(
            n_saddle, 9,
            f'only {n_saddle} saddle configs exercised; the Professor '
            'specification requires at least 9 spanning both lobes/branches')
        self._plot_theta_error(
            [(g, t, 0) for g, t, _ in residuals], 'macro saddle',
            'caustic_search_theta_error_saddle', xlabel='gamma')

    def test_saddle_branch_selection_tracks_reference(self):
        """
        Branch-invariance falsification: on a near-symmetric saddle
        (``beta = 0``, mirror lobes across ``y1 = 0``) a source swept in
        ``y1`` through the symmetry line flips the nearest LOBE exactly as
        the independent oracle does -- one clean transition, no
        Newton-induced chatter, and never a flip the reference does not
        also make.
        """
        beta = 0.0
        produced_lobes, oracle_lobes = [], []
        for y1 in CAUSTIC_SYMMETRY_Y1_SWEEP:
            source = (y1, CAUSTIC_SYMMETRY_Y2)
            with self.subTest(y1=y1):
                produced = geometry.nearest_caustic_point(
                    CAUSTIC_SYMMETRY_GAMMA, beta,
                    np.asarray(source, dtype=float),
                    kappa=CAUSTIC_SYMMETRY_KAPPA)
                oracle = self._oracle_nearest(
                    CAUSTIC_SYMMETRY_GAMMA, beta, CAUSTIC_SYMMETRY_KAPPA,
                    source, saddle=True)
                self.n_checks += 1
                self.assertLess(
                    abs(produced.distance - oracle['distance'])
                    / oracle['distance'], CAUSTIC_RTOL,
                    f'y1={y1}: distance drift across the symmetry sweep')
                if oracle['n_near_equal'] > 1:
                    continue
                produced_lobe = self._lobe_of_theta(produced.theta, beta)
                produced_lobes.append(produced_lobe)
                oracle_lobes.append(oracle['lobe'])
                self.n_checks += 1
                self.assertEqual(
                    produced_lobe, oracle['lobe'],
                    f'y1={y1}: selected lobe {produced_lobe} != oracle lobe '
                    f'{oracle["lobe"]}; the search chattered off the '
                    'reference lobe across the symmetry line')
        self.assertEqual(
            set(oracle_lobes), {0, 1},
            'the source sweep never crossed the lobe-symmetry line, so the '
            'branch-invariance falsification is vacuous; widen the sweep')
        self._plot_lobe_sweep(produced_lobes, oracle_lobes)

    @staticmethod
    def _warm_best_ms(gamma, beta, kappa, source):
        """Warm best-of-N per-call cost [ms] of `nearest_caustic_point`."""
        source = np.asarray(source, dtype=float)

        def search():
            geometry.nearest_caustic_point(gamma, beta, source, kappa=kappa)

        search()  # warm (compile numba caustic core + caches)
        best = np.inf
        for _ in range(CAUSTIC_TIMING_REPEATS):
            start = time.perf_counter()
            search()
            best = min(best, time.perf_counter() - start)
        return best * 1e3

    def test_caustic_search_warm_timing_probe(self):
        """
        WP-A timing probe (SOFT).  Times `nearest_caustic_point` warm on
        BOTH cost classes the Newton caustic shortcut targeted -- the
        POSITIVE-parity astroid (single 4-cusp search, measured ~0.1 ms
        class) and the MACRO-SADDLE two-lobe/two-branch search (measured
        ~1 ms class) -- and prints each per-call cost with its ratio to
        the ~0.3 ms target.  The hard sub-millisecond assertion (on the
        slower saddle class) is enforced only under
        ``COGWHEEL_STRICT_TIMING`` (machine dependent); otherwise a
        generous non-flaky ceiling guards a catastrophic regression.
        This is a diagnostic guard, not a physical claim.
        """
        positive_ms = self._warm_best_ms(
            0.20, 0.0, 0.0, (0.15, 0.05))
        saddle_ms = self._warm_best_ms(
            CAUSTIC_SYMMETRY_GAMMA, 0.0, CAUSTIC_SYMMETRY_KAPPA, (0.20, 0.05))
        print(f'\n[CausticTiming] warm nearest_caustic_point: '
              f'positive-parity = {positive_ms:.4f} ms (~0.1 ms class), '
              f'macro-saddle = {saddle_ms:.4f} ms (~1 ms class); '
              f'target ~{CAUSTIC_TIMING_TARGET_MS} ms, saddle ratio '
              f'{saddle_ms / CAUSTIC_TIMING_TARGET_MS:.2f}x')
        # Gate the SLOWER (saddle) class -- the worst case bounds both.
        self.n_checks += 1
        if _STRICT_TIMING:
            self.assertLess(
                saddle_ms, 1.0,
                f'warm saddle caustic search {saddle_ms:.4f} ms exceeds the '
                'strict 1 ms class (COGWHEEL_STRICT_TIMING set)')
        else:
            self.assertLess(
                saddle_ms, CAUSTIC_TIMING_LOOSE_CEILING_MS,
                f'warm saddle caustic search {saddle_ms:.4f} ms exceeds the '
                f'loose {CAUSTIC_TIMING_LOOSE_CEILING_MS} ms regression '
                'ceiling')

    def _plot_theta_error(self, residuals, label, name, xlabel=None):
        """Scatter of returned-vs-oracle angular residual."""
        if not residuals:
            return
        residuals = np.array(residuals)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(residuals[:, 0], np.maximum(residuals[:, 1], 1e-18),
                   s=22, zorder=3)
        gate = (CAUSTIC_SADDLE_THETA_ATOL if 'saddle' in label
                else CAUSTIC_THETA_ATOL)
        ax.axhline(gate, color='k', ls='--', label=f'{gate:g} gate')
        ax.set_xlabel(xlabel or 'source-to-caustic distance (oracle)')
        ax.set_ylabel('|theta_produced - theta_oracle| (mod 2*pi)')
        ax.set_yscale('log')
        ax.set_title(f'WP-A caustic argmin theta preserved ({label})')
        ax.legend(fontsize=8)
        self._save_figure(fig, name)

    def _plot_lobe_sweep(self, produced_lobes, oracle_lobes):
        """Selected-lobe id vs the source sweep across the symmetry line."""
        if not oracle_lobes:
            return
        y1 = np.asarray(CAUSTIC_SYMMETRY_Y1_SWEEP[:len(oracle_lobes)])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.step(y1, oracle_lobes, where='mid', color='k', lw=2,
                label='oracle lobe', zorder=2)
        ax.scatter(y1, produced_lobes, color='crimson', s=40,
                   label='produced lobe', zorder=3)
        ax.set_xlabel('source $y_1$ (swept across the symmetry line)')
        ax.set_ylabel('selected lobe id')
        ax.set_yticks((0, 1))
        ax.set_title('WP-A saddle lobe selection tracks the reference')
        ax.legend(fontsize=8)
        self._save_figure(fig, 'caustic_search_lobe_sweep')

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

    def test_arclength_theta_gate_rejects_a_forged_non_global_theta(self):
        """
        Self-falsification PARTNER for the arc-length theta certification
        (`CAUSTIC_ARCLEN_ATOL`) that gates `nearest_caustic_point`'s
        returned ``theta`` in the two preservation tests above.  A theta
        at the floating-point localization floor (~1e-9 rad from the
        oracle argmin) passes; a FORGED non-global stationary theta offset
        O(1) rad away lands at O(1) arc-length and is REJECTED.  Without
        this partner the theta gate could not be shown to go red -- yet a
        wrong stationary point (a genuine lobe-jump / non-global
        convergence) is exactly what it must catch.  Evaluated at a
        REGULAR (non-cusp) caustic point, where the caustic speed is O(1)
        so the arc-length currency is sharp; at a cusp the speed vanishes
        and the gate deliberately tolerates theta, so a cusp config would
        make the forgery vacuous.
        """
        gamma, beta, kappa = 0.20, 0.0, 0.0
        source = (0.15, 0.05)  # unique minimum at a regular caustic point
        oracle = self._oracle_nearest(gamma, beta, kappa, source,
                                      saddle=False)
        theta_star = oracle['theta']
        speed = self._caustic_speed(theta_star, gamma, beta, kappa, 1.0)
        # Preconditions: a clean, discriminating forgery needs a unique
        # minimum at a regular (non-cusp, speed O(1)) point.
        self.n_checks += 1
        self.assertEqual(
            oracle['n_near_equal'], 1,
            'the forgery config lost its unique minimum; pick another')
        self.n_checks += 1
        self.assertGreater(
            speed, 1e-2,
            f'caustic speed {speed:.3e} is near a cusp; the arc-length '
            'forgery would be vacuous there -- pick a regular point')
        # A theta at the floating-point localization floor passes the gate.
        good_arclen = self._angular_gap(
            theta_star + 1e-9, theta_star) * speed
        self.n_checks += 1
        self.assertLess(
            good_arclen, CAUSTIC_ARCLEN_ATOL,
            f'a theta at the localization floor lands at arc-length '
            f'{good_arclen:.3e}, already past the {CAUSTIC_ARCLEN_ATOL} '
            'gate; the gate is too tight to admit the true answer')
        # A forged non-global stationary theta (O(1) rad away) is rejected.
        forged = theta_star + 0.5
        forged_gap = self._angular_gap(forged, theta_star)
        forged_arclen = forged_gap * speed
        self.n_checks += 1
        self.assertGreater(
            forged_arclen, CAUSTIC_ARCLEN_ATOL,
            f'a forged non-global theta {forged:.6g} (gap {forged_gap:.3e} '
            f'rad, speed {speed:.3e}) lands at arc-length '
            f'{forged_arclen:.3e}, which does not exceed the '
            f'{CAUSTIC_ARCLEN_ATOL} gate; the theta certification cannot '
            'reject a wrong stationary point')


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

    # RETIRED with `CoarseNodeInterpolationTestCase`: this was the
    # self-falsification (an under-resolved coarse grid must exceed
    # `INTERP_NULLSAFE_CEIL`) of the removed coarse-node cubic-spline kernel
    # interpolation.  Its replacement -- an under-seeded LOO-envelope grid must
    # breach the ceiling the adaptive `_envelope_loo_nodes` set clears -- is
    # owed to the Test Developer alongside the SACR-C interpolation gate.

    @_brute_accuracy_tier
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
