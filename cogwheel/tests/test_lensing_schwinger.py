"""
Tests for `lensing.chang_refsdal._schwinger`, the exact 1D
Schwinger-parameter wave-branch EVALUATOR for the macro-saddle domain
(``f_schwinger``, its certify-XOR-refuse contract, the deep-band F009-S
pins, the dd-mandatory falsifications, and the warm per-point cost
measurement).  The mass-sheet / geometric branch lives in
`test_lensing_saddle_geometry.py`; the WP1/WP2 build (2026-07-19)
additionally exercises the OPERATOR-level positive-parity Schwinger
fallback dispatch and the image-census guard here (WP1/WP2 constants
block below and the classes at the foot of the file), since those are
the surfaces that consume this evaluator.

WHY THE ORACLE IS INDEPENDENT (F002)
------------------------------------
Every accuracy gate is judged against `_oracle_1d`, a PURE-mpmath
evaluation of the SAME 1D Schwinger representation (research note
Sec. 6.1)::

    F = (w / (2 pi i)) e^{i w |y|^2 / 2} (pi / Gamma(iw/2))
        Int_0^inf t^{iw/2 - 1} h(t) dt,
    h(t) = [(t - iwa/2)(t - iwb/2)]^{-1/2}
           exp[-w^2 y1^2 / (4(t - iwa/2)) - w^2 y2^2 / (4(t - iwb/2))],

principal roots, ``a = 1 - gamma'``, ``b = 1 + gamma'``, regularized at
``t = 0`` by one integration by parts and quadratured with
``mpmath.quad`` in ``u = ln t`` at ``dps = 30 + ceil(w)``.  It shares
NONE of production's derivation: no double-double arithmetic, no
Newton-refined Gauss-Legendre rule, no paired-rule certification --
just arbitrary precision.  An AST guard
(`OracleImportGuardTestCase`, the `test_lensing_gauge` idiom) proves
the oracle path references nothing from
`cogwheel.lensing.chang_refsdal`, and the guard itself is shown able
to go red.

The oracle is CERTIFIED before it judges anything:

* against the point-mass closed form
  ``e^{pi w/4 + i(w/2) ln(w/2)} Gamma(1 - iw/2) 1F1(iw/2; 1; iw|y|^2/2)``
  at ``a = b = 1`` -- measured 3.6e-23 at ``w = 10`` and, crucially,
  5.2e-19 at ``w = 30`` (the closed form exercises the SAME
  ``e^{pi w/4}`` cancellation the high band needs);
* against the literal Build-6 anchor
  ``F(3, (0.4, 0.3), gamma' = 1.3)`` (validated against an independent
  2D lens-plane oracle to 2.2e-15, research note Sec. 6.2) -- measured
  1.1e-14;
* internally: refining dps / panels / margins moves it by < 2e-17.

HIGH-BAND HISTORY (defect measured 2026-07-18, FIXED 2026-07-19)
----------------------------------------------------------------
`f_schwinger` used to fabricate SILENTLY CERTIFIED values above
``w ~ 20``, with relative error tracking ``~ eps_f64 * e^{pi w/4}``
(2.2e-9 at ``w = 20`` up to ~3.4e2 at ``w = 55``).  Two eps_f64-class
systematics, each bit-identical in the N and 2N rules (hence invisible
to the paired-rule certification) and amplified by ``e^{+pi w/4}`` on
reconstruction: (1) the IBP endpoint term was evaluated at ``t_cap``
while both quadrature domains split at ``exp(fl(math.log(t_cap)))``,
breaking the ``T``-consistency of the IBP identity by ``~ eps_f64``
absolute; (2) the ``1/s`` factor multiplying the endpoint and A pieces
(but NOT B, so no cancellation in the IBP combination) was the float64
reciprocal ``fl(1/half_w)`` treated as dd-exact.  Both are fixed in the
core (endpoint evaluated at the actual split point ``e^{u_mid}``;
``1/half_w`` carried in dd), and `HighBandKnownDefectTestCase` now
gates the high band at the unweakened spec tolerances: measured
9.1e-14 at ``w = 20``, 1.7e-14 at ``w = 30``, 4.4e-15 at ``w = 45``,
5.6e-13 at ``w = 55``.

TOLERANCES
----------
Certified band (``w <= 10``): production worst measured 8.3e-13, so the
1e-10 gate has > 100x headroom.  Deep band: |F| closed-form residual
4.4e-5 at ``w = 1e-4`` (gate 1e-3); Morse-phase intercept residual
1.5e-7 (gate 5e-4).  Falsification config ``w = 30, y = (1, 0),
gamma' = 1.3`` measures 9.0e-8 unpatched -- green under the 1e-6
falsification gate with 11x margin, so a patched RED is the
corruption's doing.
"""
from __future__ import annotations

import ast
import cmath
import functools
import inspect
import itertools
import math
import pathlib
import textwrap
import time
from unittest import TestCase, main, mock

import matplotlib
matplotlib.use('Agg')  # headless: the diagnostic plot is written, never shown
import matplotlib.pyplot as plt
import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import (
    _airy_fold, _pearcey_cusp, _schwinger, geometry, operator)
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError, W_CEILING_SCHWINGER, f_schwinger)
from cogwheel.lensing.chang_refsdal.operator import (
    F_op, F_op_grid,
    cancellation_exponent, geometric_amplification, select_branch)

#: Where the dispatch-accuracy diagnostic plot is written (the house
#: convention: ``cogwheel/tests/output/<test>_<desc>.png``).
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

#: Base oracle precision; the working dps is ``30 + ceil(w)`` (the
#: research-note scaling: mpmath's own quadrature under-resolves the
#: t-integral at high w unless dps grows with w).
ORACLE_DPS_BASE = 30

#: Oscillations of the ``t^{iw/2}`` phase per composite mpmath panel,
#: the per-panel refinement ceiling, the additive ``u``-range slack
#: past the ``pi w / 4`` cancellation depth, and the low-``w`` panel
#: floor.  Calibrated so the oracle is converged to < 2e-17 of itself
#: under refinement (dps + panels + margin) at ``w = 30, 45, 55``.
ORACLE_WAVELENGTHS_PER_PANEL = 8.0
ORACLE_MAXDEGREE = 5
ORACLE_EXTRA_MARGIN = 40.0
ORACLE_MIN_PANELS = 12

#: Oracle-certification gates. The point-mass closed form lands at
#: 3.6e-23 (w=10) / 5.2e-19 (w=30); the literal 2D-validated anchor at
#: 1.1e-14.
PM_CERT_RTOL = 1e-12
ANCHOR_RTOL = 1e-12

#: The Build-6 sanity anchor: ``f_schwinger(3, (0.4, 0.3), 1.3)``,
#: cross-validated against the independent 2D lens-plane oracle
#: (research note Sec. 6.2, 2.2e-15 class).
ANCHOR_W = 3.0
ANCHOR_Y = (0.4, 0.3)
ANCHOR_GAMMA = 1.3
ANCHOR_VALUE = complex(0.14470585550870085, 0.4065122393352838)

#: Dev-oracle grid (research Sec. 9 / build brief). The certified band
#: gates 1e-10; the high band ``w in {20, 30, 45}`` carries the SAME
#: spec gate (defect fixed 2026-07-19, module docstring).
GRID_W_CERTIFIED = (0.5, 1.0, 3.0, 5.0, 10.0)
GRID_W_HIGH = (20.0, 30.0, 45.0)
GRID_GAMMA = (1.05, 1.3, 2.0)
GRID_Y = ((0.4, 0.3), (1.0, 0.0), (0.1, 0.1))
GRID_RTOL = 1e-10

#: Single high-band spot check near the ceiling, at the RELAXED 1e-6
#: tolerance the dd law predicts there (measured rel 5.6e-13 post-fix).
HIGH_W_SPOT = 55.0
HIGH_W_SPOT_RTOL = 1e-6

#: Deep-band (F009-S) pins at ``gamma' = 1.3`` (eigenvalues
#: ``a = -0.3``, ``b = 2.3``), ``y = (0.4, 0.3)``.
DEEP_GAMMA = 1.3
DEEP_Y = (0.4, 0.3)
DEEP_WS = (1e-4, 1e-3, 1e-2)
DEEP_MAGNITUDE_RTOL = 1e-3
MORSE_PHASE_TOL = 5e-4

#: Certify-XOR-refuse sweeps (all at ``y = (0.4, 0.3), gamma' = 1.3``).
XOR_Y = (0.4, 0.3)
XOR_GAMMA = 1.3
CERTIFIED_W_SWEEP = (10.0, 30.0, 50.0, 59.9)
REFUSED_W_SWEEP = (60.5, 65.0, 80.0)

#: dd-mandatory falsification config (F010 / F005-S analog). Unpatched
#: production measures 9.0e-8 here -- green under the 1e-6 gate.
FALS_W = 30.0
FALS_Y = (1.0, 0.0)
FALS_GAMMA = 1.3
FALS_RTOL = 1e-6
PERTURBED_CEILING = 20.0

#: Production names the mpmath oracle path must never reference
#: (F002: an oracle that touches the code under test cannot fail).
FORBIDDEN_ORACLE_NAMES = frozenset({
    'cogwheel', 'lensing', 'chang_refsdal', '_schwinger', 'f_schwinger',
    'SchwingerCertificationError', 'W_CEILING_SCHWINGER',
    '_raw_t_integral_core', '_reconstruct', '_dd_gl_rule', '_h_dd',
    '_g_dd', '_dd', 'dd_add', 'dd_mul', 'dd_sub', 'dd_div',
    'dd_complex_add', 'dd_complex_mul', 'dd_complex_sub',
    'dd_complex_div', 'np', 'numpy', 'numba'})

# =====================================================================
# WP1/WP2 build (2026-07-19): the OPERATOR-level dispatch, the
# positive-parity guard relaxation, and the image-census guard.
#
# WP2 wired the strong-shear Schwinger evaluator into `operator.F_op` /
# `operator.F_op_grid`: a positive-parity node is, below the Schwinger
# ceiling, served through the exact 1D evaluator `f_schwinger` and
# reconstructed with the SAME mass-sheet identity the operator path
# uses.  WP1 added `geometry._check_image_census`, a runtime
# index-theorem guard on the solved image set.  The constants below
# drive the dispatch-accuracy, guard-relaxation, bit-freeze,
# above-ceiling-refusal, and census-falsification suites appended to
# this file.  All oracle judgements reuse the SAME AST-guarded
# `_oracle_saddle`; the pure-shear amplification ``F_{0, gamma'}`` it
# evaluates is exactly what the dispatch reconstructs.
# =====================================================================

#: Dispatch-accuracy fixtures: positive-parity strong-shear points
#: served by the Schwinger evaluator.  Every tuple
#: ``(gamma, beta, kappa, y, w)`` was confirmed (probe 2026-07-19) to
#: match the reconstructed mpmath oracle below 1e-10.  ``gamma`` sits
#: just below the positive-parity limit (``0.47, 0.49`` with
#: ``kappa = 0`` so ``lam = 1``); the on-axis ``y = (1, 0)`` points
#: reach the low-``w`` end of the [3, 60] span, and the two
#: ``kappa != 0`` rows exercise the mass-sheet prefactor.
#: ``w = 59.9`` is the worst measured case
#: ``w = 59.9`` is the worst measured case
#: (rel 1.7e-11, Professor-staked 1.6e-11).
DISPATCH_POINTS = (
    (0.47, 0.0, 0.0, (1.0, 0.0), 5.0),
    (0.47, 0.0, 0.0, (0.4, 0.3), 8.0),
    (0.49, 0.0, 0.0, (0.4, 0.3), 12.0),
    (0.47, 0.0, 0.0, (0.4, 0.3), 20.0),
    (0.49, 0.0, 0.0, (0.4, 0.3), 30.0),
    (0.47, 0.0, 0.0, (0.4, 0.3), 45.0),
    (0.49, 0.0, 0.0, (0.4, 0.3), 55.0),
    (0.47, 0.0, 0.0, (0.1, 0.1), 59.9),
    (0.35, 0.0, 0.2, (0.4, 0.3), 20.0),
    (0.40, 0.0, 0.15, (0.1, 0.1), 30.0),
)
#: The uniform dispatch tolerance the Professor staked over ``w`` in
#: ``(0, 60]`` (worst measured 1.7e-11, > 5x headroom).
DISPATCH_RTOL = 1e-10

#: One (gamma, beta, kappa, y) fixture at which to sweep ``w`` for the
#: rel-error-vs-``w`` diagnostic plot (all these ``w`` refuse the legacy
#: path and land on the fallback).
DISPATCH_PLOT_GAMMA = 0.47
DISPATCH_PLOT_Y = (0.4, 0.3)
DISPATCH_PLOT_WS = (8.0, 12.0, 20.0, 30.0, 40.0, 50.0, 55.0, 59.9)

#: Guard-relaxation fixtures for `f_schwinger` (WP2 one-line relaxation
#: from ``gamma_prime > 1`` to ``gamma_prime > 0``).  The saddle
#: (``gamma_prime > 1``) values are BYTE-FROZEN from pre-build HEAD (the
#: relaxation must not perturb any ``gamma_prime > 1`` result); positive
#: parity (``0 < gamma_prime < 1``) is now ACCEPTED and judged against
#: the oracle; ``gamma_prime <= 0`` still raises `ValueError`.
SADDLE_BITFREEZE = {
    (3.0, (0.4, 0.3), 1.3): complex(0.14470585550870085,
                                    0.40651223933528396),
    (5.0, (0.4, 0.3), 1.3): complex(-0.3166556108784056,
                                    -0.09918956656584109),
    (10.0, (1.0, 0.0), 2.0): complex(-0.36925782902015036,
                                     0.25566036688060445),
    (8.0, (0.2, 0.1), 1.05): complex(0.39738539361296416,
                                     -0.43316960492592993),
}
#: Positive-parity ``(w, y, gamma_prime)`` now accepted by the relaxed
#: guard; ``gamma_prime`` kept clear of the ``-> 1`` parity-boundary
#: pinch (measured rel <= 5.6e-14 across these).
POSITIVE_PARITY_ACCEPTED = tuple(
    (w, (0.4, 0.3), gamma_prime)
    for gamma_prime in (0.3, 0.5, 0.7)
    for w in (3.0, 8.0, 20.0))
#: ``gamma_prime <= 0`` (``det A == 0`` or wrong sign) is a DOMAIN error.
NONPOSITIVE_GAMMA_PRIME = (0.0, -0.5, -1.3)

#: Positive-parity operator-path points (moderate shear, gamma' = 0.2 > 0).
#: RE-BASELINE (Build 8d, F017 contract-flip discipline): these configs are
#: now served by the exact Schwinger evaluator (order_used == 0), NOT the
#: legacy operator series.  `CERTIFIED_BITFREEZE` holds the NEW Schwinger
#: production values (byte-frozen); `LEGACY_BITFREEZE` holds the OLD legacy
#: literals recorded from the retired operator-series contraction.
#: The re-baseline carries a WITNESS: NEW and OLD agree to
#: `BITFREEZE_WITNESS_TOL` in the max-normalized currency -- the flip is a
#: byte/contract change, not a physics change (measured max-normalized
#: residual 5.4e-15 real / 1.4e-15 imag over this grid, scale 1.615).
CERTIFIED_BITFREEZE_GAMMA = 0.2
CERTIFIED_BITFREEZE_Y = (0.4, 0.3)
#: NEW (Build 8d) Schwinger production values.
CERTIFIED_BITFREEZE = {
    3.0: complex(1.0977672009048085, 0.8231261499570348),
    5.0: complex(0.23148446460210984, -0.5479573270850424),
    8.0: complex(1.6103326490603147, -0.12532868210912673),
    10.0: complex(0.3668222757423122, -0.22370837989397843),
}
#: OLD (pre-8d) legacy operator-series values, kept for the contract-flip
#: witness only.  They were produced by the retired dd/1F1 contraction, an
#: algorithm INDEPENDENT of the production Schwinger path (F002).
LEGACY_BITFREEZE = {
    3.0: complex(1.0977672009048116, 0.8231261499570363),
    5.0: complex(0.23148446460211186, -0.5479573270850447),
    8.0: complex(1.610332649060306, -0.12532868210912573),
    10.0: complex(0.36682227574231296, -0.2237083798939786),
}
#: Max-normalized byte-flip currency (the F005/7a/8d owner-set 1e-10
#: standard): |F_new - F_old| / max(max|F_old|, 1e-15) below this proves
#: the re-baseline moved only bytes, not physics.
BITFREEZE_WITNESS_TOL = 1e-10

#: Companion pin: the shear-free ``gamma' == 0`` point lens (measure-zero
#: in the sampled prior, but reachable) that Schwinger CANNOT represent, so
#: it stays on the legacy operator series (order_used > 0).  This keeps a
#: reachable pin on the sole remaining legacy production exit (Build 8d).
POINTLENS_BITFREEZE_GAMMA = 0.0
POINTLENS_BITFREEZE_Y = (0.3, 0.0)
POINTLENS_BITFREEZE_WS = (1.0, 2.0, 3.0, 5.0)

#: Above-ceiling fixtures: positive-parity strong-shear points evaluated
#: at ``w > W_CEILING_SCHWINGER``, spanning BOTH serving-ladder outcomes.
#: RE-BASELINE (Build 8e serving ladder): the ``y = (0.4, 0.3)`` column is
#: now SERVED by the certified uniform Airy fold arm (moderate image
#: splitting -> fold argument xi large enough to certify), while the
#: near-caustic ``y = (0.1, 0.1)`` column is genuinely HARD-CORE -- both
#: the fold argument xi (~2.4) and the Pearcey radius R (~2.6) are too
#: small for either arm, so the named `SchwingerCertificationError` still
#: stands.  `RefusalAboveCeilingTestCase` asserts the CONDITIONAL contract
#: per fixture and proves both branches are exercised.
ABOVE_CEILING_GAMMAS = (0.47, 0.49)
ABOVE_CEILING_YS = ((0.4, 0.3), (0.1, 0.1))
ABOVE_CEILING_WS = (61.0, 80.0)

#: Image-census (WP1) falsification fixtures.  A positive-parity macro
#: matrix has an interior 4-image source; a saddle matrix is 2-image
#: everywhere in this build (probed 2026-07-19), so its guard red-path
#: is reached by a single-image drop rather than a mirror-pair drop.
CENSUS_POSITIVE_MATRIX_ARGS = (0.3, 0.0, 0.0)
CENSUS_POSITIVE_SOURCE = (0.05, 0.03)
CENSUS_SADDLE_MATRIX_ARGS = (1.3, 0.0, 0.0)
CENSUS_SADDLE_SOURCE = (0.1, 0.05)

# =====================================================================
# Build 8f (F028) select_branch routing fixtures.  WP1 gave
# `_positive_parity_grid` a geometric branch routed through
# `select_branch`; WP2 routed `_saddle_grid`'s above-ceiling geometric
# decision through the SAME `select_branch` predicate (boundary
# preserving).  ONE-HOME PREDICATE AGREEMENT IS NOT PINNED HERE: it is
# the canonical pin of
# `test_lensing_operator.BranchGateTestCase.test_thresholds_have_one_home`,
# which sweeps both operator grids against `select_branch` (all three
# legs, including `eta`).  The tests below pin the remaining, distinct
# claims: exact geometric serve on the F028 table configs, a
# below-ceiling accuracy anchor, below-ceiling byte-identity, the saddle
# serve boundary, the delta_min single-solve budget, and the three
# above-ceiling 'wave' outcomes.  All fixture branch labels were
# MEASURED (probe 2026-07-28), never assumed from the brief's named
# coordinates.
# =====================================================================

#: Below-ceiling nodes (acceptance #1): the exact wave batch must not move.
BELOW_CEILING_WS = (5.0, 40.0, 59.0)

#: F028 table configs (acceptance #3): positive parity, ``|y|`` MEASURED
#: (probe 2026-07-28) to be BOTH resolved AND select_branch-geometric so
#: the grid serves `geometric_amplification` exactly.  The on-axis brief
#: coordinates gave ``delta_min = 0`` (unresolved -> 'wave'); an OFF-axis
#: source restores a resolved 4-image geometry.
F028_SERVE = (  # (gamma, w, y)
    (0.70, 70.0, (1.0, 0.7)),
    (0.70, 500.0, (1.0, 0.7)),
    (0.90, 500.0, (1.5, 1.0)),
)
#: Same configs BELOW the ceiling (acceptance: geometric-vs-quadrature
#: anchor).  Here the Schwinger quadrature is a legitimate oracle, so the
#: geometric asymptote is anchored to `f_schwinger` at these w.
F028_ANCHOR = (  # (gamma, y)
    (0.70, (1.0, 0.7)),
    (0.90, (1.5, 1.0)),
)
F028_ANCHOR_WS = (45.0, 55.0, 60.0)
#: The asymptote is a leading-order stationary-phase approximation, so
#: near the ceiling it agrees with the exact quadrature only to a few
#: parts in 1e4; the spec upper bound (~1e-4) with a comfortable margin
#: (measured worst ~4.4e-6) documents this is an ACCURACY anchor, not a
#: certification claim.
F028_ANCHOR_TOL = 1e-4

#: Byte-identity reference (acceptance #1), captured as exact `float.hex()`
#: literals of the served F_op value BELOW the ceiling.  Frozen from BOTH
#: the pre-build (HEAD, git-worktree) and post-build trees, verified
#: IDENTICAL 2026-07-28 -- the select_branch insertion perturbs nothing
#: below w = 60.  (label, gamma, y, beta, kappa) -> {w: (re_hex, im_hex)}.
#: NEVER import a module from a prior revision to regenerate this (F022).
BYTEFREEZE_CONFIGS = (  # (gamma, y, beta, kappa)
    (0.2, (0.03, 0.04), 0.0, 0.0),
    (0.5, (0.18, 0.24), 0.7, 0.3),
    (0.9, (0.6, 0.8), 0.0, 0.0),
    (1.2, (0.18, 0.24), 0.0, 0.0),
    (2.0, (0.6, 0.8), 0.7, 0.0),
)
BYTEFREEZE_REFERENCE = {
    (0.2, (0.03, 0.04), 0.0, 0.0): {
        5.0: ('-0x1.7bdfd68459dc8p-2', '0x1.db774973d8f16p+1'),
        40.0: ('-0x1.a01871a342ea7p-1', '0x1.2d26878ef9dbfp-3'),
        59.0: ('0x1.3a31ed6b9835ap+0', '-0x1.0daa9d4839567p-2')},
    (0.5, (0.18, 0.24), 0.7, 0.3): {
        5.0: ('0x1.83551ffafbc32p-1', '0x1.0d39084dc0090p+2'),
        40.0: ('-0x1.4987ed76bbbf0p+0', '-0x1.1dc1d4d9d2c02p-1'),
        59.0: ('-0x1.380cac59cb467p+0', '-0x1.3646ca36ce5bfp-1')},
    (0.9, (0.6, 0.8), 0.0, 0.0): {
        5.0: ('-0x1.9cd1cb4cdde5ep+0', '0x1.9eb6646d5846cp+0'),
        40.0: ('-0x1.6e7bdca33fcb1p+1', '0x1.35e0f5e1fd3cfp-1'),
        59.0: ('0x1.2cf5933db121cp-3', '0x1.61098fb994692p+0')},
    (1.2, (0.18, 0.24), 0.0, 0.0): {
        5.0: ('-0x1.a87905cdc357ep-2', '0x1.a164b635e2c0dp-4'),
        40.0: ('-0x1.7b4202339d504p-4', '-0x1.3716832a74d41p-1'),
        59.0: ('0x1.20298e9a20a2dp-1', '0x1.0ba8eea8b55bap-2')},
    (2.0, (0.6, 0.8), 0.7, 0.0): {
        5.0: ('0x1.6b503b3c7eba2p-2', '0x1.43ce298db67bcp-4'),
        40.0: ('-0x1.4e34796c43220p-5', '-0x1.54e31a62cb16ep-6'),
        59.0: ('-0x1.d23811ac54f32p-3', '0x1.127cc1ab82d5ap-2')},
}

#: Saddle serve-boundary fixtures (anti-variant): a RESOLVED saddle config
#: straddling the ceiling.  Both w must be geometric-served, proving the
#: boundary stayed at ``w > 60`` (NOT the rejected ``pi w / 4 > 48`` ->
#: ``w > 61.115`` variant, which would leave w = 60.5 AND w = 61.5 on
#: different sides / un-served).
SADDLE_BOUNDARY = (  # (gamma, y)
    (1.2, (0.18, 0.24)),
    (2.0, (0.18, 0.24)),
)
SADDLE_BOUNDARY_WS = (60.5, 61.5)

#: Three above-ceiling 'wave' outcomes (acceptance #4/#5), MEASURED
#: 2026-07-28.  Fold-Airy-served, Pearcey-cusp-served, and both-arms-
#: refuse (named refusal with the lowest-index refuser's authentic
#: `f_schwinger` message).  All positive parity, w = 61 (just above the
#: ceiling), ce < L_MAX so select_branch routes to the wave arm.
THREE_OUTCOME_W = 61.0
THREE_OUTCOME_FOLD = (0.47, (0.4, 0.3))      # fold certifies, cusp does not
THREE_OUTCOME_CUSP = (0.5, (0.5, 0.1))       # cusp certifies, fold does not
THREE_OUTCOME_REFUSE = (0.5, (0.3, 0.2))     # neither arm certifies


# ---------------------------------------------------------------------
# The independent mpmath oracle path (AST-guarded: pure math + mpmath).
# ---------------------------------------------------------------------

def _oracle_1d(w, y1, y2, a, b):
    """
    Evaluate the 1D Schwinger representation in pure mpmath.

    ``F = (w / 2 pi i) e^{i w |y|^2 / 2} (pi / Gamma(s)) I`` with
    ``s = iw/2`` and ``I = Int_0^inf t^{s-1} h dt`` regularized by one
    integration by parts (the ``t = 0`` boundary term vanishing by the
    analytic continuation defining the identity at ``Re s = 0``)::

        I = T^s h(T)/s - (1/s) Int_0^T t^s h' dt
            + Int_T^inf t^{s-1} h dt,

    both integrals absolutely convergent in ``u = ln t`` (the first
    decays as ``e^u`` toward ``u -> -inf``, the tail as ``e^{-u}``).
    ``h'`` is hand-differentiated here from the ``h`` above -- shared
    MATHEMATICS with production, zero shared CODE.  Valid at both
    parities (used with ``a = b = 1`` for the point-mass
    certification).
    """
    dps = ORACLE_DPS_BASE + int(math.ceil(w))
    with mpmath.workdps(dps):
        w_ = mpmath.mpf(w)
        s = mpmath.mpc(0, w_ / 2)
        branch_a = mpmath.mpc(0, w_ * mpmath.mpf(a) / 2)
        branch_b = mpmath.mpc(0, w_ * mpmath.mpf(b) / 2)
        amp1 = (w_ * mpmath.mpf(y1)) ** 2 / 4
        amp2 = (w_ * mpmath.mpf(y2)) ** 2 / 4

        def kernel(t):
            da = t - branch_a
            db = t - branch_b
            return (mpmath.exp(-amp1 / da - amp2 / db)
                    / (mpmath.sqrt(da) * mpmath.sqrt(db)))

        def kernel_derivative(t):
            da = t - branch_a
            db = t - branch_b
            return kernel(t) * (amp1 / da ** 2 + amp2 / db ** 2
                                - 1 / (2 * da) - 1 / (2 * db))

        t_cap = w_ * (abs(mpmath.mpf(a)) + abs(mpmath.mpf(b)) + 2) / 2
        u_mid = mpmath.log(t_cap)
        margin = mpmath.pi * w_ / 4 + ORACLE_EXTRA_MARGIN
        wavelength = 4 * mpmath.pi / w_
        n_panels = max(
            ORACLE_MIN_PANELS,
            int(mpmath.ceil(margin / (ORACLE_WAVELENGTHS_PER_PANEL
                                      * wavelength))))
        part_a = mpmath.quad(
            lambda u: (mpmath.exp((s + 1) * u)
                       * kernel_derivative(mpmath.exp(u))),
            mpmath.linspace(u_mid - margin, u_mid, n_panels + 1),
            maxdegree=ORACLE_MAXDEGREE)
        tail = mpmath.quad(
            lambda u: mpmath.exp(s * u) * kernel(mpmath.exp(u)),
            mpmath.linspace(u_mid, u_mid + margin, n_panels + 1),
            maxdegree=ORACLE_MAXDEGREE)
        raw = t_cap ** s * kernel(t_cap) / s - part_a / s + tail

        prefactor = mpmath.mpc(0, -w_ / 2)  # (w / 2 pi i) * pi
        source_phase = mpmath.exp(
            1j * w_ * (mpmath.mpf(y1) ** 2 + mpmath.mpf(y2) ** 2) / 2)
        result = prefactor * source_phase * raw / mpmath.gamma(s)
    return result


@functools.lru_cache(maxsize=None)
def _oracle_saddle(w, y1, y2, gamma_prime):
    """Cached saddle-domain oracle: ``a = 1 - g'``, ``b = 1 + g'``."""
    return _oracle_1d(w, y1, y2, 1.0 - gamma_prime, 1.0 + gamma_prime)


def _oracle_point_mass(w, y1, y2):
    """
    The point-mass (``a = b = 1``) closed form, INDEPENDENT of the 1D
    representation: ``e^{pi w/4 + i (w/2) ln(w/2)} Gamma(1 - iw/2)
    1F1(iw/2; 1; i w |y|^2 / 2)`` (the `test_lensing_hyp1f1` carrier
    idiom), used only to CERTIFY `_oracle_1d`.
    """
    with mpmath.workdps(2 * ORACLE_DPS_BASE + int(math.ceil(w))):
        w_ = mpmath.mpf(w)
        y_sq = mpmath.mpf(y1) ** 2 + mpmath.mpf(y2) ** 2
        carrier = mpmath.exp(mpmath.pi * w_ / 4
                             + 1j * (w_ / 2) * mpmath.log(w_ / 2))
        result = (carrier * mpmath.gamma(1 - 1j * w_ / 2)
                  * mpmath.hyp1f1(1j * w_ / 2, 1, 1j * w_ * y_sq / 2))
    return result


def _reconstructed_dispatch_oracle(w, y, gamma, beta, kappa):
    """
    The value `operator.F_op` MUST return on the WP2 positive-parity
    strong-shear fallback, built INDEPENDENTLY of production.

    The mass-sheet reduction (``lam = 1 - kappa``, ``y' = y/sqrt(lam)``,
    ``gamma' = gamma/lam``, eigenframe rotation by ``beta``) is done here
    in plain arithmetic -- NOT via `operator._mass_sheet_map` -- so the
    only production-derived quantity anywhere near this oracle is nil.
    The amplification itself is the AST-guarded pure-mpmath
    ``F_{0, gamma'}(w, y_eig)`` (`_oracle_saddle`); the mass-sheet
    prefactor ``(1/lam) e^{0.5i w ln lam - 0.5i w kappa s}`` is the exact
    identity the operator path reconstructs with (the prefactor is a
    common complex factor, so it cancels in the relative error and cannot
    mask a discrepancy in the amplification itself).
    """
    lam = 1.0 - float(kappa)
    root_lam = math.sqrt(lam)
    y_scaled = (y[0] / root_lam, y[1] / root_lam)
    gamma_prime = float(gamma) / lam
    z_eig = cmath.exp(-1j * float(beta)) * complex(y_scaled[0], y_scaled[1])
    s = y_scaled[0] ** 2 + y_scaled[1] ** 2
    amplification = _oracle_saddle(w, z_eig.real, z_eig.imag, gamma_prime)
    prefactor = cmath.exp(0.5j * w * math.log(lam)
                          - 0.5j * w * float(kappa) * s) / lam
    return mpmath.mpc(prefactor) * amplification


def _referenced_names(func):
    """
    Return every name a function's own source references (the
    `test_lensing_gauge` / `test_lensing_channels` idiom): the
    ``ast.Import`` / ``ast.ImportFrom`` walk extended with ``ast.Name``
    ids and ``ast.Attribute`` attribute names, so a forbidden
    dependency entering as ``_schwinger.f_schwinger`` or a bare name is
    caught, not only as an import statement.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or '').split('.')[0])
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    names.discard('')
    return names


class SchwingerTestCase(TestCase):
    """
    Base class carrying the mpmath comparison and the anti-vacuity
    tally (`tearDown` fails a test that asserted nothing).
    """

    _expect_checks = True

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self._expect_checks and self.n_checks == 0:
            self.fail('vacuous test: no comparison ran, so nothing was '
                      'asserted')

    def assert_close(self, got, exact, tol, msg=''):
        """
        Assert `got` matches the mpmath `exact` to relative `tol`; bump
        `n_checks` BEFORE asserting (so an expected failure still
        satisfies the anti-vacuity tally) and return the relative
        error.
        """
        rel = abs(mpmath.mpc(got) - exact) / abs(exact)
        self.n_checks += 1
        self.assertLessEqual(
            rel, mpmath.mpf(tol),
            f'{msg}: relative error {mpmath.nstr(rel, 5)} > {tol}')
        return float(rel)


class OracleImportGuardTestCase(SchwingerTestCase):
    """The oracle path must not touch production code (F002)."""

    _ORACLE_PATH = (_oracle_1d, _oracle_saddle, _oracle_point_mass,
                    _reconstructed_dispatch_oracle)

    def test_oracle_path_references_no_production_names(self):
        """
        Neither the 1D oracle, its cached saddle wrapper, nor the
        point-mass certifier references any name from
        `cogwheel.lensing.chang_refsdal` (or numpy/numba, whose
        presence would signal a float64 shortcut inside the oracle).
        """
        for func in self._ORACLE_PATH:
            overlap = _referenced_names(func) & FORBIDDEN_ORACLE_NAMES
            self.n_checks += 1
            self.assertFalse(
                overlap,
                f'oracle function {func.__name__} references forbidden '
                f'production names {sorted(overlap)}; the oracle is not '
                'independent and its gates are tautological (F002)')

    def test_guard_itself_can_go_red(self):
        """
        A function that DOES reach production is caught by the same
        checker, so the guard above is not vacuous.
        """
        def tainted():
            return _schwinger.f_schwinger  # forbidden on both counts
        overlap = _referenced_names(tainted) & FORBIDDEN_ORACLE_NAMES
        self.n_checks += 1
        self.assertTrue(
            overlap,
            'the AST guard failed to flag a function that references '
            'production; the import-guard test cannot go red')


class OracleCertificationTestCase(SchwingerTestCase):
    """
    Certify the oracle BEFORE it judges production (research note
    Sec. 12: "certify the oracle against the closed forms").
    """

    def test_point_mass_closed_form(self):
        """
        At ``a = b = 1`` (pure point mass, zero shear) the 1D oracle
        reproduces the independent 1F1 closed form at ``w = 10`` (the
        brief's certification point) and at ``w = 30`` -- the latter
        proves the oracle's quadrature survives the ``e^{pi w/4}``
        cancellation regime in which it convicts production.
        """
        for w in (10.0, 30.0):
            got = _oracle_1d(w, 0.4, 0.3, 1.0, 1.0)
            exact = _oracle_point_mass(w, 0.4, 0.3)
            self.assert_close(got, exact, PM_CERT_RTOL,
                              f'oracle vs point-mass closed form, w={w}')

    def test_build_anchor(self):
        """
        The oracle reproduces the literal Build-6 anchor value
        ``F(3, (0.4, 0.3), gamma'=1.3)`` (independently validated to
        2.2e-15 against the 2D lens-plane oracle, research Sec. 6.2).
        """
        got = _oracle_saddle(ANCHOR_W, *ANCHOR_Y, ANCHOR_GAMMA)
        self.assert_close(complex(got), mpmath.mpc(ANCHOR_VALUE),
                          ANCHOR_RTOL, 'oracle vs literal 2D-validated '
                          'anchor')

    def test_production_matches_anchor(self):
        """`f_schwinger` itself reproduces the literal anchor."""
        got = f_schwinger(ANCHOR_W, np.array(ANCHOR_Y), ANCHOR_GAMMA)
        self.assert_close(got, mpmath.mpc(ANCHOR_VALUE), ANCHOR_RTOL,
                          'production vs literal anchor')


class CertifiedBandGridTestCase(SchwingerTestCase):
    """Production vs the certified dev-oracle on the certified band."""

    def test_certified_band_matches_oracle(self):
        """
        ``w <= 10`` x ``gamma' in {1.05, 1.3, 2.0}`` x three source
        positions: relative error < 1e-10 (worst measured 8.3e-13, so
        > 100x headroom; the 1.05 column approaches the parity-boundary
        pinch, the 2.0 column the strong-shear side).
        """
        for w in GRID_W_CERTIFIED:
            for gamma_prime in GRID_GAMMA:
                for y in GRID_Y:
                    with self.subTest(w=w, gamma_prime=gamma_prime, y=y):
                        exact = _oracle_saddle(w, y[0], y[1], gamma_prime)
                        got = f_schwinger(w, np.array(y), gamma_prime)
                        self.assert_close(
                            got, exact, GRID_RTOL,
                            f'w={w}, gamma\'={gamma_prime}, y={y}')


class HighBandKnownDefectTestCase(SchwingerTestCase):
    """
    The high band ``w in {20, 30, 45}`` (plus the ``w = 55`` spot) at
    the module's OWN advertised accuracy.

    See the module docstring ("HIGH-BAND HISTORY"): two eps_f64-class
    N/2N-invisible systematics (the ``t_cap``-vs-``e^{u_mid}`` IBP
    endpoint/split mismatch and the float64 ``fl(1/half_w)`` reciprocal
    on the endpoint and A pieces), each amplified by ``e^{pi w/4}`` on
    reconstruction, used to break this band; both were fixed in the
    core on 2026-07-19 and these tests now gate the fixed contract (the
    docstring's 1e-10-to-the-ceiling claim and the dd law's 1e-6 at
    w=55) as plain green tests.  Do NOT widen these tolerances.
    """

    def test_high_band_grid_meets_spec(self):
        """
        Same grid gate as the certified band (1e-10 relative against
        the mpmath oracle), on ``w in {20, 30, 45}``.
        """
        for w in GRID_W_HIGH:
            for gamma_prime in GRID_GAMMA:
                for y in ((0.1, 0.1), (0.4, 0.3), (1.0, 0.0)):
                    exact = _oracle_saddle(w, y[0], y[1], gamma_prime)
                    got = f_schwinger(w, np.array(y), gamma_prime)
                    self.assert_close(
                        got, exact, GRID_RTOL,
                        f'w={w}, gamma\'={gamma_prime}, y={y} '
                        '(high band, module docstring HIGH-BAND HISTORY)')

    def test_high_w_spot_check(self):
        """
        Single spot at ``w = 55`` against the RELAXED 1e-6 tolerance
        the dd cancellation law predicts near the ceiling (measured
        rel 5.6e-13 post-fix).
        """
        exact = _oracle_saddle(HIGH_W_SPOT, 0.4, 0.3, 1.3)
        got = f_schwinger(HIGH_W_SPOT, np.array([0.4, 0.3]), 1.3)
        self.assert_close(got, exact, HIGH_W_SPOT_RTOL,
                          f'w={HIGH_W_SPOT} spot check')


class DeepBandTestCase(SchwingerTestCase):
    """
    F009-S deep-band pins at ``gamma' = 1.3`` (``a = -0.3, b = 2.3``),
    ``y = (0.4, 0.3)``.  Both oracles are LITERAL closed forms built
    from raw eigenvalues -- never from the module (F002); F009's lesson
    applies verbatim: the limit is ``sqrt(|mu_macro|)``, not 1, and the
    Morse phase ``-pi/2`` must be pinned alongside the magnitude.
    """

    def test_magnitude_approaches_literal_closed_form(self):
        """
        ``|F(w -> 0)| -> 1 / sqrt(|a b|)`` with an O(w) correction:
        rel < 1e-3 at ``w = 1e-4`` (measured 4.4e-5), and the residual
        DECREASES monotonically toward small ``w`` across three decades
        -- the linear-vanishing signature that separates the exact
        limit from a plateau.
        """
        eig_a = 1.0 - DEEP_GAMMA   # raw eigenvalues, never the module
        eig_b = 1.0 + DEEP_GAMMA
        closed = 1.0 / math.sqrt(abs(eig_a * eig_b))
        residuals = []
        for w in DEEP_WS:
            value = f_schwinger(w, np.array(DEEP_Y), DEEP_GAMMA)
            residuals.append(abs(abs(value) - closed) / closed)
            self.n_checks += 1
        self.assertLessEqual(
            residuals[0], DEEP_MAGNITUDE_RTOL,
            f'|F| at w={DEEP_WS[0]} misses the literal macro-'
            f'magnification limit 1/sqrt|ab|: rel {residuals[0]:.3e}')
        self.assertLess(
            residuals[0], residuals[1],
            'deep-band |F| residual does not shrink from w=1e-3 to '
            'w=1e-4; the macro limit is not being approached')
        self.assertLess(
            residuals[1], residuals[2],
            'deep-band |F| residual does not shrink from w=1e-2 to '
            'w=1e-3; the macro limit is not being approached')

    def test_morse_phase_intercept(self):
        """
        The saddle Morse phase: fitting ``arg F = phi0 + a1 w ln(w/2)
        + a2 w`` (the F009-S drift model, the ``w ln(w/2)`` term being
        the point-mass core normalization -- NOT a defect) over the
        three deep-band frequencies extrapolates to ``phi0 = -pi/2``
        within 5e-4 (measured residual 1.5e-7).
        """
        phases = []
        design = []
        for w in DEEP_WS:
            value = f_schwinger(w, np.array(DEEP_Y), DEEP_GAMMA)
            phases.append(cmath.phase(value))
            design.append([1.0, w * math.log(w / 2.0), w])
        intercept = np.linalg.solve(np.array(design),
                                    np.array(phases))[0]
        self.n_checks += 1
        self.assertLess(
            abs(intercept + math.pi / 2), MORSE_PHASE_TOL,
            f'Morse-phase intercept {intercept:.8f} is not -pi/2 '
            f'within {MORSE_PHASE_TOL} (got residual '
            f'{abs(intercept + math.pi / 2):.3e}); the saddle '
            'e^{-i pi/2} deep-band phase law is violated')


class CertifyXorRefuseTestCase(SchwingerTestCase):
    """
    The evaluator either returns a FINITE certified value or raises the
    named `SchwingerCertificationError` -- never NaN, never inf, never
    an anonymous error.  (Accuracy of the returned high-``w`` values is
    the separate `HighBandKnownDefectTestCase`.)
    """

    def _assert_finite_return(self, w):
        """`f_schwinger` returns and the value is finite (no NaN/inf)."""
        value = f_schwinger(w, np.array(XOR_Y), XOR_GAMMA)
        self.n_checks += 1
        self.assertTrue(
            math.isfinite(value.real) and math.isfinite(value.imag),
            f'non-finite certified value {value} at w = {w}: the '
            'certify-XOR-refuse contract is violated (F005-S)')
        return value

    def _assert_named_refusal(self, w, *tokens):
        """The exact named error is raised, its message naming each
        token; no value (finite or otherwise) escapes."""
        with self.assertRaises(SchwingerCertificationError) as ctx:
            f_schwinger(w, np.array(XOR_Y), XOR_GAMMA)
        exc = ctx.exception
        self.n_checks += 1
        self.assertIs(type(exc), SchwingerCertificationError,
                      'raised something other than the named error')
        message = str(exc)
        for token in tokens:
            self.assertIn(token, message,
                          f'refusal does not name {token!r}: {message}')

    def test_error_type_contract(self):
        """`SchwingerCertificationError` is a RuntimeError (a refusal,
        not an input error) and domain errors stay `ValueError`.

        The bad ``gamma_prime`` cases are ``gamma_prime <= 0`` ONLY
        (``det A == 0`` or the wrong sign): after the WP2 guard
        relaxation the positive-parity band ``0 < gamma_prime < 1`` is a
        VALID domain that certifies (see `GuardRelaxationTestCase`), so
        it must NOT be listed here as a domain error.
        """
        self.n_checks += 1
        self.assertTrue(
            issubclass(SchwingerCertificationError, RuntimeError))
        for bad_args in ((0.0, XOR_Y, XOR_GAMMA),
                         (-1.0, XOR_Y, XOR_GAMMA),
                         (3.0, XOR_Y, 0.0),
                         (3.0, XOR_Y, -0.5)):
            with self.assertRaises(ValueError) as ctx:
                f_schwinger(bad_args[0], np.array(bad_args[1]),
                            bad_args[2])
            self.n_checks += 1
            self.assertNotIsInstance(
                ctx.exception, SchwingerCertificationError,
                'a domain error leaked out as the certification '
                'refusal; the two error surfaces must stay distinct')

    def test_certified_band_returns_finite(self):
        """``w in {10, 30, 50, 59.9}`` return finite certified
        values."""
        for w in CERTIFIED_W_SWEEP:
            self._assert_finite_return(w)

    def test_above_ceiling_refuses(self):
        """``w in {60.5, 65, 80}`` raise the named refusal, naming both
        the offending ``w`` and the ceiling."""
        for w in REFUSED_W_SWEEP:
            self._assert_named_refusal(w, str(w),
                                       str(W_CEILING_SCHWINGER))

    def test_ceiling_boundary_is_not_off_by_one(self):
        """
        ``w = W_CEILING_SCHWINGER`` exactly still EVALUATES (the refuse
        condition is strict ``w > ceiling``) and one ulp above refuses
        -- the F004 float64-exact-boundary lesson (60.0 is exact in
        float64, so this boundary is testable bit-for-bit).
        """
        self._assert_finite_return(W_CEILING_SCHWINGER)
        self._assert_named_refusal(
            np.nextafter(W_CEILING_SCHWINGER, np.inf),
            str(W_CEILING_SCHWINGER))


class DdMandatoryFalsificationTestCase(SchwingerTestCase):
    """
    Prove float64 fabrication is real and the gates can go red (the
    F005-S analog, via the F010 ``py_func`` idiom: numba freezes module
    globals at compile time, so every perturbation is injected by
    swapping the njit core for its ``.py_func`` body, which re-reads
    the module globals in the interpreter).
    """

    def _gate_outcome(self):
        """Run the FALS config; return ``(raised, rel_err)`` against
        the certified oracle (``rel_err = inf`` on refusal)."""
        try:
            got = f_schwinger(FALS_W, np.array(FALS_Y), FALS_GAMMA)
        except SchwingerCertificationError:
            return True, float('inf')
        exact = _oracle_saddle(FALS_W, *FALS_Y, FALS_GAMMA)
        rel = float(abs(mpmath.mpc(got) - exact) / abs(exact))
        return False, rel

    def _assert_green(self, label):
        """The gate must be green here, so a later RED is the patch's
        doing."""
        raised, rel = self._gate_outcome()
        self.n_checks += 1
        self.assertFalse(
            raised, f'{label}: f_schwinger refused the certified FALS '
            'config; the falsification precondition is broken')
        self.n_checks += 1
        self.assertLessEqual(
            rel, FALS_RTOL,
            f'{label}: rel error {rel:.3e} already exceeds '
            f'{FALS_RTOL:.0e}; the gate is not green to begin with')

    def test_float64_dd_accumulation_drives_gate_red(self):
        """
        Collapsing the dd-complex accumulation to plain float64
        (replacing `dd_complex_add` through the core's ``py_func``)
        must drive the ``w = 30`` gate RED -- here the engine's own
        paired-rule certification fires (float64 quadrature noise at
        ``eps_f64 * e^{pi w/4}`` differs between the N and 2N rules),
        which IS the designed named refusal for a float64 substrate.
        The uncorrupted ``py_func`` chain stays green, so RED is the
        corruption's doing, not the interpretation's.
        """
        self._assert_green('unpatched')

        core_pyfunc = _schwinger._raw_t_integral_core.py_func
        self.n_checks += 1
        self.assertFalse(
            hasattr(core_pyfunc, 'signatures'),
            '_raw_t_integral_core.py_func carries .signatures; it is '
            'not a plain py_func body, so the perturbation would not '
            'reach compiled code (F010 vacuity)')

        with mock.patch.object(_schwinger, '_raw_t_integral_core',
                               core_pyfunc):
            self._assert_green('uncorrupted py_func chain')

        def float64_complex_add(are_hi, are_lo, aim_hi, aim_lo,
                                bre_hi, bre_lo, bim_hi, bim_lo):
            """A float64 accumulator wearing the dd calling
            convention."""
            return (are_hi + are_lo + bre_hi + bre_lo, 0.0,
                    aim_hi + aim_lo + bim_hi + bim_lo, 0.0)

        with mock.patch.object(_schwinger, '_raw_t_integral_core',
                               core_pyfunc), \
                mock.patch.object(_schwinger, 'dd_complex_add',
                                  float64_complex_add):
            raised, rel = self._gate_outcome()
        print(f'\n[Falsification] float64 dd_complex_add: '
              f'raised={raised} rel_err={rel:.3e}')

        self.n_checks += 1
        self.assertTrue(
            raised or rel > FALS_RTOL,
            f'a float64-collapsed dd accumulation still certified '
            f'(rel_err {rel:.3e} <= {FALS_RTOL:.0e}); the dd substrate '
            'is not load-bearing or the py_func chain is incomplete '
            '(F010)')

    def test_perturbed_ceiling_refuses_previously_certified_w(self):
        """
        Lowering `W_CEILING_SCHWINGER` to 20 makes the previously
        certified ``w = 30`` config refuse by name.  `f_schwinger` is
        an interpreted function (asserted), so the module-global patch
        provably reaches it -- no compiled copy can hold the old
        ceiling (F010).
        """
        self.n_checks += 1
        self.assertFalse(
            hasattr(f_schwinger, 'signatures'),
            'f_schwinger appears to be numba-compiled; a module-global '
            'ceiling patch would not reach it (F010) and this '
            'falsification would be vacuous')
        self._assert_green('unpatched')
        with mock.patch.object(_schwinger, 'W_CEILING_SCHWINGER',
                               PERTURBED_CEILING):
            with self.assertRaises(SchwingerCertificationError) as ctx:
                f_schwinger(FALS_W, np.array(FALS_Y), FALS_GAMMA)
        self.n_checks += 1
        self.assertIn(str(PERTURBED_CEILING), str(ctx.exception),
                      'the refusal does not name the perturbed ceiling')


class WarmCostMeasurementTestCase(SchwingerTestCase):
    """
    Warm per-point cost: a MEASUREMENT, not a gate (it prices the
    envelope-surrogate decision; run pytest with ``-s`` to see the
    numbers on a passing run).
    """

    def test_report_warm_per_point_cost(self):
        summary = _schwinger._measure_warm_cost()
        for key in ('n_points', 'mean_ms', 'min_ms', 'max_ms'):
            self.n_checks += 1
            self.assertIn(key, summary)
            self.assertTrue(math.isfinite(summary[key]))
            self.assertGreater(summary[key], 0.0)

        lines = []
        for w in (10.0, 30.0):
            f_schwinger(w, np.array([0.4, 0.3]), 1.3)  # warm
            best = math.inf
            for _ in range(5):
                start = time.perf_counter()
                f_schwinger(w, np.array([0.4, 0.3]), 1.3)
                best = min(best, time.perf_counter() - start)
            lines.append(f'w={w:g}: {1e3 * best:.1f} ms/point')
        print('\n[test_lensing_schwinger] WARM PER-POINT COST '
              '(envelope-surrogate pricing) | ' + ' | '.join(lines))


class DispatchFallbackOracleTestCase(SchwingerTestCase):
    """
    Positive-parity strong-shear dispatch accuracy.

    `operator.F_op` / `operator.F_op_grid` serve a positive-parity node
    through the exact 1D evaluator and reconstruct it with the mass-sheet
    identity.  Each rerouted value is judged against the INDEPENDENT
    reconstructed mpmath oracle at the Professor-staked 1e-10 (worst
    measured 1.7e-11 at ``w = 59.9``).
    """

    def test_fop_matches_reconstructed_oracle(self):
        """
        For every dispatch fixture `F_op` returns the reconstructed
        pure-shear oracle within 1e-10, uniformly across ``w`` in
        [3, 59.9] and both ``kappa = 0`` and ``kappa != 0``.
        """
        for gamma, beta, kappa, y, w in DISPATCH_POINTS:
            with self.subTest(gamma=gamma, kappa=kappa, y=y, w=w):
                value, _ = F_op(w, np.asarray(y), gamma,
                                beta=beta, kappa=kappa)
                oracle = _reconstructed_dispatch_oracle(
                    w, y, gamma, beta, kappa)
                self.assert_close(
                    value, oracle, DISPATCH_RTOL,
                    f'F_op vs reconstructed oracle at w={w}, '
                    f'y={y}, gamma={gamma}, kappa={kappa}')

    def test_grid_and_scalar_fallback_agree(self):
        """
        `F_op_grid` and `F_op` route the SAME fallback: a one-element
        grid returns byte-identical to the scalar entry point (FINDINGS
        F005: one contraction, one certification, one fallback).
        """
        for gamma, beta, kappa, y, w in DISPATCH_POINTS[:4]:
            with self.subTest(gamma=gamma, kappa=kappa, y=y, w=w):
                scalar, _ = F_op(w, np.asarray(y), gamma,
                                 beta=beta, kappa=kappa)
                grid, _, _ = F_op_grid(np.asarray([float(w)]),
                                       np.asarray(y), gamma,
                                       beta=beta, kappa=kappa)
                self.n_checks += 1
                self.assertEqual(
                    complex(grid[0]), scalar,
                    f'grid and scalar fallback disagree at w={w}')

    def test_dispatch_accuracy_diagnostic_plot(self):
        """
        Sweep ``w`` at one fallback config and write the rel-error-vs-``w``
        diagnostic (log axis, with the 1e-10 line) to
        ``output/dispatch_fallback_oracle_accuracy.png``, revealing any
        cancellation-driven degradation as ``w -> 60``.  The plot is a
        MEASUREMENT; the accompanying assertion keeps it non-vacuous.
        """
        gamma, y = DISPATCH_PLOT_GAMMA, DISPATCH_PLOT_Y
        rel_errors = []
        for w in DISPATCH_PLOT_WS:
            value, _ = F_op(w, np.asarray(y), gamma)
            oracle = _reconstructed_dispatch_oracle(w, y, gamma, 0.0, 0.0)
            rel_errors.append(self.assert_close(
                value, oracle, DISPATCH_RTOL,
                f'dispatch plot point w={w}'))

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        axis.semilogy(DISPATCH_PLOT_WS, rel_errors, 'o-', label='|F_op - oracle| / |oracle|')
        axis.axhline(DISPATCH_RTOL, color='crimson', linestyle='--',
                     label=f'{DISPATCH_RTOL:.0e} gate')
        axis.set_xlabel('w (dimensionless frequency)')
        axis.set_ylabel('relative error')
        axis.set_title(f'WP2 fallback dispatch accuracy '
                       f'(gamma={gamma}, y={y}, kappa=0)')
        axis.legend()
        axis.grid(True, which='both', alpha=0.3)
        figure.tight_layout()
        figure.savefig(_OUTPUT_DIR / 'dispatch_fallback_oracle_accuracy.png',
                       dpi=110)
        plt.close(figure)


class DispatchSelfFalsificationTestCase(SchwingerTestCase):
    """
    Prove the 1e-10 dispatch gate is DISCRIMINATING: comparing the same
    `F_op` fallback value against an oracle evaluated at a WRONG reduced
    shear must blow the relative error far past the gate.  Without this,
    a gate that always passed (e.g. an oracle that ignored ``gamma'``)
    would read green vacuously.
    """

    def test_wrong_gamma_prime_oracle_fails_the_gate(self):
        gamma, beta, kappa, y, w = 0.47, 0.0, 0.0, (0.4, 0.3), 20.0
        value, _ = F_op(w, np.asarray(y), gamma, beta=beta, kappa=kappa)
        correct = _reconstructed_dispatch_oracle(w, y, gamma, beta, kappa)
        # A deliberately mis-set reduced shear (gamma' shifted by 5%).
        wrong = _reconstructed_dispatch_oracle(
            w, y, gamma * 1.05, beta, kappa)
        rel_correct = abs(mpmath.mpc(value) - correct) / abs(correct)
        rel_wrong = abs(mpmath.mpc(value) - wrong) / abs(wrong)
        self.n_checks += 1
        self.assertLessEqual(float(rel_correct), DISPATCH_RTOL)
        self.n_checks += 1
        self.assertGreater(
            float(rel_wrong), 1e-3,
            'the dispatch gate is not discriminating: F_op agrees with a '
            f'5%-wrong-gamma_prime oracle to {float(rel_wrong):.3e}, so a '
            'green result would carry no information')


class GuardRelaxationTestCase(SchwingerTestCase):
    """
    WP2 relaxed `_schwinger.f_schwinger`'s guard from ``gamma_prime > 1``
    (saddle only) to ``gamma_prime > 0``, admitting positive parity.

    The three arms confirm the one-line relaxation did EXACTLY that:
    ``gamma_prime > 1`` values are byte-frozen (unchanged), ``0 <
    gamma_prime < 1`` is now accepted and correct, and ``gamma_prime <=
    0`` still refuses at the domain surface (`ValueError`, NOT the
    certification refusal).
    """

    def test_saddle_values_are_byte_frozen(self):
        """Every ``gamma_prime > 1`` value equals the pre-build literal
        to full float64 precision (the relaxation must not perturb the
        certified saddle numerics)."""
        for (w, y, gamma_prime), frozen in SADDLE_BITFREEZE.items():
            with self.subTest(w=w, y=y, gamma_prime=gamma_prime):
                value = f_schwinger(w, np.asarray(y), gamma_prime)
                self.n_checks += 1
                self.assertEqual(
                    value, frozen,
                    f'saddle value at w={w}, y={y}, gamma_prime='
                    f'{gamma_prime} drifted from the frozen literal '
                    f'{frozen!r}; the guard relaxation perturbed the '
                    'gamma_prime > 1 path')

    def test_positive_parity_is_accepted_and_matches_oracle(self):
        """``0 < gamma_prime < 1`` now returns a finite certified value
        matching the AST-guarded pure-shear oracle to 1e-10."""
        for w, y, gamma_prime in POSITIVE_PARITY_ACCEPTED:
            with self.subTest(w=w, y=y, gamma_prime=gamma_prime):
                value = f_schwinger(w, np.asarray(y), gamma_prime)
                self.n_checks += 1
                self.assertTrue(
                    math.isfinite(value.real) and math.isfinite(value.imag),
                    f'positive-parity gamma_prime={gamma_prime} returned '
                    f'non-finite {value} at w={w}')
                oracle = _oracle_saddle(w, y[0], y[1], gamma_prime)
                self.assert_close(
                    value, oracle, DISPATCH_RTOL,
                    f'positive-parity f_schwinger vs oracle at w={w}, '
                    f'y={y}, gamma_prime={gamma_prime}')

    def test_nonpositive_gamma_prime_still_raises_valueerror(self):
        """``gamma_prime <= 0`` (``det A == 0`` or wrong sign) is a
        DOMAIN error, distinct from the certification refusal."""
        for gamma_prime in NONPOSITIVE_GAMMA_PRIME:
            with self.subTest(gamma_prime=gamma_prime):
                with self.assertRaises(ValueError) as ctx:
                    f_schwinger(3.0, np.asarray([0.4, 0.3]), gamma_prime)
                self.n_checks += 1
                self.assertNotIsInstance(
                    ctx.exception, SchwingerCertificationError,
                    'a gamma_prime <= 0 domain error leaked out as the '
                    'certification refusal; the surfaces must stay '
                    'distinct')


class PositiveParityBitFreezeTestCase(SchwingerTestCase):
    """
    RE-BASELINE (Build 8d, F017): the moderate-shear positive-parity
    config (``gamma' = 0.2 > 0``) that the LEGACY operator series used to
    certify is now served by the exact Schwinger evaluator on the wave
    branch (``w <= W_CEILING_SCHWINGER``), so its diagnostics report
    ``order_used == 0``.  The frozen literals are re-baselined to the NEW
    Schwinger production values; each carries a contract-flip WITNESS --
    the NEW Schwinger value and the OLD legacy literal (the retired
    operator-series contraction's recorded output, an INDEPENDENT
    algorithm, F002) agree to `BITFREEZE_WITNESS_TOL` in the
    max-normalized currency, proving the flip moved bytes, not physics.
    A companion pin keeps the shear-free ``gamma' == 0`` point lens on
    its own non-Schwinger exit (``order_used > 0``).
    """

    def test_new_schwinger_agrees_with_old_legacy_max_normalized(self):
        """Contract-flip WITNESS: the NEW Schwinger production values and
        the OLD legacy literals agree to `BITFREEZE_WITNESS_TOL` in the
        max-normalized currency -- the re-baseline is a byte flip, not a
        physics change."""
        w_array = np.asarray(sorted(CERTIFIED_BITFREEZE))
        new_arr, _orders, _conv = F_op_grid(
            w_array, np.asarray(CERTIFIED_BITFREEZE_Y),
            CERTIFIED_BITFREEZE_GAMMA)
        old_arr = np.asarray(
            [LEGACY_BITFREEZE[float(w)] for w in w_array], dtype=complex)
        scale = max(float(np.max(np.abs(old_arr))), 1e-15)
        metric_re = float(
            np.max(np.abs(new_arr.real - old_arr.real))) / scale
        metric_im = float(
            np.max(np.abs(new_arr.imag - old_arr.imag))) / scale
        self.n_checks += 1
        self.assertLess(
            max(metric_re, metric_im), BITFREEZE_WITNESS_TOL,
            f'NEW-vs-OLD disagreement {max(metric_re, metric_im):.3e} '
            f'exceeds the {BITFREEZE_WITNESS_TOL:.0e} byte-flip currency '
            f'(scale={scale:.4f}) -- this is a PHYSICS regression, not a '
            'byte re-baseline')

    def test_scalar_fop_matches_frozen_literals(self):
        for w, frozen in CERTIFIED_BITFREEZE.items():
            with self.subTest(w=w):
                value, diagnostics = F_op(
                    w, np.asarray(CERTIFIED_BITFREEZE_Y),
                    CERTIFIED_BITFREEZE_GAMMA)
                self.n_checks += 1
                self.assertEqual(
                    value, frozen,
                    f'F_op at w={w} drifted from the re-baselined Schwinger '
                    f'literal {frozen!r}')
                # order_used == 0 proves the SCHWINGER evaluator served this
                # sheared positive-parity node (the legacy series would
                # report order_used > 0).
                self.n_checks += 1
                self.assertEqual(
                    diagnostics.order_used, 0,
                    f'F_op at w={w} reports '
                    f'order_used={diagnostics.order_used} -- a sheared '
                    'positive-parity node must be Schwinger-served '
                    '(order_used == 0) since Build 8d')

    def test_grid_fop_matches_frozen_literals(self):
        w_array = np.asarray(sorted(CERTIFIED_BITFREEZE))
        values, orders, converged = F_op_grid(
            w_array, np.asarray(CERTIFIED_BITFREEZE_Y),
            CERTIFIED_BITFREEZE_GAMMA)
        for index, w in enumerate(w_array):
            with self.subTest(w=float(w)):
                self.n_checks += 1
                self.assertEqual(
                    complex(values[index]), CERTIFIED_BITFREEZE[float(w)],
                    f'certified F_op_grid at w={w} drifted from the '
                    're-baselined Schwinger literal')
                self.n_checks += 1
                self.assertEqual(
                    int(orders[index]), 0,
                    f'F_op_grid node w={w} reports order {int(orders[index])}'
                    ' -- a sheared positive-parity node must be '
                    'Schwinger-served (order 0) since Build 8d')
                self.n_checks += 1
                self.assertTrue(bool(converged[index]))

    def test_shear_free_point_lens_uses_the_closed_form(self):
        """Companion pin: the shear-free ``gamma' == 0`` point lens is
        served by the point-mass CLOSED FORM, not by any series.

        RE-BASELINE. This previously asserted ``order_used > 0`` -- that
        the shear-free config stayed on the legacy operator series, which
        was then its sole production exit. The series has been retired
        from this route: at ``gamma' = 0`` the shear operator
        ``exp[i*gamma*D_beta/(2w)]`` is the IDENTITY, so the series
        collapsed to its zeroth term, which is exactly the point-mass
        kernel `point_mass_g_derivatives` already computes. The serve is
        now that kernel times the mass-sheet prefactor.

        Equivalence is not assumed: the SHA-pinned byte-identity tests in
        `test_lensing_fast_path.py::OperatorFusionByteIdentityTestCase`
        compare the served amplification against the pre-change module and
        find it byte-identical. Only ``order_used`` moved (9 -> 0), which
        is the intended signal that no operator series ran.

        The Schwinger 1D representation still cannot represent this
        config, so the claim that it must NOT be reached is unchanged and
        is pinned in `test_lensing_surrogate.py`.
        """
        y = np.asarray(POINTLENS_BITFREEZE_Y)
        for w in POINTLENS_BITFREEZE_WS:
            with self.subTest(w=w):
                value, diagnostics = F_op(w, y, POINTLENS_BITFREEZE_GAMMA)
                self.n_checks += 1
                self.assertEqual(
                    diagnostics.order_used, 0,
                    f"gamma'==0 point lens at w={w} reports order_used="
                    f'{diagnostics.order_used} -- the closed form runs no '
                    'operator series, so the order must be 0')
                self.n_checks += 1
                self.assertTrue(bool(diagnostics.converged))
                self.n_checks += 1
                self.assertTrue(np.isfinite(value))


class RefusalAboveCeilingTestCase(SchwingerTestCase):
    """
    The serving ladder at the ceiling (F005 / Build 8e): a positive-parity
    strong-shear node evaluated at ``w > W_CEILING_SCHWINGER`` is now
    EITHER served by a certified uniform arm (fold Airy / cusp Pearcey) OR
    falls through to the SAME existing named `SchwingerCertificationError`.

    RE-BASELINE (Build 8e serving ladder): the old UNCONDITIONAL
    above-ceiling refusal pin becomes CONDITIONAL, asserted per fixture.
    A served node's ``F_op`` value must BE the serving arm's number (the
    arm called DIRECTLY, agreeing at 1e-12 -- no third path); a refusing
    node must be genuinely HARD-CORE (NO arm certifies it) and raise the
    named `SchwingerCertificationError`.  Both branches are asserted
    non-vacuous: the fixture set spans an arm-served column
    (``y = (0.4, 0.3)``) and a hard-core column (``y = (0.1, 0.1)``).
    """

    def _serving_arm(self, w, y, gamma, beta=0.0, kappa=0.0):
        """The uniform arm that serves this node, called DIRECTLY.

        Reproduces the production ladder's fixed fold-then-cusp order
        (`operator._uniform_arm_value`) by calling the arm modules
        themselves -- an INDEPENDENT path to the served value, not
        operator's own dispatcher -- so a served ``F_op`` value can be
        pinned to BE the arm's number.  Returns the complex arm value, or
        ``None`` when neither arm certifies (a genuinely hard-core node).
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

    def test_scalar_fop_refuses_above_ceiling(self):
        n_served = n_refused = 0
        for gamma, y, w in itertools.product(
                ABOVE_CEILING_GAMMAS, ABOVE_CEILING_YS, ABOVE_CEILING_WS):
            with self.subTest(gamma=gamma, y=y, w=w):
                self.assertGreater(w, W_CEILING_SCHWINGER)
                self.n_checks += 1
                arm = self._serving_arm(w, y, gamma)
                try:
                    value, _ = F_op(w, np.asarray(y), gamma)
                except SchwingerCertificationError:
                    # (b) genuine hard-core refusal: NO arm may certify it.
                    self.assertIsNone(
                        arm, f'F_op refused at w={w}, y={y}, gamma={gamma} '
                        'yet an arm certifies it -- the ladder should have '
                        'served this node')
                    n_refused += 1
                    self.n_checks += 1
                    continue
                # (a) arm-served: the served number must BE the arm's number.
                self.assertIsNotNone(
                    arm, f'F_op served {value!r} at w={w}, y={y}, '
                    f'gamma={gamma} but neither arm certifies -- served by '
                    'a non-arm path')
                self.assertAlmostEqual(
                    abs(value - arm), 0.0, delta=1e-12,
                    msg=f'served F_op {value!r} is not the serving arm value '
                    f'{arm!r} at w={w}, y={y}, gamma={gamma}')
                n_served += 1
                self.n_checks += 1
        self.assertGreater(n_refused, 0,
                           'no genuinely hard-core refusal in the fixture set')
        self.assertGreater(n_served, 0,
                           'no arm-served node in the fixture set')

    def test_grid_fop_refuses_above_ceiling(self):
        n_served = n_refused = 0
        for gamma, y, w in itertools.product(
                ABOVE_CEILING_GAMMAS, ABOVE_CEILING_YS, ABOVE_CEILING_WS):
            with self.subTest(gamma=gamma, y=y, w=w):
                arm = self._serving_arm(w, y, gamma)
                try:
                    values, *_ = F_op_grid(np.asarray([float(w)]),
                                           np.asarray(y), gamma)
                except SchwingerCertificationError:
                    self.assertIsNone(
                        arm, f'F_op_grid refused at w={w}, y={y}, '
                        f'gamma={gamma} yet an arm certifies it')
                    n_refused += 1
                    self.n_checks += 1
                    continue
                self.assertIsNotNone(
                    arm, f'F_op_grid served {values[0]!r} at w={w}, y={y}, '
                    f'gamma={gamma} but neither arm certifies')
                self.assertAlmostEqual(
                    abs(complex(values[0]) - arm), 0.0, delta=1e-12,
                    msg=f'served F_op_grid {values[0]!r} is not the serving '
                    f'arm value {arm!r} at w={w}, y={y}, gamma={gamma}')
                n_served += 1
                self.n_checks += 1
        self.assertGreater(n_refused, 0,
                           'no genuinely hard-core refusal in the fixture set')
        self.assertGreater(n_served, 0,
                           'no arm-served node in the fixture set')

    def test_mixed_grid_refuses_whole_grid(self):
        """A grid mixing a certifiable node with a HARD-CORE above-ceiling
        node refuses the WHOLE grid rather than returning a partial result
        (per-node refusal fails the batch).  RE-BASELINE (Build 8e): the
        above-ceiling node is the near-caustic hard-core ``y = (0.1, 0.1)``
        column (no arm certifies it), so the named refusal still stands --
        the arm-served ``y = (0.4, 0.3)`` column would NOT refuse the grid.
        """
        self.assertIsNone(self._serving_arm(61.0, (0.1, 0.1), 0.47),
                          'the mixed-grid above-ceiling node is no longer '
                          'hard-core -- an arm now certifies it')
        with self.assertRaises(SchwingerCertificationError):
            F_op_grid(np.asarray([5.0, 61.0]), np.asarray([0.1, 0.1]), 0.47)
        self.n_checks += 1


class ImageCensusGuardFalsificationTestCase(SchwingerTestCase):
    """
    WP1 `geometry._check_image_census` (F010-idiom): the runtime
    index-theorem guard raises on a doctored image set independently of
    the solver's internal dead zone, and passes on the faithful census.

    The signed Morse sum must equal ``sign(det A) - 1``.  Positive parity
    (``det > 0``) admits an interior 4-image source, so the mirror-pair
    drop of the brief is exercised directly.  The saddle
    (``det < 0``) macro matrix ``macro_matrix(1.3, 0, 0)`` is 2-image
    EVERYWHERE in this build (probed 2026-07-19: no 4-image region
    exists), so its red-path is reached by dropping a single image --
    honest premise repair, not a mirror-pair drop it cannot supply.
    """

    def _images_and_matrix(self, matrix_args, source):
        matrix = geometry.macro_matrix(*matrix_args)
        images = geometry.find_images_quartic(np.asarray(source), matrix)
        return images, matrix

    def test_positive_parity_mirror_pair_drop_raises(self):
        """Drop the two images of equal Morse index (a symmetric mirror
        pair) from the faithful 4-image set; the guard names the census
        defect and reports the (now wrong) signed sum."""
        images, matrix = self._images_and_matrix(
            CENSUS_POSITIVE_MATRIX_ARGS, CENSUS_POSITIVE_SOURCE)
        self.n_checks += 1
        self.assertEqual(len(images), 4,
                         'the positive-parity fixture is no longer a '
                         '4-image source; pick another interior point')
        indices = [geometry.morse_index(image, matrix) for image in images]
        # The mirror pair: the first Morse index shared by two images.
        shared = next(value for value in set(indices)
                      if indices.count(value) >= 2)
        drop = [position for position, value in enumerate(indices)
                if value == shared][:2]
        doctored = [image for position, image in enumerate(images)
                    if position not in drop]
        expected_signed = sum((-1) ** geometry.morse_index(image, matrix)
                              for image in doctored)
        with self.assertRaises(geometry.LensDomainError) as ctx:
            geometry._check_image_census(doctored, matrix)
        message = str(ctx.exception)
        self.n_checks += 1
        self.assertIn('census defect', message.lower(),
                      f'refusal does not name the census defect: {message}')
        self.n_checks += 1
        self.assertIn(str(expected_signed), message,
                      f'refusal does not report the signed sum '
                      f'{expected_signed}: {message}')

    def test_saddle_single_image_drop_raises(self):
        """The saddle matrix is 2-image; dropping one image breaks the
        signed sum (``-1 != -2``) and the guard refuses by name."""
        images, matrix = self._images_and_matrix(
            CENSUS_SADDLE_MATRIX_ARGS, CENSUS_SADDLE_SOURCE)
        self.n_checks += 1
        self.assertEqual(len(images), 2,
                         'the saddle fixture unexpectedly changed image '
                         'count; re-probe the caustic')
        doctored = images[:-1]
        with self.assertRaises(geometry.LensDomainError) as ctx:
            geometry._check_image_census(doctored, matrix)
        self.n_checks += 1
        self.assertIn('census defect', str(ctx.exception).lower())

    def test_faithful_census_passes_for_both_parities(self):
        """The guard returns None (no raise) on the solver's faithful
        image set for both parities -- so the red-path above is the
        doctoring's doing, not a guard that always fires."""
        for matrix_args, source in (
                (CENSUS_POSITIVE_MATRIX_ARGS, CENSUS_POSITIVE_SOURCE),
                (CENSUS_SADDLE_MATRIX_ARGS, CENSUS_SADDLE_SOURCE)):
            with self.subTest(matrix_args=matrix_args):
                images, matrix = self._images_and_matrix(matrix_args, source)
                self.n_checks += 1
                self.assertIsNone(
                    geometry._check_image_census(images, matrix),
                    f'the faithful census for {matrix_args} was refused; '
                    'the guard is over-firing')

class _SelectBranchRoutingTestCase(SchwingerTestCase):
    """
    FIXTURE-GUARD machinery for the Build 8f (F028) serve tests.

    NOT a routing pin.  The one-home agreement between the operator
    grids and `select_branch` is pinned ONCE, in
    `test_lensing_operator.BranchGateTestCase.test_thresholds_have_one_home`;
    the helpers here exist only so the serve tests below can guard their
    own PREMISES (``this fixture is still select_branch-geometric``)
    instead of assuming a label that a threshold change could silently
    invalidate.

    `_predicate_branch` recomputes the shared gate's arguments from the
    public helpers (`macro_matrix`, `_real_delay_min_separation`,
    `cancellation_exponent`, `geometry.nearest_caustic_point`) and
    returns `select_branch`'s label.  `_observed_branch` recovers the
    grid's routing decision from what the scalar `F_op` entry actually
    serves at that node: a value bit-equal to `geometric_amplification`
    means the grid took the geometric branch; any other value (a uniform
    arm) or a named `SchwingerCertificationError` means it took the wave
    branch; a `geometry.LensDomainError` can only come from the
    geometric handoff's census guard, so it is a geometric routing that
    the census refused.
    """

    _expect_checks = True

    @staticmethod
    def _is_positive_parity(gamma, kappa):
        return 1.0 - float(kappa) > abs(float(gamma))

    def _predicate_branch(self, w, y, gamma, beta, kappa):
        """The shared gate's label, arguments rebuilt independently."""
        source = np.asarray(y, dtype=float)
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        delta_min = operator._real_delay_min_separation(source, matrix)
        if self._is_positive_parity(gamma, kappa):
            # Positive parity: L == w*|y'| == cancellation_exponent, and the
            # third leg is eta, the distance to the caustic (F031). The grid
            # supplies eta, so this mirror must too -- omitting it silently
            # disables a live leg and the two would disagree exactly where
            # the gate does its work (near the caustic).
            exponent = cancellation_exponent(w, source, gamma, kappa)
            try:
                eta = float(geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa).distance)
            except geometry.LensDomainError:
                eta = 0.0
            return select_branch(w, delta_min, exponent, eta)
        # Saddle: infinite exponent AND infinite eta -> only the resolution
        # leg is live. F031 is positive-parity only, so the saddle boundary
        # is deliberately left where it was rather than inheriting an
        # unmeasured threshold.
        return select_branch(w, delta_min, math.inf, math.inf)

    def _observed_branch(self, w, y, gamma, beta, kappa):
        """The grid's routing, read off the served scalar `F_op` value."""
        source = np.asarray(y, dtype=float)
        try:
            geom = complex(geometric_amplification(
                w, source, gamma, beta=beta, kappa=kappa))
            geom_ok = True
        except geometry.LensDomainError:
            geom, geom_ok = None, False
        try:
            served = complex(F_op(w, source, gamma, beta=beta, kappa=kappa)[0])
        except SchwingerCertificationError:
            return 'wave', None
        except geometry.LensDomainError:
            # Only the geometric handoff census raises this above ceiling.
            return 'geometric', None
        if geom_ok and served == geom:
            return 'geometric', served
        return 'wave', served


class F028GeometricServeTestCase(_SelectBranchRoutingTestCase):
    """
    F028 GEOMETRIC SERVE + QUADRATURE ANCHOR (acceptance #3).

    On the F028 table configs (positive parity, ``|y|`` chosen so each is
    resolved AND select_branch-geometric), the positive-parity grid serves
    THROUGH `geometric_amplification`: the served value is bit-for-bit
    equal to a direct call.  This replaces the previously-measured
    60%-267% fold-arm error on exactly these well-resolved configs (F028).
    The serve is NOT asserted against the quadrature above the ceiling
    (there ``F_op`` IS the arm, so the difference is identically zero and
    the quadrature does not answer above w = 60).

    Its accuracy is anchored SEPARATELY, below/at the ceiling, where the
    Schwinger quadrature is a legitimate independent oracle: the geometric
    asymptote agrees with the exact `F_op` wave value to a few parts in
    1e4.  That anchor makes the byte-equal serve meaningful without being
    a certification claim.
    """

    def test_f028_configs_served_through_geometric_amplification(self):
        for gamma, w, y in F028_SERVE:
            with self.subTest(gamma=gamma, w=w, y=y):
                source = np.asarray(y, dtype=float)
                # Sanity: these fixtures really are positive-parity,
                # resolved AND select_branch-geometric.
                self.assertEqual(
                    self._predicate_branch(w, y, gamma, 0.0, 0.0),
                    'geometric',
                    f'F028 fixture gamma={gamma}, w={w}, y={y} is not '
                    'select_branch-geometric -- fixture drifted')
                served = complex(F_op(w, source, gamma)[0])
                direct = complex(geometric_amplification(w, source, gamma))
                self.n_checks += 1
                self.assertEqual(
                    served, direct,
                    f'F028 gamma={gamma}, w={w}, y={y}: served {served!r} '
                    f'is not the geometric_amplification value {direct!r}')

    def test_geometric_asymptote_anchors_to_quadrature_below_ceiling(self):
        """Accuracy anchor: below/at the ceiling the stationary-phase
        asymptote matches the exact Schwinger quadrature (`F_op` there) to
        ``F028_ANCHOR_TOL``.  This is an ACCURACY anchor with an
        INDEPENDENT oracle (asymptote vs exact quadrature), NOT a
        certification claim.
        """
        worst = 0.0
        for gamma, y in F028_ANCHOR:
            source = np.asarray(y, dtype=float)
            for w in F028_ANCHOR_WS:
                with self.subTest(gamma=gamma, y=y, w=w):
                    self.assertLessEqual(w, W_CEILING_SCHWINGER)
                    quadrature = complex(F_op(w, source, gamma)[0])
                    asymptote = complex(
                        geometric_amplification(w, source, gamma))
                    rel = self.assert_close(
                        asymptote, mpmath.mpc(quadrature), F028_ANCHOR_TOL,
                        f'gamma={gamma}, y={y}, w={w}')
                    worst = max(worst, rel)
        self.assertLess(worst, F028_ANCHOR_TOL)


class BelowCeilingByteIdentityTestCase(SchwingerTestCase):
    """
    BELOW-CEILING BYTE-IDENTITY (F028, acceptance #1).

    The `select_branch` insertion is above-ceiling only, so the exact
    wave batch below ``W_CEILING_SCHWINGER`` must not move a single bit.
    Each served value is compared to a pre-build reference captured as
    exact `float.hex()` literals (`BYTEFREEZE_REFERENCE`), frozen from
    BOTH the pre-build (HEAD) and post-build trees and verified IDENTICAL
    (2026-07-28).  The reference is a stored constant -- NEVER an import
    of a module from a prior git revision (F022, a build-killer).  The
    fixtures span both parities, five ``gamma``, ``beta in {0, 0.7}`` and
    ``kappa in {0, 0.3}``.
    """

    def test_served_values_are_byte_identical_below_ceiling(self):
        for gamma, y, beta, kappa in BYTEFREEZE_CONFIGS:
            source = np.asarray(y, dtype=float)
            reference = BYTEFREEZE_REFERENCE[(gamma, y, beta, kappa)]
            for w in BELOW_CEILING_WS:
                with self.subTest(gamma=gamma, y=y, beta=beta, kappa=kappa,
                                  w=w):
                    self.assertLess(w, W_CEILING_SCHWINGER)
                    served = complex(
                        F_op(w, source, gamma, beta=beta, kappa=kappa)[0])
                    ref_re, ref_im = reference[w]
                    self.n_checks += 1
                    self.assertEqual(
                        served.real, float.fromhex(ref_re),
                        f'real part moved at gamma={gamma}, y={y}, '
                        f'beta={beta}, kappa={kappa}, w={w}: '
                        f'{served.real.hex()} != {ref_re}')
                    self.assertEqual(
                        served.imag, float.fromhex(ref_im),
                        f'imag part moved at gamma={gamma}, y={y}, '
                        f'beta={beta}, kappa={kappa}, w={w}: '
                        f'{served.imag.hex()} != {ref_im}')


class SaddleServeBoundaryInvarianceTestCase(_SelectBranchRoutingTestCase):
    """
    SADDLE SERVE-BOUNDARY INVARIANCE (F028; anti-variant guard).

    This test exists to prevent the REJECTED incoherent saddle-exponent
    variant from silently returning.  WP2 routes the saddle geometric
    decision through ``select_branch(w, delta_min, math.inf)``, so the
    only live leg is resolution and the boundary is ``w > 60 AND
    resolved``.  A rejected variant fed ``pi*w/4`` as the exponent against
    ``L_MAX = 48``, which would move the frequency boundary to
    ``w > 4*48/pi ~ 61.115`` -- splitting a straddling pair at
    ``w in {60.5, 61.5}`` onto different sides.  A RESOLVED saddle config
    must therefore be geometric-served at BOTH ``w = 60.5`` and
    ``w = 61.5``; if either fell to the wave arm the incoherent variant
    would be back.
    """

    def test_resolved_saddle_served_geometric_across_the_boundary(self):
        for gamma, y in SADDLE_BOUNDARY:
            source = np.asarray(y, dtype=float)
            for w in SADDLE_BOUNDARY_WS:
                with self.subTest(gamma=gamma, y=y, w=w):
                    # Guard the premise: this config is resolved at w=61,
                    # so both straddling nodes SHOULD be geometric under
                    # the w>60 boundary.
                    self.assertEqual(
                        self._predicate_branch(w, y, gamma, 0.0, 0.0),
                        'geometric',
                        f'saddle fixture gamma={gamma}, y={y}, w={w} is not '
                        'select_branch-geometric -- fixture drifted')
                    observed, served = self._observed_branch(
                        w, y, gamma, 0.0, 0.0)
                    self.n_checks += 1
                    self.assertEqual(
                        observed, 'geometric',
                        f'resolved saddle gamma={gamma}, y={y} served '
                        f'{observed!r} at w={w} -- the boundary moved off '
                        '``w > 60`` (the rejected pi*w/4 variant returned)')
                    direct = complex(
                        geometric_amplification(w, source, gamma))
                    self.assertEqual(
                        served, direct,
                        f'saddle gamma={gamma}, y={y}, w={w}: served '
                        f'{served!r} is not the geometric value {direct!r}')


class DeltaMinComputedAtMostOnceTestCase(SchwingerTestCase):
    """
    DELTA_MIN COMPUTED-AT-MOST-ONCE / NOT-BELOW-CEILING (F028, #6).

    The image-quartic solve `_real_delay_min_separation` is the only
    expensive geometric primitive on the routing path.  Both grids guard
    it behind ``np.any(w_array > ceiling)`` and compute it ONCE per grid
    evaluation.  Spying the module-qualified symbol
    ``operator._real_delay_min_separation`` (patched on the module so the
    grid's global lookup is intercepted): a grid entirely below the
    ceiling triggers ZERO solves, and a grid with one above-ceiling node
    triggers EXACTLY ONE, for BOTH the positive-parity and saddle grids.
    """

    #: (label, w_array, gamma, y, kappa, expected_calls)
    _CASES = (
        ('positive below', (5.0, 40.0, 59.0), 0.9, (1.0, 0.7), 0.0, 0),
        ('positive one-above', (5.0, 70.0), 0.9, (1.0, 0.7), 0.0, 1),
        ('saddle below', (5.0, 40.0, 59.0), 1.2, (0.3, 0.2), 0.0, 0),
        ('saddle one-above', (5.0, 70.0), 1.2, (0.3, 0.2), 0.0, 1),
    )

    def test_delta_min_solve_count_per_grid(self):
        original = operator._real_delay_min_separation
        for label, w_tuple, gamma, y, kappa, expected in self._CASES:
            with self.subTest(case=label):
                w_array = np.asarray(w_tuple, dtype=float)
                source = np.asarray(y, dtype=float)
                with mock.patch.object(
                        operator, '_real_delay_min_separation',
                        side_effect=original) as spy:
                    try:
                        F_op_grid(w_array, source, gamma, kappa=kappa)
                    except SchwingerCertificationError:
                        # A refusing above-ceiling node still computes
                        # delta_min once BEFORE the node pre-pass.
                        pass
                    self.n_checks += 1
                    self.assertEqual(
                        spy.call_count, expected,
                        f'{label}: expected {expected} quartic solve(s), '
                        f'got {spy.call_count}')


class AboveCeilingWaveThreeOutcomeTestCase(SchwingerTestCase):
    """
    ABOVE-CEILING 'WAVE' THREE-OUTCOME COVERAGE (F028, #4/#5).

    A positive-parity node routed to the wave branch above the ceiling
    lands in exactly one of three outcomes; this exercises all three:
    (i) the uniform fold Airy arm certifies and serves;
    (ii) the fold arm refuses but the cusp Pearcey arm serves;
    (iii) BOTH arms refuse and the named `SchwingerCertificationError`
    fires with the lowest-index refuser's authentic `f_schwinger`
    message.  The refusal message at ``w > 60`` is the y-independent
    ceiling refusal, reproduced here by an INDEPENDENT direct call to
    `f_schwinger` at the same ``w`` (its ceiling guard fires before any
    y-dependent work).  Refusal-identity across the fixture matrix is
    already covered by `RefusalAboveCeilingTestCase`; this only pins that
    the third outcome is reachable and named.
    """

    def _arms(self, w, y, gamma):
        source = np.asarray(y, dtype=float)
        fold = _airy_fold.fold_amplification(w, source, gamma)
        cusp = _pearcey_cusp.cusp_amplification(w, source, gamma)
        return fold, cusp

    def test_fold_airy_arm_serves(self):
        gamma, y = THREE_OUTCOME_FOLD
        source = np.asarray(y, dtype=float)
        fold, cusp = self._arms(THREE_OUTCOME_W, y, gamma)
        self.assertIsNotNone(fold, 'fold fixture no longer fold-served')
        served = complex(F_op(THREE_OUTCOME_W, source, gamma)[0])
        self.n_checks += 1
        self.assertEqual(
            served, complex(fold),
            f'fold outcome: served {served!r} is not the fold arm value '
            f'{complex(fold)!r}')

    def test_cusp_pearcey_arm_serves_when_fold_refuses(self):
        gamma, y = THREE_OUTCOME_CUSP
        source = np.asarray(y, dtype=float)
        fold, cusp = self._arms(THREE_OUTCOME_W, y, gamma)
        self.assertIsNone(fold, 'cusp fixture is now fold-served, not cusp')
        self.assertIsNotNone(cusp, 'cusp fixture no longer cusp-served')
        served = complex(F_op(THREE_OUTCOME_W, source, gamma)[0])
        self.n_checks += 1
        self.assertEqual(
            served, complex(cusp),
            f'cusp outcome: served {served!r} is not the cusp arm value '
            f'{complex(cusp)!r}')

    def test_both_arms_refuse_raises_named_authentic_message(self):
        gamma, y = THREE_OUTCOME_REFUSE
        source = np.asarray(y, dtype=float)
        fold, cusp = self._arms(THREE_OUTCOME_W, y, gamma)
        self.assertIsNone(fold, 'refuse fixture is now fold-served')
        self.assertIsNone(cusp, 'refuse fixture is now cusp-served')
        # Independent oracle for the authentic message: f_schwinger's
        # ceiling guard fires before any y-dependent work, so a direct
        # call at the same w reproduces the exact message text.
        expected_message = None
        try:
            f_schwinger(THREE_OUTCOME_W, np.asarray([0.3, 0.2]), 0.3)
        except SchwingerCertificationError as exc:
            expected_message = str(exc)
        self.assertIsNotNone(expected_message,
                             'f_schwinger did not refuse above the ceiling')
        with self.assertRaises(SchwingerCertificationError) as caught:
            F_op(THREE_OUTCOME_W, source, gamma)
        self.n_checks += 1
        self.assertEqual(
            str(caught.exception), expected_message,
            'both-refuse node did not raise the authentic f_schwinger '
            'ceiling message')


class SelectBranchSelfFalsificationTestCase(_SelectBranchRoutingTestCase):
    """
    SELF-FALSIFICATION: the routing observation can go RED.

    A numerical routing suite is only trustworthy if its assertions have
    teeth.  `test_opposite_label_breaks_one_home` reaches into a genuine
    fixture -- a real `F_op` serve at a near-caustic node -- pins the
    label it actually produces, and shows the opposite label is
    rejected, so `_observed_branch` is not a constant.
    """

    def test_opposite_label_breaks_one_home(self):
        # A genuinely wave-routed node (near-caustic, unresolved above
        # the ceiling): asserting it is 'geometric' MUST fail.
        gamma, kappa, beta = 0.9, 0.0, 0.0
        y = (0.05 * 0.8, 0.05 * 0.6)
        observed, _ = self._observed_branch(61.0, y, gamma, beta, kappa)
        self.assertEqual(observed, 'wave', 'falsification premise drifted')
        with self.assertRaises(AssertionError):
            self.assertEqual(observed, 'geometric')
        self.n_checks += 1

    # RETIRED (vacuity audit): `test_corrupted_byte_reference_is_detected`
    # and `test_perturbed_geometric_serve_is_detected` mutated the EXPECTED
    # VALUE (`ref * (1 + 1e-12)`, `direct * (1 + 1e-9)`) and then asserted
    # that `assertEqual` rejected it.  That proves only that `assertEqual`
    # distinguishes two distinct float64s -- it exercised no cogwheel code
    # path and could not fail whatever the operator did.  A genuine
    # mutation test perturbs PRODUCTION, not the oracle; see
    # `DdMandatoryFalsificationTestCase::test_float64_dd_accumulation_
    # drives_gate_red` in this file for the correct py_func-chain pattern.
    # The surviving method here retains teeth: it reads a real served
    # value and pins a real routing label.
    #
    # DELETED (one-home consolidation): `test_select_branch_has_both_live_
    # legs` re-pinned the gate's legs from hand-built ``(w, delta_min, L)``
    # triples -- a second home for the predicate, and one that already
    # silently omitted the third (`eta`) leg.  The legs are pinned once,
    # in `test_lensing_operator.BranchGateTestCase` (`test_four_quadrants`,
    # `test_boundary_equalities` and the live-eta-leg assertion inside
    # `test_thresholds_have_one_home`).



if __name__ == '__main__':
    main()
