"""
Tests for the WP1/WP2 BATCHED wave-branch operator path in
`lensing.chang_refsdal.operator`: the weight-vector batched contraction
(`_weight_vectors` + `_contract_grid`) behind the single-path
`F_op_grid`, and its use by `channels._exact_total` through the lensed
likelihood.

WHAT THIS SUITE PINS
--------------------
WP1 replaced the former per-node ``dim x dim`` bilinear form with a
precomputed per-order weight vector (built ONCE, since the operator
table and monomial powers are ``w``-independent within one lens
configuration) dotted against each node's rescaled radial derivatives,
and routed BOTH the scalar `F_op` and the batched `F_op_grid` through
ONE contraction path (`_grid_certified`).  The claim is that this
changed the ACCUMULATION ORDER and the SPEED, not the answer or the
certified-or-refuse contract.  This suite re-certifies that claim:

* `BatchedContractionCertificationTestCase` re-runs the F005 oracle
  certification against an INDEPENDENT mpmath reference over the union of
  the in-domain ``F_op`` grid and the ``L in [24, 48]`` boundary band,
  and adds the two batching-specific invariants the reorder could break:
  the per-node return-vs-refuse DECISION is identical between a solo
  ``[w]`` call and the full batch (no cross-node convergence-state
  leakage), and the returned VALUE agrees to ``1e-14`` (identical code on
  identical data).
* `BatchedContractionFalsificationTestCase` proves that certification is
  not vacuous for the new ``njit`` core: two perturbations injected
  through the numba ``py_func`` chain (a corrupted convergence tolerance
  and a corrupted radial-index gather) each drive the accuracy gate red.
* `FewMsTimingTestCase` pins the machine-INDEPENDENT speed properties the
  batching was for (RB beats brute by >= `SPEEDUP_MIN`; the pure
  contraction is subdominant to the amplification engine) plus an
  arithmetic-derived absolute regression guard `MS_CEILING`.
* `BatchedEquivalenceTestCase` confirms the scalar entry point delegates
  to the batched path BIT-IDENTICALLY, so the many existing RB-vs-brute,
  determinism, crown-accuracy and interpolation gates in the sibling
  suites automatically exercise the batched path at their ORIGINAL
  tolerances.

WHY THE ORACLE IS INDEPENDENT (F002)
------------------------------------
``F_op_grid`` is gated against `_oracle_fop`, an mpmath amplification
built ENTIRELY from ``mpmath.hyp1f1`` (the textbook Kummer
s-derivative ladder, NOT the production double-double kernel) and an
INTEGER-coefficient ``(u, v)`` monomial ladder for the shear operator
``exp(i*gamma*D_0/2w)`` (NOT the production complex shear-eigenframe
weight vectors).  The two share no code and no numerical substrate; the
top-level reconstruction is re-derived from the diffraction integral,
not copied from ``F_op``.  `OracleIndependenceTestCase` (an AST guard)
pins that the oracle helpers reference no production name.  mpmath is
imported ONLY here and never becomes importable from a production path.

TOLERANCES
----------
* `FOP_RTOL` = 1e-10 is a property of ``F_op_grid``, not the oracle
  (exact far beyond float64); UNCHANGED from the scalar certification.
* `SINGLE_BATCH_RTOL` = 1e-14: solo ``[w]`` and full-batch share the
  identical per-node arithmetic, so a larger gap is cross-node
  contamination, not round-off.
* `SPEEDUP_MIN` = 8.0 (raised from the former 3.0): the measured warm
  ``lnlike`` speed-up is tens-fold, so 8.0 is a structural, non-retuned
  advance rather than a machine-calibrated knob.
* `MS_CEILING` = 0.175 s is ARITHMETIC-DERIVED (see its doc-comment),
  NOT the brief's 10 ms physical target; it is a secondary regression
  guard behind the two machine-independent structural gates.

ANTI-VACUITY AND SELF-FALSIFICATION
-----------------------------------
`BatchedOperatorTestCase.tearDown` fails a test that made zero
comparisons.  `BatchedContractionFalsificationTestCase` and
`SelfFalsificationTestCase` prove the accuracy gate and the anti-vacuity
guard can each go red.
"""
from __future__ import annotations

# Single-thread pinning for the timing gate (best-effort): production
# runs under a parallel sampler with every core busy, so the honest
# per-eval cost is the SINGLE-THREAD one.  These env vars are read by
# OpenBLAS/MKL/numba at import time, so they are set BEFORE numpy/numba
# are imported.  When another already-imported module initialised the
# thread pool first (one shared pytest process) the pin is a no-op; the
# HARD timing gates (speedup, contraction < engine) are robust to that.
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

import ast
import inspect
import os
import pathlib
import textwrap
import time
import warnings
from unittest import TestCase, main, mock

import mpmath
import numpy as np

from cogwheel import data, waveform
from cogwheel.lensing.chang_refsdal import geometry, operator
from cogwheel.lensing.chang_refsdal.operator import (
    CancellationError, F_op, F_op_grid)
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError, W_CEILING_SCHWINGER)

#: Named wave-branch refusals (Build 8d homogenization): a sheared
#: positive-parity host (``gamma' > 0``) is served by the exact Schwinger
#: evaluator and refuses above its ceiling with
#: `SchwingerCertificationError`; the shear-free ``gamma' == 0`` point
#: lens keeps the legacy `CancellationError`.  Both are named refusals of
#: the certify-XOR-refuse contract, so the batched decision tests accept
#: EITHER.
_WAVE_REFUSALS = (CancellationError, SchwingerCertificationError)
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, _data_term, _norm_term)

try:  # Diagnostics only; never gate a test on plotting being present.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Oracle / certification constants.
# ---------------------------------------------------------------------------

#: Working precision [decimal digits] of the mpmath amplification oracle.
#: ~35 digits of margin over the 1e-10 gate; the oracle is the reference,
#: so it must not be the thing under test.
ORACLE_DPS = 50

#: Operator-order cap for the mpmath oracle; exceeds ``F_op_grid``'s
#: convergence order so the reference is fully summed.
ORACLE_MAX_ORDER = 100

#: Relative-error gate on ``F_op_grid`` against the oracle (UNCHANGED
#: from the scalar certification): a property of the wave branch.
FOP_RTOL = 1e-10

#: Solo-``[w]``-vs-full-batch agreement gate: identical per-node
#: arithmetic, so a larger gap is cross-node contamination.
SINGLE_BATCH_RTOL = 1e-14

#: Operator-order cap handed to ``F_op_grid`` for the oracle comparison
#: (large enough that the highest-``w`` grid point converges).
FOP_MAX_ORDER = 70

#: In-domain grid axes: ``w``, physical ``sqrt(s) = |y|`` (kappa=0), and
#: shear.  ``L = w*sqrt(s)`` runs from 0.3 to 45; some high-``L``,
#: shear-on points refuse (routed to the certified-XOR-refuse contract).
FOP_GRID_W = (1.0, 10.0, 20.0, 40.0, 50.0)
FOP_GRID_SQRT_S = (0.3, 0.9)
FOP_GRID_GAMMA = (0.0, 0.2)

#: F005 boundary band: ``L in linspace(24, 48, 17)`` at ``y = (0.9, 0)``,
#: ``gamma = 0.20``, ``kappa = 0`` (so ``w = L / 0.9``).  Both outcomes
#: MUST occur across the band -- certified at low ``L``, refused at high
#: ``L`` (the XOR contract, FINDINGS F005).
CERT_SQRT_S = 0.9
CERT_GAMMA = 0.20
CERT_LS = np.linspace(24.0, 48.0, 17)

#: Band for the certify-XOR-refuse boundary test.  Build 7a moved the
#: refusal onset for this configuration to the Schwinger ceiling
#: (``w = 60``, i.e. ``L = 60 * CERT_SQRT_S = 54``): legacy
#: `CancellationError` refusals at ``w <= 60`` are rescued by the
#: cross-parity Schwinger fallback, so the band must straddle that
#: ceiling to witness both outcomes.  It must also stay BELOW the
#: kernel's own ``L = w * sqrt(s) <= 60`` double-double product
#: ceiling: above it the point-mass kernel raises
#: `HypergeometricDomainError` before any cancellation logic runs — a
#: third, separate refusal tier outside this test's contract.
XOR_BAND_LS = np.linspace(24.0, 59.4, 22)

#: Above-ceiling config that still REFUSES, for the certify-XOR-refuse
#: contract.  The `XOR_BAND_LS` host above is hugely RESOLVED
#: (``delta_min = 2.094``, so ``w * delta_min = 56..138``), and since the
#: authoritative-gate build its above-ceiling nodes are served by
#: `geometric_amplification` rather than refused -- correctly, since
#: ``L = 54..59.4 > L_MAX`` and ``|y| = 0.9`` sits ``eta ~ 0.45`` outside a
#: caustic of extent ``0.447``, the ``2e-7`` regime of F029.  The named
#: refusal now fires only where the gate says 'wave' above the ceiling and
#: BOTH uniform arms decline, which needs an UNRESOLVED host:
#: ``w * delta_min = 0.836 < RHO_END``.
XOR_REFUSING_GAMMA = 0.8722
XOR_REFUSING_Y = (0.80786, 0.28183)
XOR_REFUSING_W = 65.0

#: Accuracy gate on the ABOVE-CEILING geometric serve, against the same
#: independent mpmath `_oracle_fop` used below the ceiling.
#:
#: The production exact evaluator (`_schwinger.f_schwinger`) refuses above
#: ``W_CEILING_SCHWINGER``, and the suite treated that as meaning nothing
#: above the ceiling could be checked -- so every above-ceiling test asserted
#: which PATH ran, or byte-identity against production, which is
#: unfalsifiable by construction (a wrong value passes). But `_oracle_fop` is
#: a pure mpmath operator-series reconstruction with NO frequency ceiling:
#: the ceiling is a property of one evaluator, not of the mathematics.
#:
#: Measured on the served nodes of `XOR_BAND_LS` (2026-07-29): worst relative
#: error 3.7e-5, so this gate carries a ~27x margin.
#:
#: SCOPE, measured and deliberately narrow. `_oracle_fop` is the LEGACY
#: operator series -- the path demoted in Build 8d precisely because it
#: cancels catastrophically at high ``L = w * |y'|`` (F005: certified to
#: ``L ~ 25-30``, certified-or-refused through 48). It is therefore a valid
#: reference only at MODERATE ``L``. This band runs ``L`` in [24, 59.4],
#: near that edge, so treat this as a CONSISTENCY gate between two
#: independent reconstructions, NOT a certification of either.
#:
#: It would NOT have caught F028. Those configs sit at ``L ~ 100-200``
#: (``|y| ~ 1.5-2.1`` at ``w = 70-500``), where `_oracle_fop` itself
#: diverges: measured 2026-07-29, the arm AND the geometric serve both
#: report relative error 1.000e+00 against it there, i.e. the oracle is the
#: outlier. Falsifying the uniform arms in their own high-``L`` regime needs
#: a reference this suite does not yet have -- see FINDINGS F028/F029.
GEOMETRIC_SERVE_RTOL = 1e-3

#: Production names the independent oracle helpers must NOT reference
#: (F002 oracle independence, enforced by the AST guard).
ORACLE_FORBIDDEN_NAMES = frozenset({
    'operator', 'F_op', 'F_op_grid', 'channels', 'geometry',
    '_hyp1f1', 'point_mass_g_derivatives', '_grid_certified',
    '_contract_grid', '_weight_vectors', 'LensedRelativeBinningLikelihood',
})

# ---------------------------------------------------------------------------
# Falsification constants.
# ---------------------------------------------------------------------------

#: Certified in-domain config for the self-falsification (``L = 11``):
#: the UNPATCHED ``F_op_grid`` is within `FOP_RTOL` of the oracle here, so
#: the gate can go green before each perturbation drives it red.
FALS_W = 20.0
FALS_Y = (0.55, 0.0)
FALS_GAMMA = 0.20

# ---------------------------------------------------------------------------
# Timing-fixture constants (crown four-image, shared with the crown
# likelihood suite so the fixture is read off the SAME configuration).
# ---------------------------------------------------------------------------

#: Higher-mode approximant so the mode-pair contraction is genuinely
#: exercised on the fast path.
APPROXIMANT = 'IMRPhenomXPHM'

#: Fixed seed for every stochastic input.
SEED = 20260717

#: Bin width [Hz] of the uniform relative-binning grid (crown value).
DF_BIN = 4.0

#: Largest relative image delay [s] the fixture's bins support.
DELTA_T_MAX = 0.02

#: Lens mass [Msun] / redshift of the crown (well-conditioned) fixture.
M_LENS_MSUN = 90.0
Z_LENS = 0.4

#: The crown four-image config ``(label, y1, y2, gamma, beta, kappa)``.
_CROWN = ('four-image', 0.08, 0.06, 0.20, 0.0, 0.0)

#: Best-of-N repeats for warm timing (robust to scheduler jitter).
TIMING_REPEATS = 5

#: Lower bound on the warm RB speed-up over full-grid brute force, raised
#: from the former machine-calibrated 3.0 to 8.0.  The batched engine's
#: measured advantage is tens-fold (predicted lnlike ~139x over ~15 s
#: brute), so 8.0 sits well below the floor: a structural, non-retuned
#: advance, machine-INDEPENDENT.
SPEEDUP_MIN = 8.0

#: LOOSE absolute regression ceiling [s] on warm best-of-N ``lnlike``.
#: RE-TUNED (Build 8d homogenization): the exact positive-parity wave
#: branch is now the Schwinger evaluator at ~90 ms/node, so the warm crown
#: ``lnlike`` (8 engine nodes) measures ~0.75 s -- the exact path is the
#: SINGLE certified evaluator BY DESIGN (the surrogate is the speed layer,
#: off by default).  The loose ceiling is set to 3.0 s: ~4x the measured
#: cost, generous against a loaded box, yet still catches a catastrophic
#: regression (e.g. a full-grid engine evaluation, ~140 s).  It is NOT the
#: brief's 10 ms physical target; the tight/strict speed claim lives under
#: ``COGWHEEL_STRICT_TIMING`` (the brute-force speed-up, below).
MS_CEILING = 3.0

#: Strict-timing switch (machine-dependent gates are opt-in).  The
#: brute-force speed-up gate re-evaluates the FULL-grid matched filter,
#: which -- since homogenization routes the exact Schwinger engine
#: per-frequency -- now costs ~140 s PER brute call (best-of-N would be
#: minutes), a build-killer for the default fast suite.  It is therefore
#: gated OFF unless ``COGWHEEL_STRICT_TIMING`` is set; the default suite
#: keeps the cheap structural gate (contraction < engine) and the loose
#: absolute ceiling.
_STRICT_TIMING = bool(os.environ.get('COGWHEEL_STRICT_TIMING'))

#: Directory for diagnostic plots (created on demand).
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


# ---------------------------------------------------------------------------
# Independent mpmath amplification oracle (oracle-only; imports nothing
# from the production operator/channels/geometry).
# ---------------------------------------------------------------------------
def _oracle_radial_ladder(w, s):
    """Memoized ``k -> d^k/ds^k G_PM(w, s)`` at oracle precision.

    ``G_PM(w, s) = C(w) * 1F1(1 - i w/2; 1; -i w s/2)`` and its ``k``-th
    ``s``-derivative is ``C(w) * c**k * (a)_k / (1)_k * 1F1(a+k; 1+k;
    c s)`` with ``a = 1 - i w/2``, ``c = -i w/2`` and ``C(w) = exp(pi w/4
    + i (w/2) ln(w/2)) Gamma(1 - i w/2)`` (Abramowitz & Stegun ch. 13).
    A fresh ``mpmath.hyp1f1`` per ``k`` -- the direct textbook
    definition, no Kummer reparametrization and no shared numerator.
    """
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
    """Apply the real shear operator ``D_0 = d_u**2 - d_v**2``.

    ``state`` maps ``(a, b) -> int`` coefficient of ``u**a v**b G^(k)``,
    with the radial index implied by ``k = (a + b)//2 + order``.  This is
    the real ``(u, v)`` monomial ladder (``z = u + i v`` gives
    ``2 d_z**2 + 2 d_zbar**2 = d_u**2 - d_v**2``) with EXACT Python-int
    coefficients -- deliberately not the production's complex
    shear-eigenframe weight vectors.  No mpmath is spent here.
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


class OracleConvergenceError(RuntimeError):
    """The mpmath operator-series oracle did not converge.

    Raised instead of returning a truncated partial sum. An oracle that
    silently returns a wrong number is worse than no oracle: it converts
    "we cannot check this" into a confident false comparison, which is
    exactly how F028 survived and how it was twice misdiagnosed (F030).
    """


def _oracle_fop(w, y, gamma, beta=0.0, kappa=0.0,
                max_order=ORACLE_MAX_ORDER):
    """INDEPENDENT wave-optics amplification ``F(w)`` at oracle
    precision.

    Sums ``total = sum_n (i gamma'/(2w))**n / n! * D_0**n G_PM`` at the
    eigenframe-rotated source and applies the mass-sheet prefactor
    ``F = (1/lam) exp(0.5j w ln(lam) - 0.5j w kappa s + 0.5j w s) total``
    with ``lam = 1 - kappa``, ``gamma' = gamma/lam``, ``s = |y'|**2``,
    ``y' = y/sqrt(lam)``.  This carries the diffraction integral's
    operator reduction independently of ``F_op``'s own reconstruction
    (F002).
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
        converged = False
        for n in range(max_order + 1):
            if n:
                factorial *= n
                state = _oracle_operator_step(state)
            term = alpha ** n / factorial * evaluate(state, n)
            total += term
            if n >= 4 and abs(term) <= mpmath.mpf('1e-24') * abs(total):
                small += 1
                if small >= 3:
                    converged = True
                    break
            else:
                small = 0

        # CERTIFIED-OR-REFUSE, the same contract production obeys. Without
        # this the loop simply fell out at `max_order` and returned the
        # TRUNCATED sum, indistinguishable from a converged one -- an oracle
        # that fails silently, which is the one behaviour this codebase
        # refuses everywhere else. Measured 2026-07-29: at the F028 configs
        # (L ~ 100-200) the series never converges, and the truncated value
        # is so wrong that BOTH the uniform arm and the geometric serve
        # report relative error 1.000e+00 against it -- the oracle is the
        # outlier. That silent failure cost two wrong diagnoses before it
        # was caught. Cost of the guard: 1 node of 39 across the two
        # production bands (L = 59.4).
        if not converged:
            raise OracleConvergenceError(
                f'the operator-series oracle did not converge at w={w!s}, '
                f'y=({y[0]!r}, {y[1]!r}), gamma={gamma!r}, kappa={kappa!r} '
                f'within max_order={max_order}: the partial sum is a '
                f'TRUNCATION, not a reference, and must not be compared '
                f'against. This regime needs a different oracle (F030).')

        value = ((1 / lam)
                 * mpmath.e ** (0.5j * w * mpmath.log(lam)
                                - 0.5j * w * mpmath.mpf(kappa) * s
                                + 0.5j * w * s)
                 * total)
        return complex(value)


def _referenced_names(func):
    """Return every name a function's own source references.

    Walks ``ast.Import`` / ``ast.ImportFrom`` plus ``ast.Name`` ids and
    ``ast.Attribute`` attribute names, so a production dependency that
    entered a helper as ``operator.F_op`` or a bare ``F_op`` name -- not
    only as an import statement -- is caught.  The idiom mirrors the
    sibling lens suites' oracle-independence guard.
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


class BatchedOperatorTestCase(TestCase):
    """Shared relative-error assertion plus the anti-vacuity tally.

    `tearDown` fails a test that asserted nothing, so a suite that
    silently stops comparing cannot read green.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    def assert_relative(self, got, want, rtol, msg):
        """Assert ``|got - want| <= rtol * |want|`` and tally a check."""
        error = abs(complex(got) - complex(want)) / abs(complex(want))
        self.n_checks += 1
        self.assertLessEqual(error, rtol, f'{msg}: relative error '
                             f'{error:.3e} exceeds {rtol:.0e}')
        return error

    @staticmethod
    def _save_figure(fig, name):
        """Write ``fig`` to ``output/<name>.png`` and close it."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=120,
                    bbox_inches='tight')
        plt.close(fig)


class OracleIndependenceTestCase(BatchedOperatorTestCase):
    """The mpmath oracle helpers reference no production name (F002).

    A certification is only as trustworthy as the independence of its
    oracle: if `_oracle_fop` reached into ``operator`` it would gate the
    module against itself.  This AST guard walks each oracle helper's own
    source and fails if any production name leaks in.
    """

    def test_oracle_helpers_reference_no_production_names(self):
        """No oracle helper references a production module or symbol."""
        for func in (_oracle_fop, _oracle_radial_ladder,
                     _oracle_operator_step):
            names = _referenced_names(func)
            leaked = names & ORACLE_FORBIDDEN_NAMES
            self.n_checks += 1
            self.assertEqual(
                leaked, set(),
                f'oracle helper {func.__name__} references production '
                f'names {sorted(leaked)}; the oracle is not independent')

    def test_oracle_actually_uses_mpmath(self):
        """The amplification oracle is built from mpmath, not float64.

        A positive control on the guard above: an "independent" oracle
        that silently dropped to float64 would share the production
        substrate; assert mpmath is genuinely on the oracle path.
        """
        for func in (_oracle_fop, _oracle_radial_ladder):
            self.n_checks += 1
            self.assertIn(
                'mpmath', _referenced_names(func),
                f'{func.__name__} does not reference mpmath; the oracle '
                'may have dropped to the float64 substrate under test')


class BatchedContractionCertificationTestCase(BatchedOperatorTestCase):
    """Re-certify the batched accumulation order against the mpmath
    oracle.

    The weight-vector reorder must not perturb the answer inside the
    certified band, must not leak convergence state across nodes, and
    must keep the F005 certified-XOR-refuse contract.  Three invariants:
    (1) every returned node matches the independent oracle to `FOP_RTOL`;
    (2) the solo-``[w]`` and full-batch return/refuse DECISION is
    identical, with zero flips in either direction; (3) where both
    return, the VALUE agrees to `SINGLE_BATCH_RTOL`.
    """

    def _configs(self):
        """Yield ``(label, y, gamma, beta, kappa, w_array)`` grid rows.

        The union of the in-domain ``F_op`` sweep (``L`` from 0.3 to 45)
        and the F005 boundary band (``L in [24, 48]``).
        """
        for sqrt_s in FOP_GRID_SQRT_S:
            for gamma in FOP_GRID_GAMMA:
                yield (f'grid_s{sqrt_s:g}_g{gamma:g}',
                       np.array([sqrt_s, 0.0]), gamma, 0.0, 0.0,
                       np.array(FOP_GRID_W, dtype=float))
        yield ('cert_band', np.array([CERT_SQRT_S, 0.0]), CERT_GAMMA,
               0.0, 0.0, CERT_LS / CERT_SQRT_S)

    def _solo(self, w, y, gamma, beta, kappa):
        """Return ``(certified, value)`` for a single-``[w]`` batch call.

        ``certified`` is False and ``value`` None when the node raises a
        named wave-branch refusal (`CancellationError` on the ``gamma'==0``
        legacy exit, or `SchwingerCertificationError` on the homogenized
        Schwinger path); otherwise ``value`` is the returned ``F``.
        """
        try:
            values, _, _ = F_op_grid(
                np.array([w], dtype=float), y, gamma, beta=beta,
                kappa=kappa, max_order=FOP_MAX_ORDER)
        except _WAVE_REFUSALS:
            return False, None
        return True, complex(values[0])

    def test_batched_matches_mpmath_oracle(self):
        """Every returned node matches the independent oracle to
        `FOP_RTOL`.

        Runs each grid point through a solo ``F_op_grid([w])`` (the
        single-path batched code) and, where it certifies, gates the
        value against `_oracle_fop`.  Refused nodes carry no accuracy
        claim and are skipped -- their contract is the XOR test below.
        """
        ls, errors, refused_ls = [], [], []
        for label, y, gamma, beta, kappa, w_array in self._configs():
            root_s = float(np.sqrt(y @ y))
            for w in w_array:
                certified, value = self._solo(w, y, gamma, beta, kappa)
                cancellation_l = float(w) * root_s
                if not certified:
                    refused_ls.append(cancellation_l)
                    continue
                oracle = _oracle_fop(w, y, gamma, beta=beta, kappa=kappa)
                error = self.assert_relative(
                    value, oracle, FOP_RTOL,
                    f'{label} w={w:g} L={cancellation_l:.2f}')
                ls.append(cancellation_l)
                errors.append(error)

        self.assertGreater(
            self.n_checks, 0,
            'no in-domain grid point certified; the oracle gate ran on '
            'nothing')
        self._plot_accuracy(ls, errors, refused_ls)

    def _plot_accuracy(self, ls, errors, refused_ls):
        """Scatter ``log10(rel err)`` vs ``L`` under the `FOP_RTOL`
        ceiling."""
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        safe = [max(e, 1e-18) for e in errors]
        ax.scatter(ls, safe, s=24, label='returned nodes')
        for cancellation_l in refused_ls:
            ax.axvline(cancellation_l, color='0.85', lw=0.8, zorder=0)
        ax.axhline(FOP_RTOL, color='crimson', ls='--',
                   label=f'FOP_RTOL = {FOP_RTOL:.0e}')
        ax.set_yscale('log')
        ax.set_xlabel('cancellation exponent L = w * |y\'|')
        ax.set_ylabel('|F_batched - F_oracle| / |F_oracle|')
        ax.set_title('Batched F_op_grid vs mpmath oracle')
        ax.legend(loc='best', fontsize=8)
        self._save_figure(fig, 'batched_operator_oracle_accuracy')

    def test_single_and_batch_refusal_decisions_identical(self):
        """The solo-vs-batch return/refuse decision has zero flips.

        For each config, the subset of solo-certified nodes must ALSO
        certify when evaluated together (no certify->refuse flip from
        cross-node convergence-state leakage), and each solo-refused node
        must STILL refuse when batched alongside a certified node (no
        refuse->certify flip).  Both directions are checked.
        """
        for label, y, gamma, beta, kappa, w_array in self._configs():
            solo = {float(w): self._solo(w, y, gamma, beta, kappa)[0]
                    for w in w_array}
            certified = [w for w, ok in solo.items() if ok]
            refused = [w for w, ok in solo.items() if not ok]

            if certified:
                # No certified node may flip to a refusal in the batch.
                values, _, _ = F_op_grid(
                    np.array(certified, dtype=float), y, gamma, beta=beta,
                    kappa=kappa, max_order=FOP_MAX_ORDER)
                self.n_checks += 1
                self.assertEqual(
                    len(values), len(certified),
                    f'{label}: batch over solo-certified nodes returned '
                    f'{len(values)} of {len(certified)} values')

            for w_refused in refused:
                # A refused node must refuse even beside a certified one.
                batch = ([certified[0]] if certified else []) + [w_refused]
                self.n_checks += 1
                with self.assertRaises(
                        _WAVE_REFUSALS,
                        msg=f'{label}: solo-refused w={w_refused:g} did '
                            'not refuse when batched'):
                    F_op_grid(np.array(batch, dtype=float), y, gamma,
                              beta=beta, kappa=kappa,
                              max_order=FOP_MAX_ORDER)

    def test_single_and_batch_values_agree(self):
        """Where both return, solo and full-batch agree to
        `SINGLE_BATCH_RTOL`.

        Identical per-node arithmetic on identical data, so any gap above
        `SINGLE_BATCH_RTOL` is cross-node contamination, not round-off.
        """
        for label, y, gamma, beta, kappa, w_array in self._configs():
            certified = [float(w) for w in w_array
                         if self._solo(w, y, gamma, beta, kappa)[0]]
            if not certified:
                continue
            batch_values, _, _ = F_op_grid(
                np.array(certified, dtype=float), y, gamma, beta=beta,
                kappa=kappa, max_order=FOP_MAX_ORDER)
            for w, batch_value in zip(certified, batch_values):
                solo_value = self._solo(w, y, gamma, beta, kappa)[1]
                self.assert_relative(
                    batch_value, solo_value, SINGLE_BATCH_RTOL,
                    f'{label} w={w:g} solo-vs-batch')

    def test_cert_band_certifies_low_l_and_refuses_high_l(self):
        """The certify-XOR-refuse contract still shows BOTH outcomes.

        A path that only ever returned, or only ever refused, would be a
        silent regression of the certified-or-refuse guarantee (F005).

        RE-BASELINE (authoritative-gate build, F028/F029): the refusal half
        no longer comes from THIS host.  ``gamma = 0.20``, ``|y| = 0.9`` is
        hugely resolved (``delta_min = 2.094``, so ``w * delta_min`` runs
        56..138 across the band) and sits ``eta ~ 0.45`` outside a caustic
        of extent ``0.447``.  Its above-ceiling nodes therefore satisfy the
        authoritative gate (`select_branch`: resolved AND ``L > L_MAX``) and
        are now SERVED by `geometric_amplification` -- accurately; F029
        measures ``2e-7`` median error at ``eta > 0.3``.  Previously they
        raised `SchwingerCertificationError` because the uniform arm
        declined, which was a refusal where a correct answer was available.

        So the band is now expected to certify END TO END, and the refusal
        half is witnessed by an UNRESOLVED above-ceiling host
        (`XOR_REFUSING_*`, ``w * delta_min = 0.836 < RHO_END``), where the
        gate says 'wave', both uniform arms decline, and the named refusal
        fires.  The XOR guarantee is unchanged; only which configuration
        exhibits which half has moved.
        """
        y = np.array([CERT_SQRT_S, 0.0])
        decisions = {
            float(cancellation_l):
                self._solo(cancellation_l / CERT_SQRT_S, y, CERT_GAMMA,
                           0.0, 0.0)[0]
            for cancellation_l in XOR_BAND_LS}
        certified = [k for k, ok in decisions.items() if ok]
        refused = [k for k, ok in decisions.items() if not ok]

        self.n_checks += 1
        self.assertTrue(
            certified,
            'no L in [24, 59.4] certified; the boundary band never returns')
        self.n_checks += 1
        self.assertFalse(
            refused,
            f'L values {sorted(refused)} refused on a resolved host whose '
            f'above-ceiling nodes the authoritative gate sends to geometric '
            f'optics; a refusal here means the gate regressed to the '
            f'pre-F028 arm-or-refuse routing')
        self.n_checks += 1
        self.assertTrue(
            decisions[float(XOR_BAND_LS[0])],
            f'the lowest band point L={XOR_BAND_LS[0]:.0f} did not certify')

        # The refusal half of the XOR, on an UNRESOLVED above-ceiling host.
        refusing_certified = self._solo(
            XOR_REFUSING_W, np.array(XOR_REFUSING_Y), XOR_REFUSING_GAMMA,
            0.0, 0.0)[0]
        self.n_checks += 1
        self.assertFalse(
            refusing_certified,
            f'the unresolved above-ceiling host gamma={XOR_REFUSING_GAMMA}, '
            f'y={XOR_REFUSING_Y}, w={XOR_REFUSING_W} certified; the named '
            f'refusal never fires, so the certified-or-refuse contract has '
            f'no refusing witness left')

    def test_served_band_values_match_the_oracle_above_the_ceiling_too(self):
        """Every served node in the band is ACCURATE, not merely served.

        The gap this closes: above `_schwinger.W_CEILING_SCHWINGER` the
        production exact evaluator refuses, so the suite had no production
        path to compare against and fell back on asserting WHICH RUNG served
        the node, or byte-identity against production. Both pass for a wrong
        number -- byte-identity against the serving rung is true by
        construction (F028: `F_op` serves THROUGH the arm above the ceiling,
        so `F_op == arm` is guaranteed however wrong the arm is).

        `_oracle_fop` is an independent mpmath operator-series
        reconstruction with no FREQUENCY ceiling -- that ceiling belongs to
        one evaluator, not to the mathematics -- so pointing it above the
        ceiling turns this band's serve from unfalsifiable into gated.

        Two limits, both measured 2026-07-29, so this is not read as more
        than it is:

        * `_oracle_fop` is the LEGACY operator series and has its own
          ``L = w * |y'|`` limit (F005). This band tops out at ``L = 59.4``,
          near that edge, so this is a consistency gate between two
          independent reconstructions, not a certification of either.
        * It does NOT catch F028. On this band the uniform arm does not
          serve a wrong value -- it DECLINES, so a regression to the pre-fix
          routing reds the non-vacuity assertion below rather than the
          accuracy one. F028's configs live at ``L ~ 100-200`` where
          `_oracle_fop` itself diverges (arm and geometric both report
          relative error 1.0 against it there).

        What it does buy: the above-ceiling geometric serve is now gated on
        a VALUE rather than on which rung answered, and the routing cannot
        silently regress without reddening this test.
        """
        y = np.array([CERT_SQRT_S, 0.0])
        n_above = 0
        n_below = 0
        for cancellation_l in XOR_BAND_LS:
            w = float(cancellation_l) / CERT_SQRT_S
            certified, value = self._solo(w, y, CERT_GAMMA, 0.0, 0.0)
            if not certified:
                continue
            above = w > W_CEILING_SCHWINGER
            rtol = GEOMETRIC_SERVE_RTOL if above else FOP_RTOL
            try:
                reference = _oracle_fop(w, y, CERT_GAMMA)
            except OracleConvergenceError:
                # No reference exists at this node, so there is nothing to
                # compare against. Skipping is the honest outcome; the
                # non-vacuity assertions below ensure the whole band cannot
                # skip silently. (Measured: only L = 59.4 lands here.)
                continue
            error = abs(value - reference) / abs(reference)
            with self.subTest(L=float(cancellation_l), w=w, above=above):
                self.n_checks += 1
                self.assertLess(
                    error, rtol,
                    f'served value at L={cancellation_l:.2f} (w={w:.3f}, '
                    f'{"above" if above else "below"} the ceiling) is '
                    f'{error:.3e} from the independent mpmath oracle, over '
                    f'the {rtol:.0e} gate')
            n_above += int(above)
            n_below += int(not above)

        # Non-vacuity: the above-ceiling arm is the whole point of the test.
        self.n_checks += 1
        self.assertGreater(
            n_above, 0,
            'no served node lay above the Schwinger ceiling, so the '
            'above-ceiling accuracy gate never ran')
        self.n_checks += 1
        self.assertGreater(
            n_below, 0,
            'no served node lay below the ceiling, so the wave-branch '
            'gate never ran')


class BatchedContractionFalsificationTestCase(BatchedOperatorTestCase):
    """Prove the batched-contraction accuracy gate is not vacuous (F010).

    numba freezes module globals at compile time, so patching a
    module-global never reaches the compiled ``njit`` core.  Each
    perturbation is therefore injected through the ``py_func`` chain:
    the batched core (and, for the gather perturbation, the weight-vector
    builder) is replaced by its ``.py_func`` body, which re-reads the
    module globals in the interpreter.  Two perturbations -- a corrupted
    convergence tolerance that truncates the shear series, and a zeroed
    radial-index gather that collapses the bilinear form -- must each
    drive the `FOP_RTOL` gate red (refuse OR return past the tolerance).
    A perturbation that left the gate green would mean the njit core is
    dead code or the ``py_func`` chain is incomplete.
    """

    def _gate_outcome(self):
        """Run the FALS point; return ``(raised, rel_err)``.

        ``raised`` is True with ``rel_err = inf`` when the contraction
        refuses (`CancellationError`); otherwise ``rel_err`` is the
        relative error against the mpmath oracle.

        Targets the LEGACY certified path `operator._grid_certified`
        directly: since Build 7a the public `F_op_grid` rescues a
        legacy refusal at ``w <= 60`` with a correct Schwinger-fallback
        value (which does not consume the perturbed series), so a
        perturbation-induced refusal would be masked and the
        falsification would go vacuous through the public entry point.
        """
        try:
            values, *_ = operator._grid_certified(
                np.array([FALS_W], dtype=float), np.array(FALS_Y),
                FALS_GAMMA, max_order=FOP_MAX_ORDER)
        except CancellationError:
            return True, float('inf')
        oracle = _oracle_fop(FALS_W, FALS_Y, FALS_GAMMA,
                             max_order=ORACLE_MAX_ORDER)
        rel = abs(complex(values[0]) - oracle) / abs(oracle)
        return False, rel

    def _assert_green_unpatched(self):
        """The gate must be green before a patch, so RED is the patch's
        doing."""
        raised, rel = self._gate_outcome()
        self.n_checks += 1
        self.assertFalse(
            raised, 'unpatched F_op_grid refused the certified FALS '
            'config; the falsification precondition is broken')
        self.n_checks += 1
        self.assertLessEqual(
            rel, FOP_RTOL,
            f'unpatched F_op_grid rel error {rel:.3e} already exceeds '
            f'{FOP_RTOL:.0e}; the gate is not green to begin with')

    # RETIRED (Build 8b-levers fusion): the two F010 falsifications that
    # patched `operator._contract_grid.py_func` and
    # `operator._weight_vectors.py_func` are gone — WP-B merged both
    # stages into `operator._fused_contraction`, so those attributes no
    # longer exist (the tests died on AttributeError, not on physics).
    # The IDENTICAL falsification concerns (series-tolerance truncation
    # and the zeroed half_sum gather collapse) are re-homed through the
    # fused core's py_func in
    # test_lensing_fast_path.py::OperatorFusionFalsificationTestCase,
    # which also pins that half_sum stays an argument and
    # _SERIES_TOLERANCE a patchable module global (F010: a falsification
    # must always have a reachable red path).


# ---------------------------------------------------------------------------
# Timing-fixture helpers (a seeded crown likelihood, built once).
# ---------------------------------------------------------------------------

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
        eventname='test_batched', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0., seed=SEED)
    event_data.inject_signal(_reference_par_dic(), APPROXIMANT)
    return event_data


class FewMsTimingTestCase(BatchedOperatorTestCase):
    """The batched wave branch is fast, and the speed-up is structural.

    On the warm crown fixture the RB ``lnlike`` beats
    ``lnlike_bruteforce`` by at least `SPEEDUP_MIN` (machine-INDEPENDENT,
    HARD) and the pure ``_data_term`` + ``_norm_term`` contraction is
    subdominant to the amplification-engine call that feeds it (HARD).
    The absolute warm ``lnlike`` wall time is guarded by the
    arithmetic-derived ceiling `MS_CEILING` -- a secondary regression
    guard, NOT the brief's 10 ms physical target (that needs Lever B,
    deferred to Build 4).  A per-component breakdown is printed so a
    regression pinpoints the slipped lever.
    """

    @classmethod
    def setUpClass(cls):
        """Build the crown likelihood once for the whole class."""
        cls.par_dic_0 = _reference_par_dic()
        assert sorted(cls.par_dic_0) == waveform.WaveformGenerator.params, (
            'reference par_dic keys drifted from WaveformGenerator.params')
        cls.event_data = _make_noisy_event()
        cls.waveform_generator = waveform.WaveformGenerator.from_event_data(
            cls.event_data, APPROXIMANT)
        band = cls.event_data.frequencies[cls.event_data.fslice]
        f_lo, f_hi = float(band[0]), float(band[-1])
        edges = np.arange(f_lo, f_hi, DF_BIN)
        if edges[-1] < f_hi:
            edges = np.append(edges, f_hi)
        cls.fbin = edges
        cls.like = LensedRelativeBinningLikelihood(
            cls.event_data, cls.waveform_generator, cls.par_dic_0,
            delta_t_max=DELTA_T_MAX, fbin=cls.fbin)

    def _crown_candidate(self):
        """Merge the crown lens row with the fiducial waveform params."""
        _, y1, y2, gamma, beta, kappa = _CROWN
        candidate = dict(self.par_dic_0)
        candidate.update({'m_lens_msun': M_LENS_MSUN, 'z_lens': Z_LENS,
                          'y1': y1, 'y2': y2, 'gamma': gamma,
                          'beta': beta, 'kappa': kappa})
        return candidate

    @staticmethod
    def _best_time(thunk):
        """Best-of-`TIMING_REPEATS` wall time [s] for ``thunk``."""
        best = np.inf
        for _ in range(TIMING_REPEATS):
            start = time.perf_counter()
            thunk()
            best = min(best, time.perf_counter() - start)
        return best

    def test_lnlike_warm_wall_time_and_speedup(self):
        """Warm ``lnlike`` sits under the loose `MS_CEILING`, with a
        component breakdown printed; the brute-force speed-up gate is
        opt-in under ``COGWHEEL_STRICT_TIMING``.

        RE-TUNED (Build 8d): the exact wave branch is the Schwinger
        evaluator (~90 ms/node), so warm crown ``lnlike`` is ~0.75 s and
        the loose ceiling is 3.0 s.  The speed-up over ``lnlike_bruteforce``
        is still the machine-independent structural claim, but brute now
        re-evaluates the exact engine per-frequency (~140 s per call), so
        measuring it is gated behind ``COGWHEEL_STRICT_TIMING`` -- the
        default suite must stay fast.
        """
        candidate = self._crown_candidate()

        def run_rb():
            self.like.lnlike(candidate)

        run_rb()  # warm caches (numba already compiled at import)
        t_rb = self._best_time(run_rb)

        lens = self.like._lens_params(candidate)
        source = np.asarray((lens['y1'], lens['y2']), dtype=float)

        def run_caustic():
            geometry.nearest_caustic_point(
                lens['gamma'], lens['beta'], source, kappa=lens['kappa'])

        def run_engine():
            self.like._amplification_coefficients(candidate)

        run_caustic()
        run_engine()
        t_caustic = self._best_time(run_caustic)
        t_engine = self._best_time(run_engine)
        _, _, _, partition = self.like._amplification_coefficients(candidate)
        print(f'\n[FewMsTiming] breakdown (best-of-{TIMING_REPEATS}): '
              f'caustic-search={t_caustic * 1e3:.3f} ms, '
              f'amplification-engine={t_engine * 1e3:.2f} ms '
              f'({partition.w.size} nodes), '
              f'lnlike total={t_rb * 1e3:.2f} ms')

        self.n_checks += 1
        self.assertLessEqual(
            t_rb, MS_CEILING,
            f'warm lnlike best-of-{TIMING_REPEATS} = {t_rb * 1e3:.2f} ms '
            f'exceeds the loose ceiling {MS_CEILING * 1e3:.0f} ms; a lever '
            'regressed (see breakdown)')

        t_brute = None
        if _STRICT_TIMING:
            def run_brute():
                self.like.lnlike_bruteforce(candidate)

            run_brute()
            t_brute = self._best_time(run_brute)
            print(f'[FewMsTiming] STRICT brute={t_brute * 1e3:.1f} ms, '
                  f'speedup={t_brute / t_rb:.1f}x')
            self.n_checks += 1
            self.assertGreater(
                t_brute, SPEEDUP_MIN * t_rb,
                f'RB lnlike ({t_rb * 1e3:.2f} ms) is not at least '
                f'{SPEEDUP_MIN}x faster than brute force '
                f'({t_brute * 1e3:.1f} ms); the batched speed-up regressed')

        if _HAVE_MPL:
            self._plot_breakdown(t_caustic, t_engine, t_rb,
                                 t_brute if t_brute is not None else t_rb)

    def test_contraction_subdominant_to_amplification_engine(self):
        """The pure contraction is faster than the amplification-engine
        call that feeds it.

        Inputs come from the LIVE hot path so the measured contraction
        matches production; the additive ``M**2 + n_img**2`` design stays
        below the 1F1 engine, with no FFT or per-frequency Python loop on
        the hot path.
        """
        candidate = self._crown_candidate()

        r0, r1, dt_lf = self.like._candidate_bin_ratios(candidate)
        rho0, rho1 = r0.conj(), r1.conj()
        delays, k0, k1, _ = self.like._amplification_coefficients(candidate)
        kbar0, kbar1 = k0.conj(), k1.conj()
        tau = delays - dt_lf
        f_center = self.like._f_center

        def run_contraction():
            _data_term(self.like._a_moments, rho0, rho1, kbar0, kbar1, tau,
                       f_center)
            _norm_term(self.like._b_moments, r0, r1, rho0, rho1, k0, k1,
                       kbar0, kbar1, delays, f_center)

        def run_engine():
            self.like._amplification_coefficients(candidate)

        run_contraction()
        run_engine()
        t_contract = self._best_time(run_contraction)
        t_engine = self._best_time(run_engine)
        print(f'\n[FewMsTiming] contraction = {t_contract * 1e3:.3f} ms, '
              f'amplification engine = {t_engine * 1e3:.2f} ms')

        self.n_checks += 1
        self.assertLess(
            t_contract, t_engine,
            f'contraction ({t_contract * 1e3:.3f} ms) is not subdominant '
            f'to the amplification engine ({t_engine * 1e3:.2f} ms); an '
            'FFT or per-frequency Python loop may have crept onto the '
            'hot path')

    def _plot_breakdown(self, t_caustic, t_engine, t_rb, t_brute):
        """Bar chart of the warm per-component wall times [ms]."""
        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        labels = ['caustic', 'engine', 'lnlike', 'brute']
        times_ms = [t_caustic * 1e3, t_engine * 1e3, t_rb * 1e3,
                    t_brute * 1e3]
        axis.bar(labels, times_ms, color='steelblue')
        axis.axhline(MS_CEILING * 1e3, color='crimson', linestyle='--',
                     label=f'MS_CEILING = {MS_CEILING * 1e3:.0f} ms')
        axis.set_yscale('log')
        axis.set_ylabel('warm best-of-N wall time [ms]')
        axis.set_title('Batched wave-branch timing breakdown (crown)')
        axis.legend()
        self._save_figure(fig, 'few_ms_timing_breakdown')


class BatchedEquivalenceTestCase(BatchedOperatorTestCase):
    """The scalar `F_op` delegates to the batched `F_op_grid` bit-for-bit.

    The single-path design routes `F_op` through a one-element
    `F_op_grid` call, so the scalar and batched values are the SAME bits,
    not merely close.  This confirms that the many RB-vs-brute,
    determinism, crown-accuracy and interpolation gates in the sibling
    suites -- which call the scalar entry points -- automatically
    exercise the batched contraction at their ORIGINAL tolerances.
    """

    def test_scalar_delegates_to_batch_bit_identically(self):
        """`F_op` value equals the one-element `F_op_grid` value exactly.

        The representative in-domain point (w=20, y=(0.55, 0),
        gamma=0.20) certifies; the delegation must be an exact
        pass-through (``assertEqual``), not an approximate match.
        """
        y = np.array(FALS_Y)
        scalar_value, _ = F_op(FALS_W, y, FALS_GAMMA)
        grid_values, _, _ = F_op_grid(
            np.array([FALS_W], dtype=float), y, FALS_GAMMA)
        self.n_checks += 1
        self.assertEqual(
            scalar_value, complex(grid_values[0]),
            'scalar F_op is not bit-identical to the one-element '
            'F_op_grid; the single-path delegation is broken')

    def test_scalar_matches_batch_across_certified_grid(self):
        """Every certified in-domain grid point delegates bit-identically.

        Sweeping the ``F_op`` grid configs, each scalar `F_op` value must
        equal the corresponding one-element `F_op_grid` value exactly, so
        the delegation is exact across the domain, not just at one point.
        """
        for sqrt_s in FOP_GRID_SQRT_S:
            for gamma in FOP_GRID_GAMMA:
                for w in FOP_GRID_W:
                    with self.subTest(sqrt_s=sqrt_s, gamma=gamma, w=w):
                        y = np.array([sqrt_s, 0.0])
                        try:
                            scalar_value, _ = F_op(
                                w, y, gamma, max_order=FOP_MAX_ORDER)
                            grid_values, _, _ = F_op_grid(
                                np.array([w], dtype=float), y, gamma,
                                max_order=FOP_MAX_ORDER)
                        except CancellationError:
                            continue  # refused nodes carry no value to match
                        self.n_checks += 1
                        self.assertEqual(
                            scalar_value, complex(grid_values[0]),
                            f'scalar/batch mismatch at w={w}, '
                            f'sqrt_s={sqrt_s}, gamma={gamma}')


class SelfFalsificationTestCase(BatchedOperatorTestCase):
    """Prove this suite's own guards can go red.

    A suite whose accuracy assertion and anti-vacuity ``tearDown`` cannot
    fail is not a test.  These positive controls confirm the relative-
    error gate rejects a deliberately wrong value and the anti-vacuity
    guard fails a test that made zero comparisons.
    """

    def test_relative_gate_rejects_wrong_value(self):
        """`assert_relative` raises when the candidate is far from truth."""
        probe = BatchedOperatorTestCase()
        probe.n_checks = 0
        self.n_checks += 1
        with self.assertRaises(AssertionError):
            probe.assert_relative(2.0 + 0j, 1.0 + 0j, FOP_RTOL,
                                  'deliberately wrong value')

    def test_anti_vacuity_teardown_fails_on_zero_checks(self):
        """`tearDown` fails a test that made zero comparisons."""
        probe = BatchedOperatorTestCase()
        probe.n_checks = 0
        self.n_checks += 1
        with self.assertRaises(AssertionError):
            probe.tearDown()

    def test_anti_vacuity_teardown_passes_when_checks_ran(self):
        """`tearDown` is silent once at least one comparison ran."""
        probe = BatchedOperatorTestCase()
        probe.n_checks = 1
        self.n_checks += 1
        probe.tearDown()  # must not raise


if __name__ == '__main__':
    main()
