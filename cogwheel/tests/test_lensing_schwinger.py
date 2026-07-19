"""
Tests for `lensing.chang_refsdal._schwinger`, the exact 1D
Schwinger-parameter wave-branch EVALUATOR for the macro-saddle domain
(``f_schwinger``, its certify-XOR-refuse contract, the deep-band F009-S
pins, the dd-mandatory falsifications, and the warm per-point cost
measurement).  The saddle census / dispatch / mass-sheet / geometric
branch live in `test_lensing_saddle_geometry.py`, not here.

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
import math
import textwrap
import time
from unittest import TestCase, main, mock

import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import _schwinger
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError, W_CEILING_SCHWINGER, f_schwinger)

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

    _ORACLE_PATH = (_oracle_1d, _oracle_saddle, _oracle_point_mass)

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
        not an input error) and domain errors stay `ValueError`."""
        self.n_checks += 1
        self.assertTrue(
            issubclass(SchwingerCertificationError, RuntimeError))
        for bad_args in ((0.0, XOR_Y, XOR_GAMMA),
                         (-1.0, XOR_Y, XOR_GAMMA),
                         (3.0, XOR_Y, 1.0),
                         (3.0, XOR_Y, 0.5)):
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


if __name__ == '__main__':
    main()
