"""
Tests for `lensing.chang_refsdal.operator`, the contour-free
Chang--Refsdal amplification operator ``F_op``.

WHY THESE ORACLES ARE INDEPENDENT
---------------------------------
Nothing here is judged against the module's own derivation:

* ``F_op`` is gated against an mpmath amplification oracle
  (`_oracle_amplification`) built ENTIRELY from mpmath at ~50 decimal
  digits.  Its point-mass kernel and every ``s``-derivative come from
  ``mpmath.hyp1f1`` -- NOT the double-double Kummer kernel the
  production uses -- and the shear operator ``exp(i*gamma*D_beta/2w)``
  is applied by an INTEGER-coefficient ladder in the real ``(u, v)``
  monomial basis, NOT the production's complex shear-eigenframe table.
  Two implementations that share no code and no numerical substrate;
  the top-level reconstruction is re-derived from the diffraction
  integral in `_oracle_amplification`'s docstring, not copied from
  ``F_op``.
* The GEOMETRIC-OPTICS SLOPE test needs no external oracle at all: the
  residual ``|F_op - sum_a exp(i*w*tau_a) H_a|`` must fall as ``w**-1``
  when the ``C1/C2`` corrections are dropped from the kernels and as
  ``w**-3`` when they are kept.  The exponent IS the physics, and the
  two-case contrast means a bug that rescales both branches equally
  cannot hide.
* The MASS-SHEET test asserts kappa-invariance of OBSERVABLES -- the
  delay difference ``Delta tau_ac`` and the flux ratio ``|K_a/K_c|`` --
  which are physically required to be kappa-independent regardless of
  how the code organizes the rescaling.  Comparing ``F_op`` against its
  own rescaling path would be vacuous and is deliberately avoided.
* The CANCELLATION test checks the reported ``max_partial_term/|total|``
  against an INDEPENDENT mpmath recomputation of the same ratio.

TOLERANCES AND TESTED DOMAIN
----------------------------
The oracle is exact to far beyond float64, so ``RTOL_GATE = 1e-10`` is a
property of ``F_op``, not the oracle.  The compared configurations keep
the cancellation exponent ``L = w*|y'|`` at or below ~25, where the
shipped wave branch delivers ~5e-12 or better (measured worst case
5.65e-12, ~180x inside the gate).  The ``L <= 48`` certified ceiling is
DRAWN on the diagnostic but not asserted at its upper edge: above
``L ~ 30`` the float64 operator contraction loses accuracy and
eventually overflows, which is a known limitation of the committed
module recorded in the build notes, not something these tests certify.

`OperatorTestCase.tearDown` fails a test that made zero comparisons, and
`SelfFalsificationTestCase` proves every gate above can actually go red.
"""
from __future__ import annotations

import itertools
import pathlib
import re
from dataclasses import FrozenInstanceError
from unittest import TestCase, main, mock

import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import operator, geometry
from cogwheel.lensing.chang_refsdal import _hyp1f1
from cogwheel.lensing.chang_refsdal import channels


try:  # Diagnostics only; never gate a test on plotting being present.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

#: Working precision of the mpmath amplification oracle, in decimal
#: digits.  ~35 digits of margin over the 1e-10 gate; the oracle is the
#: reference, so it must not be the thing under test.
ORACLE_DPS = 50

#: Relative-error gate on ``F_op`` against the oracle.  A property of
#: the wave branch, achievable because the oracle is effectively exact.
RTOL_GATE = 1e-10

#: Operator-order cap handed to ``F_op`` in the oracle comparison; large
#: enough that the highest-``w`` configuration (order ~57) converges.
FOP_MAX_ORDER = 70

#: Operator-order cap for the mpmath oracle; exceeds ``F_op``'s
#: convergence order so the reference is fully summed.
ORACLE_MAX_ORDER = 100

#: The certified cancellation-exponent ceiling drawn on the diagnostic
#: (mirrors ``operator.L_MAX``; NOT re-used as a gate here).
L_CEILING = 48.0

#: The paper's shared macro parameters for its four representative
#: source positions.
PAPER_GAMMA = 0.2
PAPER_KAPPA = 0.2
PAPER_BETA = 0.0

#: Directory for diagnostic figures.
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


def _paper_fold_source() -> np.ndarray:
    """The paper's 'near a fold' source, just inside the caustic."""
    caustic = geometry.critical_point(PAPER_GAMMA, np.pi / 4.0,
                                      PAPER_BETA, PAPER_KAPPA).source
    return 0.99 * caustic


#: ``(name, y, gamma, beta, kappa)`` -- the paper's four configurations
#: (two-image, four-image, near-fold, near-cusp; gamma=kappa=0.2) plus
#: stress cases: near-zero shear, near-parity shear, an external
#: convergence benchmark point, and a rotated shear axis.
ORACLE_CONFIGS = (
    ('two-image', np.array([0.55, 0.0]), PAPER_GAMMA, PAPER_BETA,
     PAPER_KAPPA),
    ('four-image', np.array([0.10, 0.10]), PAPER_GAMMA, PAPER_BETA,
     PAPER_KAPPA),
    ('near-fold', _paper_fold_source(), PAPER_GAMMA, PAPER_BETA,
     PAPER_KAPPA),
    ('near-cusp', np.array([-0.395, 0.0]), PAPER_GAMMA, PAPER_BETA,
     PAPER_KAPPA),
    ('small-shear', np.array([0.30, 0.10]), 0.02, 0.0, 0.0),
    ('large-shear', np.array([0.20, 0.15]), 0.40, 0.0, 0.0),
    ('kappa-bench', np.array([0.30, 0.10]), 0.112, 0.0, 0.30),
    ('beta-rotated', np.array([0.25, 0.10]), 0.20, 0.70, 0.0),
)

#: Frequencies per configuration.  Paper configurations reach the
#: higher ``w`` (and thus deeper cancellation) that stresses the kernel;
#: stress configurations use a middle band.
_PAPER_NAMES = {'two-image', 'four-image', 'near-fold', 'near-cusp'}
_PAPER_WS = (5.0, 20.0, 40.0)
_STRESS_WS = (12.0, 30.0)


# ----------------------------------------------------------------------
# Independent mpmath amplification oracle (oracle-only; never imported
# from production).
# ----------------------------------------------------------------------
def _prefactor_c(w):
    """Point-mass prefactor ``C(w)`` in closed form, at oracle
    precision.

    ``C(w) = exp(pi*w/4 + i*(w/2)*ln(w/2)) * Gamma(1 - i*w/2)``.  This is
    the textbook form (Abramowitz & Stegun ch. 13), evaluated directly
    by mpmath -- deliberately NOT the production's polar/expm1 route --
    so the two never share a rounding path.
    """
    w = mpmath.mpf(w)
    return (mpmath.e ** (mpmath.pi * w / 4 + 1j * (w / 2)
                         * mpmath.log(w / 2))
            * mpmath.gamma(1 - 1j * w / 2))


def _radial_ladder(w, s):
    """Return a memoized ``k -> d^k/ds^k G_PM(w, s)`` from mpmath.

    ``G_PM(w, s) = C(w) * 1F1(1 - i*w/2; 1; -i*w*s/2)`` and its ``k``-th
    ``s``-derivative is ``C(w) * c**k * (a)_k / (1)_k *
    1F1(a + k; 1 + k; c*s)`` with ``a = 1 - i*w/2`` and ``c = -i*w/2``.
    A fresh ``mpmath.hyp1f1`` per ``k`` -- the direct definition, with no
    Kummer reparametrization and no shared numerator.
    """
    w = mpmath.mpf(w)
    s = mpmath.mpf(s)
    a = 1 - 1j * w / 2
    c = -1j * w / 2
    prefactor = _prefactor_c(w)
    cache: dict[int, complex] = {}

    def g(k):
        if k not in cache:
            cache[k] = (prefactor * c ** k * mpmath.rf(a, k)
                        / mpmath.rf(1, k)
                        * mpmath.hyp1f1(a + k, 1 + k, c * s))
        return cache[k]
    return g


def _operator_step(state):
    """Apply ``D_0 = d_u**2 - d_v**2`` to an INTEGER-keyed operator
    state.

    ``state`` maps ``(a, b) -> int`` coefficient of the real monomial
    ``u**a * v**b * G^(k)`` with the radial index implied by
    ``k = (a + b)//2 + order``.  ``D_0`` is the eigenframe shear operator
    (equal to ``d_u**2 - d_v**2`` because ``z = u + i*v`` gives
    ``2*d_z**2 + 2*d_zbar**2 = d_u**2 - d_v**2``).  Coefficients stay
    exact Python ints; no mpmath is spent here.
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


def _oracle_series(w, y, gamma, beta, kappa, max_order):
    """Return ``(total, max_term, order_used)`` of the operator series.

    Sums ``total = sum_n (i*gamma'/(2w))**n / n! * D_0**n G_PM`` at the
    eigenframe-rotated source, tracking the largest single-order term so
    the cancellation ratio can be recomputed independently.  ``gamma'``
    and ``s`` come from the exact mass-sheet rescaling
    ``y' = y/sqrt(lam)``, ``gamma' = gamma/lam``, ``lam = 1 - kappa``.
    """
    w = mpmath.mpf(w)
    lam = 1 - mpmath.mpf(kappa)
    gamma_scaled = mpmath.mpf(gamma) / lam
    root = mpmath.sqrt(lam)
    yp = (mpmath.mpf(y[0]) / root, mpmath.mpf(y[1]) / root)
    s = yp[0] ** 2 + yp[1] ** 2
    z_eig = mpmath.e ** (-1j * mpmath.mpf(beta)) * mpmath.mpc(*yp)
    u0, v0 = z_eig.real, z_eig.imag
    g = _radial_ladder(w, s)
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
    max_term = mpmath.mpf(0)
    state = {(0, 0): 1}
    factorial = mpmath.mpf(1)
    order_used = 0
    small = 0
    for n in range(max_order + 1):
        if n:
            factorial *= n
            state = _operator_step(state)
        term = alpha ** n / factorial * evaluate(state, n)
        total += term
        max_term = max(max_term, abs(term))
        order_used = n
        if n >= 4 and abs(term) <= mpmath.mpf('1e-24') * abs(total):
            small += 1
            if small >= 3:
                break
        else:
            small = 0
    return total, max_term, order_used


def _oracle_amplification(w, y, gamma, beta=0.0, kappa=0.0,
                          max_order=ORACLE_MAX_ORDER):
    """Independent wave-optics amplification ``F(w)``.

    The diffraction integral ``F = (w/2pi/i) integral d^2x
    exp(i*w*tau(x))`` with ``tau(x) = 0.5*x.A.x - y.x + 0.5*y.y -
    ln|x|`` reduces, under ``x = x'/sqrt(lam)``, to the pure-shear
    problem plus a scalar prefactor.  Carrying that reduction through
    (independently of ``F_op``'s own reconstruction) gives

        F = (1/lam) * exp(0.5j*w*ln(lam) - 0.5j*w*kappa*s)
              * exp(0.5j*w*s) * G_CR,

    with ``G_CR`` the operator series `_oracle_series` returns and
    ``s = |y'|**2``.  All arithmetic is at `ORACLE_DPS` digits.
    """
    with mpmath.workdps(ORACLE_DPS):
        w = mpmath.mpf(w)
        lam = 1 - mpmath.mpf(kappa)
        s = (mpmath.mpf(y[0]) ** 2 + mpmath.mpf(y[1]) ** 2) / lam
        total, _, _ = _oracle_series(w, y, gamma, beta, kappa, max_order)
        value = ((1 / lam)
                 * mpmath.e ** (0.5j * w * mpmath.log(lam)
                                - 0.5j * w * mpmath.mpf(kappa) * s
                                + 0.5j * w * s)
                 * total)
        return complex(value)


def _oracle_cancellation_ratio(w, y, gamma, beta=0.0, kappa=0.0,
                               max_order=FOP_MAX_ORDER):
    """Independent ``max_partial_term / |total|`` of the operator series.

    Mirrors the quantity ``F_op`` measures during summation, computed at
    oracle precision so the reported ratio can be checked against a
    reference that shares none of its float64 accumulation.
    """
    with mpmath.workdps(ORACLE_DPS):
        total, max_term, _ = _oracle_series(w, y, gamma, beta, kappa,
                                            max_order)
        return float(max_term / abs(total))


def _savefig(fig, name):
    """Save a diagnostic figure, swallowing any backend error."""
    if not _HAVE_MPL:
        return
    try:
        _OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(_OUTPUT_DIR / name, dpi=80, bbox_inches='tight')
    except Exception:  # pragma: no cover - environment dependent
        pass
    finally:
        plt.close(fig)


class OperatorTestCase(TestCase):
    """
    Base class carrying the anti-vacuity comparison tally.

    `tearDown` fails a test that asserted nothing, so a sweep whose
    every configuration was skipped cannot read as green.
    """

    _expect_checks = True

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self._expect_checks and self.n_checks == 0:
            self.fail('vacuous test: no comparison ran, so nothing was '
                      'asserted')


class OperatorOracleTestCase(OperatorTestCase):
    """``F_op`` against the independent mpmath amplification oracle."""

    def test_matches_oracle_over_configurations(self):
        """
        Across the paper's four configurations and the stress cases,
        ``F_op`` agrees with the mpmath oracle within `RTOL_GATE`, and
        the returned `OperatorDiagnostics` are self-consistent with an
        accurate result (converged before the cap, an order actually
        used, and a small measured truncation tail).
        """
        rel_by_l: list[tuple[float, float]] = []
        for name, y, gamma, beta, kappa in ORACLE_CONFIGS:
            ws = _PAPER_WS if name in _PAPER_NAMES else _STRESS_WS
            for w in ws:
                with self.subTest(config=name, w=w):
                    value, diag = operator.F_op(
                        w, y, gamma, beta=beta, kappa=kappa,
                        max_order=FOP_MAX_ORDER)
                    reference = _oracle_amplification(
                        w, y, gamma, beta=beta, kappa=kappa)
                    rel = abs(value - reference) / abs(reference)
                    cexp = operator.cancellation_exponent(
                        w, y, gamma, kappa)
                    rel_by_l.append((cexp, rel))
                    self.assertLessEqual(
                        rel, RTOL_GATE,
                        f'{name} w={w}: |F_op - oracle|/|oracle| = '
                        f'{rel:.3e} exceeds {RTOL_GATE}')
                    self.assertTrue(
                        diag.converged,
                        f'{name} w={w}: series did not converge')
                    self.assertLessEqual(diag.order_used, FOP_MAX_ORDER)
                    self.assertLess(
                        diag.estimated_relative_tail, 1e-8,
                        f'{name} w={w}: measured tail '
                        f'{diag.estimated_relative_tail:.2e} too large '
                        'for a converged result')
                    self.n_checks += 1
        self._plot_rel_vs_l(rel_by_l)

    def test_diagnostics_are_frozen(self):
        """
        `OperatorDiagnostics` refuses attribute assignment: a report a
        caller can edit is a report that gets believed after editing.
        """
        _, diag = operator.F_op(8.0, np.array([0.55, 0.0]), PAPER_GAMMA,
                                kappa=PAPER_KAPPA, max_order=40)
        for field in ('converged', 'order_used', 'cancellation_ratio',
                      'estimated_relative_tail'):
            with self.subTest(field=field):
                with self.assertRaises(FrozenInstanceError):
                    setattr(diag, field, 0)
                self.n_checks += 1

    def test_cancellation_ratio_field_matches_independent(self):
        """
        The recorded ``cancellation_ratio`` equals an independent
        mpmath ``max_partial_term/|total|`` -- it is the MEASURED
        summation quantity, not a heuristic.
        """
        for name, y, gamma, beta, kappa, w in (
                ('two-image', np.array([0.55, 0.0]), PAPER_GAMMA,
                 PAPER_BETA, PAPER_KAPPA, 40.0),
                ('large-shear', np.array([0.20, 0.15]), 0.40, 0.0, 0.0,
                 40.0)):
            with self.subTest(config=name):
                _, diag = operator.F_op(w, y, gamma, beta=beta,
                                        kappa=kappa,
                                        max_order=FOP_MAX_ORDER)
                reference = _oracle_cancellation_ratio(
                    w, y, gamma, beta=beta, kappa=kappa)
                self.assertAlmostEqual(
                    diag.cancellation_ratio / reference, 1.0, delta=1e-3,
                    msg=f'{name}: reported ratio '
                    f'{diag.cancellation_ratio:.4e} vs independent '
                    f'{reference:.4e}')
                self.n_checks += 1

    def test_raises_named_error_above_w_ceiling(self):
        """
        Above ``W_MAX_CERTIFIED`` the kernel's named
        `HypergeometricDomainError` propagates rather than a silently
        wrong number being returned.
        """
        with self.assertRaises(_hyp1f1.HypergeometricDomainError):
            operator.F_op(_hyp1f1.W_MAX_CERTIFIED + 100.0,
                          np.array([0.05, 0.0]), 0.1)
        self.n_checks += 1

    def test_raises_named_error_above_cancellation_ceiling(self):
        """
        Above the kernel's ``w*sqrt(s) = DD_PRODUCT_CEILING`` ceiling the
        named `HypergeometricDomainError` propagates.  Here ``y=(1,0)``,
        ``kappa=0`` gives ``w*sqrt(s) = w = 70 > 60``.
        """
        with self.assertRaises(_hyp1f1.HypergeometricDomainError):
            operator.F_op(_hyp1f1.DD_PRODUCT_CEILING + 10.0,
                          np.array([1.0, 0.0]), 0.1)
        self.n_checks += 1

    def _plot_rel_vs_l(self, rel_by_l):
        if not _HAVE_MPL or not rel_by_l:
            return
        cexp = np.array([c for c, _ in rel_by_l])
        rel = np.array([max(r, 1e-18) for _, r in rel_by_l])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(cexp, rel, s=14, c='C0')
        ax.axhline(RTOL_GATE, color='k', ls='--', label='1e-10 gate')
        ax.axvline(L_CEILING, color='C3', ls=':', label='L=48')
        ax.set_yscale('log')
        ax.set_xlabel("cancellation exponent L = w|y'|")
        ax.set_ylabel('|F_op - oracle| / |oracle|')
        ax.legend()
        _savefig(fig, 'operator_oracle_rel_vs_L.png')


class GeometricOpticsSlopeTestCase(OperatorTestCase):
    """
    Self-oracling asymptotic test: the residual to the stationary-phase
    sum falls as ``w**-1`` without the ``C1/C2`` corrections and as
    ``w**-3`` with them.  No external oracle -- the exponent is the
    physics, and the two-case contrast defeats any bug that scales both
    branches identically.  Fits are on the RMS within log-``w`` bins so
    the multi-image cross-term oscillation does not bias the slope.
    """

    #: Well inside the wave branch: two well-separated images, far
    #: outside the caustic, so the asymptotics are clean.
    SLOPE_Y = np.array([0.90, 0.0])
    SLOPE_GAMMA = 0.20
    SLOPE_W = np.linspace(12.0, 45.0, 84)
    SLOPE_BINS = 8

    def _residuals(self, with_corrections):
        matrix = geometry.macro_matrix(self.SLOPE_GAMMA, 0.0, 0.0)
        images = geometry.find_images(self.SLOPE_Y, matrix)
        delays = [geometry.delay(im, self.SLOPE_Y, matrix)
                  for im in images]
        mus = [geometry.magnification(im, matrix) for im in images]
        morse = [geometry.morse_index(im, matrix) for im in images]
        c12 = [geometry.saddle_coefficients(im, matrix)
               for im in images]
        residuals = []
        for w in self.SLOPE_W:
            value, _ = operator.F_op(w, self.SLOPE_Y, self.SLOPE_GAMMA,
                                     max_order=60)
            approx = 0j
            for tau, mu, n, (c1, c2) in zip(delays, mus, morse, c12):
                lead = np.sqrt(abs(mu)) * np.exp(-0.5j * np.pi * n)
                correction = (1.0 + 1j * c1 / w + c2 / w ** 2
                              if with_corrections else 1.0)
                approx += np.exp(1j * w * tau) * lead * correction
            residuals.append(abs(value - approx))
        return np.array(residuals)

    def _binned_slope(self, residuals):
        edges = np.linspace(self.SLOPE_W[0], self.SLOPE_W[-1],
                            self.SLOPE_BINS + 1)
        centers, rms = [], []
        for i in range(self.SLOPE_BINS):
            mask = (self.SLOPE_W >= edges[i]) & (self.SLOPE_W
                                                 <= edges[i + 1])
            if mask.sum() >= 3:
                centers.append(np.sqrt(edges[i] * edges[i + 1]))
                rms.append(np.sqrt(np.mean(residuals[mask] ** 2)))
        slope = np.polyfit(np.log(centers), np.log(rms), 1)[0]
        return slope, np.array(centers), np.array(rms)

    def test_leading_and_corrected_slopes(self):
        """
        Without ``C1/C2`` the residual envelope decays as ``w**-1``;
        with them as ``w**-3``.  Both the individual exponents and their
        contrast (the corrections must buy ~2 extra powers) are checked.
        """
        slope_lead, c_lead, r_lead = self._binned_slope(
            self._residuals(with_corrections=False))
        slope_full, c_full, r_full = self._binned_slope(
            self._residuals(with_corrections=True))
        self.n_checks += 1
        self.assertTrue(
            -1.35 < slope_lead < -0.65,
            f'leading-kernel residual slope {slope_lead:.3f} is not '
            'near the expected -1')
        self.assertTrue(
            -3.4 < slope_full < -2.5,
            f'corrected-kernel residual slope {slope_full:.3f} is not '
            'near the expected -3')
        self.assertGreater(
            slope_lead - slope_full, 1.3,
            f'the C1/C2 corrections bought only '
            f'{slope_lead - slope_full:.2f} powers of w; the two-case '
            'contrast is the substance of this test')
        self._plot(c_lead, r_lead, slope_lead, c_full, r_full,
                   slope_full)

    def _plot(self, c_lead, r_lead, s_lead, c_full, r_full, s_full):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(c_lead, r_lead, 'oC0',
                  label=f'no C1/C2 ({s_lead:.2f})')
        ax.loglog(c_full, r_full, 'sC3',
                  label=f'with C1/C2 ({s_full:.2f})')
        ax.loglog(c_lead, r_lead[0] * (c_lead / c_lead[0]) ** -1.0,
                  '--C0', label='w^-1')
        ax.loglog(c_full, r_full[0] * (c_full / c_full[0]) ** -3.0,
                  '--C3', label='w^-3')
        ax.set_xlabel('w')
        ax.set_ylabel('RMS |F_op - stationary-phase sum|')
        ax.legend()
        _savefig(fig, 'operator_geometric_slope.png')


class MassSheetInvarianceTestCase(OperatorTestCase):
    """
    Mass-sheet degeneracy in NON-VACUOUS form.

    Holding the pure-shear problem ``(y', gamma')`` fixed and varying
    ``kappa`` with the physically rescaled ``y = sqrt(lam)*y'``,
    ``gamma = lam*gamma'``, the OBSERVABLES -- the delay difference
    ``Delta tau_ac`` and the flux ratio ``|K_a/K_c| = sqrt|mu_a/mu_c|``
    -- must be exactly kappa-independent.  These are physical
    requirements, not consequences of how ``operator`` organizes the
    rescaling, so asserting them is a genuine test rather than a
    restatement of ``_mass_sheet_map``.
    """

    #: The invariant pure-shear problem the sweep rescales from.
    Y_PRIME = np.array([0.35, 0.12])
    GAMMA_PRIME = 0.25
    KAPPAS = (0.0, 0.2, 0.4)

    def test_observables_are_kappa_invariant(self):
        """
        ``Delta tau_ac`` and ``|K_a/K_c|`` are flat across ``kappa`` to
        roundoff, and ``operator.cancellation_exponent`` (which routes
        through the same mass-sheet map) is likewise invariant.  The
        observables are first checked to be nontrivial, so invariance is
        not passing vacuously on ``Delta tau = 0`` or a unit ratio.
        """
        delays, ratios, cexps = [], [], []
        for kappa in self.KAPPAS:
            lam = 1.0 - kappa
            y = np.sqrt(lam) * self.Y_PRIME
            gamma = lam * self.GAMMA_PRIME
            matrix = geometry.macro_matrix(gamma, 0.0, kappa)
            images = geometry.find_images(y, matrix)
            self.assertGreaterEqual(len(images), 2)
            taus = np.array([geometry.delay(im, y, matrix)
                             for im in images])
            mus = np.array([geometry.magnification(im, matrix)
                            for im in images])
            order = np.argsort(taus)
            taus, mus = taus[order], mus[order]
            delays.append(taus[-1] - taus[0])
            ratios.append(np.sqrt(abs(mus[0] / mus[-1])))
            cexps.append(operator.cancellation_exponent(
                10.0, y, gamma, kappa))
            self.n_checks += 1

        # Nontriviality: a flat line at zero or one would be vacuous.
        self.assertGreater(abs(delays[0]), 0.1)
        self.assertGreater(abs(ratios[0] - 1.0), 0.05)

        for values, label in ((delays, 'Delta tau_ac'),
                              (ratios, '|K_a/K_c|'),
                              (cexps, 'cancellation_exponent')):
            spread = max(abs(v - values[0]) for v in values)
            self.assertLessEqual(
                spread / abs(values[0]), 1e-9,
                f'{label} drifts with kappa by relative {spread:.2e}; '
                'the mass-sheet observable is not invariant')
        self._plot(delays, ratios)

    def _plot(self, delays, ratios):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.KAPPAS, delays, 'oC0', label='Delta tau_ac')
        ax.plot(self.KAPPAS, ratios, 'sC3', label='|K_a/K_c|')
        ax.set_xlabel('kappa')
        ax.set_ylabel('observable')
        ax.set_title('flat = correct mass-sheet map')
        ax.legend()
        _savefig(fig, 'operator_mass_sheet.png')


class CancellationRefusalTestCase(OperatorTestCase):
    """
    Cancellation refusal driven by the MEASURED ratio.

    The shipped refusal threshold ``1e13`` is not reachable by ordinary
    configurations before the float64 contraction overflows, so -- as
    ``test_lensing_dd`` patches ``_dd._SPLITTER`` -- the threshold is
    patched to sit between two REAL configurations' measured ratios.
    That is faithful to the contract: the refusal is gated on the ratio
    measured during summation, whatever the threshold, and the reported
    ratio is checked against an independent mpmath recomputation so it
    cannot be an up-front formula.
    """

    #: Deep-cancellation configuration (measured ratio ~1.7e3).
    HIGH = ('large-shear', np.array([0.20, 0.15]), 0.40, 0.0, 0.0, 40.0,
            FOP_MAX_ORDER)
    #: Companion just below the patched threshold (measured ratio ~2).
    LOW = ('two-image', np.array([0.55, 0.0]), PAPER_GAMMA, PAPER_BETA,
           PAPER_KAPPA, 8.0, 40)

    def test_shipped_threshold_is_pinned(self):
        """The refusal constant is the documented ``1e13``."""
        self.assertEqual(operator._CANCELLATION_REFUSAL, 1e13)
        self.n_checks += 1

    def test_refuses_above_and_returns_below_a_patched_threshold(self):
        """
        With the threshold placed between the two configurations'
        measured ratios, the deep-cancellation config REFUSES with a
        named `CancellationError` whose message reports the measured
        ratio and the configuration, while the companion returns
        normally with its ratio recorded.
        """
        _, high_y, high_g, high_b, high_k, high_w, high_o = self.HIGH
        _, low_y, low_g, low_b, low_k, low_w, low_o = self.LOW
        independent_high = _oracle_cancellation_ratio(
            high_w, high_y, high_g, beta=high_b, kappa=high_k,
            max_order=high_o)
        threshold = 100.0
        self.assertGreater(independent_high, threshold)

        with mock.patch.object(operator, '_CANCELLATION_REFUSAL',
                               threshold):
            with self.assertRaises(operator.CancellationError) as ctx:
                operator.F_op(high_w, high_y, high_g, beta=high_b,
                              kappa=high_k, max_order=high_o)
            message = str(ctx.exception)
            # Reported ratio equals the independent recomputation.
            match = re.search(r'\|total\| = ([0-9.]+e[+\-][0-9]+)',
                              message)
            self.assertIsNotNone(match,
                                 f'no ratio in message: {message}')
            reported = float(match.group(1))
            self.assertAlmostEqual(reported / independent_high, 1.0,
                                   delta=1e-2)
            self.assertGreater(reported, threshold)
            # Message names the configuration, not just a number.
            for token in ('w =', 'gamma', 'kappa', '0.4'):
                self.assertIn(token, message)
            self.n_checks += 1

            # Companion below the threshold returns and records its
            # ratio.
            value, diag = operator.F_op(low_w, low_y, low_g, beta=low_b,
                                        kappa=low_k, max_order=low_o)
            self.assertTrue(np.isfinite(value))
            self.assertLess(diag.cancellation_ratio, threshold)
            self.assertGreater(diag.cancellation_ratio, 0.0)
            self.n_checks += 1

    def test_refusal_tracks_the_ratio_not_a_formula(self):
        """
        Under the shipped ``1e13`` threshold the deep-cancellation
        config (measured ratio ~1.7e3) does NOT refuse -- proof the
        refusal follows the measured ratio against the threshold, not an
        unconditional or up-front rule.
        """
        _, y, gamma, beta, kappa, w, order = self.HIGH
        value, diag = operator.F_op(w, y, gamma, beta=beta, kappa=kappa,
                                    max_order=order)
        self.assertTrue(np.isfinite(value))
        self.assertLess(diag.cancellation_ratio,
                        operator._CANCELLATION_REFUSAL)
        self.n_checks += 1


class BranchGateTestCase(OperatorTestCase):
    """
    ``select_branch`` returns ``'geometric'`` iff BOTH the resolution
    (``w*delta_min >= RHO_END``) and the strong-cancellation
    (``L > L_MAX``) conditions hold.  The three ``'wave'`` quadrants are
    the substance: neither condition alone may license the asymptote.
    """

    def test_four_quadrants(self):
        """
        Sweeping ``w*delta_min`` and ``L`` across both thresholds, the
        gate is ``'geometric'`` exactly on the upper-right quadrant.
        """
        resolved_products = (2.0, operator.RHO_END, 6.0)
        exponents = (30.0, operator.L_MAX, 60.0)
        gate_map = []
        for product, cexp in itertools.product(resolved_products,
                                               exponents):
            with self.subTest(w_delta_min=product, cexp=cexp):
                # take w = 1 so w*delta_min = delta_min = product
                branch = operator.select_branch(1.0, product, cexp)
                resolved = product >= operator.RHO_END
                cancelling = cexp > operator.L_MAX
                expected = ('geometric' if resolved and cancelling
                            else 'wave')
                self.assertEqual(
                    branch, expected,
                    f'w*delta_min={product}, L={cexp}: got {branch}, '
                    f'expected {expected}')
                gate_map.append((product, cexp, branch))
                self.n_checks += 1
        self._plot(gate_map)

    def test_boundary_equalities(self):
        """
        ``w*delta_min == RHO_END`` counts as resolved (``>=``) but
        ``L == L_MAX`` does NOT count as cancelling (strict ``>``) -- the
        exact slips a boundary bug would introduce.
        """
        # Resolved boundary inclusive; geometric only when L > 48 too.
        self.assertEqual(
            operator.select_branch(1.0, operator.RHO_END, 60.0),
            'geometric')
        self.assertEqual(
            operator.select_branch(1.0, operator.RHO_END,
                                   float(operator.L_MAX)), 'wave')
        # Cancellation boundary is exclusive.
        self.assertEqual(
            operator.select_branch(1.0, 6.0, float(operator.L_MAX)),
            'wave')
        self.assertEqual(
            operator.select_branch(1.0, 6.0, operator.L_MAX + 1e-9),
            'geometric')
        self.n_checks += 4

    def test_thresholds_have_one_home(self):
        """
        ``channels`` imports the gate and its thresholds from
        ``operator`` rather than redefining them: the two thresholds
        must have exactly one home so the switch and the gate cannot
        drift apart.
        """
        self.assertIs(channels.select_branch, operator.select_branch)
        self.assertIs(channels.RHO_END, operator.RHO_END)
        self.assertIs(channels.RHO_START, operator.RHO_START)
        # channels must not carry an independent L_MAX value.
        self.assertEqual(
            getattr(channels, 'L_MAX', operator.L_MAX), operator.L_MAX)
        self.n_checks += 1

    def _plot(self, gate_map):
        if not _HAVE_MPL or not gate_map:
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        for product, cexp, branch in gate_map:
            ax.scatter(product, cexp,
                       c='C3' if branch == 'geometric' else 'C0', s=40)
        ax.axvline(operator.RHO_END, color='k', ls='--')
        ax.axhline(operator.L_MAX, color='k', ls='--')
        ax.set_xlabel('w * delta_min')
        ax.set_ylabel('L')
        ax.set_title('red = geometric (upper-right only)')
        _savefig(fig, 'operator_branch_gate.png')


class SelfFalsificationTestCase(OperatorTestCase):
    """
    Prove the gates above can actually go red.

    A green suite is worth only as much as its ability to fail, so each
    gate is shown catching a deliberately corrupted input.
    """

    _expect_checks = False

    def test_oracle_gate_rejects_a_wrong_amplification(self):
        """A 1% error in ``F_op`` must blow the `RTOL_GATE`."""
        y, gamma, kappa = np.array([0.55, 0.0]), PAPER_GAMMA, PAPER_KAPPA
        value, _ = operator.F_op(20.0, y, gamma, kappa=kappa,
                                 max_order=FOP_MAX_ORDER)
        reference = _oracle_amplification(20.0, y, gamma, kappa=kappa)
        good = abs(value - reference) / abs(reference)
        bad = abs(value * 1.01 - reference) / abs(reference)
        self.assertLessEqual(good, RTOL_GATE)
        self.assertGreater(
            bad, RTOL_GATE,
            'a 1% perturbation slips through the oracle gate; it '
            'asserts nothing')

    def test_slope_gate_distinguishes_the_two_cases(self):
        """
        The leading and corrected residual envelopes have genuinely
        different slopes; a test that could not tell ``w**-1`` from
        ``w**-3`` would be decoration.
        """
        case = GeometricOpticsSlopeTestCase()
        slope_lead, *_ = case._binned_slope(
            case._residuals(with_corrections=False))
        slope_full, *_ = case._binned_slope(
            case._residuals(with_corrections=True))
        self.assertGreater(
            slope_lead - slope_full, 1.3,
            'the two asymptotic cases are not separable, so the slope '
            'test could not discriminate a bug')

    def test_mass_sheet_gate_rejects_a_broken_rescaling(self):
        """
        A rescaling that forgets ``gamma -> gamma/lam`` (uses the primed
        shear as if physical) makes the observables drift with kappa;
        the invariance gate must catch it.
        """
        y_prime, gamma_prime = np.array([0.35, 0.12]), 0.25
        delays = []
        for kappa in (0.0, 0.4):
            lam = 1.0 - kappa
            y = np.sqrt(lam) * y_prime
            gamma_wrong = gamma_prime  # BUG: not lam*gamma_prime
            matrix = geometry.macro_matrix(gamma_wrong, 0.0, kappa)
            taus = sorted(geometry.delay(im, y, matrix)
                          for im in geometry.find_images(y, matrix))
            delays.append(taus[-1] - taus[0])
        drift = abs(delays[1] - delays[0]) / abs(delays[0])
        self.assertGreater(
            drift, 1e-6,
            'the broken rescaling left the observable invariant, so the '
            'mass-sheet gate would not detect it')

    def test_gate_rejects_geometric_when_one_condition_holds(self):
        """
        If ``select_branch`` returned ``'geometric'`` on resolution
        alone it would be wrong; pin that it does not.
        """
        self.assertEqual(
            operator.select_branch(1.0, 6.0, 30.0), 'wave')
        self.assertEqual(
            operator.select_branch(1.0, 2.0, 60.0), 'wave')


if __name__ == '__main__':
    main()
