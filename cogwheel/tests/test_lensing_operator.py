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

TOLERANCES AND TESTED DOMAIN
----------------------------
The oracle is exact to far beyond float64, so ``RTOL_GATE = 1e-10`` is a
property of ``F_op``, not the oracle.  The compared configurations keep
the cancellation exponent ``L = w*|y'|`` at or below ~25, where the
shipped wave branch delivers ~5e-12 or better (measured worst case
5.65e-12, ~180x inside the gate).  Above the shipped ceilings ``F_op``
raises a NAMED refusal (`_schwinger.SchwingerCertificationError`)
rather than returning a silent
``nan`` or a finite-but-wrong value; the ``L <= 25`` oracle gates here
are the accuracy half of that certified-or-refuse guarantee.

`OperatorTestCase.tearDown` fails a test that made zero comparisons, and
`SelfFalsificationTestCase` proves every gate above can actually go red.
"""
from __future__ import annotations

import itertools
import math
import pathlib
from dataclasses import FrozenInstanceError
from unittest import TestCase, main

import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import operator, geometry
from cogwheel.lensing.chang_refsdal import _hyp1f1
from cogwheel.lensing.chang_refsdal import _schwinger
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

#: Ceiling the reported ``order_used`` is held under in the oracle
#: comparison; large enough that the highest-``w`` configuration
#: (order ~57) converged under the retired operator series.
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
                        w, y, gamma, beta=beta, kappa=kappa)
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
                                kappa=PAPER_KAPPA)
        for field in ('converged', 'order_used', 'cancellation_ratio',
                      'estimated_relative_tail'):
            with self.subTest(field=field):
                with self.assertRaises(FrozenInstanceError):
                    setattr(diag, field, 0)
                self.n_checks += 1
                self.n_checks += 1

    def test_raises_named_error_above_w_ceiling(self):
        """
        Above ``W_MAX_CERTIFIED`` the kernel's named
        `HypergeometricDomainError` propagates rather than a silently
        wrong number being returned.

        RE-TARGET (Build 8d): the kernel wave path is reached only on
        the shear-free ``gamma' == 0`` legacy exit now (a sheared
        ``gamma' > 0`` host is served by Schwinger, which refuses above
        its OWN ceiling with `SchwingerCertificationError` -- pinned by
        `test_sheared_host_above_ceiling_refuses_schwinger`).  So the
        kernel-ceiling contract is exercised at ``gamma = 0.0``, the sole
        remaining path that consumes the 1F1 kernel.
        """
        with self.assertRaises(_hyp1f1.HypergeometricDomainError):
            operator.F_op(_hyp1f1.W_MAX_CERTIFIED + 100.0,
                          np.array([0.05, 0.0]), 0.0)
        self.n_checks += 1

    def test_raises_named_error_above_cancellation_ceiling(self):
        """
        Above the kernel's ``w*sqrt(s) = DD_PRODUCT_CEILING`` ceiling the
        named `HypergeometricDomainError` propagates.  Here ``y=(1,0)``,
        ``kappa=0`` gives ``w*sqrt(s) = w = 70 > 60``.

        RE-TARGET (Build 8d): as for the ``w`` ceiling above, the 1F1
        kernel is consumed only on the shear-free ``gamma' == 0`` legacy
        exit now, so the product-ceiling contract is pinned at
        ``gamma = 0.0`` (a sheared host at this ``w`` is Schwinger-served
        and refuses with `SchwingerCertificationError`).
        """
        with self.assertRaises(_hyp1f1.HypergeometricDomainError):
            operator.F_op(_hyp1f1.DD_PRODUCT_CEILING + 10.0,
                          np.array([1.0, 0.0]), 0.0)
        self.n_checks += 1

    def test_sheared_host_above_ceiling_refuses_schwinger(self):
        """
        NEW-contract pin (Build 8d homogenization): a sheared
        positive-parity host (``gamma' > 0``) above the QD Schwinger
        ceiling (``w > _schwinger.W_CEILING_SCHWINGER_QD = 150``) that is
        NOT geometric-resolved is served by the exact 1D Schwinger
        evaluator, which refuses by name with
        `SchwingerCertificationError` -- with NO legacy fallback.  The
        mpmath extension (Build QD) serves ``60 < w <= 150``; only
        ``w > 150`` is unconditionally refused.

        F028 re-point: both fixtures are small-radius on-axis sources
        (``|y| = 0.05`` and ``0.08``) that are genuinely hard-core --
        unresolved (``w*delta_min < RHO_END``) and declined by BOTH
        uniform arms -- so `select_branch` stays on the WAVE branch and
        the named refusal fires.
        """
        w_above = _schwinger.W_CEILING_SCHWINGER_QD + 10.0
        for y in (np.array([0.05, 0.0]), np.array([0.08, 0.0])):
            with self.subTest(y=tuple(y)):
                with self.assertRaises(
                        _schwinger.SchwingerCertificationError):
                    operator.F_op(w_above, y, 0.1)
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
    #: Frequency sweep.  The top is capped so the cancellation exponent
    #: ``L = w*|y'| = 0.9*w`` stays <= ~24.3 -- inside the region the
    #: oracle gates certify as accurate.  Above it ``F_op`` may refuse by
    #: name, which would ERROR this asymptotic sweep.  The retained
    #: 12 -> 27 span still gives the binned slope fit enough leverage to
    #: separate ``w**-1`` from ``w**-3``.
    SLOPE_W = np.linspace(12.0, 27.0, 84)
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
            value, _ = operator.F_op(w, self.SLOPE_Y, self.SLOPE_GAMMA)
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


#: Config grid over which the ROUTING PREDICATE is pinned (the predicate
#: half of `BranchGateTestCase.test_thresholds_have_one_home`).  Positive
#: parity keeps ``1 - kappa > |gamma|``; the macro saddle uses
#: ``kappa = 0`` so ``|gamma| > 1``.  The ``|y|`` set spans inside and
#: outside the astroid caustic (4- and 2-image census) and therefore
#: ``eta`` both BELOW and ABOVE ``operator.ETA_MIN_GEOMETRIC``; ``beta``
#: spans the un-sheared and the sheared micro-image.  Sources are
#: OFF-AXIS (fixed unit direction ``(0.8, 0.6)`` scaled to each ``|y|``):
#: an on-axis source has mirror-degenerate Fermat delays, so
#: ``delta_min = 0``, the resolution leg is dead everywhere and the
#: geometric outcome would never appear (MEASURED 2026-07-28).
_ONEHOME_DIR = (0.8, 0.6)  # unit vector: 0.8**2 + 0.6**2 == 1
_ONEHOME_YMAGS = (0.05, 0.3, 1.0, 2.0)
_ONEHOME_YS = tuple((mag * _ONEHOME_DIR[0], mag * _ONEHOME_DIR[1])
                    for mag in _ONEHOME_YMAGS)
ONEHOME_POSITIVE = tuple(  # (gamma, kappa, y, beta)
    (gamma, kappa, y, beta)
    for gamma in (0.2, 0.5, 0.9)
    for kappa in (0.0, 0.3)
    for y in _ONEHOME_YS
    for beta in (0.0, 0.7)
    if 1.0 - kappa > abs(gamma))
ONEHOME_SADDLE = tuple(  # (gamma, kappa, y, beta)
    (gamma, 0.0, y, beta)
    for gamma in (1.2, 2.0)
    for y in _ONEHOME_YS
    for beta in (0.0, 0.7))

#: ``w`` nodes: below the wave ceiling, astride it, and deep.  The
#: above-QD-ceiling nodes (w > W_CEILING_SCHWINGER_QD = 150) include both
#: resolved and unresolved cases so both routing outcomes appear.  (Below
#: the QD ceiling the Schwinger evaluator serves every node — DD for w<=60,
#: mpmath for 60<w<=150 — so there is no branch decision and those nodes
#: are skipped by the routing sweep.)  Nodes in (60, 150] are excluded to
#: avoid invoking the slow mpmath path in the fast-tier test.
ONEHOME_WS = (5.0, 40.0, 59.0, 500.0, 1000.0)


class BranchGateTestCase(OperatorTestCase):
    """
    ``select_branch`` returns ``'geometric'`` iff ALL THREE legs hold --
    resolution (``w*delta_min >= RHO_END``), strong cancellation
    (``L > L_MAX``) and distance from the caustic
    (``eta >= ETA_MIN_GEOMETRIC``, F031).  The ``'wave'`` outcomes are
    the substance: no leg alone may license the asymptote.

    THIS CLASS IS THE ONE HOME OF THE GEOMETRIC-VS-WAVE ROUTING
    DECISION.  `test_thresholds_have_one_home` pins BOTH halves of it --
    the constants (`channels` imports them rather than redefining them)
    and the predicate itself (both operator grids' actual routing must
    agree with `select_branch` node for node).  No other test may
    re-derive the branch condition; a hand-rolled mirror elsewhere is a
    place to drift, and the ``eta`` leg is exactly the kind of change
    that silently invalidates one.
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
                # take w = 1 so w*delta_min = delta_min = product, and an
                # infinite eta so the caustic leg is vacuously satisfied
                # and the other two are isolated.
                branch = operator.select_branch(1.0, product, cexp,
                                                math.inf)
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
        exact slips a boundary bug would introduce.  ``eta`` matches the
        resolution leg: ``eta == ETA_MIN_GEOMETRIC`` is inclusive.
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
        # Caustic-distance boundary is inclusive (``>=``).
        self.assertEqual(
            operator.select_branch(1.0, 6.0, 60.0,
                                   operator.ETA_MIN_GEOMETRIC),
            'geometric')
        self.assertEqual(
            operator.select_branch(
                1.0, 6.0, 60.0,
                np.nextafter(operator.ETA_MIN_GEOMETRIC, 0.0)),
            'wave')
        self.n_checks += 6

    def test_thresholds_have_one_home(self):
        """
        The routing decision lives in exactly ONE place -- constants AND
        predicate.

        (a) CONSTANTS: ``channels`` imports the gate and its thresholds
        from ``operator`` rather than redefining them, so the switch and
        the gate cannot drift apart.

        (b) PREDICATE: for every above-ceiling node of the config grid,
        the routing the operator ACTUALLY performed -- recovered from
        what it SERVED, not from a mock or a spy -- equals the label of
        `select_branch` fed arguments rebuilt from the public helpers.
        Both operator grids are swept (`_positive_parity_grid` and
        `_saddle_grid`, reached through the scalar `F_op` wrapper), and
        the ``eta`` leg is shown to be LIVE on the positive-parity grid,
        so a mirror that drops it cannot pass.
        """
        # (a) The constants and the gate object itself have one home.
        self.assertIs(channels.select_branch, operator.select_branch)
        self.assertIs(channels.RHO_END, operator.RHO_END)
        self.assertIs(channels.RHO_START, operator.RHO_START)
        # channels must not carry an independent L_MAX value.
        self.assertEqual(
            getattr(channels, 'L_MAX', operator.L_MAX), operator.L_MAX)
        self.n_checks += 1

        # (b) The predicate itself has one home.
        self._assert_grid_routing(ONEHOME_POSITIVE, 'positive')
        self._assert_grid_routing(ONEHOME_SADDLE, 'saddle')
        self._assert_eta_leg_is_live()

    # ---------------------------------------------------------------
    # Routing machinery for the predicate half of the one-home test.
    # ---------------------------------------------------------------

    @staticmethod
    def _is_positive_parity(gamma, kappa):
        return 1.0 - float(kappa) > abs(float(gamma))

    @staticmethod
    def _caustic_distance(gamma, beta, source, kappa):
        """``eta``, with the grids' own refusal convention.

        A refusing caustic search means no geometric admission, so the
        grids fall back to ``eta = 0.0`` (every node routed to 'wave',
        the conservative direction).  This mirror must do the same.
        """
        try:
            return float(geometry.nearest_caustic_point(
                gamma, beta, source, kappa=kappa).distance)
        except geometry.LensDomainError:
            return 0.0

    def _predicate_branch(self, w, y, gamma, beta, kappa):
        """The shared gate's label, its arguments rebuilt independently.

        Recomputes the gate's inputs from the PUBLIC helpers
        (`geometry.macro_matrix`, `_real_delay_min_separation`,
        `cancellation_exponent`, `geometry.nearest_caustic_point`) --
        never from the grid's internals -- and returns
        `select_branch`'s label.
        """
        source = np.asarray(y, dtype=float)
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        delta_min = operator._real_delay_min_separation(source, matrix)
        if self._is_positive_parity(gamma, kappa):
            # Positive parity: L == w*|y'| == cancellation_exponent, and
            # the third leg is eta, the distance to the caustic (F031).
            # The grid supplies eta, so this mirror must too -- omitting
            # it silently disables a live leg and the two would disagree
            # exactly where the gate does its work (near the caustic).
            exponent = operator.cancellation_exponent(
                w, source, gamma, kappa)
            eta = self._caustic_distance(gamma, beta, source, kappa)
            return operator.select_branch(w, delta_min, exponent, eta)
        # Saddle: the cancellation exponent is positive-parity bookkeeping
        # with no saddle analogue, so that leg stays vacuous; the eta leg
        # is LIVE and measured on the saddle in its own right (F034 --
        # median 57% error below eta = 0.1, worst case 484x).
        eta = self._caustic_distance(gamma, beta, source, kappa)
        return operator.select_branch(w, delta_min, math.inf, eta)

    def _observed_branch(self, w, y, gamma, beta, kappa):
        """The grid's routing, read off the served scalar ``F_op`` value.

        No mock and no spy: the branch is recovered from WHAT WAS
        SERVED.  A value bit-equal to `geometric_amplification` means
        the grid took the geometric branch; any other finite value (a
        uniform arm) or a named `SchwingerCertificationError` means it
        took the wave branch; a `geometry.LensDomainError` can only come
        from the geometric handoff's census guard, so it is a geometric
        routing that the census refused.
        """
        source = np.asarray(y, dtype=float)
        try:
            geom = complex(operator.geometric_amplification(
                w, source, gamma, beta=beta, kappa=kappa))
            geom_ok = True
        except geometry.LensDomainError:
            geom, geom_ok = None, False
        try:
            served = complex(operator.F_op(
                w, source, gamma, beta=beta, kappa=kappa)[0])
        except _schwinger.SchwingerCertificationError:
            return 'wave'
        except geometry.LensDomainError:
            # Only the geometric handoff census raises this above ceiling.
            return 'geometric'
        if geom_ok and served == geom:
            return 'geometric'
        return 'wave'

    def _assert_grid_routing(self, configs, parity_name):
        """Grid routing == `select_branch`, node for node, non-vacuously."""
        n_geometric = n_wave = 0
        for gamma, kappa, y, beta in configs:
            for w in ONEHOME_WS:
                if w <= _schwinger.W_CEILING_SCHWINGER_QD:
                    continue  # no branch decision below the QD ceiling
                with self.subTest(parity=parity_name, gamma=gamma,
                                  kappa=kappa, y=y, beta=beta, w=w):
                    predicted = self._predicate_branch(w, y, gamma, beta,
                                                       kappa)
                    observed = self._observed_branch(w, y, gamma, beta,
                                                     kappa)
                    self.n_checks += 1
                    self.assertEqual(
                        observed, predicted,
                        f'{parity_name} node gamma={gamma}, '
                        f'kappa={kappa}, y={y}, beta={beta}, w={w}: the '
                        f'grid routed {observed!r} but select_branch '
                        f'says {predicted!r} -- the predicate has more '
                        'than one home')
                    if predicted == 'geometric':
                        n_geometric += 1
                    else:
                        n_wave += 1
        # Non-vacuity: BOTH branch labels must be exercised, else the
        # agreement is trivially satisfied by a constant predicate.
        self.assertGreater(n_geometric, 0,
                           f'no geometric-routed {parity_name} node')
        self.assertGreater(n_wave, 0,
                           f'no wave-routed {parity_name} node')

    def _assert_eta_leg_is_live(self):
        """The grid spans ``eta`` on BOTH sides of ``ETA_MIN_GEOMETRIC``.

        Stronger than a span check: at least one positive-parity node
        must be admitted by the first two legs and REFUSED by ``eta``.
        Without such a node the agreement above would still pass for a
        mirror that dropped the third leg, and the sweep would not
        witness F031 at all.
        """
        n_below = n_above = n_eta_blocked = 0
        for gamma, kappa, y, beta in ONEHOME_POSITIVE:
            source = np.asarray(y, dtype=float)
            eta = self._caustic_distance(gamma, beta, source, kappa)
            if eta < operator.ETA_MIN_GEOMETRIC:
                n_below += 1
            else:
                n_above += 1
                continue
            matrix = geometry.macro_matrix(gamma, beta, kappa)
            delta_min = operator._real_delay_min_separation(source, matrix)
            for w in ONEHOME_WS:
                if w <= _schwinger.W_CEILING_SCHWINGER_QD:
                    continue
                two_leg = operator.select_branch(
                    w, delta_min,
                    operator.cancellation_exponent(w, source, gamma, kappa),
                    math.inf)
                if two_leg == 'geometric':
                    n_eta_blocked += 1
        self.n_checks += 1
        self.assertGreater(n_below, 0,
                           'no config with eta < ETA_MIN_GEOMETRIC: the '
                           'caustic leg is untested below its threshold')
        self.assertGreater(n_above, 0,
                           'no config with eta >= ETA_MIN_GEOMETRIC: the '
                           'caustic leg is untested above its threshold')
        self.assertGreater(
            n_eta_blocked, 0,
            'no node is admitted by resolution+cancellation and refused '
            'by eta, so the sweep cannot tell the two-leg gate from the '
            'three-leg one')

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


class MacroMagnificationLimitTestCase(OperatorTestCase):
    """
    DECISIVE closed-form ``w -> 0`` gate -- the Build 2d rediagnosis.

    ``F_op`` normalizes the amplification to no lens at all (not to the
    macro image), so the flat ``|F| - 1`` the engine reports at tiny
    ``w`` is NOT a numerical singularity or a ``gamma/(2w)`` prefactor
    blow-up: it is the EXACT geometric-optics macro magnification limit

        F(w -> 0) -> sqrt(mu_macro) = 1/sqrt((1 - kappa)**2 - gamma**2),

    a mass/frequency-INDEPENDENT constant.  The two signatures that this
    is the physical limit and not a bug are that ``|F_op|`` (a) equals
    the closed form to relative `LIMIT_RTOL` and (b) does not move across
    three decades of tiny ``w`` (mass/frequency independence).  For
    ``gamma = kappa = 0`` the closed form is exactly ``1`` (the unsheared
    control).

    The expected value is written LITERALLY as
    ``1/sqrt((1 - kappa)**2 - gamma**2)`` -- never built from ``F_op``,
    ``channels``, ``geometry``, or any engine path (FINDINGS F002
    oracle-tautology trap).  A named engine refusal at these tiny ``w``
    is itself NEWS: it is recorded as a refusal and FAILS the test with
    its message, never skipped.

    If this test FAILS, it is not to be adjusted to pass -- the failure
    is the finding, to be reported verbatim.
    """

    #: Positive-parity grid (``1 - kappa > |gamma|`` at every point),
    #: crossing sheared/unsheared and converged/unconverged.
    LIMIT_GAMMAS = (0.0, 0.1, 0.2, 0.3)
    LIMIT_KAPPAS = (0.0, 0.3)
    #: Shear orientation.  ``|F|`` is invariant under the eigenframe
    #: rotation, so both values must land on the same closed form.
    LIMIT_BETAS = (0.0, 0.7)
    LIMIT_Y = np.array([0.30, 0.10])
    #: Three decades of tiny frequency; the ``|F|`` plateau must not
    #: move across them.
    LIMIT_WS = (1e-8, 1e-10, 1e-12)
    #: The limit is EXACT; this gate is a property of ``F_op``.  Loosen
    #: only on a MEASURED need and never past ``1e-6`` -- anything looser
    #: is a finding to report.
    LIMIT_RTOL = 1e-8

    def test_small_w_limit_is_macro_magnification(self):
        """
        Across the positive-parity grid and the three tiny frequencies,
        ``|F_op|`` equals the literal closed form
        ``1/sqrt((1 - kappa)**2 - gamma**2)`` to `LIMIT_RTOL` and is
        independent of ``w`` to the same tolerance.  A refusal at tiny
        ``w`` is recorded and FAILS the test rather than being skipped.
        """
        refusals: list[str] = []
        max_rel = 0.0
        sheared_control = None  # measured |F_op| at (gamma=0.2, kappa=0)
        for gamma, kappa, beta in itertools.product(
                self.LIMIT_GAMMAS, self.LIMIT_KAPPAS, self.LIMIT_BETAS):
            # Closed-form macro magnification, written LITERALLY -- never
            # via F_op/channels/geometry (FINDINGS F002).
            closed = 1.0 / np.sqrt((1.0 - kappa) ** 2 - gamma ** 2)
            if gamma == 0.0 and kappa == 0.0:
                # Unsheared control: the closed form is exactly 1.
                self.assertEqual(closed, 1.0)
            magnitudes: list[float] = []
            for w in self.LIMIT_WS:
                with self.subTest(gamma=gamma, kappa=kappa, beta=beta,
                                  w=w):
                    try:
                        value, _ = operator.F_op(
                            w, self.LIMIT_Y, gamma, beta=beta,
                            kappa=kappa)
                    except _schwinger.SchwingerCertificationError as exc:
                        # A refusal at w -> 0 is news, not a skip.
                        refusals.append(
                            f'gamma={gamma}, kappa={kappa}, beta={beta}, '
                            f'w={w:g}: {exc}')
                        continue
                    mag = abs(value)
                    magnitudes.append(mag)
                    rel = abs(mag - closed) / abs(closed)
                    max_rel = max(max_rel, rel)
                    self.assertLessEqual(
                        rel, self.LIMIT_RTOL,
                        f'gamma={gamma}, kappa={kappa}, beta={beta}, '
                        f'w={w:g}: |F_op| = {mag:.12f} != closed form '
                        f'1/sqrt((1-kappa)^2 - gamma^2) = {closed:.12f} '
                        f'(rel {rel:.3e} > {self.LIMIT_RTOL:.0e}); '
                        'F(w->0) should equal sqrt(mu_macro)')
                    if (gamma, kappa, beta, w) == (0.2, 0.0, 0.0, 1e-10):
                        sheared_control = mag
                    self.n_checks += 1
            # Mass/frequency independence: the |F| plateau must not drift
            # across the three decades of w.  This is the signature that
            # distinguishes the exact macro limit from a 1/w singularity.
            if len(magnitudes) >= 2:
                spread = max(abs(m - magnitudes[0]) for m in magnitudes)
                self.assertLessEqual(
                    spread / abs(magnitudes[0]), self.LIMIT_RTOL,
                    f'gamma={gamma}, kappa={kappa}, beta={beta}: |F_op| '
                    f'drifts by relative '
                    f'{spread / abs(magnitudes[0]):.3e} across '
                    'w in {1e-8, 1e-10, 1e-12}; a true macro '
                    'magnification limit is frequency-independent')
                self.n_checks += 1

        # A refusal at tiny w is a finding, not an expected skip: fail
        # with the recorded messages rather than passing on absence.
        self.assertEqual(
            refusals, [],
            'F_op refused at tiny w (a refusal at w -> 0 is itself news, '
            'not an expected skip):\n' + '\n'.join(refusals))

        # Self-falsification companion (the suite's anti-vacuity idiom):
        # the MEASURED |F_op| at a real sheared point matches the true
        # closed form but NOT one with the shear perturbed 1%, so the
        # 1e-8 gate genuinely discriminates a wrong macro magnification
        # rather than passing on any nearby constant.
        self.assertIsNotNone(
            sheared_control,
            'the sheared control point (gamma=0.2, kappa=0, w=1e-10) did '
            'not return, so the self-falsification companion cannot run')
        true_closed = 1.0 / np.sqrt(1.0 ** 2 - 0.2 ** 2)
        perturbed_closed = 1.0 / np.sqrt(1.0 ** 2 - (0.2 * 1.01) ** 2)
        self.assertLessEqual(
            abs(sheared_control - true_closed) / true_closed,
            self.LIMIT_RTOL)
        self.assertGreater(
            abs(sheared_control - perturbed_closed) / perturbed_closed,
            self.LIMIT_RTOL,
            'a closed form with the shear perturbed 1% still matches '
            '|F_op| within 1e-8; the gate would not catch a 1% error in '
            'the macro magnification')
        self.n_checks += 1


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
        value, _ = operator.F_op(20.0, y, gamma, kappa=kappa)
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
