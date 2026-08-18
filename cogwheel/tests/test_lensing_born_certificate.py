"""Carrier-only Born truncation certificate (``_born.born_carrier_omitted_term``).

This suite blesses the WP1/WP2 certificate that decides when the Born far
exterior may be served CARRIER-ONLY (residual identically zero, only
``_born.born_lead_carrier`` kept) beyond the trained chart box, on BOTH
parities.  It pins three properties of
``_born.born_carrier_omitted_term`` and the lead-only serve it certifies:

* **w-direction (guards the ``w_hi`` vs ``w_lo`` convention flip).**  The
  omitted-term modulus is ``hypot(a0, 0.5*w*b1) / q2r`` with ``a0``,
  ``b1``, ``q2r`` frequency-INDEPENDENT, hence STRICTLY INCREASING in
  ``w``.  Its worst case over a band ``[w_lo, w_hi]`` is therefore at the
  CEILING ``w_hi`` -- the OPPOSITE convention to the saddle-c3 gate's
  band-FLOOR ``w_lo``.  A gate keyed on ``w_lo`` would silently
  under-certify; the monotonicity pin catches a mis-derived derivative
  sign or a flipped convention.

* **Parity-agnostic factor continuity.**  ``_born_factors`` and hence the
  certificate read the SAME closed form on the positive-parity host
  (``det_a > 0``) and the macro saddle (``det_a < 0``): the exact algebra
  identity ``b1 - a0 == -lam**2 * mu_macro`` (``mu_macro = 1/det_a``,
  ``lam = 1 - kappa``) holds on BOTH sides of the parity wall, confirming
  the certificate is valid on the saddle WITHOUT routing through the
  positive-parity-only policy guard in ``born_amplification``.

* **Carrier-only accuracy (the escalate-not-iterate acceptance gate).**
  At every point the certificate ADMITS
  (``_SADDLE_FARFIELD_SAFETY * omitted_term(w_hi) <=
  _SADDLE_FARFIELD_CERT_BAR``), the served lead-only carrier matches the
  EXACT engine ``operator.F_op`` to within the certificate bar
  ``1e-3`` over the whole band, on both parities.  A point above the bar
  falsifies the certificate/derivation and must trigger ESCALATION, never
  a bar widening.

Independence of oracles (house rule).
    The accuracy oracle is ``operator.F_op`` -- the contour-free exact
    Chang-Refsdal amplification, which shares NO algebra with ``_born``.
    ``born_lead_carrier`` and ``F_op`` both live in the ABSOLUTE
    Fermat-delay frame and are both normalized to no-lens
    (``F(w->0) = sqrt(mu_macro)``), so they are directly comparable with
    no demodulation (the "pair the frames" rule): a frame mismatch would
    surface as an O(1) error, not the measured ~2e-4.  The invariant pin
    is exact float64 algebra evaluated two ways.  No test imports a module
    from a git revision.

Fixture derivation (no pinned-literal boundaries).
    Every accuracy fixture asserts its own admission PREMISE from the live
    shipping certificate and the shipped production constants
    (``_SADDLE_FARFIELD_SAFETY``, ``_SADDLE_FARFIELD_CERT_BAR`` imported
    from ``likelihood``) before it measures accuracy, and the saddle
    fixtures additionally assert the production resolution fence
    ``w_lo * delta_min >= RHO_END`` from ``operator``.  If the gate drifts
    a fixture out of the served domain the premise fails LOUDLY rather
    than measuring a refused source.

Tolerance justification.
    * The monotonicity pin is a strict inequality (``est(w_{i+1}) >
      est(w_i)``), no absolute target.
    * The invariant ``b1 - a0 == -lam**2 * mu_macro`` is exact algebra
      evaluated two ways; the residual is float64 round-off.  Measured
      worst case over the gamma sweep is ~1e-16 relative; the saddle
      absolute pin uses ``1e-13`` (>2 orders of headroom).
    * The carrier-only accuracy bar is the SHIPPED
      ``_SADDLE_FARFIELD_CERT_BAR = 1e-3``; measured worst case over all
      six admitted fixtures is ~2e-4 (~5x headroom).  The bar is NOT
      re-pinned as a literal here -- it is imported.

Serve-routing pins (WP2 lift, this shard).
    Three additional pins bless the way ``_born_residual_analytic`` ROUTES
    a query through the certificate, WITHOUT running the heavy
    reconstruction tail (``_born_reconstruct`` is spied and the amplitude
    engine is never called):

    * **Null-identity (in-box byte-identity).**  On a positive-parity
      in-box query (``rho > 2`` AND ``covers() == True``) the residual fed
      to ``_born_reconstruct`` is EXACTLY the chart's interpolated residual
      -- never zeros -- and the carrier-only certificate is NEVER consulted.
      The carrier-only lift cannot perturb the in-box serve.

    * **No-``covers()``-refusal.**  EVERY ``covers() == False`` query (both
      parities) consults ``_born_carrier_certificate_serves`` before
      deciding: it serves carrier-only (zero residual) when the certificate
      admits, or falls through to the engine (``None``) when it refuses.
      There is NO surviving straight-refusal path that returns ``None`` on a
      ``covers()`` miss without first asking the certificate -- the exact
      defect this build closes.

    * **Saddle resolution fence.**  A macro-saddle beyond-box lens with
      known smallest pairwise Fermat-delay separation ``delta_min`` is
      REFUSED as a whole when ``w_lo * delta_min < RHO_END`` (=4.0) and
      SERVES carrier-only for the twin just above; the transition sits
      exactly at ``4.0``.  Positive parity is NOT subject to this fence
      (a positive low-``w`` admitted point stays admitted as ``w_lo -> 0``).

Cost.
    Fast tier.  The only engine calls are the accuracy sweep:
    (3 positive + 3 saddle points) x 5 band w-nodes = 30 ``F_op``
    evaluations, all at ``w <= 0.75`` (the exact DD path, ``w <= 60``),
    plus ~6 more in the accuracy self-falsification.  A few seconds total.
    The monotonicity, invariant and degenerate-sentinel tests make no
    engine calls.  The three serve-routing pins are engine-free: they run
    the real cheap ``geometry_partition`` + ``caustic_rho`` and the real
    certificate, but spy ``_born_reconstruct`` to skip the tail.
"""
from __future__ import annotations

import functools
import math
import pathlib
import types
import unittest
from unittest import mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing.chang_refsdal import _born, geometry, operator
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing import likelihood
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood,
    _SADDLE_FARFIELD_SAFETY, _SADDLE_FARFIELD_CERT_BAR)
from cogwheel.lensing.ppgo_map import caustic_rho

#: Directory for diagnostic plots (created lazily by ``_save_plot``).
OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

# --------------------------------------------------------------------------- #
# Admitted beyond-box accuracy fixtures (both parities).
#
# Each is (gamma, y1, y2, w_lo, w_hi).  The band ceiling is w_hi = 60/|y|, so
# every oracle node stays inside the certifiable exterior w*|y| <= 60 (the
# exact DD path).  Admission is NOT pinned here -- every fixture asserts the
# live certificate premise at test time (see CarrierOnlyAccuracyTestCase).
# --------------------------------------------------------------------------- #
#: Positive-parity (gamma < 1) admitted far-exterior points.  |y| = 80..100
#: is where a0/q2r falls below the admission floor (5e-5).
POS_POINTS = (
    (0.30, 80.0, 0.0, 0.05, 0.75),
    (0.50, 80.0, 0.0, 0.05, 0.75),
    (0.35, 100.0, 0.0, 0.05, 0.60),
)
#: Macro-saddle (gamma > 1) admitted far-exterior points.  The saddle needs
#: a larger |y| for the same admission floor (measured), and its real-image
#: Fermat separation is huge (delta_min ~ 1e4), so the resolution fence
#: w_lo * delta_min >= RHO_END is satisfied with room to spare.
SAD_POINTS = (
    (1.5, 150.0, 0.0, 0.01, 0.40),
    (2.0, 200.0, 0.0, 0.01, 0.30),
    (3.0, 300.0, 0.0, 0.01, 0.20),
)
#: Band w-node count for the accuracy sweep.
N_W_NODES = 5

#: Admission threshold on the raw omitted term: SAFETY * est <= BAR.
_ADMIT_EST_MAX = _SADDLE_FARFIELD_CERT_BAR / _SADDLE_FARFIELD_SAFETY

# --------------------------------------------------------------------------- #
# w-direction (monotonicity) fixture.
# --------------------------------------------------------------------------- #
#: Fixed positive-parity far-exterior lens + source for the w-sweep.
MONO_GAMMA = 0.30
MONO_Y1, MONO_Y2 = 80.0, 0.0
#: Ascending frequency grid (well inside w*|y| <= 60 at |y| = 80).
MONO_WS = tuple(np.linspace(1e-4, 0.75, 12))

# --------------------------------------------------------------------------- #
# Parity-agnostic invariant fixture.
# --------------------------------------------------------------------------- #
#: Gamma sweep straddling the parity wall gamma = 1 (skipping the wall strip
#: |gamma - 1| < 0.02 where the macro image degenerates).
INVARIANT_GAMMAS = (0.20, 0.45, 0.70, 0.90, 1.10, 1.50, 2.00, 3.00)
#: Source for the invariant sweep (any y != 0; matches the exterior radius).
INVARIANT_Y1, INVARIANT_Y2 = 3.6, 0.0
#: Absolute tolerance on the saddle invariant b1 - a0 == -lam**2 * mu_macro.
INVARIANT_ABS_TOL = 1e-13
#: Relative tolerance on the same invariant across the wall-straddling sweep
#: (near the wall both sides diverge, so absolute diff scales with magnitude).
INVARIANT_REL_TOL = 1e-12


def _save_plot(fig, name: str) -> None:
    """Write a diagnostic figure to ``cogwheel/tests/output/``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _mu_macro(gamma: float, kappa: float = 0.0) -> float:
    """Independent macro magnification ``1 / ((1-kappa)**2 - gamma**2)``."""
    lam = 1.0 - kappa
    return 1.0 / (lam ** 2 - gamma ** 2)


class _BornCertTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison tally.

    ``CarrierOnlyAccuracyTestCase`` records every engine comparison; a
    sweep whose fixture list silently emptied would otherwise pass without
    asserting anything.  Tests that make no comparison leave the counter
    at zero and are unaffected (``tearDown`` only fires when a test
    DECLARED it would compare via ``requires_comparison``).
    """

    requires_comparison = False

    def setUp(self):
        """Reset the per-test engine-comparison tally."""
        self.n_compared = 0

    def tearDown(self):
        """Fail a comparison test that asserted nothing."""
        if self.requires_comparison and not self.n_compared:
            self.fail('the accuracy sweep made zero comparisons; the test '
                      'asserted nothing (empty fixture list?)')


class CarrierOmittedTermMonotoneWTestCase(_BornCertTestCase):
    """Spec 1: ``born_carrier_omitted_term`` is strictly increasing in ``w``.

    The omitted-term modulus is ``hypot(a0, 0.5*w*b1) / q2r`` with
    ``a0, b1, q2r`` frequency-INDEPENDENT, so it rises monotonically with
    ``w`` and its worst case over ``[w_lo, w_hi]`` sits at the CEILING
    ``w_hi``.  This guards the convention flip: a gate mistakenly keyed on
    the band FLOOR ``w_lo`` (the saddle-c3 gate's convention) would
    under-certify, and a mis-derived derivative sign would make the trend
    flat or downward.  No engine calls.
    """

    def test_omitted_term_strictly_increases_with_w(self):
        """Each successive ``w`` yields a strictly larger omitted term."""
        ws = np.asarray(MONO_WS, dtype=float)
        est = np.array([
            _born.born_carrier_omitted_term(float(w), MONO_Y1, MONO_Y2,
                                            MONO_GAMMA)
            for w in ws])
        # All finite (the fixture is a shear-admitted far-exterior point).
        self.assertTrue(np.all(np.isfinite(est)),
                        f'omitted term went non-finite: {est}')
        # Strict monotone increase over every adjacent pair.
        diffs = np.diff(est)
        for i, (w_lo, w_hi, d) in enumerate(zip(ws[:-1], ws[1:], diffs)):
            with self.subTest(pair=i, w_lo=w_lo, w_hi=w_hi):
                self.assertGreater(
                    d, 0.0,
                    f'omitted term not increasing from w={w_lo} to w={w_hi}: '
                    f'delta={d}')

    def test_band_worst_case_is_at_ceiling_not_floor(self):
        """The band maximum is at ``w_hi``, never at the floor ``w_lo``."""
        ws = np.asarray(MONO_WS, dtype=float)
        est = np.array([
            _born.born_carrier_omitted_term(float(w), MONO_Y1, MONO_Y2,
                                            MONO_GAMMA)
            for w in ws])
        # argmax is the last node; argmin is the first.  A gate keyed on the
        # floor would read est[0] (the smallest) as the band's worst case.
        self.assertEqual(int(np.argmax(est)), est.size - 1)
        self.assertEqual(int(np.argmin(est)), 0)
        self.assertGreater(est[-1], est[0])

    def test_w_direction_diagnostic_plot(self):
        """Emit the omitted-term-vs-``w`` trend plot (upward => correct)."""
        ws = np.asarray(MONO_WS, dtype=float)
        est = np.array([
            _born.born_carrier_omitted_term(float(w), MONO_Y1, MONO_Y2,
                                            MONO_GAMMA)
            for w in ws])
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.plot(ws, est, 'o-', color='tab:blue')
        ax.set_xlabel('w (dimensionless)')
        ax.set_ylabel('|omitted term|  (carrier-relative)')
        ax.set_title(f'Spec 1: omitted term increasing in w\n'
                     f'gamma={MONO_GAMMA}, y=({MONO_Y1}, {MONO_Y2})')
        ax.grid(True, alpha=0.3)
        _save_plot(fig, 'born_certificate_w_direction_monotone.png')


class BornFactorsParityInvariantTestCase(_BornCertTestCase):
    """Spec 2: ``b1 - a0 == -lam**2 * mu_macro`` on BOTH sides of gamma=1.

    ``_born_factors`` reads one closed form for the positive-parity host
    (``det_a = lam**2 - gamma**2 > 0``) and the macro saddle
    (``det_a < 0``).  The exact macro-limit identity
    ``b1 - a0 == -lam**2 * mu_macro`` (``mu_macro = 1/det_a``) therefore
    holds continuously across the parity wall -- proving the certificate
    is valid on the saddle WITHOUT routing through the positive-parity
    policy guard.  Exact float64 algebra, no engine calls.
    """

    def test_invariant_holds_across_parity_wall(self):
        """The identity holds to float round-off over the gamma sweep."""
        for gamma in INVARIANT_GAMMAS:
            with self.subTest(gamma=gamma):
                _, _, _, b1, a0 = _born._born_factors(
                    INVARIANT_Y1, INVARIANT_Y2, gamma, 0.0, 0.0)
                lam = 1.0
                expected = -(lam ** 2) * _mu_macro(gamma)
                self.assertAlmostEqual(
                    b1 - a0, expected,
                    delta=INVARIANT_REL_TOL * max(1.0, abs(expected)),
                    msg=f'invariant broken at gamma={gamma}: '
                        f'b1-a0={b1 - a0!r}, -lam^2 mu={expected!r}')

    def test_invariant_on_macro_saddle_absolute(self):
        """On a fixed macro saddle (gamma>1, det_a<0) pin to 1e-13 absolute."""
        gamma = 1.5
        det_a = 1.0 - gamma ** 2
        self.assertLess(det_a, 0.0, 'fixture is not a macro saddle')
        _, _, _, b1, a0 = _born._born_factors(
            INVARIANT_Y1, INVARIANT_Y2, gamma, 0.0, 0.0)
        expected = -(1.0 ** 2) * (1.0 / det_a)
        self.assertAlmostEqual(b1 - a0, expected, delta=INVARIANT_ABS_TOL)

    def test_parity_invariant_diagnostic_table(self):
        """Emit the (b1-a0) vs -lam^2 mu_macro table across the wall."""
        gammas, lhs, rhs = [], [], []
        for gamma in INVARIANT_GAMMAS:
            _, _, _, b1, a0 = _born._born_factors(
                INVARIANT_Y1, INVARIANT_Y2, gamma, 0.0, 0.0)
            gammas.append(gamma)
            lhs.append(b1 - a0)
            rhs.append(-(1.0 ** 2) * _mu_macro(gamma))
        gammas = np.asarray(gammas)
        lhs = np.asarray(lhs)
        rhs = np.asarray(rhs)
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.2, 5.0), sharex=True)
        ax0.plot(gammas, lhs, 'o', label='b1 - a0', color='tab:blue')
        ax0.plot(gammas, rhs, 'x', label='-lam^2 mu_macro', color='tab:red')
        ax0.axvline(1.0, color='k', ls=':', alpha=0.5, label='parity wall')
        ax0.set_ylabel('invariant value')
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)
        ax1.semilogy(gammas, np.abs(lhs - rhs) + 1e-300, 's-',
                     color='tab:green')
        ax1.axvline(1.0, color='k', ls=':', alpha=0.5)
        ax1.set_xlabel('gamma')
        ax1.set_ylabel('|LHS - RHS|')
        ax1.set_title('Spec 2: parity-agnostic factor continuity')
        ax1.grid(True, alpha=0.3)
        _save_plot(fig, 'born_certificate_parity_invariant_table.png')


@functools.lru_cache(maxsize=1)
def _accuracy_sweep():
    """Run the both-parity carrier-only accuracy sweep once.

    Shared by the accuracy pin and its diagnostic plot so the engine
    evaluations are paid once per file.  Returns a tuple of per-fixture
    dicts with the live admission premise, the saddle resolution fence,
    the certificate estimate and the per-node relative errors of the
    served lead-only carrier against ``operator.F_op``.

    Cost: (3 positive + 3 saddle) x ``N_W_NODES`` = 30 ``F_op`` calls,
    all at ``w <= 0.75`` (the exact DD path).  Seconds.
    """
    results = []
    for parity, points in (('positive', POS_POINTS), ('saddle', SAD_POINTS)):
        for gamma, y1, y2, w_lo, w_hi in points:
            source = np.array([y1, y2])
            # Live admission premise: SAFETY * omitted_term(w_hi) <= BAR.
            cert_est = _born.born_carrier_omitted_term(w_hi, y1, y2, gamma)
            admitted = (_SADDLE_FARFIELD_SAFETY * cert_est
                        <= _SADDLE_FARFIELD_CERT_BAR)
            # Saddle resolution fence w_lo * delta_min >= RHO_END (positive
            # parity has no fence -> delta_min reported as None).
            if parity == 'saddle':
                matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
                delta_min = operator._real_delay_min_separation(source, matrix)
            else:
                delta_min = None
            node_ws = np.linspace(w_lo, w_hi, N_W_NODES)
            rel_errs = []
            for w in node_ws:
                served = _born.born_lead_carrier(float(w), y1, y2, gamma)
                exact, _ = operator.F_op(float(w), source, gamma)
                rel_errs.append(abs(served - exact) / abs(exact))
            results.append({
                'parity': parity,
                'gamma': gamma, 'y1': y1, 'y2': y2,
                'w_lo': w_lo, 'w_hi': w_hi,
                'cert_est': cert_est,
                'safety_est': _SADDLE_FARFIELD_SAFETY * cert_est,
                'admitted': admitted,
                'delta_min': delta_min,
                'node_ws': node_ws,
                'rel_errs': np.asarray(rel_errs),
            })
    return tuple(results)


class CarrierOnlyAccuracyTestCase(_BornCertTestCase):
    """Spec 3: carrier-only serve matches the exact engine within the bar.

    At every certificate-ADMITTED beyond-box point on BOTH parities, the
    served lead-only carrier ``born_lead_carrier`` (residual identically
    zero) matches the exact ``operator.F_op`` to within the SHIPPED bar
    ``_SADDLE_FARFIELD_CERT_BAR`` over the whole band.  A point above the
    bar falsifies the certificate and must ESCALATE, never widen the bar
    (this test asserts the bar, it does not choose it).  Each fixture
    proves its own admission premise from the live certificate + shipped
    constants before it measures accuracy.
    """

    requires_comparison = True

    def test_every_admitted_point_meets_the_bar(self):
        """Served == exact within 1e-3 at every admitted node, both parities."""
        for res in _accuracy_sweep():
            with self.subTest(parity=res['parity'], gamma=res['gamma'],
                              y=(res['y1'], res['y2'])):
                # Premise 1: the LIVE certificate admits this point.
                self.assertLessEqual(
                    res['safety_est'], _SADDLE_FARFIELD_CERT_BAR,
                    f"fixture no longer admitted: SAFETY*est="
                    f"{res['safety_est']:.3e} > BAR="
                    f"{_SADDLE_FARFIELD_CERT_BAR:.3e}; move the fixture "
                    f"back inside the served domain")
                # Premise 2 (saddle only): the resolution fence holds.
                if res['parity'] == 'saddle':
                    self.assertGreaterEqual(
                        res['w_lo'] * res['delta_min'], operator.RHO_END,
                        f"saddle fence violated: w_lo*delta_min="
                        f"{res['w_lo'] * res['delta_min']:.3e} < RHO_END="
                        f"{operator.RHO_END}")
                # Accuracy: served lead-only carrier vs exact engine.
                worst = float(np.max(res['rel_errs']))
                for w, err in zip(res['node_ws'], res['rel_errs']):
                    self.n_compared += 1
                    self.assertLessEqual(
                        err, _SADDLE_FARFIELD_CERT_BAR,
                        f"carrier-only error {err:.3e} exceeds bar "
                        f"{_SADDLE_FARFIELD_CERT_BAR:.3e} at w={w:.4f}, "
                        f"gamma={res['gamma']}: ESCALATE (do not widen "
                        f"the bar) -- worst on this fixture {worst:.3e}")

    def test_accuracy_diagnostic_scatter(self):
        """Scatter served-vs-exact worst error against the certificate est."""
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for res in _accuracy_sweep():
            worst = float(np.max(res['rel_errs']))
            # Count the plotted worst-node comparison for anti-vacuity.
            self.n_compared += 1
            marker = 'o' if res['parity'] == 'positive' else '^'
            colour = 'tab:blue' if res['parity'] == 'positive' else 'tab:red'
            ax.loglog(res['safety_est'], worst, marker, color=colour,
                      label=res['parity'])
        ax.axhline(_SADDLE_FARFIELD_CERT_BAR, color='k', ls='--',
                   label=f'bar={_SADDLE_FARFIELD_CERT_BAR:g}')
        # De-duplicate legend entries.
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), fontsize=8)
        ax.set_xlabel('certificate estimate  SAFETY * omitted_term(w_hi)')
        ax.set_ylabel('worst served-vs-exact relative error')
        ax.set_title('Spec 3: carrier-only accuracy vs certificate')
        ax.grid(True, which='both', alpha=0.3)
        _save_plot(fig, 'born_certificate_accuracy_scatter.png')


class CarrierOmittedTermDegenerateTestCase(_BornCertTestCase):
    """The certificate refuses (``+inf``) on the degenerate geometries.

    ``gamma == 0`` (no shear, outside the sheared far-field domain) and
    the source at the origin (``q2r == 0``) both make
    ``born_carrier_omitted_term`` return ``math.inf`` -- explicitly ``+inf``,
    never a ``NaN`` and never a ``ZeroDivisionError`` -- so a
    ``SAFETY * est <= bar`` gate refuses loudly rather than admitting a
    garbage point.  This class ALSO pins the END-TO-END serve decision: the
    real ``_born_carrier_certificate_serves`` REFUSES both degenerate
    geometries, routing them to the exact engine, and does so specifically
    BECAUSE of the degeneracy (a non-degenerate source with the SAME
    backstop-passing images is ADMITTED).  A silent admit or a ``NaN``
    reveals a missing guard.  No engine calls.
    """

    #: Two well-separated dummy real images (sep = 2.0 >> the 0.05 backstop),
    #: so IF a degenerate query reached the separation backstop it would PASS
    #: -- isolating any refusal to the degenerate-geometry guard itself,
    #: never an incidental image-count / separation failure.
    _BACKSTOP_IMAGES = np.array([[1.0, 0.0], [-1.0, 0.0]])

    def test_zero_shear_returns_inf(self):
        """``gamma == 0`` is outside the domain -> ``+inf``."""
        self.assertEqual(
            _born.born_carrier_omitted_term(0.5, 80.0, 0.0, 0.0), math.inf)

    def test_source_at_origin_returns_inf(self):
        """Source on the macro image (``q2r == 0``) -> ``+inf``."""
        self.assertEqual(
            _born.born_carrier_omitted_term(0.5, 0.0, 0.0, 0.30), math.inf)

    def test_degenerate_terms_are_positive_inf_not_nan(self):
        """Both degeneracies give ``+inf`` explicitly, never ``NaN``.

        Diagnostic: a ``NaN`` (from a ``0/0`` or ``inf-inf``) would sneak
        past an ``est <= bar`` comparison as ``False`` on BOTH sides, which
        reads like a refusal but is actually undefined behaviour.  Pin the
        sign and finiteness so the refusal is provably a ``+inf`` refusal.
        """
        for label, args in (
                ('zero_shear', (0.5, 80.0, 0.0, 0.0)),
                ('source_at_origin', (0.5, 0.0, 0.0, 0.30))):
            with self.subTest(case=label):
                est = _born.born_carrier_omitted_term(*args)
                self.assertFalse(math.isnan(est),
                                 f'{label}: omitted term is NaN (missing '
                                 f'degenerate guard)')
                self.assertTrue(math.isinf(est) and est > 0.0,
                                f'{label}: omitted term is not +inf: {est!r}')

    def test_inf_fails_any_admission_gate(self):
        """A ``+inf`` estimate can never satisfy ``SAFETY * est <= bar``."""
        est = _born.born_carrier_omitted_term(0.5, 0.0, 0.0, 0.0)
        self.assertFalse(
            _SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR)

    def test_certificate_refuses_source_at_origin(self):
        """``_born_carrier_certificate_serves`` refuses the ``q2r == 0`` source.

        The lens is a valid gamma=0.30 far-field surface (kappa=beta=0) so
        the domain guard passes; the refusal comes from the ``+inf``
        certificate estimate at the origin, NOT the domain check.  The
        backstop images are well-separated, so the separation backstop
        would PASS -- proof the refusal is the degenerate certificate.
        """
        lens = _route_lens(0.30, 0.0, 0.0)
        self.assertFalse(
            likelihood._born_carrier_certificate_serves(
                lens, 0.05, 0.75, self._BACKSTOP_IMAGES),
            'origin source admitted a carrier-only serve (q2r==0 divide-by-'
            'zero point); the certificate must refuse')

    def test_certificate_refuses_zero_shear(self):
        """``_born_carrier_certificate_serves`` refuses the ``gamma == 0`` lens.

        Zero shear has no caustic frame; the domain guard refuses before
        the certificate estimate is even formed.  The backstop images are
        well-separated so no incidental image failure confounds the refusal.
        """
        lens = _route_lens(0.0, 80.0, 0.0)
        self.assertFalse(
            likelihood._born_carrier_certificate_serves(
                lens, 0.05, 0.75, self._BACKSTOP_IMAGES),
            'zero-shear lens admitted a carrier-only serve; the certificate '
            'must refuse (no shear => outside the sheared far-field domain)')

    def test_same_images_nondegenerate_source_is_admitted(self):
        """Teeth: the SAME backstop images + a valid source ADMIT.

        Proves the two refusals above are attributable to the DEGENERACY,
        not to a certificate that refuses everything: swapping in a genuine
        positive-parity far-exterior source (gamma=0.30, |y|=80, the
        accuracy-fixture point) with the identical images and band ADMITS a
        carrier-only serve.  Without this contrast the refuse pins could
        pass vacuously.
        """
        lens = _route_lens(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2)
        # Premise: the raw certificate estimate at w_hi is under the bar.
        est = _born.born_carrier_omitted_term(
            ROUTE_POS_WHI, ROUTE_POS_Y1, ROUTE_POS_Y2, ROUTE_POS_GAMMA)
        self.assertLessEqual(
            _SADDLE_FARFIELD_SAFETY * est, _SADDLE_FARFIELD_CERT_BAR,
            'non-degenerate fixture is no longer accuracy-admitted; the '
            'contrast has lost its point')
        self.assertTrue(
            likelihood._born_carrier_certificate_serves(
                lens, ROUTE_POS_WLO, ROUTE_POS_WHI, self._BACKSTOP_IMAGES),
            'non-degenerate positive far-exterior source was refused with '
            'backstop-passing images; the refuse pins may be vacuous')


class CarrierCertificateSelfFalsificationTestCase(_BornCertTestCase):
    """Prove the three pins can go RED -- they are not vacuous.

    Each check re-runs the exact predicate of a sibling pin against a
    deliberately corrupted input and asserts it FAILS, so a future change
    that silently breaks the physics cannot slip past a green suite.
    """

    def test_monotone_pin_rejects_a_decreasing_sequence(self):
        """The strict-increase predicate raises on a decreasing sequence."""
        decreasing = np.array([3.0, 2.0, 1.0])
        with self.assertRaises(AssertionError):
            for a, b in zip(decreasing[:-1], decreasing[1:]):
                self.assertGreater(b - a, 0.0)

    def test_invariant_pin_rejects_a_perturbed_factor(self):
        """A 1e-6 perturbation of ``b1`` breaks the 1e-13 invariant pin."""
        _, _, _, b1, a0 = _born._born_factors(
            INVARIANT_Y1, INVARIANT_Y2, 1.5, 0.0, 0.0)
        expected = -(1.0 ** 2) * (1.0 / (1.0 - 1.5 ** 2))
        # Unperturbed passes (sanity), perturbed fails (teeth).
        self.assertAlmostEqual(b1 - a0, expected, delta=INVARIANT_ABS_TOL)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual((b1 + 1e-6) - a0, expected,
                                   delta=INVARIANT_ABS_TOL)

    def test_accuracy_pin_rejects_a_corrupted_carrier(self):
        """A 5% carrier corruption blows the served-vs-exact bar."""
        gamma, y1, y2, w_lo, w_hi = POS_POINTS[0]
        source = np.array([y1, y2])
        w = 0.5 * (w_lo + w_hi)
        served = _born.born_lead_carrier(float(w), y1, y2, gamma)
        exact, _ = operator.F_op(float(w), source, gamma)
        self.n_compared += 1
        # The true carrier is under the bar (this is a real admitted point).
        self.assertLessEqual(
            abs(served - exact) / abs(exact), _SADDLE_FARFIELD_CERT_BAR)
        # A 5% corruption is far above the bar -- the comparison has teeth.
        corrupted = served * 1.05
        self.assertGreater(
            abs(corrupted - exact) / abs(exact), _SADDLE_FARFIELD_CERT_BAR)


# =========================================================================== #
# Serve-routing pins (WP2 lift): null-identity, no-covers()-refusal, saddle
# resolution fence.  ENGINE-FREE -- the real cheap ``geometry_partition`` +
# ``caustic_rho`` and the real certificate run, but ``_born_reconstruct`` (the
# heavy carrier / Rung-P / kernel-reduction tail) is SPIED and returns a
# sentinel, and ``operator.F_op`` is never called.
# =========================================================================== #

#: Positive-parity far-exterior routing fixture: rho ~ 111 (> 2, two real
#: images) and certificate-admitted at the band ceiling.  Reused as an in-box
#: query (covers()==True stub -> null-identity) and a beyond-box query
#: (covers()==False stub -> carrier-only admit).
ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2 = 0.30, 80.0, 0.0
ROUTE_POS_WLO, ROUTE_POS_WHI = 0.05, 0.75

#: Macro-saddle (gamma > 1) beyond-box routing fixture.  rho ~ 79 (> 2) and
#: the real-image Fermat separation is huge, so the resolution fence
#: ``w_lo * delta_min >= RHO_END`` is the binding gate.
ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2 = 1.5, 150.0, 0.0
ROUTE_SAD_WHI = 0.40

#: Dense-grid node count for the routing probes (only cheap geometry + the
#: real certificate run; the reconstruction tail is spied).
ROUTE_N = 8

#: A DISTINCTIVE non-zero complex residual the stub chart returns in box, so
#: the null-identity pin can prove it flows through ``_born_reconstruct``
#: unperturbed (and is NOT the carrier-only zero residual).
_DISTINCT_RESIDUAL = (np.arange(1, ROUTE_N + 1, dtype=float)
                      + 1j * np.arange(ROUTE_N, 0, -1, dtype=float))


def _route_lens(gamma, y1, y2):
    """A ``kappa=0``, ``beta=0`` lens dict with the keys the serve reads."""
    return {'kappa': 0.0, 'beta': 0.0, 'gamma': float(gamma),
            'y1': float(y1), 'y2': float(y2)}


def _route_rho(gamma, y1, y2):
    """Live caustic-frame rho for the routing premise assertions."""
    return caustic_rho(float(gamma), float(math.hypot(y1, y2)), 0.0)


def _real_images(gamma, y1, y2, dense_w):
    """Real image positions from the cheap production geometry partition."""
    geom = ChangRefsdalChannels(dense_w).geometry_partition(
        gamma=float(gamma), y=(float(y1), float(y2)), beta=0.0, kappa=0.0)
    return np.asarray(geom.images)


class _StubChart:
    """A Born residual chart with controllable ``covers`` / ``evaluate``.

    ``covers_result`` is a bool used for BOTH the 2-arg containment probe
    and the 3-arg trained-band probe, so ``covers()==True`` routes fully
    in box and ``covers()==False`` routes to the carrier-only certificate.
    ``evaluate`` returns the fixed distinctive residual and records its
    call count, so the null-identity pin can prove the interpolated
    residual (not a zero) reached ``_born_reconstruct``.
    """

    def __init__(self, covers_result: bool, residual: np.ndarray):
        self._covers_result = bool(covers_result)
        self._residual = residual
        self.evaluate_calls = 0

    def covers(self, gamma, rho, w=None):  # noqa: D401 - stub
        """Constant containment verdict for both the 2- and 3-arg probes."""
        return self._covers_result

    def evaluate(self, dense_w, gamma, rho):  # noqa: D401 - stub
        """Return the distinctive in-box residual (records the call)."""
        self.evaluate_calls += 1
        return np.array(self._residual, copy=True)


class _ReconstructSpy:
    """Records the residual/masks fed to ``_born_reconstruct``, skips tail.

    Returns a unique sentinel so the caller can distinguish "served"
    (sentinel) from "declined to the engine" (``None``) without running
    the heavy reconstruction.
    """

    def __init__(self):
        self.sentinel = object()
        self.calls: list[dict] = []

    def __call__(self, lens, dense_w, geom, residual, below_mask,
                 bottom_mask):
        self.calls.append({
            'residual': np.array(residual, copy=True),
            'below_mask': np.array(below_mask, copy=True),
            'bottom_mask': np.array(bottom_mask, copy=True),
        })
        return self.sentinel


def _make_probe(chart: _StubChart):
    """Bind the real ``_born_residual_analytic`` onto a minimal object.

    The map-consult / cell-ceiling / diffractive-bottom helpers are stubbed
    to their no-split identity (``None``) so ``below_mask`` is all-True and
    ``bottom_mask`` all-False -- the whole band is served by Born and no
    band split confounds the routing pin.  ``_born_reconstruct`` is the spy.
    """
    probe = types.SimpleNamespace()
    probe.born_residual_chart = chart
    probe._ppgo_band_split = lambda lens: None
    probe._ppgo_cell_ceiling = lambda lens: None
    probe._diffractive_bottom_ceiling = lambda lens: None
    probe.spy = _ReconstructSpy()
    probe._born_reconstruct = probe.spy
    probe.serve = types.MethodType(
        LensedRelativeBinningLikelihood._born_residual_analytic, probe)
    return probe


class BornInBoxNullIdentityTestCase(_BornCertTestCase):
    """Null-identity: the carrier-only lift never perturbs the in-box serve.

    On a positive-parity in-box query (``rho > 2`` AND ``covers() == True``)
    the residual fed to ``_born_reconstruct`` is EXACTLY the chart's
    interpolated residual -- never the carrier-only zero residual -- and the
    certificate ``_born_carrier_certificate_serves`` is NEVER consulted.
    Any perturbation of the in-box residual localizes to the reconstruction
    tail refactor.  Engine-free (the tail is spied).
    """

    def setUp(self):
        super().setUp()
        self.dense_w = np.linspace(ROUTE_POS_WLO, ROUTE_POS_WHI, ROUTE_N)
        self.lens = _route_lens(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2)
        # Premise: a genuine far-exterior in-box query (rho > 2).
        self.rho = _route_rho(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2)
        self.assertGreater(self.rho, 2.0,
                           'fixture is not far-exterior (rho <= 2)')

    def test_in_box_residual_is_the_interpolated_chart_residual(self):
        """The residual reconstructed in box equals ``chart.evaluate``."""
        chart = _StubChart(covers_result=True, residual=_DISTINCT_RESIDUAL)
        probe = _make_probe(chart)
        with mock.patch.object(
                likelihood, '_born_carrier_certificate_serves',
                wraps=likelihood._born_carrier_certificate_serves) as cert:
            result = probe.serve(self.lens, self.dense_w)
        # Served (sentinel), the in-box branch ran chart.evaluate exactly once.
        self.assertIs(result, probe.spy.sentinel)
        self.assertEqual(chart.evaluate_calls, 1)
        self.assertEqual(len(probe.spy.calls), 1)
        fed = probe.spy.calls[0]['residual']
        # BYTE-IDENTICAL to the chart's interpolated residual.
        np.testing.assert_array_equal(fed, _DISTINCT_RESIDUAL)
        # And emphatically NOT the carrier-only zero residual.
        self.assertTrue(np.any(fed != 0.0),
                        'in-box residual was zeroed -- the carrier-only path '
                        'perturbed the in-box serve')
        # The certificate is NEVER consulted on an in-box serve.
        self.assertEqual(cert.call_count, 0,
                         'in-box serve consulted the carrier-only certificate')

    def test_in_box_serve_uses_full_band_no_split(self):
        """No map/diffractive split: below_mask all-True, bottom all-False."""
        chart = _StubChart(covers_result=True, residual=_DISTINCT_RESIDUAL)
        probe = _make_probe(chart)
        probe.serve(self.lens, self.dense_w)
        call = probe.spy.calls[0]
        self.assertTrue(np.all(call['below_mask']),
                        'below_mask not all-True with the split helpers off')
        self.assertFalse(np.any(call['bottom_mask']),
                         'bottom_mask not all-False with the split helpers off')


def _fence_w(gamma, y1, y2):
    """Live saddle resolution-fence floor ``RHO_END / delta_min``."""
    source = np.array([float(y1), float(y2)])
    matrix = geometry.macro_matrix(float(gamma), 0.0, 0.0)
    delta_min = operator._real_delay_min_separation(source, matrix)
    return operator.RHO_END / delta_min, delta_min


class BornNoCoversRefusalTestCase(_BornCertTestCase):
    """Every ``covers()==False`` query consults the certificate first.

    The exact defect this build closes: HEAD refused a bare ``covers()``
    miss straight to the exact engine.  After the lift, EVERY
    ``covers()==False`` query (both parities) must route through
    ``_born_carrier_certificate_serves`` -- serving carrier-only (zero
    residual) when it admits, declining to the engine (``None``) only when
    it refuses.  There is NO surviving path that returns ``None`` on a
    ``covers()`` miss without asking the certificate.  Engine-free.
    """

    def _route_covers_false(self, lens, dense_w):
        """Serve a covers()==False query; return (result, verdicts, calls)."""
        chart = _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL)
        probe = _make_probe(chart)
        verdicts: list = []
        real = likelihood._born_carrier_certificate_serves

        def _wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            verdicts.append(out)
            return out

        with mock.patch.object(
                likelihood, '_born_carrier_certificate_serves',
                side_effect=_wrapper) as cert:
            result = probe.serve(lens, dense_w)
        return result, verdicts, cert.call_count, probe.spy

    def test_every_covers_false_query_consults_certificate(self):
        """Admit -> carrier-only serve; refuse -> engine; always consulted."""
        fence_w, _ = _fence_w(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        # (label, lens, dense_w, expect_admit)
        family = (
            ('positive_admit',
             _route_lens(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2),
             np.linspace(ROUTE_POS_WLO, ROUTE_POS_WHI, ROUTE_N), True),
            ('saddle_above_fence_admit',
             _route_lens(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2),
             np.linspace(fence_w * 1.5, ROUTE_SAD_WHI, ROUTE_N), True),
            ('saddle_below_fence_refuse',
             _route_lens(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2),
             np.linspace(fence_w * 0.1, fence_w * 0.9, ROUTE_N), False),
        )
        for label, lens, dense_w, expect_admit in family:
            with self.subTest(query=label):
                # Premise: rho > 2, so the serve REACHES the certificate
                # branch rather than the early rho <= 2 fallthrough.
                rho = _route_rho(lens['gamma'], lens['y1'], lens['y2'])
                self.assertGreater(rho, 2.0,
                                   f'{label}: fixture is not far-exterior')
                result, verdicts, n_calls, spy = self._route_covers_false(
                    lens, dense_w)
                # The certificate was consulted EXACTLY once -- no covers()
                # miss reaches a decision without asking it.
                self.assertEqual(
                    n_calls, 1,
                    f'{label}: covers()==False query did not consult the '
                    f'certificate exactly once (n={n_calls})')
                verdict = verdicts[0]
                self.assertEqual(verdict, expect_admit,
                                 f'{label}: certificate verdict {verdict!r} '
                                 f'!= expected {expect_admit!r}')
                if verdict:
                    # Admitted -> served carrier-only: sentinel + ZERO
                    # residual (the lift keeps ONLY the lead carrier).
                    self.assertIs(result, spy.sentinel,
                                  f'{label}: admitted but not served')
                    fed = spy.calls[0]['residual']
                    np.testing.assert_array_equal(
                        fed, np.zeros(ROUTE_N, dtype=complex))
                else:
                    # Refused -> declines to the exact engine, and did so
                    # ONLY after consulting the certificate (n_calls == 1
                    # above proves no straight-refusal path survived).
                    self.assertIsNone(
                        result,
                        f'{label}: refused certificate but did not decline')
                    self.assertEqual(len(spy.calls), 0,
                                     f'{label}: reconstructed despite refusal')

    def test_no_straight_refusal_path_on_covers_miss(self):
        """A refused covers()-miss returns None ONLY via the certificate.

        If a surviving straight-refusal path existed, the serve would
        return ``None`` with the certificate call_count still zero.  This
        asserts the refuse case's ``None`` is accompanied by exactly one
        certificate consult -- the negation of that defect.
        """
        fence_w, _ = _fence_w(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        lens = _route_lens(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        dense_w = np.linspace(fence_w * 0.1, fence_w * 0.9, ROUTE_N)
        result, verdicts, n_calls, _ = self._route_covers_false(lens, dense_w)
        self.assertIsNone(result)
        self.assertEqual(n_calls, 1)
        self.assertEqual(verdicts, [False])


class SaddleResolutionFenceTestCase(_BornCertTestCase):
    """The macro-saddle serve is fenced at ``w_lo * delta_min == RHO_END``.

    For a macro saddle (``gamma > 1``) with smallest pairwise real-image
    Fermat separation ``delta_min``, the certificate REFUSES the whole band
    when ``w_lo * delta_min < RHO_END`` (=4.0) and ADMITS carrier-only for
    the twin just above -- with every OTHER admission criterion (accuracy
    certificate at ``w_hi``, min-image-separation backstop) held fixed, so
    the fence is the sole thing that flips.  Positive parity is NOT subject
    to the fence.  Engine-free (predicate only).
    """

    def setUp(self):
        super().setUp()
        self.lens = _route_lens(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        self.w_hi = ROUTE_SAD_WHI
        self.fence_w, self.delta_min = _fence_w(
            ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        # Real images for the min-separation backstop (band-independent).
        dense = np.linspace(self.fence_w * 0.1, self.w_hi, ROUTE_N)
        self.images = _real_images(
            ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2, dense)

    def test_transition_sits_exactly_at_rho_end(self):
        """Below the fence refuses; the twin just above admits."""
        # Premise: the accuracy certificate ADMITS at w_hi and the backstop
        # holds, so the ONLY gate that can flip across the fence is the
        # resolution fence itself.
        est = _born.born_carrier_omitted_term(
            self.w_hi, self.lens['y1'], self.lens['y2'], self.lens['gamma'])
        self.assertLessEqual(
            _SADDLE_FARFIELD_SAFETY * est, _SADDLE_FARFIELD_CERT_BAR,
            'saddle fixture no longer accuracy-admitted at w_hi; the fence '
            'is not the sole flipping gate')
        # Below the fence: w_lo * delta_min < RHO_END -> refuse whole band.
        w_lo_below = self.fence_w * 0.999
        self.assertLess(w_lo_below * self.delta_min, operator.RHO_END)
        self.assertFalse(
            likelihood._born_carrier_certificate_serves(
                self.lens, w_lo_below, self.w_hi, self.images),
            'below-fence saddle draw was not refused')
        # Just above the fence: admits carrier-only.
        w_lo_above = self.fence_w * 1.001
        self.assertGreaterEqual(w_lo_above * self.delta_min, operator.RHO_END)
        self.assertTrue(
            likelihood._born_carrier_certificate_serves(
                self.lens, w_lo_above, self.w_hi, self.images),
            'above-fence saddle draw was not admitted')

    def test_positive_parity_has_no_fence(self):
        """A positive-parity admitted point stays admitted as ``w_lo -> 0``."""
        lens = _route_lens(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2)
        # Real images are band-independent; build them from a strictly
        # positive grid (the geometry validator rejects a zero node) while
        # the certificate is separately probed at w_lo == 0.0 below.
        dense = np.linspace(1e-6, ROUTE_POS_WHI, ROUTE_N)
        images = _real_images(
            ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2, dense)
        for w_lo in (1e-12, 0.0):
            with self.subTest(w_lo=w_lo):
                self.assertTrue(
                    likelihood._born_carrier_certificate_serves(
                        lens, w_lo, ROUTE_POS_WHI, images),
                    f'positive parity refused at w_lo={w_lo}: a fence leaked '
                    f'onto gamma < 1')

    def test_fence_transition_diagnostic_plot(self):
        """Admit/refuse vs ``w_lo * delta_min``; the step sits at 4.0."""
        ratios = np.linspace(1.0, 8.0, 40)
        w_los = ratios / self.delta_min
        admitted = np.array([
            likelihood._born_carrier_certificate_serves(
                self.lens, float(w_lo), self.w_hi, self.images)
            for w_lo in w_los], dtype=float)
        # The step is exactly at the ratio == RHO_END: all refused below,
        # all admitted at/above.
        below = ratios < operator.RHO_END
        self.assertTrue(np.all(admitted[below] == 0.0))
        self.assertTrue(np.all(admitted[~below] == 1.0))
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.step(ratios, admitted, where='post', color='tab:red')
        ax.axvline(operator.RHO_END, color='k', ls='--',
                   label=f'RHO_END={operator.RHO_END:g}')
        ax.set_xlabel('w_lo * delta_min')
        ax.set_ylabel('carrier-only admitted (1) / refused (0)')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Spec: saddle resolution fence transition')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save_plot(fig, 'born_certificate_saddle_fence_transition.png')


class ServeRoutingSelfFalsificationTestCase(_BornCertTestCase):
    """Prove the three serve-routing pins can go RED -- not vacuous.

    Each check corrupts the exact quantity a sibling pin asserts and shows
    the assertion FAILS, so a regression that silently rewires the routing
    cannot slip past a green suite.
    """

    def test_null_identity_pin_rejects_a_zero_residual(self):
        """The byte-identity assertion fails against a zeroed residual.

        Flipping ``covers()`` to False on the SAME positive lens routes to
        the carrier-only ZERO residual; the null-identity pin's exact
        equality against the distinctive residual then raises -- proof the
        pin distinguishes an in-box residual from a carrier-only zero.
        """
        lens = _route_lens(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2)
        dense_w = np.linspace(ROUTE_POS_WLO, ROUTE_POS_WHI, ROUTE_N)
        probe = _make_probe(
            _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL))
        probe.serve(lens, dense_w)
        fed_zero = probe.spy.calls[0]['residual']
        # The carrier-only route feeds zeros; the null-identity pin's exact
        # equality against the distinctive residual MUST fail.
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(fed_zero, _DISTINCT_RESIDUAL)

    def test_routing_follows_certificate_verdict(self):
        """Forcing the certificate True/False flips serve served/declined.

        If the routing ignored the certificate (a surviving straight
        refusal), patching its verdict would not change the outcome.  Here
        it does: forced-False declines to the engine (``None``), forced-True
        serves carrier-only (sentinel + zero residual).
        """
        lens = _route_lens(ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2)
        dense_w = np.linspace(ROUTE_POS_WLO, ROUTE_POS_WHI, ROUTE_N)
        # Forced refusal -> decline.
        probe_refuse = _make_probe(
            _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL))
        with mock.patch.object(likelihood,
                               '_born_carrier_certificate_serves',
                               return_value=False):
            self.assertIsNone(probe_refuse.serve(lens, dense_w))
        self.assertEqual(len(probe_refuse.spy.calls), 0)
        # Forced admission -> carrier-only serve.
        probe_admit = _make_probe(
            _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL))
        with mock.patch.object(likelihood,
                               '_born_carrier_certificate_serves',
                               return_value=True):
            self.assertIs(probe_admit.serve(lens, dense_w),
                          probe_admit.spy.sentinel)
        np.testing.assert_array_equal(
            probe_admit.spy.calls[0]['residual'],
            np.zeros(ROUTE_N, dtype=complex))

    def test_fence_pin_rejects_below_fence_as_served(self):
        """Asserting the below-fence saddle draw serves must raise."""
        lens = _route_lens(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        fence_w, delta_min = _fence_w(
            ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        dense = np.linspace(fence_w * 0.1, ROUTE_SAD_WHI, ROUTE_N)
        images = _real_images(
            ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2, dense)
        serves_below = likelihood._born_carrier_certificate_serves(
            lens, fence_w * 0.999, ROUTE_SAD_WHI, images)
        self.assertFalse(serves_below)
        with self.assertRaises(AssertionError):
            self.assertTrue(serves_below)


if __name__ == '__main__':
    unittest.main()
