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

import dataclasses
import functools
import math
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing.chang_refsdal import _born, _schwinger, geometry, operator
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, born_carrier_from_partition)
from cogwheel.lensing import likelihood
from cogwheel.lensing import ppgo_map
from cogwheel.lensing import serve_route_census
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood,
    _band_split_mask,
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
    the heavy reconstruction.  The trained-floor band split (WP1) reaches
    ``_born_reconstruct`` with two ADDITIONAL keyword arguments --
    ``engine_envelope`` and ``engine_mask`` -- carrying the exact-engine
    envelope that hosts the untrained ``[w_low, trained_floor)`` remainder;
    they are recorded (as copies, or ``None`` on Routes 1/3 which never
    pass them) so the tier-routing pin can prove which nodes each source
    populates without running the engine.
    """

    def __init__(self):
        self.sentinel = object()
        self.calls: list[dict] = []

    def __call__(self, lens, dense_w, geom, residual, below_mask,
                 bottom_mask, engine_envelope=None, engine_mask=None):
        self.calls.append({
            'residual': np.array(residual, copy=True),
            'below_mask': np.array(below_mask, copy=True),
            'bottom_mask': np.array(bottom_mask, copy=True),
            'engine_envelope': (None if engine_envelope is None
                                else np.array(engine_envelope, copy=True)),
            'engine_mask': (None if engine_mask is None
                            else np.array(engine_mask, copy=True)),
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
    probe._diffractive_bottom_ceiling = lambda lens, *, w_lo=None, w_hi=None: None
    probe.spy = _ReconstructSpy()
    probe._born_reconstruct = probe.spy
    probe.serve = types.MethodType(
        LensedRelativeBinningLikelihood._born_residual_analytic, probe)
    return probe


class _WBandChart:
    """A Born residual chart whose ``covers`` enforces a log-w trained band.

    Faithful to the shipped ``BornResidualChart`` interface the trained-
    floor band split reads: a 2-argument ``covers(gamma, rho)`` box probe
    (always True here -- the synthetic ``(gamma, rho)`` box is orthogonal to
    this suite's concern, the LOG-W split) and a 3-argument
    ``covers(gamma, rho, w)`` that additionally requires the whole served
    band ``w`` to lie inside the trained ``[floor_w, ceil_w]`` range, so a
    served band dipping below ``floor_w`` reads as a trained-band escape.
    ``log_w_grid`` exposes ``[log(floor_w), log(ceil_w)]`` so production
    reads ``trained_floor = exp(log_w_grid[0]) == floor_w`` from the
    artifact -- NEVER a literal.  ``evaluate`` returns the fixed sentinel
    residual over whatever sub-band it is handed, recording each sub-band.
    """

    def __init__(self, floor_w: float, ceil_w: float, sentinel: complex):
        self.floor_w = float(floor_w)
        self.ceil_w = float(ceil_w)
        self.sentinel = complex(sentinel)
        #: trained log-w coverage; production reads exp(log_w_grid[0]).
        self.log_w_grid = np.array(
            [math.log(self.floor_w), math.log(self.ceil_w)], dtype=float)
        self.evaluate_calls: list[np.ndarray] = []

    def covers(self, gamma, rho, w=None):  # noqa: D401 - stub
        """Box always covers; the 3-arg probe enforces the log-w band."""
        if w is None:
            return True
        w = np.asarray(w, dtype=float)
        if w.size == 0:
            return True
        return bool(w.min() >= self.floor_w and w.max() <= self.ceil_w)

    def evaluate(self, w, gamma, rho):  # noqa: D401 - stub
        """Return the sentinel residual over ``w`` (records the sub-band)."""
        w = np.asarray(w, dtype=float)
        self.evaluate_calls.append(np.array(w, copy=True))
        return np.full(w.shape, self.sentinel, dtype=complex)


def _make_floor_probe(chart, *, w_trust, w_low, engine_value=5.0 + 2.0j):
    """Bind ``_born_residual_analytic`` with an ACTIVE trained-floor split.

    Unlike ``_make_probe`` (no-split identity), this drives the map-consult
    ``w_trust`` and the diffractive-bottom ``w_low`` to concrete floats so
    the host band splits into the four tiers WP1 routes.  The engine
    sub-envelope helper ``_engine_envelope_below_split`` is a SPY returning
    ``engine_value`` on the mask nodes (and zero elsewhere, matching the
    production full-length shape) and recording the ``(dense_w, mask)`` it
    is handed; ``_born_reconstruct`` is the sentinel spy.  Engine-free.
    """
    probe = types.SimpleNamespace()
    probe.born_residual_chart = chart
    probe._ppgo_band_split = lambda lens: w_trust
    probe._ppgo_cell_ceiling = lambda lens: None
    probe._diffractive_bottom_ceiling = lambda lens, *, w_lo=None, w_hi=None: w_low
    probe.engine_env_calls = []
    probe.engine_value = complex(engine_value)

    def _engine_env(lens, dense_w, mask):
        probe.engine_env_calls.append({
            'dense_w': np.array(dense_w, copy=True),
            'mask': np.array(mask, copy=True),
        })
        out = np.zeros(np.shape(dense_w), dtype=complex)
        out[np.asarray(mask, dtype=bool)] = probe.engine_value
        return out

    probe._engine_envelope_below_split = _engine_env
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


# =========================================================================== #
# Trained-floor band-split pins (WP1/WP2).  A box-covered far-exterior draw
# whose served host sub-band DROPS BELOW the chart's trained log-w floor is a
# LOW-EDGE escape: instead of refusing the whole band (the Fact-2 regression),
# the chart serves the trained sub-band ``[trained_floor, w_trust]`` and the
# exact engine hosts the untrained remainder ``[w_low, trained_floor)`` below
# it.  These pins are ENGINE-FREE: the real cheap ``geometry_partition`` +
# ``caustic_rho`` run, but ``_engine_envelope_below_split`` and
# ``_born_reconstruct`` are spied so ``operator.F_op`` is never called.
# =========================================================================== #

#: Positive-parity far-exterior lens for the trained-floor pins (rho ~ 111 >
#: 2, gamma < 1 so the ceiling is the astroid wall >> the tiny w_hi here).
FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2 = ROUTE_POS_GAMMA, ROUTE_POS_Y1, ROUTE_POS_Y2
#: Dense band spanning all four tiers: nodes 0.05,0.15,0.25,0.35,0.45,...,0.75.
FLOOR_DENSE = np.linspace(0.05, 0.75, 8)
#: Map-consult trusted floor (below_mask boundary): w_trust = 0.60.
FLOOR_WTRUST = 0.60
#: Diffractive-bottom ceiling (bottom_mask boundary): w_low = 0.20.
FLOOR_WLOW = 0.20
#: Trained log-w floor the chart advertises via ``exp(log_w_grid[0])`` = 0.40;
#: it lies STRICTLY between w_low and w_trust so the host band splits into a
#: genuine engine tier below it and a chart tier above it.
FLOOR_TRAINED = 0.40
#: Trained log-w ceiling (well above w_hi so the chart sub-band is covered).
FLOOR_CEIL = 4.0
#: Distinctive residual the chart serves on its trained tier (NOT zero, NOT
#: the engine_value, so both an all-zero and a mislabelled tier are caught).
FLOOR_SENTINEL = 7.0 - 3.0j
#: Distinctive engine-hosted envelope value on the engine tier.
FLOOR_ENGINE_VALUE = 5.0 + 2.0j


def _floor_tier_masks(dense_w, *, w_trust, w_low, trained_floor):
    """Derive the four trained-floor tier masks from the LIVE band-split.

    Mirrors ``_born_residual_analytic`` exactly using the shipped
    ``_band_split_mask`` (never a hand-written comparison), returning
    ``(bottom_mask, engine_mask, chart_mask, above_mask)`` -- the diffractive
    bottom ``[w_lo, w_low)``, the engine-hosted ``[w_low, trained_floor)``,
    the chart-served ``[trained_floor, w_trust]`` and the bare-ppGO carrier
    ``(w_trust, w_hi]`` -- so a test can assert the partition PREMISE before
    it drives the serve.
    """
    _band_split, below_mask = _band_split_mask(dense_w, w_trust)
    band_split_low, below_low = _band_split_mask(dense_w, w_low)
    bottom_mask = below_low & below_mask if band_split_low \
        else np.zeros(dense_w.shape, dtype=bool)
    host_mask = below_mask & ~bottom_mask
    _band_split_floor, below_floor = _band_split_mask(dense_w, trained_floor)
    engine_mask = host_mask & below_floor
    chart_mask = host_mask & ~below_floor
    above_mask = ~below_mask
    return bottom_mask, engine_mask, chart_mask, above_mask


class BornTrainedFloorTierRoutingTestCase(_BornCertTestCase):
    """FLOOR-SPLIT TIER ROUTING: the three tiers map to the correct w segments.

    A box-covered draw whose chart trained log-w range is a STRICT sub-band
    ``[trained_floor, w_trust]`` of the requested band routes into four
    disjoint w-tiers.  Realised ENGINE-FREE via the spy-recorded residual /
    masks / engine_envelope / engine_mask fed to ``_born_reconstruct``:

    * ``[trained_floor, w_trust]`` -> the chart's sentinel residual
      (chart-served);
    * ``[w_low, trained_floor)`` -> the engine-hosted envelope value
      (engine-served, sentinel ABSENT);
    * ``[w_lo, w_low)`` -> the diffractive ``F_P`` bottom (``bottom_mask``);
    * ``(w_trust, w_hi]`` -> the bare ppGO carrier (above ``below_mask``).

    An off-by-one in the inner ``chart_mask`` / ``engine_mask`` split, or a
    polarity flip, moves a tier boundary to the wrong ``w`` and this pin
    fails on the mis-labelled node.
    """

    def setUp(self):
        super().setUp()
        self.lens = _route_lens(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        # Premise: far-exterior (rho > 2) so the serve REACHES the routing
        # branches rather than the early rho <= 2 fallthrough.
        self.rho = _route_rho(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        self.assertGreater(self.rho, 2.0, 'fixture is not far-exterior')
        # Premise: trained_floor is read from the artifact, not a literal.
        self.chart = _WBandChart(FLOOR_TRAINED, FLOOR_CEIL, FLOOR_SENTINEL)
        self.assertEqual(
            math.exp(float(self.chart.log_w_grid[0])), FLOOR_TRAINED,
            'trained_floor not recoverable from log_w_grid[0]')
        (self.bottom_mask, self.engine_mask, self.chart_mask,
         self.above_mask) = _floor_tier_masks(
            FLOOR_DENSE, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
            trained_floor=FLOOR_TRAINED)

    def test_four_tiers_partition_the_band(self):
        """Premise: the four tiers are non-empty and partition the band."""
        tiers = (self.bottom_mask, self.engine_mask, self.chart_mask,
                 self.above_mask)
        for name, mask in zip(
                ('bottom', 'engine', 'chart', 'above'), tiers):
            with self.subTest(tier=name):
                self.assertTrue(mask.any(), f'{name} tier is empty')
        union = np.zeros(FLOOR_DENSE.shape, dtype=bool)
        overlap = np.zeros(FLOOR_DENSE.shape, dtype=int)
        for mask in tiers:
            union |= mask
            overlap += mask.astype(int)
        self.assertTrue(np.all(union), 'tiers do not cover the whole band')
        np.testing.assert_array_equal(
            overlap, np.ones(FLOOR_DENSE.shape, dtype=int),
            'tiers are not disjoint (a node belongs to two tiers)')

    def test_tiers_route_to_the_correct_sources(self):
        """Chart sentinel on [floor,trust]; engine value on [low,floor); ..."""
        probe = _make_floor_probe(
            self.chart, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
            engine_value=FLOOR_ENGINE_VALUE)
        with mock.patch.object(
                likelihood, '_born_carrier_certificate_serves') as cert:
            result = probe.serve(self.lens, FLOOR_DENSE)
        # Route 2 taken: the certificate (Route 3 only) was never consulted.
        self.assertIs(result, probe.spy.sentinel)
        self.assertEqual(cert.call_count, 0,
                         'trained-floor split leaked into the Route-3 '
                         'certificate')
        self.assertEqual(len(probe.spy.calls), 1)
        call = probe.spy.calls[0]
        residual = call['residual']
        # Chart tier: the sentinel residual, and ONLY there.
        np.testing.assert_array_equal(
            residual[self.chart_mask],
            np.full(int(self.chart_mask.sum()), FLOOR_SENTINEL))
        # Engine + bottom + above tiers: the residual is ZERO (sentinel
        # ABSENT) -- the engine envelope, not the residual, hosts them.
        non_chart = ~self.chart_mask
        np.testing.assert_array_equal(
            residual[non_chart],
            np.zeros(int(non_chart.sum()), dtype=complex),
            'sentinel residual leaked outside the chart tier')
        # Engine tier: engine_mask recorded, engine_envelope carries the
        # engine value there and zero elsewhere.
        self.assertIsNotNone(call['engine_mask'])
        np.testing.assert_array_equal(call['engine_mask'], self.engine_mask)
        env = call['engine_envelope']
        self.assertIsNotNone(env, 'Route 2 did not pass an engine_envelope')
        np.testing.assert_array_equal(
            env[self.engine_mask],
            np.full(int(self.engine_mask.sum()), FLOOR_ENGINE_VALUE))
        np.testing.assert_array_equal(
            env[~self.engine_mask],
            np.zeros(int((~self.engine_mask).sum()), dtype=complex))
        # Bottom + above tiers: the shared below/bottom masks locate them.
        np.testing.assert_array_equal(call['bottom_mask'], self.bottom_mask)
        np.testing.assert_array_equal(
            ~call['below_mask'], self.above_mask)
        # The chart evaluated EXACTLY the chart-tier sub-band, nothing else.
        self.assertEqual(len(self.chart.evaluate_calls), 1)
        np.testing.assert_array_equal(
            self.chart.evaluate_calls[0], FLOOR_DENSE[self.chart_mask])
        # The engine sub-envelope helper was consulted with the engine mask.
        self.assertEqual(len(probe.engine_env_calls), 1)
        np.testing.assert_array_equal(
            probe.engine_env_calls[0]['mask'], self.engine_mask)

    def test_tier_routing_diagnostic_plot(self):
        """Stacked w-segment plot coloring each node by its serving tier."""
        probe = _make_floor_probe(
            self.chart, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
            engine_value=FLOOR_ENGINE_VALUE)
        with mock.patch.object(
                likelihood, '_born_carrier_certificate_serves'):
            probe.serve(self.lens, FLOOR_DENSE)
        colors = {'bottom (F_P)': ('tab:purple', self.bottom_mask),
                  'engine': ('tab:blue', self.engine_mask),
                  'chart': ('tab:green', self.chart_mask),
                  'ppGO carrier': ('tab:orange', self.above_mask)}
        fig, ax = plt.subplots(figsize=(6.0, 2.6))
        for label, (color, mask) in colors.items():
            ax.scatter(FLOOR_DENSE[mask], np.zeros(int(mask.sum())),
                       s=120, color=color, label=label)
        for edge, name in ((FLOOR_WLOW, 'w_low'),
                           (FLOOR_TRAINED, 'trained_floor'),
                           (FLOOR_WTRUST, 'w_trust')):
            ax.axvline(edge, color='k', ls='--', alpha=0.5)
            ax.text(edge, 0.02, name, rotation=90, fontsize=7, va='bottom')
        ax.set_yticks([])
        ax.set_xlabel('w')
        ax.set_title('Spec B: trained-floor tier routing')
        ax.legend(fontsize=7, ncol=2, loc='upper center')
        _save_plot(fig, 'born_certificate_trained_floor_tier_routing.png')


#: Disjoint-HIGH trained range: entirely ABOVE the requested band
#: [0.05, 0.75], so EVERY served node escapes the trained log-w coverage --
#: a genuine full escape, not a low-edge split.  trained_floor = 2.0 sits
#: above w_hi, so the inner ``_band_split_mask`` at trained_floor is inactive
#: and Route 2 is correctly skipped in favour of the Route-3 carrier-only lift.
DISJOINT_FLOOR, DISJOINT_CEIL = 2.0, 4.0


class BornDisjointEscapeNullSplitTestCase(_BornCertTestCase):
    """NULL-SPLIT BYTE-IDENTITY: a full escape never engages Route 2.

    A box-covered draw whose chart trained log-w range is DISJOINT from the
    requested band (the whole served band escapes) must NOT trigger the
    trained-floor band split -- it must behave BYTE-IDENTICALLY to a plain
    ``covers()`` box-miss (the HEAD Route-3 fall-through / carrier-only lift):
    the residual fed to ``_born_reconstruct`` is the carrier-only ZERO
    residual, ``engine_envelope`` / ``engine_mask`` are ``None`` (Route 2
    never ran), and the chart's ``evaluate`` is NEVER called.  A future
    refactor of the tiering that let a disjoint range dribble into Route 2
    would perturb the residual or attach an engine envelope; this pin
    catches it.  Engine-free.
    """

    def setUp(self):
        super().setUp()
        self.lens = _route_lens(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        self.rho = _route_rho(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        self.assertGreater(self.rho, 2.0, 'fixture is not far-exterior')

    def _serve(self, chart):
        """Serve the disjoint/box-miss chart on the positive lens (Route 3)."""
        probe = _make_probe(chart)
        with mock.patch.object(
                likelihood, '_born_carrier_certificate_serves',
                wraps=likelihood._born_carrier_certificate_serves) as cert:
            result = probe.serve(self.lens, FLOOR_DENSE)
        return result, probe.spy, cert

    def test_disjoint_escape_is_byte_identical_to_box_miss(self):
        """Disjoint full-escape feeds the SAME thing as a plain covers-miss."""
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        # Premise: the trained range really is disjoint-HIGH (above w_hi).
        self.assertGreater(math.exp(float(disjoint.log_w_grid[0])),
                           float(FLOOR_DENSE.max()),
                           'trained floor is not above the band (not a full '
                           'escape)')
        # Reference: a plain covers()==False box-miss -- the HEAD Route-3 path.
        box_miss = _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL)
        res_new, spy_new, cert_new = self._serve(disjoint)
        res_ref, spy_ref, cert_ref = self._serve(box_miss)
        # Same certificate verdict (admitted) -> both serve carrier-only.
        self.assertIs(res_new, spy_new.sentinel)
        self.assertIs(res_ref, spy_ref.sentinel)
        self.assertEqual(cert_new.call_count, 1)
        self.assertEqual(cert_ref.call_count, 1)
        call_new, call_ref = spy_new.calls[0], spy_ref.calls[0]
        # BYTE-IDENTITY of every argument fed to _born_reconstruct.
        np.testing.assert_array_equal(
            call_new['residual'], call_ref['residual'])
        np.testing.assert_array_equal(
            call_new['below_mask'], call_ref['below_mask'])
        np.testing.assert_array_equal(
            call_new['bottom_mask'], call_ref['bottom_mask'])
        # The residual is the carrier-only ZERO residual (not the chart's).
        np.testing.assert_array_equal(
            call_new['residual'], np.zeros(FLOOR_DENSE.shape, dtype=complex))
        # Route 2 NEVER ran: no engine envelope / mask on either path.
        self.assertIsNone(call_new['engine_envelope'])
        self.assertIsNone(call_new['engine_mask'])
        self.assertIsNone(call_ref['engine_envelope'])
        # The disjoint chart's evaluate was NEVER invoked.
        self.assertEqual(len(disjoint.evaluate_calls), 0,
                         'disjoint full escape invoked chart interpolation')

    def test_disjoint_escape_evaluate_never_called_via_mock(self):
        """Explicit mock of the chart's evaluate: it is never called."""
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        with mock.patch.object(disjoint, 'evaluate',
                               wraps=disjoint.evaluate) as ev:
            self._serve(disjoint)
        self.assertEqual(ev.call_count, 0,
                         'chart interpolation invoked on a full escape')

    def test_disjoint_escape_refused_certificate_declines_like_box_miss(self):
        """A refused saddle full-escape declines to the engine (None), same."""
        lens = _route_lens(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        fence_w, _ = _fence_w(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        dense = np.linspace(fence_w * 0.1, fence_w * 0.9, ROUTE_N)
        # Premise: far-exterior saddle below the resolution fence.
        rho = _route_rho(ROUTE_SAD_GAMMA, ROUTE_SAD_Y1, ROUTE_SAD_Y2)
        self.assertGreater(rho, 2.0, 'saddle fixture is not far-exterior')
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        box_miss = _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL)
        probe_new = _make_probe(disjoint)
        probe_ref = _make_probe(box_miss)
        res_new = probe_new.serve(lens, dense)
        res_ref = probe_ref.serve(lens, dense)
        # Both decline to the exact engine; neither reconstructs.
        self.assertIsNone(res_new)
        self.assertIsNone(res_ref)
        self.assertEqual(len(probe_new.spy.calls), 0)
        self.assertEqual(len(probe_ref.spy.calls), 0)
        self.assertEqual(len(disjoint.evaluate_calls), 0)

    def test_disjoint_vs_box_miss_diagnostic_plot(self):
        """Residual-vs-w overlay: the disjoint and box-miss paths coincide."""
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        box_miss = _StubChart(covers_result=False, residual=_DISTINCT_RESIDUAL)
        _r, spy_new, _c = self._serve(disjoint)
        _r, spy_ref, _c = self._serve(box_miss)
        res_new = spy_new.calls[0]['residual']
        res_ref = spy_ref.calls[0]['residual']
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.plot(FLOOR_DENSE, res_new.real, 'o-', color='tab:green',
                label='disjoint escape (new path)')
        ax.plot(FLOOR_DENSE, res_ref.real, 'x--', color='tab:orange',
                label='box-miss (HEAD path)')
        ax.plot(FLOOR_DENSE, (res_new - res_ref).real, 's:',
                color='tab:red', label='difference (must be 0)')
        ax.set_xlabel('w')
        ax.set_ylabel('Re(residual fed to reconstruct)')
        ax.set_title('Spec A: disjoint escape == box-miss (null split)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save_plot(fig, 'born_certificate_disjoint_null_split.png')


# =========================================================================== #
# Spec C: FLOOR-SPLIT REVIVAL -- the census ``classify_draw`` routes a
# LOW-EDGE trained-floor escape to ``born_analytic`` (WP1 Route 2), NOT to a
# whole-band engine refusal (``engine_residual`` -- the Fact-2 regression a
# future gate tweak could reintroduce).
#
# Drives the REAL production ``serve_route_census.classify_draw`` waterfall
# against a SYNTHETIC ``_ProductionModules`` bundle (``dataclasses.replace``
# on the real ``_load_production_modules()`` output) swapping ONLY the four
# accessors that fix the tiering geometry -- the Born chart, the
# dimensionless-w map (pinned to ``FLOOR_DENSE``), the certified-ppGO
# ``w_trust`` split and the diffractive-bottom ``w_low`` -- so the geometry
# partition, the shipped ``_band_split_mask`` arithmetic, the
# ``_born_trained_floor_route`` mirror and the carrier certificate all stay
# the production objects.  Engine-free: ``dimensionless_frequency`` is
# stubbed and intercepts 1-5 return before the exact-wave node pass.
# =========================================================================== #


@functools.lru_cache(maxsize=1)
def _census_base_mods():
    """The real production-module bundle (loaded once, engine-free)."""
    return serve_route_census._load_production_modules()


#: Frequency grid handed to ``classify_draw``; ignored by the stubbed
#: ``dimensionless_frequency`` (which returns ``FLOOR_DENSE``), present only
#: because the signature requires it.
_CENSUS_F_GRID = np.geomspace(20.0, 1024.0, FLOOR_DENSE.size)


def _census_mods(chart, *, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
                 dense=FLOOR_DENSE, born_carrier_serves=None):
    """Synthetic ``_ProductionModules`` fixing a deterministic low-edge tiering.

    Swaps ONLY the chart, the dimensionless-w map (pinned to ``dense`` so the
    band is deterministic and engine-free), the certified-ppGO ``w_trust``
    split and the diffractive-bottom ``w_low`` ceiling.  Every other field --
    the real geometry class, the shipped ``_band_split_mask``, the parity
    walls, the ``_born_trained_floor_route`` mirror, the carrier certificate
    -- is the production object, so ``classify_draw`` runs the shipped
    waterfall.  ``born_carrier_serves`` may be overridden (Route-3 control)
    for the disjoint contrast.
    """
    base = _census_base_mods()
    overrides = dict(
        born_chart=chart,
        dimensionless_frequency=lambda f_grid, m, z: np.array(dense,
                                                              copy=True),
        ppgo_band_split=lambda lens: w_trust,
        ppgo_cell_ceiling=lambda lens: None,
        diffractive_bottom_ceiling=(
            lambda lens, *, w_lo=None, w_hi=None: w_low))
    if born_carrier_serves is not None:
        overrides['born_carrier_serves'] = born_carrier_serves
    return dataclasses.replace(base, **overrides)


def _classify_floor(chart, *, born_carrier_serves=None):
    """Run the real ``classify_draw`` on the far-exterior floor fixture."""
    mods = _census_mods(chart, born_carrier_serves=born_carrier_serves)
    return serve_route_census.classify_draw(
        mods, gamma=FLOOR_GAMMA, m_lens_msun=1.0, y1=FLOOR_Y1, y2=FLOOR_Y2,
        f_grid=_CENSUS_F_GRID, gamma_edges=ppgo_map._gamma_band_edges())


class BornTrainedFloorCensusRevivalTestCase(_BornCertTestCase):
    """FLOOR-SPLIT REVIVAL: census routes a low-edge escape to born_analytic.

    A box-covered draw whose HOST sub-band drops BELOW the chart's trained
    ``log_w`` floor (a genuine LOW-EDGE strict sub-band escape) must be
    classified ``born_analytic`` -- partially chart-served via WP1 Route 2 --
    NOT whole-refused to the engine (``engine_residual`` / the Fact-2
    regression).  A DISJOINT-HIGH escape (trained range entirely above the
    band) must NOT trigger Route 2 and instead falls to the carrier-only
    certificate (Route 3) -- proving the revival is specific to the low-edge
    split, not a blanket ``born_analytic`` verdict.  Driven through the REAL
    ``serve_route_census.classify_draw`` with only the chart + tiering
    accessors swapped; engine-free.
    """

    def setUp(self):
        super().setUp()
        # Premise: the fixture is the far-exterior Born box the intercept
        # requires (rho > the live _BORN_RHO_FLOOR) -- else the whole Born
        # rung is skipped and the route says nothing about Route 2.
        self.rho = _route_rho(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        self.assertGreater(self.rho, serve_route_census._BORN_RHO_FLOOR,
                           'fixture is not far-exterior; Born rung skipped')
        self.low_edge = _WBandChart(FLOOR_TRAINED, FLOOR_CEIL, FLOOR_SENTINEL)
        # Premise: a genuine STRICT sub-band -- trained floor read from the
        # artifact (never a literal) sits between w_low and w_trust.
        self.assertEqual(
            math.exp(float(self.low_edge.log_w_grid[0])), FLOOR_TRAINED,
            'trained_floor not recoverable from log_w_grid[0]')
        self.assertTrue(FLOOR_WLOW < FLOOR_TRAINED < FLOOR_WTRUST,
                        'trained floor is not a strict inner sub-band')

    def test_low_edge_escape_route_is_born_analytic(self):
        """Low-edge trained-floor escape -> born_analytic (Route 2 revival)."""
        # A cert stub that WOULD admit (Route 3): if Route 2 mistakenly
        # declined, the draw would still be born_carrier_only, so a
        # born_analytic verdict proves Route 2 fired, and call_count == 0
        # proves it fired BEFORE the certificate was ever consulted.
        cert = mock.Mock(return_value=True)
        result = _classify_floor(self.low_edge, born_carrier_serves=cert)
        self.assertEqual(result.route, 'born_analytic',
                         'low-edge trained-floor escape was not revived to '
                         'born_analytic')
        self.assertEqual(cert.call_count, 0,
                         'Route 2 leaked into the Route-3 carrier certificate')
        # The exact regression the spec guards: it must NOT whole-refuse.
        self.assertNotIn(result.route,
                         ('engine_residual', 'wave_refused', 'engine_refused'),
                         'low-edge escape whole-refused to the engine '
                         '(Fact-2 regression)')

    def test_disjoint_escape_is_not_born_analytic(self):
        """Disjoint-HIGH escape skips Route 2 -> carrier-only (Route 3)."""
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        # Premise: the trained range is entirely ABOVE the band (full escape).
        self.assertGreater(math.exp(float(disjoint.log_w_grid[0])),
                           float(FLOOR_DENSE.max()),
                           'disjoint fixture is not above the band')
        # Cert admits -> the ONLY way to born_carrier_only is Route 3, so the
        # route being carrier_only (not analytic) proves Route 2 was skipped.
        cert = mock.Mock(return_value=True)
        result = _classify_floor(disjoint, born_carrier_serves=cert)
        self.assertNotEqual(result.route, 'born_analytic',
                            'a disjoint full escape was mislabelled '
                            'born_analytic (Route 2 fired wrongly)')
        self.assertEqual(result.route, 'born_carrier_only')
        self.assertEqual(cert.call_count, 1,
                         'Route 3 certificate was not consulted for the '
                         'disjoint escape')

    def test_disjoint_escape_engine_refuses_without_certificate(self):
        """Disjoint escape + declining cert -> engine demand (the pre-WP1 fate).

        With Route 2 unavailable (disjoint) AND the carrier certificate
        declining, the draw falls through to the engine node pass -- the
        whole-refuse fate WP1 Route 2 rescues the low-edge population FROM.
        This is the mass that moves out of ``engine_residual`` into
        ``born_analytic`` once the trained floor is a strict inner sub-band.
        """
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        cert = mock.Mock(return_value=False)
        result = _classify_floor(disjoint, born_carrier_serves=cert)
        self.assertNotEqual(result.route, 'born_analytic')
        self.assertNotEqual(result.route, 'born_carrier_only')
        # It landed on the engine / analytic-fallthrough side, never revived.
        self.assertIn(
            result.route,
            ('engine_residual', 'analytics_engine_hosted', 'wave_refused',
             'diffractive_analytic', 'diffractive_engine_hosted'),
            'disjoint escape with a declining cert did not fall through to '
            'the engine node pass')

    def test_trained_floor_route_predicate_discriminates(self):
        """``_born_trained_floor_route`` is True low-edge, False disjoint.

        Self-falsification of the mirror predicate itself: the same host mask
        and w-grid yield True for the strict inner sub-band and False for the
        disjoint-high range (the inner split at ``trained_floor`` is inactive
        there), so the census route difference is the predicate's doing, not
        an incidental waterfall side effect.
        """
        _bottom, engine_mask, chart_mask, _above = _floor_tier_masks(
            FLOOR_DENSE, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
            trained_floor=FLOOR_TRAINED)
        host_mask = engine_mask | chart_mask
        self.assertTrue(engine_mask.any() and chart_mask.any(),
                        'the low-edge fixture does not split both tiers')
        mods_low = _census_mods(self.low_edge)
        self.assertTrue(
            serve_route_census._born_trained_floor_route(
                mods_low, FLOOR_GAMMA, self.rho, host_mask, FLOOR_DENSE),
            'the low-edge strict sub-band did not serve via Route 2')
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        mods_hi = _census_mods(disjoint)
        self.assertFalse(
            serve_route_census._born_trained_floor_route(
                mods_hi, FLOOR_GAMMA, self.rho, host_mask, FLOOR_DENSE),
            'a disjoint-high range wrongly served via Route 2')

    def test_revival_route_histogram_diagnostic_plot(self):
        """Route vs trained-floor sweep: revived born_analytic vs escape.

        Sweeps the chart's trained floor from inside the band (a strict inner
        sub-band -> born_analytic, Route 2) up past the band ceiling (a full
        escape -> born_carrier_only, Route 3, cert stubbed to admit), and
        histograms the resulting census route -- the born_analytic bar is
        exactly the population WP1 revives out of the engine.
        """
        floors = np.linspace(0.25, 3.0, 12)
        cert = mock.Mock(return_value=True)
        routes = []
        for floor in floors:
            chart = _WBandChart(float(floor), FLOOR_CEIL, FLOOR_SENTINEL)
            routes.append(
                _classify_floor(chart, born_carrier_serves=cert).route)
        # A genuine mix must appear, else the sweep pins nothing.
        self.assertIn('born_analytic', routes,
                      'no trained floor produced a revived born_analytic')
        self.assertIn('born_carrier_only', routes,
                      'no trained floor produced a Route-3 escape')
        labels = sorted(set(routes))
        counts = [routes.count(label) for label in labels]
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.bar(labels, counts, color='tab:green')
        ax.set_ylabel('trained floors in sweep')
        ax.set_title('Spec C: trained-floor sweep -> census route')
        ax.tick_params(axis='x', labelrotation=20, labelsize=8)
        fig.tight_layout()
        _save_plot(fig, 'born_certificate_revival_route_histogram.png')


# --------------------------------------------------------------------------- #
# NULL-RESIDUAL RECONSTRUCTION IDENTITY (Spec: R=0 -> bare carrier to
# round-off; the DRY ``_born_reconstruct`` tail stays exact under the new
# tiering, with the diffractive bottom overwrite and above-below_mask zeroing
# applied identically to HEAD).
# --------------------------------------------------------------------------- #

#: Relative tolerance for the R=0 reconstruction identity.  The
#: ``_born_reconstruct`` demodulation is an exact algebraic round-trip
#: (``(f_total - ppgo) * exp(1j w t_min)`` reversed by
#: ``reconstruct_farfield``'s ``exp(-1j w t_min)`` re-modulation) so the only
#: error is float64 round-off in the phase; the spec fixes the bar at 1e-13.
NULL_RECON_RTOL = 1e-13


def _make_reconstruct_probe():
    """Bind the REAL ``_born_reconstruct`` tail behind a Route-3 serve.

    Drives ``_born_residual_analytic`` on a NON-covering chart so the serve
    takes the certificate-gated carrier-only Route 3 -- which feeds an
    identically ZERO residual to ``_born_reconstruct`` -- while the
    ``w_trust`` / ``w_low`` split helpers are pinned so the reconstruction
    exercises BOTH the diffractive-bottom overwrite (``[w_lo, w_low)``) and
    the above-``w_trust`` envelope zeroing.  Route 3 passes
    ``engine_envelope=None`` so no exact-engine call is made -- the whole
    reconstruction is analytic (carrier + diffractive series + demodulation).
    The two post-reconstruction reducers (``_reduce_dense_kernels`` /
    ``_image_delays``) need heavy instance/lens state and run AFTER
    ``reconstruct_farfield``, so they are stubbed to no-ops; the captured
    total is untouched by them.
    """
    probe = types.SimpleNamespace()
    probe.born_residual_chart = _StubChart(
        covers_result=False, residual=np.zeros(FLOOR_DENSE.size))
    probe._ppgo_band_split = lambda lens: FLOOR_WTRUST
    probe._ppgo_cell_ceiling = lambda lens: None
    probe._diffractive_bottom_ceiling = (
        lambda lens, *, w_lo=None, w_hi=None: FLOOR_WLOW)
    probe._reduce_dense_kernels = lambda kernels: (None, None)
    probe._image_delays = lambda lens, geom: None
    probe._born_reconstruct = types.MethodType(
        LensedRelativeBinningLikelihood._born_reconstruct, probe)
    probe.serve = types.MethodType(
        LensedRelativeBinningLikelihood._born_residual_analytic, probe)
    return probe


class BornNullResidualReconstructionTestCase(_BornCertTestCase):
    """R=0 reconstruction reproduces the bare carrier to round-off.

    The ``_born_reconstruct`` tail is shared by the in-box serve (fed the
    interpolated residual) and the carrier-only serve (fed R=0).  With an
    identically zero residual ``f_total = carrier + 0`` must reconstruct,
    node-for-node, to the pieces HEAD produced: the analytic carrier on the
    trained/host interior, the diffractive series ``F_P`` on the overwritten
    bottom, and the bare ppGO image-kernel sum above the trusted floor.  The
    oracle is INDEPENDENT of the reconstruction: ``born_carrier_from_partition``
    on a freshly rebuilt partition, ``diffractive_amplification`` on the
    bottom nodes, and the closed-form image-kernel sum above -- none reads the
    captured total.  Engine-free (Route 3, ``engine_envelope=None``).
    """

    requires_comparison = True

    def setUp(self):
        super().setUp()
        self.dense_w = np.array(FLOOR_DENSE, copy=True)
        self.lens = _route_lens(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        # Rebuild the deterministic geometry the serve computes internally so
        # the oracle carrier / ppGO come from the identical partition.
        self.geom = ChangRefsdalChannels(self.dense_w).geometry_partition(
            gamma=FLOOR_GAMMA, y=(FLOOR_Y1, FLOOR_Y2), beta=0.0, kappa=0.0)
        # Premise: a genuine far-exterior draw the carrier certificate admits
        # (else Route 3 declines to None and the tail never runs).
        self.rho = _route_rho(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        self.assertGreater(self.rho, 2.0,
                           'fixture is not far-exterior (rho <= 2)')
        w_lo, w_hi = float(self.dense_w.min()), float(self.dense_w.max())
        self.assertTrue(
            likelihood._born_carrier_certificate_serves(
                self.lens, w_lo, w_hi, self.geom.images),
            'premise lost: the carrier certificate no longer admits the '
            'floor fixture, so Route 3 cannot be reached')
        # The three tiers, derived from the LIVE band-split (never literals).
        # This Route-3 serve uses R=0 (a non-covering chart), so there is NO
        # trained-floor engine sub-band; the tiers are bottom / host-interior
        # / above only.
        _band_split, below_mask = _band_split_mask(self.dense_w, FLOOR_WTRUST)
        band_split_low, below_low = _band_split_mask(self.dense_w, FLOOR_WLOW)
        self.assertTrue(band_split_low, 'bottom split inactive -- fixture '
                        'no longer straddles w_low')
        self.bottom_mask = below_low & below_mask
        self.host_mask = below_mask & ~self.bottom_mask
        self.above_mask = ~below_mask
        # All three tiers must be populated or the identity is vacuous.
        for name, mask in (('bottom', self.bottom_mask),
                           ('host-interior', self.host_mask),
                           ('above', self.above_mask)):
            self.assertTrue(mask.any(),
                            f'{name} tier empty -- reconstruction identity '
                            f'would assert nothing there')

    def _capture_total(self, probe):
        """Serve via Route 3, returning the discarded reconstruction total.

        Patches ONLY ``likelihood.reconstruct_farfield`` (the tail); the
        carrier build calls ``channels.reconstruct_farfield`` in a different
        namespace, so exactly one capture -- the tail -- is recorded.
        """
        captured = {}
        real_rf = likelihood.reconstruct_farfield

        def _capture_rf(*args, **kwargs):
            kernels, total = real_rf(*args, **kwargs)
            captured.setdefault('calls', 0)
            captured['calls'] += 1
            captured['total'] = np.array(total, copy=True)
            return kernels, total

        with mock.patch.object(
                likelihood, 'reconstruct_farfield', _capture_rf):
            result = probe.serve(self.lens, self.dense_w)
        self.assertIsNotNone(result, 'Route 3 declined -- the tail never ran')
        self.assertEqual(captured.get('calls'), 1,
                         'the reconstruction tail did not run exactly once')
        return captured['total']

    def test_r0_reconstruction_matches_piecewise_oracle(self):
        """Each tier reconstructs to its independent closed-form oracle."""
        probe = _make_reconstruct_probe()
        total = self._capture_total(probe)

        # Independent oracle pieces (none reads ``total``).
        partition_ns = types.SimpleNamespace(
            w=self.dense_w,
            source=np.array([FLOOR_Y1, FLOOR_Y2]),
            gamma=FLOOR_GAMMA, beta=0.0, kappa=0.0,
            matrix=likelihood.macro_matrix(FLOOR_GAMMA, 0.0, 0.0),
            t_min=self.geom.t_min, delays=self.geom.delays,
            saddle_kernels=self.geom.saddle_kernels,
            real_mask=self.geom.real_mask, images=self.geom.images)
        carrier = born_carrier_from_partition(partition_ns)
        real = np.asarray(self.geom.real_mask, dtype=bool)
        ppgo = np.sum(
            self.geom.saddle_kernels[:, real]
            * np.exp(1j * self.dense_w[:, None]
                     * self.geom.delays[real][None, :]),
            axis=1)
        diffractive = np.array([
            likelihood.diffractive_amplification(
                float(self.dense_w[idx]), (FLOOR_Y1, FLOOR_Y2),
                FLOOR_GAMMA, 0.0, 0.0)
            for idx in np.flatnonzero(self.bottom_mask)], dtype=complex)

        # Host-interior: R=0 -> the bare carrier to round-off (the core claim).
        host = self.host_mask
        rel_host = np.abs(total[host] - carrier[host]) / np.abs(carrier[host])
        self.n_compared += int(host.sum())
        self.assertLessEqual(
            float(rel_host.max()), NULL_RECON_RTOL,
            f'host-interior R=0 reconstruction deviates from the bare '
            f'carrier by {float(rel_host.max()):.2e} > {NULL_RECON_RTOL:.0e}')

        # Bottom: overwritten by the diffractive series F_P.
        rel_bottom = (np.abs(total[self.bottom_mask] - diffractive)
                      / np.abs(diffractive))
        self.n_compared += int(self.bottom_mask.sum())
        self.assertLessEqual(
            float(rel_bottom.max()), NULL_RECON_RTOL,
            'diffractive-bottom overwrite not reproduced to round-off')

        # Above w_trust: envelope zeroed -> the bare ppGO image-kernel sum.
        rel_above = (np.abs(total[self.above_mask] - ppgo[self.above_mask])
                     / np.abs(ppgo[self.above_mask]))
        self.n_compared += int(self.above_mask.sum())
        self.assertLessEqual(
            float(rel_above.max()), NULL_RECON_RTOL,
            'above-w_trust nodes did not telescope to the bare ppGO sum')

        # Diagnostic: |total - piecewise oracle| vs w on a log scale.
        oracle = carrier.copy()
        oracle[self.above_mask] = ppgo[self.above_mask]
        oracle[self.bottom_mask] = diffractive
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        resid = np.abs(total - oracle) / np.abs(oracle)
        ax.semilogy(self.dense_w, np.maximum(resid, 1e-18), 'o-',
                    color='tab:purple')
        ax.axhline(NULL_RECON_RTOL, color='k', ls='--', lw=0.8,
                   label=f'bar {NULL_RECON_RTOL:.0e}')
        ax.set_xlabel('dimensionless w')
        ax.set_ylabel('|total - oracle| / |oracle|')
        ax.set_title('R=0 reconstruction identity (piecewise oracle)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_plot(fig, 'born_certificate_null_residual_reconstruction.png')


# --------------------------------------------------------------------------- #
# ENGINE-FREE GUARANTEE ON ANALYTIC ROUTES (Spec: the census must never call
# the amplitude engine or freshly import mpmath while classifying a draw onto
# an analytic Born route; import-time module load is unavoidable, a CALL is a
# defect).
# --------------------------------------------------------------------------- #


class _EngineDoorTripwire(Exception):
    """Sentinel raised by every booby-trapped engine/mpmath entry point.

    Deliberately NOT a subclass of any refusal error the census catches, so a
    stray engine call surfaces as a hard failure rather than being swallowed
    as a refusal (which would silently reroute the draw and hide the defect).
    """


class BornCensusEngineFreeTestCase(_BornCertTestCase):
    """The census classifies Born analytic routes without touching the engine.

    ``serve_route_census.classify_draw`` is a pure CLASSIFIER: it consults the
    cheap geometry partition, the chart's ``covers`` box, the band-split masks
    and the analytic carrier certificate, and returns a route label -- it must
    never evaluate a waveform or run the mpmath quadrature.  Every engine door
    (``ChangRefsdalChannels.evaluate``, ``_schwinger.f_schwinger`` and its
    mpmath fallback) is booby-trapped to raise a sentinel that the census
    CANNOT catch, and ``mpmath`` must not freshly enter ``sys.modules`` while a
    born route is classified.  The three charts drive Routes 1/2/3 (two
    ``born_analytic`` + one ``born_carrier_only``).
    """

    def setUp(self):
        super().setUp()
        # The census catch tuples must not swallow the tripwire (a widened
        # catch that caught it would let an engine call masquerade as a
        # refusal).  Assert disjointness against the LIVE tuples.
        base = _census_base_mods()
        for tup_name in ('refusal_errors', 'diffractive_refusal_errors'):
            for caught in getattr(base, tup_name):
                self.assertFalse(
                    issubclass(_EngineDoorTripwire, caught),
                    f'tripwire is catchable by {tup_name} ({caught!r}) -- an '
                    f'engine call would be swallowed as a refusal')

    def _run_engine_trapped(self, charts):
        """Classify each chart with every engine door booby-trapped.

        Returns ``(routes, doors, mpmath_freshly_imported)``.
        """
        door_evaluate = mock.Mock(side_effect=_EngineDoorTripwire('evaluate'))
        door_fschw = mock.Mock(side_effect=_EngineDoorTripwire('f_schwinger'))
        door_mpmath = mock.Mock(
            side_effect=_EngineDoorTripwire('_f_schwinger_mpmath'))
        mpmath_pre = 'mpmath' in sys.modules
        routes = []
        with mock.patch.object(ChangRefsdalChannels, 'evaluate',
                               door_evaluate), \
                mock.patch.object(_schwinger, 'f_schwinger', door_fschw), \
                mock.patch.object(_schwinger, '_f_schwinger_mpmath',
                                  door_mpmath):
            for chart in charts:
                routes.append(_classify_floor(chart).route)
        mpmath_fresh = ('mpmath' in sys.modules) and not mpmath_pre
        doors = {'evaluate': door_evaluate, 'f_schwinger': door_fschw,
                 '_f_schwinger_mpmath': door_mpmath}
        return routes, doors, mpmath_fresh

    def test_born_routes_never_call_the_engine_or_import_mpmath(self):
        """No engine door fires and mpmath is not freshly imported."""
        charts = [
            _WBandChart(0.05, FLOOR_CEIL, FLOOR_SENTINEL),      # Route 1
            _WBandChart(FLOOR_TRAINED, FLOOR_CEIL, FLOOR_SENTINEL),  # Route 2
            _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL),  # R3
        ]
        routes, doors, mpmath_fresh = self._run_engine_trapped(charts)

        # The three charts must land on the analytic Born routes (else the
        # engine-free guarantee is vacuous -- a refused draw touches nothing).
        self.assertEqual(routes[0], 'born_analytic', 'Route 1 misrouted')
        self.assertEqual(routes[1], 'born_analytic', 'Route 2 misrouted')
        self.assertEqual(routes[2], 'born_carrier_only', 'Route 3 misrouted')

        # No engine door fired.
        for name, door in doors.items():
            self.assertEqual(
                door.call_count, 0,
                f'the census called the engine door {name!r} '
                f'{door.call_count}x while classifying an analytic route')

        # mpmath did not freshly enter sys.modules during classification.
        self.assertFalse(
            mpmath_fresh,
            'mpmath was freshly imported while classifying a born route')

        # Diagnostic: the per-door call-count bar (all zero on a clean run).
        names = list(doors)
        counts = [doors[n].call_count for n in names]
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.bar(names, counts, color='tab:red')
        ax.set_ylabel('census calls (must be 0)')
        ax.set_ylim(0, 1)
        ax.set_title('Engine-free guarantee: engine-door call counts')
        ax.tick_params(axis='x', labelrotation=15, labelsize=8)
        fig.tight_layout()
        _save_plot(fig, 'born_certificate_census_engine_free.png')

# --------------------------------------------------------------------------- #
# CENSUS-MIRROR FAITHFULNESS (Spec: the census mirror must track the
# production Born route EXACTLY -- and, crucially, by DELEGATING to the
# production accessor objects (``born_chart.covers``, ``log_w_grid``,
# ``_band_split_mask``) rather than re-typing the decision.  The route
# OUTCOMES themselves are already pinned by ``BornTrainedFloorCensusRevival
# TestCase``; the NOVEL teeth here are (a) the delegation assertions and
# (b) a covered/uncovered x sub-band/escape route-agreement MATRIX between
# the census ``classify_draw`` and the production ``_born_residual_analytic``
# itself.)
# --------------------------------------------------------------------------- #


def _production_born_label(chart, lens):
    """The Born route label implied by the production ``_born_residual_analytic``.

    Drives the REAL serve (bound onto ``_make_floor_probe``) with the
    carrier certificate patched to ADMIT, and reads back which of the three
    routes fired purely from observable production side effects -- never
    from the census:

    * ``result is None``            -> ``'declined'`` (Route 3 refusal);
    * served, certificate untouched -> ``'born_analytic'`` (Route 1 or 2);
    * served, certificate consulted -> ``'born_carrier_only'`` (Route 3).

    The vocabulary matches the census ``ClassifiedDraw.route`` exactly so the
    two independent implementations can be compared cell-for-cell.  A second
    return value carries the certificate call-count for the diff table.
    """
    probe = _make_floor_probe(chart, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
                              engine_value=FLOOR_ENGINE_VALUE)
    with mock.patch.object(
            likelihood, '_born_carrier_certificate_serves',
            mock.Mock(return_value=True)) as cert:
        result = probe.serve(lens, FLOOR_DENSE)
    if result is None:
        return 'declined', cert.call_count
    if cert.call_count == 0:
        return 'born_analytic', cert.call_count
    return 'born_carrier_only', cert.call_count


class BornCensusMirrorFaithfulnessTestCase(_BornCertTestCase):
    """CENSUS-MIRROR FAITHFULNESS: the mirror DELEGATES and tracks production.

    Two invariants the recurring laggard (a mirror that drifts because it
    re-typed the gate) fails:

    * DELEGATION -- ``_born_trained_floor_route`` reads the trained floor
      from the artifact ``log_w_grid[0]``, hands it to the PRODUCTION
      ``_band_split_mask`` object, and probes the PRODUCTION
      ``born_chart.covers`` with the chart sub-band.  Verified by wrapping
      those two callables in ``mock.Mock(wraps=...)`` spies and asserting
      the exact arguments -- a re-typed comparison would never touch them.
    * ROUTE AGREEMENT -- across a covered/uncovered x sub-band/escape grid,
      the census ``classify_draw`` route equals the route the production
      ``_born_residual_analytic`` itself takes (``_production_born_label``),
      row for row, with a genuine mix of verdicts present so the match is
      not vacuous.

    Engine-free: the production serve rides ``_make_floor_probe`` (spy
    reconstruct, no engine call) and the census rides ``_classify_floor``.
    """

    def setUp(self):
        super().setUp()
        self.lens = _route_lens(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        self.rho = _route_rho(FLOOR_GAMMA, FLOOR_Y1, FLOOR_Y2)
        # Premise: far-exterior Born box (rho > the live floor), else the
        # whole rung is skipped and no route claim is meaningful.
        self.assertGreater(self.rho, serve_route_census._BORN_RHO_FLOOR,
                           'fixture is not far-exterior; Born rung skipped')
        self.assertGreater(self.rho, 2.0, 'fixture below the production rho>2')
        self.low_edge = _WBandChart(FLOOR_TRAINED, FLOOR_CEIL, FLOOR_SENTINEL)
        # The host sub-band and its trained-floor split (shipped arithmetic).
        _bottom, self.engine_mask, self.chart_mask, _above = _floor_tier_masks(
            FLOOR_DENSE, w_trust=FLOOR_WTRUST, w_low=FLOOR_WLOW,
            trained_floor=FLOOR_TRAINED)
        self.host_mask = self.engine_mask | self.chart_mask
        self.assertTrue(self.engine_mask.any() and self.chart_mask.any(),
                        'the low-edge fixture does not split both tiers')

    def test_mirror_delegates_to_production_covers_and_band_split(self):
        """The mirror calls the PRODUCTION ``covers`` / ``_band_split_mask``.

        Wrap ``born_chart.covers`` and the module ``_band_split_mask`` in
        ``wraps`` spies and drive ``_born_trained_floor_route`` directly.
        The predicate must (i) split the band at ``trained_floor =
        exp(log_w_grid[0])`` -- read from the artifact -- via the production
        ``_band_split_mask``, and (ii) probe ``covers`` with exactly the
        chart sub-band ``FLOOR_DENSE[chart_mask]``.  A mirror that re-typed
        either comparison would leave a spy uncalled: that is the teeth.
        """
        chart = self.low_edge
        original_covers = chart.covers
        covers_spy = mock.Mock(wraps=original_covers)
        chart.covers = covers_spy
        # Wrap the PRODUCTION band-split object the mirror actually consults
        # (``mods.band_split_mask`` == ``likelihood._band_split_mask``), so a
        # re-typed comparison would leave this spy uncalled.
        base = _census_mods(chart)
        bsm_spy = mock.Mock(wraps=base.band_split_mask)
        mods = dataclasses.replace(base, band_split_mask=bsm_spy)

        served = serve_route_census._born_trained_floor_route(
            mods, FLOOR_GAMMA, self.rho, self.host_mask, FLOOR_DENSE)
        self.assertTrue(served, 'the low-edge strict sub-band did not serve')
        self.n_compared += 1

        # (i) The band split ran on the PRODUCTION helper, once, at the
        # artifact-derived trained floor -- never a literal.
        trained_floor = math.exp(float(chart.log_w_grid[0]))
        self.assertEqual(bsm_spy.call_count, 1,
                         'the mirror did not delegate to the production '
                         '_band_split_mask (re-typed the split?)')
        bsm_args, bsm_kwargs = bsm_spy.call_args
        self.assertEqual(bsm_kwargs, {})
        np.testing.assert_array_equal(bsm_args[0], FLOOR_DENSE)
        self.assertEqual(bsm_args[1], trained_floor,
                         'the split was not taken at exp(log_w_grid[0])')

        # (ii) The coverage probe ran on the PRODUCTION chart object with the
        # chart sub-band FLOOR_DENSE[chart_mask] and this draw's (gamma, rho).
        self.assertEqual(covers_spy.call_count, 1,
                         'the mirror did not delegate to born_chart.covers')
        cov_args, cov_kwargs = covers_spy.call_args
        self.assertEqual(cov_kwargs, {})
        self.assertEqual(cov_args[0], FLOOR_GAMMA)
        self.assertEqual(cov_args[1], self.rho)
        np.testing.assert_array_equal(cov_args[2], FLOOR_DENSE[self.chart_mask])
    def test_census_route_matches_production_across_the_grid(self):
        """Census route == production ``_born_residual_analytic`` route, row by row.

        A covered/uncovered x sub-band/escape grid of charts:

        * in-box (trained floor below the whole band)   -> born_analytic;
        * low-edge escape (strict inner sub-band)        -> born_analytic;
        * disjoint-high escape (trained range above)     -> born_carrier_only.

        For each the production label (``_production_born_label``, cert
        ADMIT) must equal the census route (``_classify_floor`` with a cert
        that ADMITS), and a genuine mix of ``born_analytic`` /
        ``born_carrier_only`` must appear so the agreement is not vacuous.
        """
        floor_min = float(FLOOR_DENSE.min())
        charts = {
            'in_box': _WBandChart(floor_min, FLOOR_CEIL, FLOOR_SENTINEL),
            'low_edge': _WBandChart(FLOOR_TRAINED, FLOOR_CEIL, FLOOR_SENTINEL),
            'disjoint_high': _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL,
                                         FLOOR_SENTINEL),
        }
        cert = mock.Mock(return_value=True)
        table = []
        for name, chart in charts.items():
            prod_label, cert_calls = _production_born_label(chart, self.lens)
            census_route = _classify_floor(
                chart, born_carrier_serves=cert).route
            table.append({'draw': name, 'production': prod_label,
                          'census': census_route, 'cert_calls': cert_calls})
            with self.subTest(draw=name):
                self.assertEqual(
                    census_route, prod_label,
                    f'census route {census_route!r} disagrees with the '
                    f'production _born_residual_analytic route {prod_label!r} '
                    f'for the {name} draw (mirror drifted from production)')
            self.n_compared += 1
        labels = {row['production'] for row in table}
        self.assertIn('born_analytic', labels,
                      'no draw exercised the analytic route')
        self.assertIn('born_carrier_only', labels,
                      'no draw exercised the carrier-only route')

        # Diagnostic: the per-draw route-diff table -- any disagreeing row is
        # a mirror that reimplemented rather than delegated.
        fig, ax = plt.subplots(figsize=(6.4, 2.2))
        ax.axis('off')
        cells = [[r['draw'], r['production'], r['census'],
                  'OK' if r['production'] == r['census'] else 'MISMATCH']
                 for r in table]
        tbl = ax.table(
            cellText=cells,
            colLabels=['draw', 'production route', 'census route', 'agree'],
            loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        ax.set_title('CENSUS-MIRROR FAITHFULNESS: route-agreement matrix')
        fig.tight_layout()
        _save_plot(fig, 'born_certificate_census_route_agreement.png')

    def test_declining_certificate_agrees_on_fallthrough(self):
        """A disjoint escape + DECLINING cert: production declines, census too.

        The one row where production returns ``None`` (Route 3 refusal): the
        census must NOT label it a Born route either -- it falls through to
        the engine node pass.  Pins that the mirror tracks production on the
        REFUSAL edge as well as the serve edge (a mirror that admitted here
        would over-count born routes).
        """
        disjoint = _WBandChart(DISJOINT_FLOOR, DISJOINT_CEIL, FLOOR_SENTINEL)
        probe = _make_floor_probe(disjoint, w_trust=FLOOR_WTRUST,
                                  w_low=FLOOR_WLOW)
        with mock.patch.object(
                likelihood, '_born_carrier_certificate_serves',
                mock.Mock(return_value=False)) as cert:
            result = probe.serve(self.lens, FLOOR_DENSE)
        self.assertIsNone(result, 'production did not decline the disjoint '
                          'escape under a refusing certificate')
        self.assertEqual(cert.call_count, 1)
        census_route = _classify_floor(
            disjoint,
            born_carrier_serves=mock.Mock(return_value=False)).route
        self.assertNotIn(
            census_route, ('born_analytic', 'born_carrier_only'),
            'census kept a Born route where production declined')
        self.n_compared += 1



if __name__ == '__main__':
    unittest.main()
