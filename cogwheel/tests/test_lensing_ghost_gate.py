"""Certification suite for the RE-KEYED far-field ghost gate.

This build replaced the ghost subtraction's admit/refuse criterion in
``cogwheel.lensing.chang_refsdal.channels.farfield_ghost_term`` from a
frequency-DEPENDENT decay test ``w_min * Im tau_c >= _FARFIELD_WINDOW_RADIANS``
(=2.0) to a frequency-INDEPENDENT GEOMETRIC separation test

    min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN    (=0.7),

a bare complex Euclidean distance between every real image ``x_a``
(``geometry.find_images``) and the complex ghost position ``x_c``
(``geometry.ghost_kernel(...).position``, imaginary part KEPT).  The single-
saddle stationary-phase expansion the ghost kernel relies on is valid only
where the ghost is resolved from every real image; near a cusp the ghost
coalesces with a real image (separation -> 0) and the gate must refuse.

What this suite certifies, and the ORACLE for each claim:

* GATE BOUNDARY (`GhostGateBoundaryTestCase`).  The refuse/admit BEHAVIOUR
  (``geometry.GhostDomainError`` raised vs a finite return) is the primary
  oracle.  The independently-recomputed separation is a *tripwire*, not the
  gate: refuse configs must satisfy ``sep < SEP_REFUSE_MAX`` (=0.5) and admit
  configs ``sep > SEP_ADMIT_MIN`` (=1.0).  Both bounds are 1.5x-margined away
  from the production threshold 0.7 (the refuse cluster sits ~0.20-0.29, the
  admit cluster ~1.82-2.88), so a wrong currency/norm that merely rescaled the
  distance would be caught.  Only 4 of the brief's 7 refuse configs are
  reproducible in-session (the other 3 need the driver's 2026-07-27 table),
  so we test the 4 and assert the general property.

* TRAIN/SERVE DECISION AGREEMENT (`TrainServeDecisionAgreementTestCase`).  The
  gate reads only ``(source, matrix)``; therefore a small-``min(w)`` "training"
  grid and a larger-``min(w)`` "serve" sub-band MUST reach the identical
  admit/refuse boolean.  Asserted on the BOOLEAN, never proxied through a
  magnitude collapse.  The old decay gate's ``w``-dependence (which this
  property retired) is exhibited red in the self-falsification class.

* DO-NOTHING CONTROL (`DoNothingControlTestCase`).  Where the gate ADMITS,
  additionally subtracting the ghost must never make the far-field label
  worse.  The F-normalized residual of a label is ``max|E_ff| / max|F|`` --
  ``E_ff`` is exactly ``F`` minus the analytic terms it removed, so its
  magnitude IS the un-modelled remainder relative to the exact operator
  total ``partition.exact_total`` (the independent engine oracle, which shares
  no code with the analytic labels).  We require
  ``resid(MINUS_GHOST) <= resid(KERNEL_SUM) + DO_NOTHING_SLACK`` (=1e-12,
  additive; both residuals can be tiny, so a ratio is inappropriate).

* REACHABLE-RED for the constant (`GhostSeparationConstantReachableRedTestCase`).
  Both ``_GHOST_SEPARATION_MIN`` and ``_GHOST_DECAY_IM_THRESHOLD`` are
  monkeypatched to 0.0 (a refuse config wrongly admits) and ``_GHOST_SEPARATION_MIN``
  is patched to 2.0 (an admit config wrongly refuses), proving the thresholds are
  load-bearing.  Near-cusp configs simultaneously fail BOTH gates (low Im(tau_c)
  ~0.001 and low separation), so both must be patched to demonstrate reachability.

* BETA GUARD (`BetaGuardTestCase`).  ``_surrogate_coefficients`` must fall
  through to the exact engine (return ``None``) for ``beta != 0`` exactly as
  for ``kappa != 0`` -- the surrogate is a ``beta = kappa = 0`` surface.  The
  guard is isolated by spying the very next call after it
  (``likelihood.dimensionless_frequency``): un-called => the guard returned;
  called => execution passed the guard.

* FRAME SINGLE-SOURCE REGRESSION (`FrameSingleSourceRegressionTestCase`).
  ``channels.real_image_delays`` now routes its frame origin through
  ``_frame_delays`` instead of an inline ``.min()``.  The returned delays must
  be BYTE-IDENTICAL (max abs diff exactly 0.0) to an independently computed
  ``np.sort(absolute_delays - min(absolute_delays))``, and the smallest
  element must be exactly 0.0 -- the min-subtraction convention is unchanged.

* SELF-FALSIFICATION (`GhostGateSelfFalsificationTestCase`).  An independent
  reimplementation of the retired decay gate DISAGREES between the train and
  serve grids on a near-threshold skew config (proving the boolean-equality
  test has teeth), and a refused near-cusp config forced through the
  MINUS_GHOST label (by patching both ``_GHOST_SEPARATION_MIN`` and
  ``_GHOST_DECAY_IM_THRESHOLD`` to 0) makes the DO-NOTHING residual WORSE
  (proving the gates protect that control).

Oracle independence: the separation tripwire and the old-gate foil are built
from the ``geometry`` primitives directly (no reuse of the gate's decision
branch); the DO-NOTHING oracle is the engine's ``exact_total``; the frame
regression oracle is an inline ``.min()`` subtraction with no call into
``_frame_delays``.
"""
from __future__ import annotations

import itertools
import pathlib
import unittest
from unittest import mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing import likelihood as lensing_likelihood
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.chang_refsdal import channels, geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, farfield_envelope_from_partition, farfield_ghost_term,
    real_image_delays, FARFIELD_KERNEL_SUM, FARFIELD_KERNEL_SUM_MINUS_GHOST)


#: Production geometric threshold under test (``min_a |x_a - x_c| >= this``).
GHOST_SEPARATION_MIN = channels._GHOST_SEPARATION_MIN

#: Retired decay-gate threshold (``w_min * Im tau_c >= this``); used only as
#: the self-falsification foil, never as a production reference.
FARFIELD_WINDOW_RADIANS = channels._FARFIELD_WINDOW_RADIANS

#: Tripwire bounds on the independently-recomputed separation.  1.5x-margined
#: around the 0.7 threshold: refuse configs cluster ~0.20-0.29 (< 0.5), admit
#: configs cluster ~1.82-2.88 (> 1.0).  Any config straddling 0.7 flags a
#: wrong-currency threshold rather than silently passing.
SEP_REFUSE_MAX = 0.5
SEP_ADMIT_MIN = 1.0
SEP_DIAGNOSTIC_THRESHOLD = 0.7

#: Additive slack on the F-normalized DO-NOTHING residual comparison.  Both
#: residuals can be ~1e-4, so an additive floor (not a ratio) is used.
DO_NOTHING_SLACK = 1.0e-12

#: (gamma, theta_c [deg], offset) configs, positive parity, beta=kappa=0,
#: source at |y| = r_caustic(gamma, theta_c) + offset.  REFUSE configs sit
#: just outside the astroid caustic near a cusp (offset bisected on |y| so the
#: separation lands well below 0.5; Im(tau_c) also small ~0.001 near axis).
#: ADMIT configs are far-from-cusp exterior points (separation ~1.4-2.9,
#: Im(tau_c) in 0.40-0.87 > _GHOST_DECAY_IM_THRESHOLD).
#: Offsets for admit configs verified to satisfy both gates (2026-08-01).
REFUSE_CONFIGS = (
    (0.30, 0.3, 0.04),
    (0.50, 0.3, 0.04),
    (0.70, 0.3, 0.06),
    (0.90, 0.3, 0.06),
)
ADMIT_CONFIGS = (
    (0.50, 45.0, 0.65),   # Im(tau_c)~0.43, sep~1.98 (offset raised for margin above 0.4 threshold)
    (0.90, 45.0, 1.00),   # Im(tau_c)=0.690, sep=1.819
    (0.30, 45.0, 0.80),   # Im(tau_c)=0.873, sep=2.877
    (0.30, 20.0, 1.50),   # Im(tau_c)=1.714, sep=4.070
)
#: One admitted + one refused config for the train/serve decision test.
TRAIN_SERVE_ADMIT_CONFIG = (0.50, 45.0, 0.65)
TRAIN_SERVE_REFUSE_CONFIG = (0.90, 0.3, 0.06)

#: Near-threshold skew config for the retired-decay-gate foil: sep=1.807
#: (the NEW gate ADMITS on both grids), Im(tau_c)=0.5375 in [0.5, 1.0), so the
#: old ``w_min * Im tau_c >= 2.0`` gate flips between min(w)=2 (refuse) and
#: min(w)=4 (admit).
NEAR_THRESHOLD_CONFIG = (0.50, 22.5, 0.85)

#: A "training" grid with a small ``min(w)`` near the diffractive floor and a
#: "serve" sub-band with a strictly larger ``min(w)``; both strictly inside
#: (2, 60).  The old-gate flip window [2/min(w_serve), 2/min(w_train)) = [0.5,
#: 1.0) is non-empty, which is what makes the decay foil bite.
TRAIN_W = np.geomspace(2.0, 50.0, 30)
SERVE_W = np.geomspace(4.0, 50.0, 20)

#: Mid-band evaluation grid for the DO-NOTHING label residuals (below w=60).
MID_BAND_W = np.geomspace(5.0 * 1.003, 60.0 * 0.997, 40)

#: A generic evaluation grid for the boundary gate calls.
GATE_W = np.array([15.0, 25.0, 35.0])

#: Configs for the frame single-source regression (varied gamma, theta_c).
FRAME_CONFIGS = (
    (0.50, 45.0, 0.60),
    (0.70, 30.0, 0.40),
    (0.30, 60.0, 0.50),
    (0.90, 15.0, 0.35),
)

#: Directory for diagnostic plots.
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'


def _source_and_matrix(gamma: float, theta_deg: float, offset: float):
    """Positive-parity source just outside the caustic and its macro matrix.

    Places the source on the ray at ``theta_deg`` with modulus
    ``r_caustic(gamma, theta) + offset`` (beta = kappa = 0), the geometry the
    brief specifies for every gate probe.
    """
    theta = np.deg2rad(theta_deg)
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    radius = geometry.r_caustic(gamma, theta) + offset
    source = radius * np.array([np.cos(theta), np.sin(theta)])
    return source, matrix


def _ghost_separation(source: np.ndarray, matrix: np.ndarray) -> float:
    """``min_a |x_a - x_c|`` recomputed from the geometry primitives.

    Independent of ``farfield_ghost_term``'s decision branch: the ghost
    position ``x_c`` comes from ``geometry.ghost_kernel`` and the real images
    ``x_a`` from ``geometry.find_images``; the complex norm keeps ``Im(x_c)``.
    """
    contribution = geometry.ghost_kernel(GATE_W, source, matrix)
    x_c = contribution.position
    real_images = geometry.find_images(source, matrix)
    return min(
        float(np.sqrt(np.sum(np.abs(x_a - x_c) ** 2))) for x_a in real_images)


def _ghost_admits(w: np.ndarray, source: np.ndarray,
                  matrix: np.ndarray) -> bool:
    """True iff ``farfield_ghost_term`` returns (admits) on this grid."""
    try:
        farfield_ghost_term(w, source, matrix)
    except geometry.GhostDomainError:
        return False
    return True


def _build_partition(gamma: float, theta_deg: float, offset: float,
                     w: np.ndarray):
    """A fully-evaluated exterior partition (its ``exact_total`` is required)."""
    source, _matrix = _source_and_matrix(gamma, theta_deg, offset)
    engine = ChangRefsdalChannels(w)
    engine.reset()
    return engine.evaluate(gamma=gamma, y=(float(source[0]), float(source[1])),
                           beta=0.0, kappa=0.0)


def _f_normalized_residuals(partition) -> tuple[float, float]:
    """``(resid_KERNEL_SUM, resid_MINUS_GHOST)`` F-normalized by ``max|F|``."""
    denom = float(np.max(np.abs(partition.exact_total))) or 1.0
    e_ks = farfield_envelope_from_partition(partition, FARFIELD_KERNEL_SUM)
    e_mg = farfield_envelope_from_partition(
        partition, FARFIELD_KERNEL_SUM_MINUS_GHOST)
    return (float(np.max(np.abs(e_ks))) / denom,
            float(np.max(np.abs(e_mg))) / denom)


class GhostGateTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity guard shared by every gate suite.

    ``comparisons`` counts the substantive assertions actually executed; the
    ``tearDown`` FAILS if a test body ran zero of them, so a silently-skipping
    or short-circuited suite cannot read green.
    """

    def setUp(self):
        self.comparisons = 0

    def tearDown(self):
        self.assertGreater(
            self.comparisons, 0,
            'anti-vacuity: no ghost-gate comparison executed in this test -- '
            'the suite would have passed vacuously')


class GhostGateBoundaryTestCase(GhostGateTestCase):
    """Refuse near-cusp configs; admit far-from-cusp configs (fact-2/fact-1).

    The gate BEHAVIOUR is the oracle: a refuse config must raise
    ``geometry.GhostDomainError`` and independently read ``sep < 0.5``; an
    admit config must return and read ``sep > 1.0``.  The separation tripwire
    is 1.5x-margined from the 0.7 threshold on both sides.
    """

    @classmethod
    def setUpClass(cls):
        cls._records = []  # (separation, admitted) for the diagnostic scatter.

    def test_near_cusp_configs_refuse_and_read_below_half(self):
        """Every fact-2 config raises GhostDomainError with sep < 0.5."""
        for gamma, theta_deg, offset in REFUSE_CONFIGS:
            with self.subTest(gamma=gamma, theta_c=theta_deg, offset=offset):
                source, matrix = _source_and_matrix(gamma, theta_deg, offset)
                separation = _ghost_separation(source, matrix)
                with self.assertRaises(geometry.GhostDomainError):
                    farfield_ghost_term(GATE_W, source, matrix)
                self.assertLess(
                    separation, SEP_REFUSE_MAX,
                    f'refuse config sep={separation:.4f} is not a genuine '
                    f'near-cusp point (>= {SEP_REFUSE_MAX}); the tripwire '
                    'suggests a wrong separation currency')
                self._records.append((separation, False))
                self.comparisons += 1

    def test_far_from_cusp_configs_admit_and_read_above_one(self):
        """Every fact-1 config returns finitely with sep > 1.0."""
        for gamma, theta_deg, offset in ADMIT_CONFIGS:
            with self.subTest(gamma=gamma, theta_c=theta_deg, offset=offset):
                source, matrix = _source_and_matrix(gamma, theta_deg, offset)
                separation = _ghost_separation(source, matrix)
                ghost = farfield_ghost_term(GATE_W, source, matrix)
                self.assertEqual(np.asarray(ghost).shape, GATE_W.shape)
                self.assertTrue(np.all(np.isfinite(ghost)))
                self.assertGreater(
                    separation, SEP_ADMIT_MIN,
                    f'admit config sep={separation:.4f} is not far-from-cusp '
                    f'(<= {SEP_ADMIT_MIN}); the tripwire suggests a wrong '
                    'separation currency')
                self._records.append((separation, True))
                self.comparisons += 1

    def test_refuse_and_admit_clusters_straddle_the_threshold(self):
        """Refuse cluster sits below and admit cluster above 0.7.

        A single monotone check that no config lands on the wrong side of the
        production threshold -- if any refuse config swept to sep > 0.7 or any
        admit config to sep < 0.7, the constant 0.7 would need revisiting.
        """
        refuse_seps = [
            _ghost_separation(*_source_and_matrix(g, th, off))
            for g, th, off in REFUSE_CONFIGS]
        admit_seps = [
            _ghost_separation(*_source_and_matrix(g, th, off))
            for g, th, off in ADMIT_CONFIGS]
        self.assertLess(
            max(refuse_seps), SEP_DIAGNOSTIC_THRESHOLD,
            f'a refuse config swept above the 0.7 threshold '
            f'(max={max(refuse_seps):.4f}); revisit the gate constant')
        self.assertGreater(
            min(admit_seps), SEP_DIAGNOSTIC_THRESHOLD,
            f'an admit config swept below the 0.7 threshold '
            f'(min={min(admit_seps):.4f}); revisit the gate constant')
        self.comparisons += 1
        self._plot_separation_scatter(refuse_seps, admit_seps)

    @classmethod
    def _plot_separation_scatter(cls, refuse_seps, admit_seps):
        """Scatter separation vs admit/refuse with the 0.7 threshold line."""
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(refuse_seps, np.zeros(len(refuse_seps)), c='r',
                   label='refused (fact-2)', zorder=3)
        ax.scatter(admit_seps, np.ones(len(admit_seps)), c='b',
                   label='admitted (fact-1)', zorder=3)
        ax.axvline(SEP_DIAGNOSTIC_THRESHOLD, color='k', ls='--',
                   label=f'_GHOST_SEPARATION_MIN={SEP_DIAGNOSTIC_THRESHOLD}')
        ax.set_xlabel(r'separation  $\min_a |x_a - x_c|$')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['refuse', 'admit'])
        ax.set_title('ghost gate boundary: separation vs decision')
        ax.legend(loc='center right')
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'ghost_gate_boundary_separation.png', dpi=110)
        plt.close(fig)


class TrainServeDecisionAgreementTestCase(GhostGateTestCase):
    """The re-keyed gate decides identically for a train vs serve w grid.

    Because ``min_a |x_a - x_c|`` contains no ``w``, the admit/refuse boolean
    is provably identical for the small-``min(w)`` training grid and the
    larger-``min(w)`` serve sub-band.  Asserted on the boolean itself for BOTH
    an admitted and a refused config (fact 4), never via a magnitude collapse.
    """

    def test_admit_and_refuse_configs_decide_identically_across_w(self):
        """train-grid decision == serve-grid decision for both configs."""
        cases = ((TRAIN_SERVE_ADMIT_CONFIG, True),
                 (TRAIN_SERVE_REFUSE_CONFIG, False))
        for (gamma, theta_deg, offset), expected_admit in cases:
            with self.subTest(gamma=gamma, theta_c=theta_deg, offset=offset):
                source, matrix = _source_and_matrix(gamma, theta_deg, offset)
                # Sanity: the grids genuinely differ in their smallest node.
                self.assertLess(float(TRAIN_W.min()), float(SERVE_W.min()))
                train_admit = _ghost_admits(TRAIN_W, source, matrix)
                serve_admit = _ghost_admits(SERVE_W, source, matrix)
                self.assertEqual(
                    train_admit, serve_admit,
                    f'train/serve gate skew: train admit={train_admit}, '
                    f'serve admit={serve_admit} -- the re-keyed gate must be '
                    'frequency-independent')
                self.assertEqual(
                    train_admit, expected_admit,
                    f'config decided admit={train_admit}, expected '
                    f'{expected_admit}')
                self.comparisons += 1


class DoNothingControlTestCase(GhostGateTestCase):
    """Where the gate admits, subtracting the ghost never worsens the label.

    On every admitted config the F-normalized residual of
    ``FARFIELD_KERNEL_SUM_MINUS_GHOST`` must not exceed that of
    ``FARFIELD_KERNEL_SUM`` (the subtract-nothing-extra control) beyond an
    additive ``DO_NOTHING_SLACK``.  The oracle is the engine's ``exact_total``.
    """

    @classmethod
    def setUpClass(cls):
        cls._ratios = []  # resid_mg / resid_ks per admitted config.

    def test_minus_ghost_residual_never_exceeds_kernel_sum(self):
        """resid(MINUS_GHOST) <= resid(KERNEL_SUM) + 1e-12 on admit configs."""
        for gamma, theta_deg, offset in ADMIT_CONFIGS:
            with self.subTest(gamma=gamma, theta_c=theta_deg, offset=offset):
                partition = _build_partition(gamma, theta_deg, offset,
                                             MID_BAND_W)
                resid_ks, resid_mg = _f_normalized_residuals(partition)
                self.assertLessEqual(
                    resid_mg, resid_ks + DO_NOTHING_SLACK,
                    f'subtracting the ghost worsened the admitted label: '
                    f'resid(MINUS_GHOST)={resid_mg:.3e} > '
                    f'resid(KERNEL_SUM)={resid_ks:.3e}')
                self._ratios.append((f'{gamma}/{theta_deg:g}/{offset:g}',
                                     resid_mg / resid_ks))
                self.comparisons += 1
        self._plot_ratio_bars()

    @classmethod
    def _plot_ratio_bars(cls):
        """Bar plot of resid(MINUS_GHOST)/resid(KERNEL_SUM); any bar>1 fails."""
        if not cls._ratios:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        labels = [name for name, _ in cls._ratios]
        values = [ratio for _, ratio in cls._ratios]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(values)), values, color='b')
        ax.axhline(1.0, color='r', ls='--', label='violation line (=1)')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('resid(MINUS_GHOST) / resid(KERNEL_SUM)')
        ax.set_title('DO-NOTHING control on admitted configs')
        ax.legend()
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'ghost_gate_donothing_ratios.png', dpi=110)
        plt.close(fig)


class GhostSeparationConstantReachableRedTestCase(GhostGateTestCase):
    """The 0.7 threshold is load-bearing: patching it flips decisions.

    ``_GHOST_SEPARATION_MIN`` is read at call time inside
    ``farfield_ghost_term``, so ``mock.patch.object`` on the module global
    reaches the live gate.  Set to 0.0 a refuse config wrongly ADMITS; set to
    2.0 an admit config wrongly REFUSES -- proving the boundary assertions
    bite on both sides.
    """

    def test_lowering_constant_to_zero_admits_a_refuse_config(self):
        """With MIN=0.0 and decay threshold=0.0 a fact-2 config admits.

        Near-cusp configs simultaneously have low separation (sep < 0.5) and
        low Im(tau_c) (~0.001) — the decay gate fires first.  To isolate the
        separation gate as load-bearing, both constants are patched to zero:
        once the decay gate is also disabled the config must admit because its
        separation alone was the only remaining block (at MIN=0.7 after the
        decay gate is cleared it would still be refused by separation, but
        here we zero both to confirm neither alone is vacuous).
        """
        gamma, theta_deg, offset = REFUSE_CONFIGS[3]
        source, matrix = _source_and_matrix(gamma, theta_deg, offset)
        self.assertFalse(_ghost_admits(GATE_W, source, matrix),
                         'precondition: this config must refuse normally')
        with mock.patch.object(channels, '_GHOST_SEPARATION_MIN', 0.0), \
             mock.patch.object(channels, '_GHOST_DECAY_IM_THRESHOLD', 0.0):
            self.assertTrue(
                _ghost_admits(GATE_W, source, matrix),
                'with both gates zeroed the config must admit; at least one '
                'constant is not load-bearing on the refuse side')
        self.comparisons += 1

    def test_raising_constant_to_two_refuses_an_admit_config(self):
        """With MIN=2.0 a fact-1 far-from-cusp config no longer admits."""
        gamma, theta_deg, offset = ADMIT_CONFIGS[0]
        source, matrix = _source_and_matrix(gamma, theta_deg, offset)
        self.assertTrue(_ghost_admits(GATE_W, source, matrix),
                        'precondition: this config must admit at MIN=0.7')
        with mock.patch.object(channels, '_GHOST_SEPARATION_MIN', 2.0):
            self.assertFalse(
                _ghost_admits(GATE_W, source, matrix),
                'with MIN=2.0 the separation gate must refuse; the constant '
                'is not load-bearing on the admit side')
        self.comparisons += 1


class _Sentinel(Exception):
    """Marker raised by the spy so the beta-guard test can catch it."""


class _SurrogateProbe:
    """Lightweight probe binding the REAL surrogate-coefficients methods.

    ``_surrogate_coefficients`` reaches the kappa/beta guards using only
    ``self._lens_params``; downstream it needs ``self._kernel_dense_f`` (an
    argument to ``dimensionless_frequency``, which the test spies out).  No
    heavy waveform/event construction is required to exercise the guard.
    """

    _lens_params = LensedRelativeBinningLikelihood._lens_params
    _surrogate_coefficients = \
        LensedRelativeBinningLikelihood._surrogate_coefficients
    _kernel_dense_f = np.array([1.0, 2.0])


def _servable_par_dic(*, beta: float, kappa: float) -> dict:
    """A par_dic with all lens keys; otherwise-servable but for beta/kappa."""
    source, _matrix = _source_and_matrix(*TRAIN_SERVE_ADMIT_CONFIG)
    return {
        'm_lens_msun': 1.0e3, 'z_lens': 0.5,
        'y1': float(source[0]), 'y2': float(source[1]),
        'gamma': 0.5, 'beta': beta, 'kappa': kappa,
    }


class BetaGuardTestCase(GhostGateTestCase):
    """``_surrogate_coefficients`` falls through for beta != 0 (as for kappa).

    The guard is isolated by spying ``likelihood.dimensionless_frequency`` --
    the first call AFTER the beta guard.  A blocked candidate returns ``None``
    with the spy un-called; an unblocked candidate reaches (and trips) the spy.
    """

    def test_nonzero_beta_returns_none_without_passing_guard(self):
        """beta=0.3 => None, and dimensionless_frequency is never called."""
        probe = _SurrogateProbe()
        par_dic = _servable_par_dic(beta=0.3, kappa=0.0)
        spy = mock.Mock(side_effect=_Sentinel)
        with mock.patch.object(lensing_likelihood, 'dimensionless_frequency',
                               spy):
            result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(
            result, 'beta != 0 must fall through to the exact engine (None)')
        self.assertEqual(
            spy.call_count, 0,
            'execution passed the beta guard for beta != 0 (spy was called)')
        self.comparisons += 1

    def test_nonzero_kappa_returns_none_without_passing_guard(self):
        """kappa=0.5 => None (companion guard), spy never called."""
        probe = _SurrogateProbe()
        par_dic = _servable_par_dic(beta=0.0, kappa=0.5)
        spy = mock.Mock(side_effect=_Sentinel)
        with mock.patch.object(lensing_likelihood, 'dimensionless_frequency',
                               spy):
            result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result, 'kappa != 0 must fall through (None)')
        self.assertEqual(spy.call_count, 0,
                         'execution passed the kappa guard for kappa != 0')
        self.comparisons += 1

    def test_zero_beta_proceeds_past_the_guard(self):
        """beta=0.0 (kappa=0) must NOT be stopped by the beta guard.

        The spy raises the moment execution passes the guard, proving the
        guard keys specifically on ``beta != 0`` and does not block beta = 0.
        """
        probe = _SurrogateProbe()
        par_dic = _servable_par_dic(beta=0.0, kappa=0.0)
        spy = mock.Mock(side_effect=_Sentinel)
        with mock.patch.object(lensing_likelihood, 'dimensionless_frequency',
                               spy):
            with self.assertRaises(_Sentinel):
                probe._surrogate_coefficients(par_dic)
        self.assertGreaterEqual(
            spy.call_count, 1,
            'beta = 0 must proceed past the guard and reach '
            'dimensionless_frequency')
        self.comparisons += 1


class FrameSingleSourceRegressionTestCase(GhostGateTestCase):
    """``real_image_delays`` keeps the min-subtracted convention byte-for-byte.

    Independently recomputes ``np.sort(absolute - min(absolute))`` from
    ``geometry.find_images``/``geometry.delay`` (no call into
    ``_frame_delays``) and requires exact agreement with a smallest element of
    exactly 0.0 -- routing the frame origin through ``_frame_delays`` must not
    have changed the returned convention.
    """

    def test_real_image_delays_match_inline_min_subtraction(self):
        """Byte-identical to the inline oracle; smallest element == 0.0."""
        for gamma, theta_deg, offset in FRAME_CONFIGS:
            with self.subTest(gamma=gamma, theta_c=theta_deg, offset=offset):
                source, matrix = _source_and_matrix(gamma, theta_deg, offset)
                produced = real_image_delays(
                    gamma, (float(source[0]), float(source[1])),
                    beta=0.0, kappa=0.0)
                images = geometry.find_images(source, matrix)
                absolute = np.array(
                    [geometry.delay(image, source, matrix) for image in images])
                expected = np.sort(absolute - absolute.min())
                self.assertEqual(produced.shape, expected.shape)
                self.assertEqual(
                    float(np.max(np.abs(produced - expected))), 0.0,
                    'real_image_delays diverged from the inline '
                    'min-subtracted oracle')
                self.assertEqual(
                    float(produced.min()), 0.0,
                    'the smallest real-image delay is not exactly 0.0')
                self.comparisons += 1


def _old_decay_gate_admits(w: np.ndarray, source: np.ndarray,
                           matrix: np.ndarray) -> bool:
    """RETIRED gate reimplemented independently: ``w_min*Im tau_c >= 2.0``.

    The foil the re-keyed geometric gate replaced.  ``Im tau_c`` comes from
    ``geometry.ghost_kernel(...).delay.imag`` -- the same holomorphic ghost
    delay, read without any separation branch.  Used only to exhibit the
    train/serve skew the new gate removes; never a production reference.
    """
    contribution = geometry.ghost_kernel(w, source, matrix)
    return float(w.min()) * float(contribution.delay.imag) \
        >= FARFIELD_WINDOW_RADIANS


class GhostGateSelfFalsificationTestCase(GhostGateTestCase):
    """Proof the suite can go red: the retired gate skews, the gate protects.

    Two independent red-witnesses.  (1) The retired decay gate DISAGREES
    between the train and serve grids on the near-threshold skew config, so
    ``TrainServeDecisionAgreementTestCase``'s boolean-equality assertion would
    have FAILED under the old code -- it has teeth.  (2) A refused near-cusp
    config forced through the MINUS_GHOST label (gate bypassed via MIN=0.0)
    makes the DO-NOTHING residual strictly WORSE, so the gate is exactly what
    protects ``DoNothingControlTestCase``.
    """

    def test_retired_decay_gate_skews_train_versus_serve(self):
        """Old gate: refuse on the small-min(w) grid, admit on the larger one.

        Also confirms the NEW gate ADMITS both (frequency-independent), so the
        disagreement is a property of the retired gate alone.
        """
        source, matrix = _source_and_matrix(*NEAR_THRESHOLD_CONFIG)
        # New gate: identical (admit) on both grids.
        self.assertTrue(_ghost_admits(TRAIN_W, source, matrix))
        self.assertTrue(_ghost_admits(SERVE_W, source, matrix))
        # Old gate: train refuses, serve admits -- the skew the test catches.
        old_train = _old_decay_gate_admits(TRAIN_W, source, matrix)
        old_serve = _old_decay_gate_admits(SERVE_W, source, matrix)
        self.assertFalse(
            old_train,
            'retired decay gate should refuse the small-min(w) train grid')
        self.assertTrue(
            old_serve,
            'retired decay gate should admit the larger-min(w) serve grid')
        self.assertNotEqual(
            old_train, old_serve,
            'the retired gate must skew train vs serve; without a skew the '
            'boolean-equality test would have no teeth')
        self.comparisons += 1

    def test_forcing_a_refused_config_worsens_the_donothing_residual(self):
        """Bypassing both gates on a near-cusp config: resid(MINUS_GHOST) > ks.

        Near-cusp configs have low Im(tau_c) (~0.001) so the decay gate fires
        before the separation gate.  To force the ghost through the
        MINUS_GHOST label, both ``_GHOST_SEPARATION_MIN`` and
        ``_GHOST_DECAY_IM_THRESHOLD`` are patched to zero — this is the only
        way to reach ``farfield_ghost_term``'s return path for a config that
        fails both gates.  The residual comparison proves the gate is
        protective: subtracting the undecayed, inseparable ghost WORSENS the
        label.
        """
        gamma, theta_deg, offset = TRAIN_SERVE_REFUSE_CONFIG
        partition = _build_partition(gamma, theta_deg, offset, MID_BAND_W)
        # Sanity: this config genuinely refuses at the production thresholds.
        source, matrix = _source_and_matrix(gamma, theta_deg, offset)
        self.assertFalse(_ghost_admits(MID_BAND_W, source, matrix))
        with mock.patch.object(channels, '_GHOST_SEPARATION_MIN', 0.0), \
             mock.patch.object(channels, '_GHOST_DECAY_IM_THRESHOLD', 0.0):
            resid_ks, resid_mg = _f_normalized_residuals(partition)
        self.assertGreater(
            resid_mg, resid_ks,
            f'forcing the refused ghost through MINUS_GHOST did not worsen '
            f'the label (resid_mg={resid_mg:.3e} <= resid_ks={resid_ks:.3e}); '
            'the gate would then have no protective effect')
        self.comparisons += 1


if __name__ == '__main__':
    unittest.main()
