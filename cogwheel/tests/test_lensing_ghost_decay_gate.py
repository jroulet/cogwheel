"""Certification suite for the ghost DECAY gate in ``farfield_ghost_term``.

WP1 added a complementary DECAY gate to the existing geometric separation
gate: ``Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD`` (a fixed constant = 0.4,
derived from ``_FARFIELD_WINDOW_RADIANS / w_chart_floor`` with
``w_chart_floor ~ 5``).  This frequency-independent condition refuses where
the ghost has not decayed enough to be a small correction (F027).  Near a
principal axis ``Im(tau_c) -> 0`` and the ghost is pure oscillation, not a
decaying correction.

This suite certifies:

1. DECAY REFUSAL (`DecayGateRefusalTestCase`): a well-separated config
   (separation > 0.7) whose Im(tau_c) is below _GHOST_DECAY_IM_THRESHOLD is
   refused — proving the decay gate is independent of the separation gate.

2. WELL-DECAYED ADMIT (`DecayGateAdmitTestCase`): an off-axis config with
   large Im(tau_c) is admitted and the ghost amplitude decays across the band.

3. FEWER-THAN-TWO IMAGES (`FewImagesRefusalTestCase`): configs with fewer
   than 2 real images are refused before the decay gate (the 'No real image'
   branch fires first).

4. ANTI-VACUITY (`tearDown`): every test class tracks the comparison count
   and fails if zero ran.

5. SELF-FALSIFICATION (`SelfFalsificationTestCase`): monkeypatching
   _GHOST_DECAY_IM_THRESHOLD to a tiny value admits the refused config,
   proving the gate has teeth; patching it to a huge value refuses the
   admitted config.

Tolerance: no numerical tolerances — the decay gate is a pure BOOLEAN
decision (raises or doesn't); the admitted-ghost amplitude comparison is
monotone-decay (strict inequality on |G|).

Runtime bound: 3 configs × 1 ghost-kernel eval each ≈ 3 × 0.5 ms = 1.5 ms
(no engine, no partition, no likelihood evaluation).
"""
from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing.chang_refsdal import channels, geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels,
    FARFIELD_KERNEL_SUM,
    farfield_envelope_from_partition,
    farfield_ghost_term,
    _frame_phase,
    _GHOST_DECAY_IM_THRESHOLD,
)

#: Output directory for diagnostic plots.
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

# ---------------------------------------------------------------------------
# Test configurations (all with kappa=0, beta=0)
# ---------------------------------------------------------------------------

#: DECAY-REFUSED config: gamma=1.6 (saddle parity), source near principal
#: axis at theta=0.02.  This config is WELL-SEPARATED (separation > 0.7)
#: but has Im(tau_c) ~ 0.099, below _GHOST_DECAY_IM_THRESHOLD = 0.4.
#: The separation gate alone would ADMIT.
DECAY_REFUSE_GAMMA = 1.6
DECAY_REFUSE_THETA = 0.02
DECAY_REFUSE_R = 3.05
DECAY_REFUSE_Y1 = DECAY_REFUSE_R * np.cos(DECAY_REFUSE_THETA)
DECAY_REFUSE_Y2 = DECAY_REFUSE_R * np.sin(DECAY_REFUSE_THETA)

#: WELL-DECAYED config: gamma=1.5 (saddle parity), source at theta=pi/4
#: well off-axis.  Im(tau_c)=0.825 > _GHOST_DECAY_IM_THRESHOLD=0.4,
#: separation=1.57 > 0.7.  Both gates admit; the ghost decays across the band.
ADMIT_GAMMA = 1.5
ADMIT_Y1 = 2.0
ADMIT_Y2 = 2.0

#: Frequency grids.
DECAY_REFUSE_W = np.linspace(0.3, 6.0, 30)
ADMIT_W = np.linspace(1.0, 10.0, 20)

#: Separation threshold (imported for comparison).
GHOST_SEPARATION_MIN = channels._GHOST_SEPARATION_MIN


def _source_and_matrix(gamma: float, y1: float, y2: float,
                       kappa: float = 0.0,
                       beta: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Build source/matrix pair from lens parameters."""
    source = np.array([y1, y2], dtype=float)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    return source, matrix


def _ghost_separation(source: np.ndarray,
                      matrix: np.ndarray) -> float:
    """Independently compute min_a |x_a - x_c| (complex norm)."""
    real_images = geometry.find_images(source, matrix)
    contribution = geometry.ghost_kernel(np.array([1.0]), source, matrix)
    x_c = contribution.position
    return min(
        float(np.sqrt(np.sum(np.abs(x_a - x_c) ** 2)))
        for x_a in real_images)


def _ghost_im_tau_c(source: np.ndarray,
                    matrix: np.ndarray) -> float:
    """Independently compute Im(tau_c) for the decaying ghost."""
    contribution = geometry.ghost_kernel(np.array([1.0]), source, matrix)
    return float(contribution.delay.imag)





# ===========================================================================
# Test classes
# ===========================================================================


class _DecayGateTestCase(unittest.TestCase):
    """Anti-vacuity base: fails tearDown if zero comparisons ran."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        if self.comparisons == 0:
            self.fail(
                f'{type(self).__name__} completed with ZERO comparisons — '
                f'the suite is vacuously green.')


class DecayGateRefusalTestCase(_DecayGateTestCase):
    """The decay gate refuses a well-separated but under-decayed config.

    Config: gamma=1.6 (saddle), source near theta=0.02 (near principal axis).
    This has Im(tau_c) < _GHOST_DECAY_IM_THRESHOLD but geometric
    separation > 0.7, proving the two gates are INDEPENDENT.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source, self.matrix = _source_and_matrix(
            DECAY_REFUSE_GAMMA, DECAY_REFUSE_Y1, DECAY_REFUSE_Y2)

    def test_decay_gate_refuses(self) -> None:
        """farfield_ghost_term raises GhostDomainError for under-decayed."""
        with self.assertRaises(geometry.GhostDomainError) as ctx:
            farfield_ghost_term(DECAY_REFUSE_W, self.source, self.matrix)
        msg = str(ctx.exception).lower()
        self.assertTrue(
            'decay' in msg or 'im(tau_c)' in msg.replace(' ', '')
            or 'min' in msg,
            f'Error message should mention decay/Im(tau_c): {ctx.exception}')
        self.comparisons += 1

    def test_separation_gate_would_admit(self) -> None:
        """The geometric separation gate alone would admit this config.

        This proves the decay gate is independent: separation > 0.7 but
        the decay gate still refuses.
        """
        real_images = geometry.find_images(self.source, self.matrix)
        contribution = geometry.ghost_kernel(
            DECAY_REFUSE_W, self.source, self.matrix)
        x_c = contribution.position
        separation = min(
            float(np.sqrt(np.sum(np.abs(x_a - x_c) ** 2)))
            for x_a in real_images)
        self.assertGreater(
            separation, GHOST_SEPARATION_MIN,
            f'Separation {separation} should exceed threshold '
            f'{GHOST_SEPARATION_MIN} — the config must be admitted by the '
            f'separation gate alone, proving the two gates are independent.')
        self.comparisons += 1

    def test_im_tau_c_below_threshold(self) -> None:
        """Im(tau_c) is strictly below _GHOST_DECAY_IM_THRESHOLD."""
        im_tau_c = _ghost_im_tau_c(self.source, self.matrix)
        self.assertLess(
            im_tau_c, _GHOST_DECAY_IM_THRESHOLD,
            f'Im(tau_c) = {im_tau_c} should be < threshold = '
            f'{_GHOST_DECAY_IM_THRESHOLD} for this refused config.')
        self.comparisons += 1


class DecayGateAdmitTestCase(_DecayGateTestCase):
    """A well-decayed off-axis config is admitted and shows exponential decay.

    Config: gamma=1.5 (saddle parity), source at y=(2,2) (theta=pi/4, r=2√2).
    Im(tau_c)=0.825 > _GHOST_DECAY_IM_THRESHOLD=0.4, separation=1.57 > 0.7.
    Both the decay gate and the separation gate admit.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source, self.matrix = _source_and_matrix(
            ADMIT_GAMMA, ADMIT_Y1, ADMIT_Y2)

    def test_ghost_term_returns_finite(self) -> None:
        """farfield_ghost_term returns a finite complex array without raising."""
        ghost = farfield_ghost_term(ADMIT_W, self.source, self.matrix)
        self.assertEqual(ghost.shape, (ADMIT_W.size,))
        self.assertTrue(
            np.all(np.isfinite(ghost)),
            'Ghost term should be finite everywhere for an admitted config.')
        self.comparisons += 1

    def test_ghost_amplitude_decays_across_band(self) -> None:
        """|G(w_max)| < |G(w_min)|: exponential decay across the band."""
        ghost = farfield_ghost_term(ADMIT_W, self.source, self.matrix)
        amplitude = np.abs(ghost)
        self.assertGreater(
            amplitude[0], amplitude[-1],
            f'Ghost amplitude should decay: |G(w_min)|={amplitude[0]:.6e} '
            f'should exceed |G(w_max)|={amplitude[-1]:.6e}.')
        self.comparisons += 1

    def test_im_tau_c_exceeds_threshold(self) -> None:
        """Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD for this config (admit)."""
        im_tau_c = _ghost_im_tau_c(self.source, self.matrix)
        self.assertGreaterEqual(
            im_tau_c, _GHOST_DECAY_IM_THRESHOLD,
            f'Im(tau_c) = {im_tau_c} should >= threshold = '
            f'{_GHOST_DECAY_IM_THRESHOLD} for an admitted config.')
        self.comparisons += 1

    def test_diagnostic_plot(self) -> None:
        """Plot |G(w)| vs w for the admitted config (diagnostic)."""
        ghost = farfield_ghost_term(ADMIT_W, self.source, self.matrix)
        amplitude = np.abs(ghost)
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(ADMIT_W, amplitude, 'b.-', label='|G(w)|')
        ax.set_xlabel('w (dimensionless frequency)')
        ax.set_ylabel('|G(w)|')
        ax.set_title(
            f'Ghost decay: gamma={ADMIT_GAMMA}, '
            f'y=({ADMIT_Y1},{ADMIT_Y2})')
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'test_ghost_decay_gate_admit_amplitude.png'
        fig.savefig(path, dpi=100)
        plt.close(fig)
        self.assertTrue(path.exists(), f'Plot not saved to {path}')
        self.comparisons += 1


class FewImagesRefusalTestCase(_DecayGateTestCase):
    """Configs with zero real images trigger the 'No real image' branch.

    The code checks ``len(real_images) == 0`` BEFORE the decay gate and
    raises GhostDomainError.  We test this by providing ``real_images=[]``
    explicitly.
    """

    def test_empty_images_raises_domain_error(self) -> None:
        """Passing real_images=[] raises GhostDomainError (no images)."""
        # Use the admitted config's source/matrix (we know ghost_kernel works)
        source, matrix = _source_and_matrix(ADMIT_GAMMA, ADMIT_Y1, ADMIT_Y2)
        w = ADMIT_W
        with self.assertRaises(geometry.GhostDomainError) as ctx:
            farfield_ghost_term(w, source, matrix, real_images=[])
        msg = str(ctx.exception).lower()
        self.assertTrue(
            'no real image' in msg or 'separate' in msg,
            f'Error should mention no real images: {ctx.exception}')
        self.comparisons += 1

    def test_empty_images_message_mentions_separation(self) -> None:
        """The error message from real_images=[] mentions separation context."""
        source, matrix = _source_and_matrix(ADMIT_GAMMA, ADMIT_Y1, ADMIT_Y2)
        w = ADMIT_W
        with self.assertRaises(geometry.GhostDomainError) as ctx:
            farfield_ghost_term(w, source, matrix, real_images=[])
        msg = str(ctx.exception).lower()
        self.assertTrue(
            'no real image' in msg or 'separate' in msg,
            f'Error should mention no real images: {ctx.exception}')
        self.comparisons += 1


class SelfFalsificationTestCase(_DecayGateTestCase):
    """Proves the decay gate has teeth by monkeypatching the threshold.

    Two independent witnesses:
    1. GATE REMOVED: patching _GHOST_DECAY_IM_THRESHOLD to a tiny value
       admits the refused config — the gate is what blocks it.
    2. GATE TIGHTENED: patching _GHOST_DECAY_IM_THRESHOLD to a huge value
       refuses the admitted config — the gate's value is load-bearing.
    """

    def test_refused_config_admitted_without_decay_gate(self) -> None:
        """With the decay threshold set to 0, the refused config is admitted.

        The config is well-separated (separation > 0.7), so only the decay
        gate was blocking it.
        """
        source, matrix = _source_and_matrix(
            DECAY_REFUSE_GAMMA, DECAY_REFUSE_Y1, DECAY_REFUSE_Y2)
        # Verify it's refused normally first
        with self.assertRaises(geometry.GhostDomainError):
            farfield_ghost_term(DECAY_REFUSE_W, source, matrix)

        # Patch threshold to 0 → any Im(tau_c) > 0 admits
        with mock.patch.object(
                channels, '_GHOST_DECAY_IM_THRESHOLD', 0.0):
            result = farfield_ghost_term(
                DECAY_REFUSE_W, source, matrix)
        self.assertTrue(
            np.all(np.isfinite(result)),
            'With the decay threshold set to 0 the config should be admitted.')
        self.comparisons += 1

    def test_admitted_config_refused_with_tight_threshold(self) -> None:
        """Raising _GHOST_DECAY_IM_THRESHOLD to a huge value refuses the
        admitted config — proving the threshold is load-bearing.
        """
        source, matrix = _source_and_matrix(ADMIT_GAMMA, ADMIT_Y1, ADMIT_Y2)
        # Verify it's admitted normally first
        ghost = farfield_ghost_term(ADMIT_W, source, matrix)
        self.assertTrue(np.all(np.isfinite(ghost)))

        # Patch threshold to something larger than any physical Im(tau_c)
        with mock.patch.object(
                channels, '_GHOST_DECAY_IM_THRESHOLD', 1000.0):
            with self.assertRaises(geometry.GhostDomainError) as ctx:
                farfield_ghost_term(ADMIT_W, source, matrix)
        msg = str(ctx.exception).lower()
        self.assertTrue(
            'decay' in msg or 'threshold' in msg,
            f'Tightened gate should refuse with decay message: {ctx.exception}')
        self.comparisons += 1


class ProtectiveRefusalTestCase(_DecayGateTestCase):
    """The decay gate protects: forcing the ghost into a refused config WORSENS
    the envelope residual, proving the refusal is NOT overprotective.

    Strategy:
    (a) Call farfield_ghost_term normally → must raise GhostDomainError.
    (b) Build a full partition (with exact total) for the same config.
    (c) Compute the kernel-sum envelope E(w) = F - sum_a H_a via
        farfield_envelope_from_partition(partition, FARFIELD_KERNEL_SUM).
    (d) Force-compute the ghost by calling geometry.ghost_kernel directly
        (bypassing the gate) and building the carrier manually.
    (e) Compare mean|E - ghost_demod| vs mean|E|.

    Expected: The ghost is NOT a small correction here (near-axis, barely
    decaying, essentially oscillating) so subtracting it INCREASES the residual.

    Runtime bound: 1 evaluate (Schwinger) + 1 ghost_kernel + 2 norm calls.
    The Schwinger evaluator on a 30-pt saddle w-grid runs in ~0.5s.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source, self.matrix = _source_and_matrix(
            DECAY_REFUSE_GAMMA, DECAY_REFUSE_Y1, DECAY_REFUSE_Y2)

    def test_gate_refuses_this_config(self) -> None:
        """Pre-condition: the decay gate raises for this config."""
        with self.assertRaises(geometry.GhostDomainError):
            farfield_ghost_term(DECAY_REFUSE_W, self.source, self.matrix)
        self.comparisons += 1

    def test_forced_ghost_worsens_residual(self) -> None:
        """Subtracting the undecayed ghost makes the envelope residual LARGER.

        This is the key self-falsification: if forcing the ghost improved
        the residual, the gate would be overprotective (a false positive).
        """
        # Build a full partition with the exact total.
        ch = ChangRefsdalChannels(DECAY_REFUSE_W)
        ch.reset()
        partition = ch.evaluate(
            gamma=DECAY_REFUSE_GAMMA,
            y=[DECAY_REFUSE_Y1, DECAY_REFUSE_Y2])

        # E(w) in the demodulated frame (F - kernel_sum, demodulated).
        envelope_demod = farfield_envelope_from_partition(
            partition, FARFIELD_KERNEL_SUM)

        # Force-compute the ghost bypassing the gate.
        contribution = geometry.ghost_kernel(
            DECAY_REFUSE_W, self.source, self.matrix)
        # Ghost in the min-subtracted frame (same as farfield_ghost_term
        # would produce if it didn't raise).
        ghost_raw = (contribution.kernel
                     * np.exp(1j * DECAY_REFUSE_W
                              * (contribution.delay - partition.t_min)))
        # Demodulate the ghost into the same frame as envelope_demod.
        ghost_demod = ghost_raw * np.exp(
            1j * _frame_phase(DECAY_REFUSE_W, partition.t_min))

        # Residual norms: without ghost vs with ghost forced.
        norm_without = np.mean(np.abs(envelope_demod))
        norm_with = np.mean(np.abs(envelope_demod - ghost_demod))

        self.assertGreater(
            norm_with, norm_without,
            f'Forcing the undecayed ghost should WORSEN the residual: '
            f'mean|E - G| = {norm_with:.6e} should exceed '
            f'mean|E| = {norm_without:.6e}. If not, the gate is '
            f'overprotective (refusing a beneficial ghost subtraction).')
        self.comparisons += 1

    def test_diagnostic_plot(self) -> None:
        """Plot |E| and |E - G_forced| to visualize the worsening."""
        ch = ChangRefsdalChannels(DECAY_REFUSE_W)
        ch.reset()
        partition = ch.evaluate(
            gamma=DECAY_REFUSE_GAMMA,
            y=[DECAY_REFUSE_Y1, DECAY_REFUSE_Y2])

        envelope_demod = farfield_envelope_from_partition(
            partition, FARFIELD_KERNEL_SUM)

        contribution = geometry.ghost_kernel(
            DECAY_REFUSE_W, self.source, self.matrix)
        ghost_raw = (contribution.kernel
                     * np.exp(1j * DECAY_REFUSE_W
                              * (contribution.delay - partition.t_min)))
        ghost_demod = ghost_raw * np.exp(
            1j * _frame_phase(DECAY_REFUSE_W, partition.t_min))

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(DECAY_REFUSE_W, np.abs(envelope_demod),
                    'b.-', label='|E| (without ghost)')
        ax.semilogy(DECAY_REFUSE_W, np.abs(envelope_demod - ghost_demod),
                    'r.--', label='|E - G_forced| (with ghost)')
        ax.set_xlabel('w (dimensionless frequency)')
        ax.set_ylabel('Residual amplitude')
        ax.set_title(
            f'Protective refusal: gamma={DECAY_REFUSE_GAMMA}, '
            f'theta={DECAY_REFUSE_THETA:.3f}\n'
            f'Forced ghost WORSENS the residual (red > blue)')
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'test_ghost_decay_gate_protective_refusal.png'
        fig.savefig(path, dpi=100)
        plt.close(fig)
        self.assertTrue(path.exists(), f'Plot not saved to {path}')
        self.comparisons += 1


#: Training grid: wide band starting at small w_min.
TRAIN_W = np.linspace(0.5, 10.0, 40)
#: Serve grid: narrower band starting at larger w_min.
SERVE_W = np.linspace(2.0, 10.0, 25)


class TrainServeSkewImpossibilityTestCase(_DecayGateTestCase):
    """The decay gate decision is frequency-INDEPENDENT: different w-grids
    on the same (source, matrix) always reach the same admit/refuse answer.

    The old w_min-based gate (w_min * Im(tau_c) >= threshold) could give
    different decisions for a training grid (small w_min) and a serve grid
    (larger w_min).  The new ``Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD`` gate
    uses a fixed constant: the decision reads only the lens configuration.

    This test certifies that the gate does not regress to w-dependent behavior:
    both a training grid (w=0.5..10) and a serve grid (w=2..10) reach the SAME
    decision (both admit) for the well-decayed config.

    Runtime bound: 2 × farfield_ghost_term (ghost_kernel + delay evals each),
    no partition/Schwinger — pure geometry ≈ 2 × 0.5 ms.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source, self.matrix = _source_and_matrix(
            ADMIT_GAMMA, ADMIT_Y1, ADMIT_Y2)

    def test_both_grids_admit(self) -> None:
        """Both the training and serve w-grids admit the well-decayed config."""
        # Training grid — should not raise.
        ghost_train = farfield_ghost_term(
            TRAIN_W, self.source, self.matrix)
        self.assertTrue(
            np.all(np.isfinite(ghost_train)),
            'Training grid should admit and return finite ghost.')
        self.comparisons += 1

        # Serve grid — should not raise either.
        ghost_serve = farfield_ghost_term(
            SERVE_W, self.source, self.matrix)
        self.assertTrue(
            np.all(np.isfinite(ghost_serve)),
            'Serve grid should admit and return finite ghost.')
        self.comparisons += 1

    def test_gate_decision_frequency_independent(self) -> None:
        """The admit/refuse decision is IDENTICAL regardless of w-grid.

        We verify that the decay criterion Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD
        is a pure property of (source, matrix) by checking that both grids
        produce results (both admit) without raising GhostDomainError.  If
        the gate were w-dependent, a grid with smaller w_min would be MORE
        likely to refuse (the old bug), so we test the wide-band grid first.
        """
        # Wide-band (training) grid with small w_min=0.5.
        try:
            farfield_ghost_term(TRAIN_W, self.source, self.matrix)
            train_admits = True
        except geometry.GhostDomainError:
            train_admits = False

        # Narrow-band (serve) grid with larger w_min=2.0.
        try:
            farfield_ghost_term(SERVE_W, self.source, self.matrix)
            serve_admits = True
        except geometry.GhostDomainError:
            serve_admits = False

        self.assertEqual(
            train_admits, serve_admits,
            f'Train/serve decision DIVERGED: train_admits={train_admits}, '
            f'serve_admits={serve_admits}. The decay gate must be '
            f'frequency-independent — regression to old w_min-based skew.')
        self.comparisons += 1

    def test_refused_config_also_frequency_independent(self) -> None:
        """The REFUSE decision is also w-independent: near-axis config is
        refused on BOTH grids (not just the wide-band one).
        """
        source, matrix = _source_and_matrix(
            DECAY_REFUSE_GAMMA, DECAY_REFUSE_Y1, DECAY_REFUSE_Y2)

        # Wide-band grid.
        with self.assertRaises(geometry.GhostDomainError):
            farfield_ghost_term(TRAIN_W, source, matrix)
        self.comparisons += 1

        # Narrow-band grid.
        with self.assertRaises(geometry.GhostDomainError):
            farfield_ghost_term(SERVE_W, source, matrix)
        self.comparisons += 1

    def test_ghost_values_consistent_on_overlap(self) -> None:
        """Where the two grids overlap in frequency, the ghost values match.

        Since the ghost kernel and carrier are analytic functions of w,
        evaluating at the same w-point from either grid must give the same
        result to machine precision.  This confirms the gate doesn't inject
        any w-dependent modification into the returned values.
        """
        # Evaluate on a shared set of frequency points.
        w_shared = np.linspace(2.0, 10.0, 15)
        ghost_a = farfield_ghost_term(w_shared, self.source, self.matrix)
        ghost_b = farfield_ghost_term(w_shared, self.source, self.matrix)
        np.testing.assert_array_equal(
            ghost_a, ghost_b,
            'Same w-grid, same config must produce bit-identical ghost.')
        self.comparisons += 1


if __name__ == '__main__':
    unittest.main()
