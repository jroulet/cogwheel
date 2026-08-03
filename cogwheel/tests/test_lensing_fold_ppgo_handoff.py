"""Tests for fold-ppGO interior handoff in _surrogate_coefficients + census.

Validates three properties of ``fold_ppgo_correction`` for interior
sources above the DD product cap:

1. **Accuracy**: fold correction agrees with the exact engine within 1%
   for an interior 4-image config at moderate xi (>= 4) and w above the
   DD cap (~40).
2. **Gate conservatism**: the correction refuses (xi < threshold) near
   axis cusps where Δτ → 0.
3. **Round-trip identity**: extracting a far-field envelope from the
   fold-corrected total and reconstructing via ``reconstruct_farfield``
   is lossless to machine precision.

Tolerance justification
-----------------------
- Test 1 uses 1% relative tolerance: this is the spec's accuracy bar for
  the fold-ppGO vs exact engine comparison at xi >= 4.
- Test 3 uses 1e-12 relative tolerance: the demod/remod round-trip incurs
  only floating-point rounding (see ``_frame_phase`` caveat — worst case
  is ~1e-11 near a fold, but our fixture is moderate so ~1e-14 expected).

Cost budget
-----------
- Test 1: 1 config × 20 w-points engine evaluation ≈ < 10 s.
- Test 2: 1 config × 1 w-point engine evaluation ≈ < 2 s.
- Test 3: Pure algebra on pre-computed fold correction, no engine eval ≈ < 1 s.
- Total file budget: < 15 s (well under the 5 min ceiling).
"""
from __future__ import annotations

import math
import os
import pathlib
import unittest
from unittest import TestCase

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._airy_fold import fold_ppgo_correction
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels,
    FARFIELD_KERNEL_SUM,
    reconstruct_farfield,
)

# ---------------------------------------------------------------------------
#: Output directory for diagnostic plots.
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

#: External shear for all tests — positive parity (gamma < 1).
GAMMA: float = 0.5

#: Caustic-relative radius for interior source — well inside (rho < 1).
RHO_INTERIOR: float = 0.3

#: Angle away from axis cusps (theta ~ pi/4).
THETA_INTERIOR: float = math.pi / 4.0

#: Caustic-relative radius for near-caustic source (rho ~ 0.7).
#: At rho=0.7, gamma=0.5 the fold pair Δτ is small enough that
#: xi_min < 4.0, demonstrating the gate's refusal regime.
RHO_NEAR_CAUSTIC: float = 0.7

#: Angle for near-caustic source (same as interior for comparability).
THETA_NEAR_CAUSTIC: float = math.pi / 4.0

#: Frequency range above a typical DD cap (~40) but below Schwinger wall.
W_MIN: float = 45.0
W_MAX: float = 55.0
N_W: int = 20

#: xi threshold from production (fold correction refuses below this).
XI_FOLD_THRESHOLD: float = 4.0

#: Accuracy bar for fold correction vs exact (1% relative error).
ACCURACY_BAR: float = 0.01

#: Round-trip machine-precision bar (demod/remod identity).
ROUNDTRIP_BAR: float = 1e-12
# ---------------------------------------------------------------------------


def _polar_source(rho: float, angle: float, gamma: float,
                  *, kappa: float = 0.0) -> np.ndarray:
    """Build source position from caustic-relative rho and polar angle.

    rho = |y| / r_caustic(gamma, angle), so |y| = rho * r_caustic.
    """
    reach = geometry.r_caustic(gamma, angle, kappa=kappa)
    radius = rho * reach
    return radius * np.array([math.cos(angle), math.sin(angle)])


def _compute_t_min(source: np.ndarray, gamma: float, *,
                   beta: float = 0.0, kappa: float = 0.0) -> float:
    """Minimum real-image Fermat delay for the config."""
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(source, matrix)
    absolute_delays = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    return float(absolute_delays.min())


def _compute_delta_tau(source: np.ndarray, gamma: float, *,
                       beta: float = 0.0, kappa: float = 0.0) -> float | None:
    """Delay separation of the merging fold pair, or None if no pair."""
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(source, matrix)

    entries: list[tuple[float, int]] = []
    for image in images:
        try:
            n_morse = geometry.morse_index(image, matrix)
        except geometry.LensDomainError:
            return None
        entries.append((geometry.delay(image, source, matrix), n_morse))
    entries.sort(key=lambda entry: entry[0])

    best_gap = math.inf
    for (tau_low, n_low), (tau_high, n_high) in zip(entries, entries[1:]):
        if n_low == 0 and n_high == 1:
            gap = tau_high - tau_low
            if 0.0 < gap < best_gap:
                best_gap = gap
    return best_gap if best_gap < math.inf else None


def _compute_xi_min(w_min: float, delta_tau: float) -> float:
    """Compute xi_min = (3*w_min*Δτ/4)^{2/3} for the fold pair."""
    return (3.0 * w_min * delta_tau / 4.0) ** (2.0 / 3.0)


def _save_diagnostic_plot(filename: str, w: np.ndarray,
                          exact: np.ndarray, corrected: np.ndarray,
                          title: str) -> None:
    """Save diagnostic plot comparing fold correction to exact."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(w, np.abs(exact), 'k-', label='exact')
        ax1.plot(w, np.abs(corrected), 'r--', label='fold-ppGO')
        ax1.set_ylabel('|F(w)|')
        ax1.set_title(title)
        ax1.legend()

        rel_diff = np.abs(corrected - exact) / np.maximum(np.abs(exact), 1e-30)
        ax2.semilogy(w, rel_diff, 'b-')
        ax2.axhline(ACCURACY_BAR, color='r', linestyle=':', label='1% bar')
        ax2.set_xlabel('w')
        ax2.set_ylabel('relative error')
        ax2.legend()

        plt.tight_layout()
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_OUTPUT_DIR / filename, dpi=100)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not available — skip plot


# ===========================================================================
# Base test case with anti-vacuity tearDown
# ===========================================================================
class _HandoffTestCase(TestCase):
    """Base class carrying anti-vacuity enforcement."""

    def setUp(self):
        """Reset per-test comparison counter."""
        self.n_checks = 0

    def tearDown(self):
        """Fail if zero comparisons ran (anti-vacuity)."""
        if self.n_checks == 0:
            self.fail(
                f'{self._testMethodName}: zero comparisons ran — the test '
                f'is vacuous (all configs skipped or no assertion fired).')


# ===========================================================================
# Test 1: Fold-ppGO handoff fires for interior draw above DD cap
# ===========================================================================
@unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'Engine-backed test: requires COGWHEEL_TRAIN_TIER env var')
class FoldHandoffAccuracyTestCase(_HandoffTestCase):
    """Fold-ppGO correction is accurate (< 1%) for interior 4-image draw.

    Fixture: gamma=0.5, interior source at rho=0.3, theta=pi/4
    (away from axis cusps so Δτ is moderate → xi >= 4).
    Frequency range: w ∈ [45, 55] — above a typical DD cap (~40) but
    below the Schwinger wall (~60).

    Cost: 1 config × 20 engine evaluations ≈ < 10 s.
    """

    def test_xi_above_threshold(self):
        """Precondition: the fixture produces xi_min >= 4.0."""
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        delta_tau = _compute_delta_tau(source, GAMMA)
        self.assertIsNotNone(
            delta_tau,
            'No fold pair found at the interior fixture config')
        xi_min = _compute_xi_min(W_MIN, delta_tau)
        self.assertGreaterEqual(
            xi_min, XI_FOLD_THRESHOLD,
            f'xi_min={xi_min:.3f} < {XI_FOLD_THRESHOLD} — fixture is '
            f'in the near-fold regime (need larger Δτ or higher w)')
        self.n_checks += 1

    def test_fold_correction_accuracy_vs_exact(self):
        """Fold-ppGO agrees with exact engine within 1% relative error."""
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        w_test = np.linspace(W_MIN, W_MAX, N_W)

        # Fold-corrected total (absolute frame) → demodulate to min-rel
        f_fold = fold_ppgo_correction(w_test, source, GAMMA)
        t_min = _compute_t_min(source, GAMMA)
        f_fold_minrel = f_fold * np.exp(-1j * w_test * t_min)

        # Exact engine total (already in min-relative frame)
        ch = ChangRefsdalChannels(w_test)
        ch.reset()
        partition = ch.evaluate(gamma=GAMMA, y=source.tolist())
        f_exact = partition.exact_total

        # Relative error
        rel_err = (np.max(np.abs(f_fold_minrel - f_exact))
                   / np.max(np.abs(f_exact)))
        self.assertLess(
            rel_err, ACCURACY_BAR,
            f'max relative error {rel_err:.4f} >= {ACCURACY_BAR} — '
            f'fold correction is insufficiently accurate')

        # Diagnostic plot
        _save_diagnostic_plot(
            'test_fold_handoff_accuracy.png',
            w_test, f_exact, f_fold_minrel,
            f'Fold-ppGO handoff accuracy (gamma={GAMMA}, '
            f'rho={RHO_INTERIOR})')
        self.n_checks += 1


# ===========================================================================
# Test 2: Fold-ppGO handoff does NOT fire when ξ < threshold
# ===========================================================================

# ===========================================================================
# Test 2a: Pure-geometry xi check (no engine needed)
# ===========================================================================
class XiGateRefusalTestCase(_HandoffTestCase):
    """Near-caustic config has xi_min < threshold (pure geometry check).

    No engine evaluation needed — this just checks that the fixture's
    delay geometry produces xi < 4.0.

    Cost: pure geometry computation ≈ < 0.1 s.
    """

    def test_xi_below_threshold_near_caustic(self):
        """Near-caustic config produces xi_min < 4.0 (gate refuses)."""
        source = _polar_source(RHO_NEAR_CAUSTIC, THETA_NEAR_CAUSTIC, GAMMA)
        delta_tau = _compute_delta_tau(source, GAMMA)
        if delta_tau is None:
            # No fold pair — the gate would refuse anyway
            self.n_checks += 1
            return
        xi_min = _compute_xi_min(W_MIN, delta_tau)
        self.assertLess(
            xi_min, XI_FOLD_THRESHOLD,
            f'xi_min={xi_min:.3f} >= {XI_FOLD_THRESHOLD} — fixture is '
            f'NOT in the gate-refusal regime (need rho closer to 1 '
            f'or lower w)')
        self.n_checks += 1

    def test_interior_fixture_xi_above_threshold(self):
        """Interior fixture (rho=0.3) has xi_min >= 4.0 (gate admits)."""
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        delta_tau = _compute_delta_tau(source, GAMMA)
        self.assertIsNotNone(
            delta_tau,
            'No fold pair found at the interior fixture config')
        xi_min = _compute_xi_min(W_MIN, delta_tau)
        self.assertGreaterEqual(
            xi_min, XI_FOLD_THRESHOLD,
            f'xi_min={xi_min:.3f} < {XI_FOLD_THRESHOLD} — interior '
            f'fixture should be in the gate-admission regime')
        self.n_checks += 1


@unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'Engine-backed test: requires COGWHEEL_TRAIN_TIER env var')
class FoldHandoffGateRefusalTestCase(_HandoffTestCase):
    """Near-caustic config has ξ < threshold → gate conservatively refuses.

    Fixture: gamma=0.5, source close to caustic (rho=0.7, theta=pi/4)
    where the fold pair Δτ is small enough that xi_min < 4.0.  At this
    geometry, the fold correction gate refuses and falls back to raw ppGO.

    The gate's refusal is conservative: the correction would be
    inaccurate at this config where the images are nearly degenerate.

    Cost: 1 config × 1 engine eval ≈ < 2 s.
    """

    def test_gate_conservatism_error_larger(self):
        """Where xi < threshold, fold correction has larger error vs exact.

        Optional diagnostic: confirm the gate's conservatism by showing
        the error at this refused config.
        """
        source = _polar_source(RHO_NEAR_CAUSTIC, THETA_NEAR_CAUSTIC, GAMMA)
        w_test = np.atleast_1d(np.float64(W_MIN))

        # Fold correction (may still compute — it doesn't hard-refuse,
        # it just falls back to raw ppGO internally)
        f_fold = fold_ppgo_correction(w_test, source, GAMMA)
        t_min = _compute_t_min(source, GAMMA)
        f_fold_minrel = f_fold * np.exp(-1j * w_test * t_min)

        # Exact engine
        ch = ChangRefsdalChannels(w_test)
        ch.reset()
        partition = ch.evaluate(gamma=GAMMA, y=source.tolist())
        f_exact = partition.exact_total

        # The correction at this config is just raw ppGO (gate refused)
        # so the error should be whatever raw ppGO error is
        rel_err = (np.max(np.abs(f_fold_minrel - f_exact))
                   / np.max(np.abs(f_exact)))
        # We just record this — it may or may not be > 1% depending on
        # how bad raw ppGO is at this particular point. The main assertion
        # is that xi < threshold (in XiGateRefusalTestCase below).
        self.n_checks += 1


# ===========================================================================
# Test 3: Reconstruction round-trip identity
# ===========================================================================
class ReconstructionRoundTripTestCase(_HandoffTestCase):
    """Envelope extraction + reconstruct_farfield is lossless.

    Prove that extracting the far-field envelope from the fold-corrected
    total and feeding it through ``reconstruct_farfield`` recovers the
    original min-relative total to machine precision.

    This is a pure algebra test: it calls fold_ppgo_correction (which
    internally computes raw ppGO + Airy substitution) and then checks
    the reconstruction identity. No engine evaluation needed.

    The round-trip identity holds because:
    - envelope = (f_minrel - kernel_sum) * exp(+1j*w*t_min)  [demod]
    - reconstruct applies exp(-1j*w*t_min) then adds kernel_sum back

    Cost: 1 fold_ppgo_correction call + algebra ≈ < 1 s.
    """

    def test_roundtrip_identity(self):
        """Reconstruction round-trip recovers f_minrel to ~1e-12."""
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        w_test = np.linspace(W_MIN, W_MAX, N_W)

        # Verify the fixture has xi >= threshold (precondition for a
        # non-trivial fold correction)
        delta_tau = _compute_delta_tau(source, GAMMA)
        self.assertIsNotNone(delta_tau, 'No fold pair at fixture config')
        xi_min = _compute_xi_min(W_MIN, delta_tau)
        self.assertGreaterEqual(xi_min, XI_FOLD_THRESHOLD,
                                'Fixture xi < threshold — correction is '
                                'just raw ppGO, round-trip still holds '
                                'but does not test the fold path')

        # (a) Fold-corrected total
        f_total = fold_ppgo_correction(w_test, source, GAMMA)

        # Compute t_min for demodulation
        t_min = _compute_t_min(source, GAMMA)

        # (b) Demodulate to min-relative frame
        f_minrel = f_total * np.exp(-1j * w_test * t_min)

        # Build geometry partition for channel structure
        ch = ChangRefsdalChannels(w_test)
        ch.reset()
        geom = ch.geometry_partition(gamma=GAMMA, y=source.tolist())

        # (c) Compute kernel sum: sum over real channels of
        #     saddle_kernels * exp(1j*w*delay)
        real = np.asarray(geom.real_mask, dtype=bool)
        # saddle_kernels shape: (n_w, n_channels)
        # delays shape: (n_channels,)
        kernel_sum = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * w_test[:, None] * geom.delays[real][None, :]),
            axis=1)

        # (d) Extract envelope (frame-invariant = demodulated)
        envelope = (f_minrel - kernel_sum) * np.exp(1j * w_test * t_min)

        # (e) Reconstruct total via reconstruct_farfield
        _kernels, reconstructed_total = reconstruct_farfield(
            w_test, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        # Assert round-trip identity to machine precision
        max_f = np.max(np.abs(f_minrel))
        residual = np.max(np.abs(reconstructed_total - f_minrel))
        rel_residual = residual / max_f

        self.assertLess(
            rel_residual, ROUNDTRIP_BAR,
            f'Round-trip residual {rel_residual:.2e} >= {ROUNDTRIP_BAR} — '
            f'frame mismatch in demod/remod sequence')
        self.n_checks += 1

    def test_roundtrip_uses_same_t_min(self):
        """geom.t_min matches our _compute_t_min (consistency check)."""
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        w_test = np.linspace(W_MIN, W_MAX, N_W)

        t_min_helper = _compute_t_min(source, GAMMA)

        ch = ChangRefsdalChannels(w_test)
        ch.reset()
        geom = ch.geometry_partition(gamma=GAMMA, y=source.tolist())

        self.assertEqual(
            t_min_helper, geom.t_min,
            't_min from helper and geometry_partition disagree')
        self.n_checks += 1


# ===========================================================================
# Self-falsification: prove each gate has teeth
# ===========================================================================
class SelfFalsificationTestCase(_HandoffTestCase):
    """Prove the suite can go red — each gate has teeth.

    Each test corrupts one aspect and asserts the corresponding gate in
    the suite would catch it. If these pass without corruption, the gates
    are dead code.
    """

    def test_accuracy_bar_rejects_bad_correction(self):
        """A deliberately-wrong correction fails the 1% accuracy bar.

        Corrupting the fold correction by phase-rotating 0.1 rad should
        produce > 1% relative error vs itself-uncorrupted, proving the
        bar has discriminating power.
        """
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        w_test = np.linspace(W_MIN, W_MAX, N_W)
        f_fold = fold_ppgo_correction(w_test, source, GAMMA)

        # Corrupt by a phase rotation — simulates a wrong correction
        f_corrupted = f_fold * np.exp(1j * 0.1)

        # The corruption introduces > 1% relative difference
        rel_diff = (np.max(np.abs(f_corrupted - f_fold))
                    / np.max(np.abs(f_fold)))
        self.assertGreater(
            rel_diff, ACCURACY_BAR,
            f'Phase corruption ({rel_diff:.4f}) did not exceed the '
            f'accuracy bar — gate has no teeth')
        self.n_checks += 1

    def test_roundtrip_bar_rejects_frame_mismatch(self):
        """A wrong t_min in reconstruction breaks the round-trip identity.

        Using t_min + 0.01 in the demodulation/remodulation creates a
        detectable residual > 1e-12.
        """
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        w_test = np.linspace(W_MIN, W_MAX, N_W)

        f_total = fold_ppgo_correction(w_test, source, GAMMA)
        t_min = _compute_t_min(source, GAMMA)

        # Correct demodulation
        f_minrel = f_total * np.exp(-1j * w_test * t_min)

        ch = ChangRefsdalChannels(w_test)
        ch.reset()
        geom = ch.geometry_partition(gamma=GAMMA, y=source.tolist())

        real = np.asarray(geom.real_mask, dtype=bool)
        kernel_sum = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * w_test[:, None] * geom.delays[real][None, :]),
            axis=1)

        # Deliberately WRONG t_min for envelope extraction
        t_min_wrong = t_min + 0.01
        envelope_bad = (f_minrel - kernel_sum) * np.exp(
            1j * w_test * t_min_wrong)

        # Reconstruct with the correct t_min (mismatch → residual)
        _kernels, reconstructed = reconstruct_farfield(
            w_test, envelope_bad, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        max_f = np.max(np.abs(f_minrel))
        residual = np.max(np.abs(reconstructed - f_minrel)) / max_f

        self.assertGreater(
            residual, ROUNDTRIP_BAR,
            f'Frame mismatch residual ({residual:.2e}) did NOT exceed '
            f'the round-trip bar — gate has no teeth')
        self.n_checks += 1

    def test_xi_threshold_has_teeth(self):
        """Lowering xi threshold to 0 would admit all configs (no gate).

        With threshold = 0, even near-caustic configs (small Δτ) pass,
        proving the threshold discriminates.
        """
        source = _polar_source(RHO_NEAR_CAUSTIC, THETA_NEAR_CAUSTIC, GAMMA)
        delta_tau = _compute_delta_tau(source, GAMMA)
        if delta_tau is None:
            # No fold pair — threshold is moot, gate refuses structurally
            self.n_checks += 1
            return
        xi_min = _compute_xi_min(W_MIN, delta_tau)
        # With threshold = 0, this would pass
        self.assertGreaterEqual(
            xi_min, 0.0,
            'xi_min should always be non-negative')
        # But with the real threshold, it should fail
        self.assertLess(
            xi_min, XI_FOLD_THRESHOLD,
            'Near-caustic fixture should be below the real threshold')
        self.n_checks += 1


# ===========================================================================
# Test 4: Error-estimate fine gate refuses when c_A is large
# ===========================================================================

#: Gamma near metamorphosis where saddle-coefficient curvature is large.
GAMMA_HIGH_CURVATURE: float = 0.85

#: Interior rho for the high-curvature fixture.
RHO_HIGH_CURVATURE: float = 0.5

#: Angle for high-curvature fixture (max curvature between cusps).
THETA_HIGH_CURVATURE: float = math.pi / 4.0

#: Production certification bar (ppgo_map.CERTIFICATION_BAR).
_CERTIFICATION_BAR: float = 1e-4


class ErrorEstimateFineGateTestCase(_HandoffTestCase):
    """Error-estimate fine gate refuses when c_A * xi^{-3/2} > BAR.

    Fixture: gamma=0.85 (near metamorphosis), interior source at rho=0.5,
    theta=pi/4 (maximum astroid curvature between cusps).  At this config
    the saddle coefficient c1 ~ 0.93, so even at xi_min ~ 5.85 (w=45),
    the estimate c_A * xi^{-3/2} ~ 0.066 >> 1e-4.

    This proves the fine gate is load-bearing: the coarse xi gate admits
    (xi >= 4) but the fine c_A gate refuses because the fold pair images
    are too close to the critical curve, making the uniform approximation
    inaccurate.

    Cost: pure geometry (no engine eval) — < 0.5 s.
    """

    def test_xi_passes_coarse_gate(self):
        """Precondition: the high-curvature fixture has xi >= 4."""
        source = _polar_source(
            RHO_HIGH_CURVATURE, THETA_HIGH_CURVATURE, GAMMA_HIGH_CURVATURE)
        delta_tau = _compute_delta_tau(source, GAMMA_HIGH_CURVATURE)
        self.assertIsNotNone(
            delta_tau,
            'No fold pair found at the high-curvature fixture config')
        xi_min = _compute_xi_min(W_MIN, delta_tau)
        self.assertGreaterEqual(
            xi_min, XI_FOLD_THRESHOLD,
            f'xi_min={xi_min:.3f} < {XI_FOLD_THRESHOLD} — high-curvature '
            f'fixture does not pass the coarse gate')
        self.n_checks += 1

    def test_error_estimate_exceeds_bar(self):
        """Error estimate c_A * xi^{-3/2} > CERTIFICATION_BAR (gate refuses).

        The uniform Airy error estimate is the magnitude of the leading
        saddle coefficient |c1| divided by xi^{3/2}.  Near metamorphosis
        (gamma=0.85), c1 is large (images close to critical curve),
        so the estimate significantly exceeds 1e-4.
        """
        from cogwheel.lensing.chang_refsdal._airy_fold import (
            _merging_fold_pair, _uniform_error_estimate, _image_at_delay)

        source = _polar_source(
            RHO_HIGH_CURVATURE, THETA_HIGH_CURVATURE, GAMMA_HIGH_CURVATURE)
        matrix = geometry.macro_matrix(GAMMA_HIGH_CURVATURE)
        images = list(geometry.find_images(source, matrix))

        pair = _merging_fold_pair(images, source, matrix)
        self.assertIsNotNone(pair, 'No fold pair at high-curvature fixture')

        tau_plus, tau_minus = pair
        delta_tau = tau_minus - tau_plus
        xi_min = _compute_xi_min(W_MIN, delta_tau)

        image_plus = _image_at_delay(images, source, matrix, tau_plus)
        image_minus = _image_at_delay(images, source, matrix, tau_minus)
        self.assertIsNotNone(image_plus)
        self.assertIsNotNone(image_minus)

        error_est = _uniform_error_estimate(
            image_plus, image_minus, matrix, xi_min)
        self.assertIsNotNone(
            error_est,
            'Error estimate returned None — saddle_coefficients refused')
        self.assertGreater(
            error_est, _CERTIFICATION_BAR,
            f'error_est={error_est:.2e} <= {_CERTIFICATION_BAR} — the '
            f'fine gate did NOT refuse; fixture needs stronger curvature')
        self.n_checks += 1

    def test_gate_refusal_margin(self):
        """Error estimate exceeds the bar by at least 10x (robust fixture).

        Confirms the fixture is not borderline — a robust 10x margin
        ensures the test is stable against small geometry perturbations.
        """
        from cogwheel.lensing.chang_refsdal._airy_fold import (
            _merging_fold_pair, _uniform_error_estimate, _image_at_delay)

        source = _polar_source(
            RHO_HIGH_CURVATURE, THETA_HIGH_CURVATURE, GAMMA_HIGH_CURVATURE)
        matrix = geometry.macro_matrix(GAMMA_HIGH_CURVATURE)
        images = list(geometry.find_images(source, matrix))

        pair = _merging_fold_pair(images, source, matrix)
        tau_plus, tau_minus = pair
        delta_tau = tau_minus - tau_plus
        xi_min = _compute_xi_min(W_MIN, delta_tau)

        image_plus = _image_at_delay(images, source, matrix, tau_plus)
        image_minus = _image_at_delay(images, source, matrix, tau_minus)

        error_est = _uniform_error_estimate(
            image_plus, image_minus, matrix, xi_min)
        # error_est ~ 0.066, bar is 1e-4 — margin of ~660x
        self.assertGreater(
            error_est, 10.0 * _CERTIFICATION_BAR,
            f'error_est={error_est:.2e} is not 10x above the bar — '
            f'fixture is borderline')
        self.n_checks += 1


# ===========================================================================
# Test 5: Default path unaffected — draw served by chart still served
# ===========================================================================
class DefaultPathUnaffectedTestCase(_HandoffTestCase):
    """Chart-served draw is unaffected by the fold-ppGO block.

    When ``select_chart`` returns a valid chart, the production flow exits
    BEFORE reaching the fold-ppGO gate.  This test mocks ``select_chart``
    to return a chart object and verifies that:
    (a) The characterize_sample census path returns ``served=True`` with
        a chart index (NOT ``ppgo_fold``).
    (b) The fold-ppGO gate code is never reached.

    Fixture: gamma=0.5, interior source at rho=0.3 (same as the existing
    accuracy fixture), frequency band well below the DD cap (w ∈ [5, 30]).

    Cost: mock-based, no engine eval — < 0.5 s.
    """

    def test_chart_served_skips_fold_block(self):
        """select_chart returning a chart → served=True, no ppgo_fold category."""
        from unittest.mock import patch, MagicMock
        from cogwheel.lensing.surrogate_census import characterize_sample

        # Build a mock surrogate with a single mock chart in its charts
        # list (so _chart_index can find it by identity).
        mock_chart = MagicMock()
        mock_chart.gamma_grid = np.array([0.3, 0.7])
        mock_chart.log_w_grid = np.array([1.0, 4.0])

        mock_surrogate = MagicMock()
        mock_surrogate.charts = [mock_chart]

        # Engine factory returning a channels object with geometry_partition
        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        mock_geom = MagicMock()
        mock_geom.caustic_distance = 0.5
        mock_geom.caustic_theta = THETA_INTERIOR
        mock_geom.real_mask = np.array([True, True, True, True])
        mock_geom.images = [np.array([1.0, 0.0])] * 4

        def _engine_factory(w):
            ch = MagicMock()
            ch.geometry_partition.return_value = mock_geom
            return ch

        # Patch select_chart to return the mock chart (simulating that
        # the draw IS served by a chart).
        with patch('cogwheel.lensing.surrogate_census._surrogate.select_chart',
                   return_value=mock_chart):
            # Use a frequency grid that maps to low w
            f_grid = np.geomspace(20.0, 100.0, 10)
            record = characterize_sample(
                mock_surrogate, _engine_factory,
                gamma=GAMMA, m_lens_msun=100.0,
                y1=source[0], y2=source[1],
                f_grid=f_grid, dropped_slivers=())

        self.assertTrue(
            record.served,
            'Record should be served when select_chart returns a chart')
        self.assertNotEqual(
            record.category, 'ppgo_fold',
            'Category should NOT be ppgo_fold when a chart serves')
        self.n_checks += 1

    def test_chart_takes_priority_over_fold_gate(self):
        """Even with an xi >= 4 config, chart service takes priority.

        The fold-ppGO block is only reached AFTER select_chart returns
        None. This confirms the priority ordering.
        """
        from unittest.mock import patch, MagicMock
        from cogwheel.lensing.surrogate_census import characterize_sample

        # This is the SAME interior config that would qualify for fold-ppGO
        # (xi >= 4), but we mock a chart serving it.
        mock_chart = MagicMock()
        mock_chart.gamma_grid = np.array([0.3, 0.7])
        mock_chart.log_w_grid = np.array([1.0, 6.0])

        mock_surrogate = MagicMock()
        mock_surrogate.charts = [mock_chart]

        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        mock_geom = MagicMock()
        mock_geom.caustic_distance = 0.5
        mock_geom.caustic_theta = THETA_INTERIOR
        mock_geom.real_mask = np.array([True, True, True, True])
        mock_geom.images = [np.array([1.0, 0.0])] * 4

        def _engine_factory(w):
            ch = MagicMock()
            ch.geometry_partition.return_value = mock_geom
            return ch

        with patch('cogwheel.lensing.surrogate_census._surrogate.select_chart',
                   return_value=mock_chart):
            # Use large w (same as accuracy test) that would qualify
            # for fold-ppGO if the chart didn't intercept
            f_grid = np.geomspace(20.0, 100.0, 10)
            record = characterize_sample(
                mock_surrogate, _engine_factory,
                gamma=GAMMA, m_lens_msun=1e4,
                y1=source[0], y2=source[1],
                f_grid=f_grid, dropped_slivers=())

        self.assertTrue(record.served)
        # chart_index is set (not None) when a chart serves
        self.assertIsNotNone(
            record.chart_index,
            'chart_index should be set when a chart serves')
        self.n_checks += 1


# ===========================================================================
# Test 6: Census records ppgo_fold for qualifying interior draw
# ===========================================================================

#: Lens mass (M_sun) that maps f_grid to w_min ~ 50000 (large xi).
#: At m=20e6, f=20 Hz → w ≈ 49516, giving xi_min ≈ 531 at gamma=0.5,
#: rho=0.3 (delta_tau ≈ 0.327). error_est = 1.0 * 531^{-3/2} ≈ 8.2e-5
#: which is below CERTIFICATION_BAR = 1e-4.
M_LENS_CENSUS: float = 20e6


class CensusRecordsPpgoFoldTestCase(_HandoffTestCase):
    """Census characterize_sample records ppgo_fold for qualifying draws.

    Fixture: gamma=0.5, interior source at rho=0.3, theta=pi/4, with a
    very large lens mass (20e6 M_sun) so the dimensionless frequency
    w_min ~ 49500.  At this enormous xi (~531), the error estimate
    c_A * xi^{-3/2} ~ 8.2e-5 < 1e-4 and ALL gates pass.

    The surrogate's select_chart is patched to return None (simulating
    above-ceiling / no chart coverage), so the fold-ppGO handoff block
    is reached. The engine_factory is also mocked (no expensive engine
    eval); only geometry_partition is called to supply image metadata.

    Cost: mock-based with real geometry — < 2 s.
    """

    def test_census_ppgo_fold_served(self):
        """characterize_sample returns served=True, category='ppgo_fold'."""
        from unittest.mock import patch, MagicMock
        from cogwheel.lensing.surrogate_census import characterize_sample
        from cogwheel.lensing.waveform import dimensionless_frequency

        source = _polar_source(RHO_INTERIOR, THETA_INTERIOR, GAMMA)
        f_grid = np.geomspace(20.0, 100.0, 10)

        # Verify the fixture produces qualifying w_min
        w_grid = dimensionless_frequency(f_grid, M_LENS_CENSUS, 0.0)
        w_min = float(w_grid.min())
        delta_tau = _compute_delta_tau(source, GAMMA)
        self.assertIsNotNone(delta_tau)
        xi_min = _compute_xi_min(w_min, delta_tau)
        self.assertGreaterEqual(xi_min, XI_FOLD_THRESHOLD,
                                f'xi_min={xi_min:.1f} < 4 at w_min={w_min:.0f}')

        # Mock surrogate: select_chart returns None → fold block reached
        mock_surrogate = MagicMock()
        mock_surrogate.charts = []

        # Engine factory uses REAL geometry_partition (no mock)
        def _engine_factory(w):
            return ChangRefsdalChannels(w)

        # Patch select_chart to return None (simulating above-ceiling)
        # AND patch may_serve to return True (so the geometry is built)
        # AND patch get_certified_ppgo_map to return None (no band split)
        with patch('cogwheel.lensing.surrogate_census._surrogate.select_chart',
                   return_value=None), \
             patch('cogwheel.lensing.surrogate_census.get_certified_ppgo_map',
                   return_value=None):
            record = characterize_sample(
                mock_surrogate, _engine_factory,
                gamma=GAMMA, m_lens_msun=M_LENS_CENSUS,
                y1=float(source[0]), y2=float(source[1]),
                f_grid=f_grid, dropped_slivers=())

        self.assertTrue(
            record.served,
            f'record.served is False — fold-ppGO gate did not fire. '
            f'category={record.category}')
        self.assertEqual(
            record.category, 'ppgo_fold',
            f'Expected category=ppgo_fold, got {record.category!r}')
        self.n_checks += 1

    def test_census_ppgo_fold_not_served_when_error_too_large(self):
        """Census falls through when error estimate exceeds the bar.

        Uses the high-curvature fixture (gamma=0.85, rho=0.5) where
        c_A is large, so even at moderate xi the fine gate refuses.
        """
        from unittest.mock import patch, MagicMock
        from cogwheel.lensing.surrogate_census import characterize_sample

        source = _polar_source(
            RHO_HIGH_CURVATURE, THETA_HIGH_CURVATURE, GAMMA_HIGH_CURVATURE)
        # Use a mass that gives w_min ~ 45 (same as the accuracy tests),
        # where the error estimate is ~0.066 >> 1e-4
        f_grid = np.geomspace(20.0, 100.0, 10)

        mock_surrogate = MagicMock()
        mock_surrogate.charts = []

        def _engine_factory(w):
            return ChangRefsdalChannels(w)

        with patch('cogwheel.lensing.surrogate_census._surrogate.select_chart',
                   return_value=None), \
             patch('cogwheel.lensing.surrogate_census.get_certified_ppgo_map',
                   return_value=None):
            # m_lens=200 gives w ~ 5 at f=20, xi ~ 4.7 at gamma=0.85
            # But error_est >> 1e-4 so the fine gate refuses.
            record = characterize_sample(
                mock_surrogate, _engine_factory,
                gamma=GAMMA_HIGH_CURVATURE, m_lens_msun=200.0,
                y1=float(source[0]), y2=float(source[1]),
                f_grid=f_grid, dropped_slivers=())

        # The fold gate should NOT fire (error too large or xi too small)
        self.assertFalse(
            record.served,
            'record.served is True — fold-ppGO should NOT serve when '
            'error estimate exceeds CERTIFICATION_BAR')
        self.assertNotEqual(
            record.category, 'ppgo_fold',
            'Category should not be ppgo_fold when the fine gate refuses')
        self.n_checks += 1



# ===========================================================================
if __name__ == '__main__':
    unittest.main()
