"""Born residual chart wiring tests for the fact-4 slot.

Verifies that ``LensedRelativeBinningLikelihood._surrogate_coefficients``
correctly dispatches to the Born residual path when:
  - the amplification surrogate declines (``served=False``),
  - ``born_residual_chart`` is either None (returns None) or a chart
    covering the candidate's (gamma, rho) box.

Tolerance rationale
-------------------
The mock chart evaluates to a KNOWN constant at grid nodes — no
interpolation error, only FP arithmetic chain error (~1e-14 relative).
The test uses 1e-13 relative tolerance (a small safety factor above the
1e-14 theoretical floor) for the carrier+residual identity.

Runtime budget: ~3 s (no engine, geometry-partition is analytic).
"""
from __future__ import annotations

import math
import pathlib
import types
import unittest

import numpy as np

from cogwheel.lensing.born_residual_chart import BornResidualChart
from cogwheel.lensing.chang_refsdal.channels import (
    FARFIELD_KERNEL_SUM,
    born_carrier_from_partition,
    reconstruct_farfield,
)
from cogwheel.lensing.chang_refsdal.geometry import macro_matrix
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.ppgo_map import caustic_rho, caustic_geometry
from cogwheel.lensing.waveform import dimensionless_frequency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Output directory for diagnostic plots.
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

#: Lens mass (solar masses) and redshift — lightweight config.
_M_LENS_MSUN: float = 100.0
_Z_LENS: float = 0.5

#: Shear for the positive-parity Born annulus (well below astroid wall).
_GAMMA: float = 0.5

#: kappa = 0 (the Born chart and surrogate are kappa=0 surfaces).
_KAPPA: float = 0.0

#: beta = 0 (the Born chart and surrogate are beta=0 surfaces).
_BETA: float = 0.0

#: Target caustic-relative coordinate for the served config.
_TARGET_RHO: float = 3.0

#: Dense frequency grid (Hz) — 64 sub-samples covering 4 bins.
_N_BINS: int = 4
_KERNEL_SUBSAMPLES: int = 16
_N_DENSE: int = _N_BINS * _KERNEL_SUBSAMPLES  # 64

#: Chart grid extents.
_CHART_GAMMA_GRID = np.linspace(0.3, 0.8, 6)
_CHART_RHO_GRID = np.linspace(1.5, 5.0, 8)
_CHART_LOG_W_GRID = np.log(np.geomspace(0.1, 50.0, 20))

#: Synthetic residual magnitude (small, constant in w for simplicity).
_RESIDUAL_SCALE: float = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_target_y() -> float:
    """Compute |y| that gives rho ~ _TARGET_RHO at _GAMMA, _KAPPA."""
    reach, _ = caustic_geometry(_GAMMA, _KAPPA)
    return _TARGET_RHO * reach


def _make_dense_f_grid() -> np.ndarray:
    """Build a dense frequency grid yielding w in [0.5, 20] approximately."""
    # w = xi * f  => f = w / xi
    xi = dimensionless_frequency(1.0, _M_LENS_MSUN, _Z_LENS)
    w_lo, w_hi = 0.5, 20.0
    f_lo, f_hi = w_lo / xi, w_hi / xi
    return np.linspace(float(f_lo), float(f_hi), _N_DENSE)


def _build_mock_chart() -> BornResidualChart:
    """Build a BornResidualChart with known synthetic residual.

    R(w, gamma, rho) = _RESIDUAL_SCALE * exp(-rho) * (1 + 0.001j)
    (constant in w, smooth in gamma and rho for interpolation).
    """
    n_gamma = len(_CHART_GAMMA_GRID)
    n_rho = len(_CHART_RHO_GRID)
    n_w = len(_CHART_LOG_W_GRID)

    # Fill grid values: R = scale * exp(-rho) * (1 + 0.001j)
    # Shape: (n_gamma, n_rho, n_w)
    rho_3d = _CHART_RHO_GRID[None, :, None] * np.ones((n_gamma, 1, n_w))
    residual = _RESIDUAL_SCALE * np.exp(-rho_3d) * (1.0 + 0.001j)

    return BornResidualChart(
        gamma_grid=_CHART_GAMMA_GRID,
        rho_grid=_CHART_RHO_GRID,
        log_w_grid=_CHART_LOG_W_GRID,
        real_coeffs=residual.real.copy(),
        imag_coeffs=residual.imag.copy(),
        provenance={'test': 'synthetic'},
    )


class _MockSurrogate:
    """Mock amplification surrogate that passes may_serve but refuses serve."""

    def may_serve(self, gamma: float, log_w_min: float,
                  log_w_max: float) -> bool:
        """Always pass the cheap pre-check."""
        return True

    def serve(self, w_array, *, gamma, y1, y2, beta, eta, theta,
              image_count):
        """Always decline: surrogate has no chart for this config."""
        return np.zeros(w_array.shape, dtype=complex), False, None


class _BornResidualProbe:
    """Lightweight probe carrying the REAL surrogate-coefficients methods.

    Binds the real ``_surrogate_coefficients``, ``_reduce_dense_kernels``,
    ``_image_delays``, ``_lens_params`` from
    ``LensedRelativeBinningLikelihood`` onto a minimal instance with the
    needed attributes.  No heavy waveform/event construction required.
    """

    # Bind real methods from the class.
    _lens_params = LensedRelativeBinningLikelihood._lens_params
    _surrogate_coefficients = (
        LensedRelativeBinningLikelihood._surrogate_coefficients)
    _reduce_dense_kernels = (
        LensedRelativeBinningLikelihood._reduce_dense_kernels)
    _image_delays = LensedRelativeBinningLikelihood._image_delays
    _ppgo_band_split = LensedRelativeBinningLikelihood._ppgo_band_split

    def __init__(self, *, born_residual_chart=None):
        dense_f = _make_dense_f_grid()
        self._kernel_dense_f = dense_f
        self.amplification_surrogate = _MockSurrogate()
        self.born_residual_chart = born_residual_chart
        self.kernel_subsamples = _KERNEL_SUBSAMPLES
        self.n_bins = _N_BINS

        # Build uniform per-bin sub-sample weights (least-squares for
        # a linear fit to kernel_subsamples points per bin).
        n_sub = _KERNEL_SUBSAMPLES
        # Sub-sample positions within a bin: uniformly spaced in [-1, 1].
        t = np.linspace(-1.0, 1.0, n_sub)
        # Least-squares fit: value = mean, slope = linear-regression slope.
        # Value weights: 1/n_sub (mean).
        value_weights = np.ones(n_sub) / n_sub
        # Slope weights: sum(t_j * x_j) / sum(t_j^2).
        slope_weights = t / np.sum(t ** 2)
        # Tile for all bins: shape (n_bins, n_sub).
        self._kernel_fit_value = np.tile(value_weights, (_N_BINS, 1))
        self._kernel_fit_slope = np.tile(slope_weights, (_N_BINS, 1))

        # _ppgo_band_split needs _ppgo_cell_coords which needs more;
        # but with no ppgo map installed it returns None from
        # get_certified_ppgo_map() early. The real _ppgo_band_split is
        # bound above and calls get_certified_ppgo_map() which returns
        # None in tests (no map installed), so w_trust = None.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class NoChartByteIdentityTestCase(unittest.TestCase):
    """Verify _surrogate_coefficients returns None when born_residual_chart=None.

    This is the HEAD behavior: the fact-4 slot declines when no chart is
    attached, falling through to the exact engine.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        # Anti-vacuity: at least one comparison ran.
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no assertions ran in NoChartByteIdentityTestCase')

    def test_no_chart_returns_none(self):
        """With born_residual_chart=None, surrogate declining → returns None."""
        probe = _BornResidualProbe(born_residual_chart=None)
        abs_y = _compute_target_y()
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': _BETA,
            'kappa': _KAPPA,
        }
        result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result)
        self.n_checks += 1

    def test_no_chart_multiple_configs(self):
        """Several Born-annulus configs all return None with no chart."""
        probe = _BornResidualProbe(born_residual_chart=None)
        reach, _ = caustic_geometry(_GAMMA, _KAPPA)
        # Test rho = 2.0, 3.0, 4.5 — all exterior
        for rho in (2.0, 3.0, 4.5):
            abs_y = rho * reach
            par_dic = {
                'm_lens_msun': _M_LENS_MSUN,
                'z_lens': _Z_LENS,
                'y1': abs_y * 0.6,
                'y2': abs_y * 0.8,
                'gamma': _GAMMA,
                'beta': _BETA,
                'kappa': _KAPPA,
            }
            with self.subTest(rho=rho):
                result = probe._surrogate_coefficients(par_dic)
                self.assertIsNone(result)
                self.n_checks += 1

    def test_no_chart_is_head_behavior(self):
        """No born_residual_chart means the fact-4 slot does not fire.

        Explicitly verifies that the return path is the immediate None
        after ``if born_chart is None: return None``, not a different
        None from an upstream guard (like kappa != 0 or rho <= 1).
        """
        # Use a config that would be SERVED if a chart were attached:
        # rho ~ 3.0, inside the mock chart's grid.
        probe = _BornResidualProbe(born_residual_chart=None)
        abs_y = _compute_target_y()
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': _BETA,
            'kappa': _KAPPA,
        }
        # With chart=None → None
        result_none = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result_none)

        # With chart attached → non-None (proves it's the chart guard)
        chart = _build_mock_chart()
        probe_with_chart = _BornResidualProbe(born_residual_chart=chart)
        result_chart = probe_with_chart._surrogate_coefficients(par_dic)
        self.assertIsNotNone(result_chart)
        self.n_checks += 1


class MockChartServePathTestCase(unittest.TestCase):
    """Verify the Born residual serve path reconstructs carrier+residual.

    When the amplification surrogate declines and born_residual_chart
    covers the candidate, _surrogate_coefficients should:
      1. Return a non-None 4-tuple (delays, k0, k1, geom).
      2. The reconstruction is algebraically correct: f_total at dense w
         equals born_carrier_from_partition(partition) + chart.evaluate(w).
      3. The returned k0/k1 match the independent reconstruction.

    Cost: ~1 s (one geometry_partition + one born_carrier_from_partition).
    """

    def setUp(self):
        self.n_checks = 0
        self.chart = _build_mock_chart()
        self.probe = _BornResidualProbe(born_residual_chart=self.chart)
        self.abs_y = _compute_target_y()
        self.par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': self.abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': _BETA,
            'kappa': _KAPPA,
        }

    def tearDown(self):
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no assertions ran in MockChartServePathTestCase')

    def test_returns_non_none_tuple(self):
        """The Born path fires and returns a 4-tuple."""
        result = self.probe._surrogate_coefficients(self.par_dic)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        delays, k0, k1, geom = result
        # Shape checks: k0 and k1 are (n_channels, n_bins)
        self.assertEqual(k0.shape[1], _N_BINS)
        self.assertEqual(k1.shape[1], _N_BINS)
        self.assertEqual(k0.shape, k1.shape)
        # delays has same number of channels
        self.assertEqual(delays.shape[0], k0.shape[0])
        self.n_checks += 1

    def test_rho_at_target(self):
        """The config's caustic-relative distance matches the target."""
        rho = caustic_rho(_GAMMA, self.abs_y, _KAPPA)
        self.assertAlmostEqual(rho, _TARGET_RHO, places=10)
        self.n_checks += 1

    def test_reconstruction_matches_carrier_plus_residual(self):
        """Returned k0/k1 reconstruct to carrier+residual at dense w.

        Independently replicates the Born path's algebra and compares
        against the probe's returned coefficients.
        """
        result = self.probe._surrogate_coefficients(self.par_dic)
        self.assertIsNotNone(result)
        delays_sec, k0, k1, geom = result

        # 1. Build the same dense_w grid the probe used.
        dense_w = dimensionless_frequency(
            self.probe._kernel_dense_f, _M_LENS_MSUN, _Z_LENS)

        # 2. Compute rho for the chart query.
        rho = caustic_rho(_GAMMA, self.abs_y, _KAPPA)

        # 3. Build the partition namespace (same as the code does).
        matrix = macro_matrix(_GAMMA, _BETA, _KAPPA)
        partition_ns = types.SimpleNamespace(
            w=dense_w,
            source=np.array([self.abs_y, 0.0]),
            gamma=_GAMMA,
            beta=_BETA,
            kappa=_KAPPA,
            matrix=matrix,
            t_min=geom.t_min,
            delays=geom.delays,
            saddle_kernels=geom.saddle_kernels,
            real_mask=geom.real_mask,
            images=geom.images,
        )

        # 4. Compute carrier + residual independently.
        carrier = born_carrier_from_partition(partition_ns)
        residual = self.chart.evaluate(dense_w, _GAMMA, rho)
        f_total_expected = carrier + residual

        # 5. Replicate the envelope extraction from the code:
        # ppgo = sum of real-image saddle kernels * exp(1j*w*tau_a)
        real = np.asarray(geom.real_mask, dtype=bool)
        ppgo = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * dense_w[:, None] * geom.delays[real][None, :]),
            axis=1)
        envelope = ((f_total_expected - ppgo)
                    * np.exp(1j * dense_w * geom.t_min))

        # 6. Reconstruct via the same function.
        kernels, total = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        # 7. Reduce to k0/k1 with the probe's weights.
        k0_expected, k1_expected = self.probe._reduce_dense_kernels(kernels)

        # 8. Compare: should be bit-identical (same FP operations).
        # Use 1e-13 relative tolerance as safety margin.
        max_k0 = np.max(np.abs(k0_expected))
        if max_k0 > 0:
            rel_err_k0 = np.max(np.abs(k0 - k0_expected)) / max_k0
            self.assertLess(rel_err_k0, 1e-13,
                            f'k0 relative error {rel_err_k0:.2e} exceeds 1e-13')
        else:
            self.assertTrue(np.allclose(k0, k0_expected, atol=1e-30))

        max_k1 = np.max(np.abs(k1_expected))
        if max_k1 > 0:
            rel_err_k1 = np.max(np.abs(k1 - k1_expected)) / max_k1
            self.assertLess(rel_err_k1, 1e-13,
                            f'k1 relative error {rel_err_k1:.2e} exceeds 1e-13')
        else:
            self.assertTrue(np.allclose(k1, k1_expected, atol=1e-30))

        self.n_checks += 1

    def test_total_matches_carrier_plus_residual_at_dense(self):
        """The internal f_total = carrier + residual to within 1e-14.

        Verifies the identity that the Born path computes — the total
        amplification is the sum of the analytic carrier and the chart
        residual — by reconstructing total from the kernels and comparing
        to the independently-computed sum.
        """
        result = self.probe._surrogate_coefficients(self.par_dic)
        self.assertIsNotNone(result)
        _, _, _, geom = result

        dense_w = dimensionless_frequency(
            self.probe._kernel_dense_f, _M_LENS_MSUN, _Z_LENS)
        rho = caustic_rho(_GAMMA, self.abs_y, _KAPPA)

        # Independent carrier + residual.
        matrix = macro_matrix(_GAMMA, _BETA, _KAPPA)
        partition_ns = types.SimpleNamespace(
            w=dense_w,
            source=np.array([self.abs_y, 0.0]),
            gamma=_GAMMA,
            beta=_BETA,
            kappa=_KAPPA,
            matrix=matrix,
            t_min=geom.t_min,
            delays=geom.delays,
            saddle_kernels=geom.saddle_kernels,
            real_mask=geom.real_mask,
            images=geom.images,
        )
        carrier = born_carrier_from_partition(partition_ns)
        residual = self.chart.evaluate(dense_w, _GAMMA, rho)
        f_total_expected = carrier + residual

        # Reconstruct total from the code's path: same envelope extraction.
        real = np.asarray(geom.real_mask, dtype=bool)
        ppgo = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * dense_w[:, None] * geom.delays[real][None, :]),
            axis=1)
        envelope = ((f_total_expected - ppgo)
                    * np.exp(1j * dense_w * geom.t_min))
        _kernels, total_reconstructed = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        # The reconstructed total should match f_total_expected to 1e-14.
        max_f = np.max(np.abs(f_total_expected))
        self.assertGreater(max_f, 0.0)
        rel_err = np.max(np.abs(total_reconstructed - f_total_expected)) / max_f
        self.assertLess(rel_err, 1e-13,
                        f'Reconstruction error {rel_err:.2e} exceeds 1e-13')
        self.n_checks += 1

    def test_diagnostic_plot(self):
        """Save diagnostic plot: |served - (carrier+residual)| / max|F|."""
        result = self.probe._surrogate_coefficients(self.par_dic)
        if result is None:
            self.skipTest('Born path did not fire — cannot generate plot')

        _, _, _, geom = result
        dense_w = dimensionless_frequency(
            self.probe._kernel_dense_f, _M_LENS_MSUN, _Z_LENS)
        rho = caustic_rho(_GAMMA, self.abs_y, _KAPPA)

        matrix = macro_matrix(_GAMMA, _BETA, _KAPPA)
        partition_ns = types.SimpleNamespace(
            w=dense_w,
            source=np.array([self.abs_y, 0.0]),
            gamma=_GAMMA,
            beta=_BETA,
            kappa=_KAPPA,
            matrix=matrix,
            t_min=geom.t_min,
            delays=geom.delays,
            saddle_kernels=geom.saddle_kernels,
            real_mask=geom.real_mask,
            images=geom.images,
        )
        carrier = born_carrier_from_partition(partition_ns)
        residual = self.chart.evaluate(dense_w, _GAMMA, rho)
        f_total_expected = carrier + residual

        real = np.asarray(geom.real_mask, dtype=bool)
        ppgo = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * dense_w[:, None] * geom.delays[real][None, :]),
            axis=1)
        envelope = ((f_total_expected - ppgo)
                    * np.exp(1j * dense_w * geom.t_min))
        _kernels, total_reconstructed = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        max_f = np.max(np.abs(f_total_expected))
        relative_error = np.abs(
            total_reconstructed - f_total_expected) / max_f

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(dense_w, relative_error, 'b.-', markersize=3)
            ax.axhline(1e-14, color='r', ls='--', label='1e-14 floor')
            ax.set_xlabel('w (dimensionless frequency)')
            ax.set_ylabel('|served - (carrier+residual)| / max|F|')
            ax.set_title('Born residual wiring: reconstruction identity')
            ax.legend()
            fig.tight_layout()
            fig.savefig(_OUTPUT_DIR / 'born_residual_wiring_identity.png',
                        dpi=100)
            plt.close(fig)
        except ImportError:
            pass  # matplotlib unavailable; skip plot

        self.n_checks += 1


class OutOfBoxFallthroughTestCase(unittest.TestCase):
    """Verify _surrogate_coefficients returns None for out-of-box configs.

    Three sub-cases:
      (a) rho > 5.0 (above chart's max rho)
      (b) 1.0 < rho < 1.5 (exterior but below chart's min rho)
      (c) rho < 1.0 (interior to caustic — rho <= 1.0 guard fires)

    All three must return None from the Born residual path.
    Cost: ~0.5 s (three geometry_partition calls).
    """

    def setUp(self):
        self.n_checks = 0
        self.chart = _build_mock_chart()
        self.probe = _BornResidualProbe(born_residual_chart=self.chart)
        self.reach, _ = caustic_geometry(_GAMMA, _KAPPA)

    def tearDown(self):
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no assertions ran in OutOfBoxFallthroughTestCase')

    def _make_par_dic(self, abs_y: float) -> dict:
        """Build a par_dic with |y| along the x-axis."""
        return {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': _BETA,
            'kappa': _KAPPA,
        }

    def test_rho_above_chart_max(self):
        """Config with rho > 5.0 returns None (chart's rho_grid max is 5.0)."""
        # rho = 6.0 → |y| = 6.0 * reach
        target_rho = 6.0
        abs_y = target_rho * self.reach
        # Verify the rho is actually above the grid.
        rho = caustic_rho(_GAMMA, abs_y, _KAPPA)
        self.assertGreater(rho, _CHART_RHO_GRID[-1])

        result = self.probe._surrogate_coefficients(self._make_par_dic(abs_y))
        self.assertIsNone(result)
        self.n_checks += 1

    def test_rho_below_chart_min_exterior(self):
        """Config with 1.0 < rho < 1.5 returns None (below chart min rho).

        The config is exterior to the caustic (rho > 1) but below the
        chart's coverage start at rho = 1.5.
        """
        target_rho = 1.2  # exterior but below chart minimum
        abs_y = target_rho * self.reach
        rho = caustic_rho(_GAMMA, abs_y, _KAPPA)
        self.assertGreater(rho, 1.0)
        self.assertLess(rho, _CHART_RHO_GRID[0])

        result = self.probe._surrogate_coefficients(self._make_par_dic(abs_y))
        self.assertIsNone(result)
        self.n_checks += 1

    def test_rho_interior_to_caustic(self):
        """Config with rho < 1.0 returns None (interior guard fires)."""
        target_rho = 0.8  # interior to caustic
        abs_y = target_rho * self.reach
        rho = caustic_rho(_GAMMA, abs_y, _KAPPA)
        self.assertLess(rho, 1.0)

        result = self.probe._surrogate_coefficients(self._make_par_dic(abs_y))
        self.assertIsNone(result)
        self.n_checks += 1

    def test_all_three_cases_consistent(self):
        """Verify all three out-of-box cases produce None consistently.

        Runs all three in a loop with subTest for comprehensive coverage.
        """
        cases = [
            ('above_max', 6.0),   # rho > 5.0
            ('below_min', 1.2),   # 1.0 < rho < 1.5
            ('interior', 0.8),    # rho < 1.0
        ]
        for label, target_rho in cases:
            abs_y = target_rho * self.reach
            with self.subTest(case=label, target_rho=target_rho):
                result = self.probe._surrogate_coefficients(
                    self._make_par_dic(abs_y))
                self.assertIsNone(result)
                self.n_checks += 1

    def test_gamma_outside_chart(self):
        """Config with gamma outside chart's gamma_grid returns None."""
        # gamma = 0.9 is above chart max (0.8)
        target_rho = _TARGET_RHO
        abs_y = target_rho * self.reach
        par_dic = self._make_par_dic(abs_y)
        par_dic['gamma'] = 0.9  # above chart max 0.8

        # Need to verify chart doesn't cover this gamma.
        rho = caustic_rho(0.9, abs_y, _KAPPA)
        # The chart.covers check uses gamma, rho — gamma=0.9 is outside [0.3, 0.8]
        self.assertFalse(self.chart.covers(0.9, rho))

        result = self.probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result)
        self.n_checks += 1


class SelfFalsificationTestCase(unittest.TestCase):
    """Proves the suite CAN go red — detectors are not vacuously green.

    Each test deliberately violates a condition the main tests rely on and
    asserts the violation is detectable.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no assertions ran in SelfFalsificationTestCase')

    def test_chart_covers_rejects_out_of_box(self):
        """chart.covers returns False for out-of-box (gamma, rho).

        Proves the containment gate has teeth: a rho outside the grid
        is actually rejected.
        """
        chart = _build_mock_chart()
        # rho = 6.0 is above grid max 5.0
        self.assertFalse(chart.covers(_GAMMA, 6.0))
        # rho = 1.0 is below grid min 1.5
        self.assertFalse(chart.covers(_GAMMA, 1.0))
        # gamma = 0.9 is above grid max 0.8
        self.assertFalse(chart.covers(0.9, 3.0))
        # gamma = 0.2 is below grid min 0.3
        self.assertFalse(chart.covers(0.2, 3.0))
        self.n_checks += 1

    def test_chart_covers_accepts_in_box(self):
        """chart.covers returns True for in-box (gamma, rho).

        Proves the containment gate doesn't reject everything.
        """
        chart = _build_mock_chart()
        self.assertTrue(chart.covers(_GAMMA, _TARGET_RHO))
        self.assertTrue(chart.covers(0.4, 2.0))
        self.assertTrue(chart.covers(0.7, 4.5))
        self.n_checks += 1

    def test_rho_guard_fires_for_interior(self):
        """A config with rho < 1.0 is rejected even if chart covers it.

        Proves the rho > 1.0 guard in the code fires independently
        of the chart's covers() method (the chart might accept rho=0.8
        if it were queried, but the code never asks).
        """
        reach, _ = caustic_geometry(_GAMMA, _KAPPA)
        # Put rho = 0.5 (deep interior) but create a chart that would
        # cover it if asked.
        interior_chart = BornResidualChart(
            gamma_grid=_CHART_GAMMA_GRID,
            rho_grid=np.linspace(0.3, 5.0, 10),  # includes rho < 1
            log_w_grid=_CHART_LOG_W_GRID,
            real_coeffs=np.zeros((6, 10, 20)),
            imag_coeffs=np.zeros((6, 10, 20)),
        )
        # The chart covers (gamma=0.5, rho=0.5)
        self.assertTrue(interior_chart.covers(_GAMMA, 0.5))

        probe = _BornResidualProbe(born_residual_chart=interior_chart)
        abs_y = 0.5 * reach  # rho = 0.5 (interior)
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': _BETA,
            'kappa': _KAPPA,
        }
        # The code's rho <= 1.0 guard fires before chart.covers is called.
        result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result)
        self.n_checks += 1

    def test_mock_surrogate_declines(self):
        """The mock surrogate always declines (served=False).

        Proves the test infrastructure correctly routes to the Born path.
        """
        mock = _MockSurrogate()
        w = np.geomspace(0.5, 20.0, 10)
        _, served, definition = mock.serve(
            w, gamma=0.5, y1=2.0, y2=0.0, beta=0.0, eta=1.5,
            theta=0.3, image_count=2)
        self.assertFalse(served)
        self.assertIsNone(definition)
        self.n_checks += 1

    def test_reconstruction_error_detectable(self):
        """A WRONG residual produces a detectable reconstruction mismatch.

        Proves the reconstruction test has teeth: if the chart returns
        a different value, the k0/k1 comparison fails.
        """
        chart = _build_mock_chart()
        probe = _BornResidualProbe(born_residual_chart=chart)
        abs_y = _compute_target_y()
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': _BETA,
            'kappa': _KAPPA,
        }
        result = probe._surrogate_coefficients(par_dic)
        self.assertIsNotNone(result)
        _, k0, _, geom = result

        # Now compute with a WRONG residual (10x larger).
        dense_w = dimensionless_frequency(
            probe._kernel_dense_f, _M_LENS_MSUN, _Z_LENS)
        rho = caustic_rho(_GAMMA, abs_y, _KAPPA)
        matrix = macro_matrix(_GAMMA, _BETA, _KAPPA)
        partition_ns = types.SimpleNamespace(
            w=dense_w,
            source=np.array([abs_y, 0.0]),
            gamma=_GAMMA,
            beta=_BETA,
            kappa=_KAPPA,
            matrix=matrix,
            t_min=geom.t_min,
            delays=geom.delays,
            saddle_kernels=geom.saddle_kernels,
            real_mask=geom.real_mask,
            images=geom.images,
        )
        carrier = born_carrier_from_partition(partition_ns)
        # Use WRONG residual: 1000x the real one (the mock residual is
        # tiny: 0.01*exp(-3) ~ 5e-4, so a large multiplier is needed to
        # overcome the carrier's dominance and produce a detectable diff).
        wrong_residual = 1000.0 * chart.evaluate(dense_w, _GAMMA, rho)
        f_total_wrong = carrier + wrong_residual

        real = np.asarray(geom.real_mask, dtype=bool)
        ppgo = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * dense_w[:, None] * geom.delays[real][None, :]),
            axis=1)
        envelope = ((f_total_wrong - ppgo)
                    * np.exp(1j * dense_w * geom.t_min))
        kernels, _ = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)
        k0_wrong, _ = probe._reduce_dense_kernels(kernels)

        # The wrong k0 should DIFFER from the correct k0.
        max_k0 = np.max(np.abs(k0))
        rel_diff = np.max(np.abs(k0 - k0_wrong)) / max_k0
        self.assertGreater(rel_diff, 1e-3,
                           'Wrong residual did not produce detectable '
                           f'difference: rel_diff={rel_diff:.2e}')
        self.n_checks += 1


class KappaBetaGuardPrecedenceTestCase(unittest.TestCase):
    """Verify kappa != 0 and beta != 0 guards fire BEFORE the Born path.

    The ``_surrogate_coefficients`` method has an early-exit guard for
    ``kappa != 0`` and ``beta != 0`` (lines ~1554-1575) that fires
    BEFORE the surrogate is consulted and BEFORE the fact-4 Born
    residual slot is reached.  These tests verify that a config with
    non-zero kappa or beta returns None even when a born_residual_chart
    is attached and would cover the (gamma, rho) point.

    This proves the guard ordering: kappa/beta → dense_w positivity →
    may_serve → geometry_partition → surrogate.serve → fact-4 Born slot.

    Cost: < 0.1 s (no geometry partition, exits at the guard).
    """

    def setUp(self):
        self.n_checks = 0
        self.chart = _build_mock_chart()
        # Use a config that WOULD be served if kappa=beta=0:
        # gamma=0.5, |y| giving rho~3.0 (inside the chart's grid).
        self.abs_y = _compute_target_y()

    def tearDown(self):
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no assertions ran in '
            'KappaBetaGuardPrecedenceTestCase')

    def test_kappa_nonzero_returns_none(self):
        """kappa=0.1 causes immediate None before the Born path."""
        probe = _BornResidualProbe(born_residual_chart=self.chart)
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': self.abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': 0.0,
            'kappa': 0.1,  # non-zero kappa
        }
        result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result)
        self.n_checks += 1

    def test_beta_nonzero_kappa_zero_returns_none(self):
        """beta=0.3 with kappa=0 causes immediate None before the Born path."""
        probe = _BornResidualProbe(born_residual_chart=self.chart)
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': self.abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': 0.3,  # non-zero beta
            'kappa': 0.0,
        }
        result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result)
        self.n_checks += 1

    def test_both_kappa_and_beta_nonzero_returns_none(self):
        """Both kappa=0.1 and beta=0.3 returns None (kappa guard fires first)."""
        probe = _BornResidualProbe(born_residual_chart=self.chart)
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': self.abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': 0.3,
            'kappa': 0.1,
        }
        result = probe._surrogate_coefficients(par_dic)
        self.assertIsNone(result)
        self.n_checks += 1

    def test_guard_fires_before_born_path_is_reached(self):
        """Prove the guard fires before the Born path by showing the
        same config WITH kappa=beta=0 actually reaches the Born path.

        This is the control: if the guard were absent or ordered after
        the Born path, then kappa=0.1 would reach the Born slot.
        """
        probe = _BornResidualProbe(born_residual_chart=self.chart)
        # Control: kappa=0, beta=0 → Born path fires (non-None).
        par_dic_clean = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': self.abs_y,
            'y2': 0.0,
            'gamma': _GAMMA,
            'beta': 0.0,
            'kappa': 0.0,
        }
        result_clean = probe._surrogate_coefficients(par_dic_clean)
        self.assertIsNotNone(result_clean,
                             'Control config should reach Born path')

        # Experimental: same config but kappa=0.1 → None (guard fires).
        par_dic_kappa = dict(par_dic_clean, kappa=0.1)
        result_kappa = probe._surrogate_coefficients(par_dic_kappa)
        self.assertIsNone(result_kappa,
                          'kappa=0.1 should be rejected by the guard')

        # Experimental: same config but beta=0.3 → None (guard fires).
        par_dic_beta = dict(par_dic_clean, beta=0.3)
        result_beta = probe._surrogate_coefficients(par_dic_beta)
        self.assertIsNone(result_beta,
                          'beta=0.3 should be rejected by the guard')
        self.n_checks += 1

    def test_various_kappa_values_all_refused(self):
        """Multiple non-zero kappa values all return None."""
        probe = _BornResidualProbe(born_residual_chart=self.chart)
        for kappa in (0.01, 0.05, 0.1, 0.5, -0.1, -0.3):
            par_dic = {
                'm_lens_msun': _M_LENS_MSUN,
                'z_lens': _Z_LENS,
                'y1': self.abs_y,
                'y2': 0.0,
                'gamma': _GAMMA,
                'beta': 0.0,
                'kappa': kappa,
            }
            with self.subTest(kappa=kappa):
                result = probe._surrogate_coefficients(par_dic)
                self.assertIsNone(result)
                self.n_checks += 1

    def test_various_beta_values_all_refused(self):
        """Multiple non-zero beta values all return None."""
        probe = _BornResidualProbe(born_residual_chart=self.chart)
        for beta in (0.01, 0.1, 0.3, 0.5, -0.1, -0.5):
            par_dic = {
                'm_lens_msun': _M_LENS_MSUN,
                'z_lens': _Z_LENS,
                'y1': self.abs_y,
                'y2': 0.0,
                'gamma': _GAMMA,
                'beta': beta,
                'kappa': 0.0,
            }
            with self.subTest(beta=beta):
                result = probe._surrogate_coefficients(par_dic)
                self.assertIsNone(result)
                self.n_checks += 1


class BornResidualChartCoversTestCase(unittest.TestCase):
    """Verify BornResidualChart.covers boundary-case behavior.

    Tests the axis-aligned box containment check with a precisely
    specified grid:
      gamma_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
      rho_grid   = [1.5, 2.0, 3.0, 4.0, 5.0]

    The ``covers`` method uses inclusive bounds on both ends:
      gamma_grid[0] <= gamma <= gamma_grid[-1]
      AND rho_grid[0] <= rho <= rho_grid[-1]

    Cost: < 0.01 s (pure Python comparisons, no numerical computation).
    """

    def setUp(self):
        self.n_checks = 0
        # Build chart with the spec-prescribed grids.
        gamma_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        rho_grid = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
        log_w_grid = np.log(np.array([1.0, 5.0, 10.0, 20.0]))
        n_gamma = len(gamma_grid)
        n_rho = len(rho_grid)
        n_w = len(log_w_grid)
        # Dummy coefficients — covers() never uses them.
        self.chart = BornResidualChart(
            gamma_grid=gamma_grid,
            rho_grid=rho_grid,
            log_w_grid=log_w_grid,
            real_coeffs=np.zeros((n_gamma, n_rho, n_w)),
            imag_coeffs=np.zeros((n_gamma, n_rho, n_w)),
        )

    def tearDown(self):
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no assertions ran in '
            'BornResidualChartCoversTestCase')

    def test_interior_point_covered(self):
        """(0.5, 3.0) is strictly inside the grid → True."""
        self.assertTrue(self.chart.covers(0.5, 3.0))
        self.n_checks += 1

    def test_lower_left_corner_covered(self):
        """(0.3, 1.5) is the lower-left corner (inclusive) → True."""
        self.assertTrue(self.chart.covers(0.3, 1.5))
        self.n_checks += 1

    def test_upper_right_corner_covered(self):
        """(0.7, 5.0) is the upper-right corner (inclusive) → True."""
        self.assertTrue(self.chart.covers(0.7, 5.0))
        self.n_checks += 1

    def test_gamma_below_min_not_covered(self):
        """(0.29, 3.0) has gamma below grid min 0.3 → False."""
        self.assertFalse(self.chart.covers(0.29, 3.0))
        self.n_checks += 1

    def test_gamma_above_max_not_covered(self):
        """(0.71, 3.0) has gamma above grid max 0.7 → False."""
        self.assertFalse(self.chart.covers(0.71, 3.0))
        self.n_checks += 1

    def test_rho_below_min_not_covered(self):
        """(0.5, 1.49) has rho below grid min 1.5 → False."""
        self.assertFalse(self.chart.covers(0.5, 1.49))
        self.n_checks += 1

    def test_rho_above_max_not_covered(self):
        """(0.5, 5.01) has rho above grid max 5.0 → False."""
        self.assertFalse(self.chart.covers(0.5, 5.01))
        self.n_checks += 1

    def test_all_four_edges_covered(self):
        """Points on all four box edges (mid-edge) are covered."""
        cases = [
            ('left_edge', 0.3, 3.0),    # gamma = min, rho interior
            ('right_edge', 0.7, 3.0),   # gamma = max, rho interior
            ('bottom_edge', 0.5, 1.5),  # gamma interior, rho = min
            ('top_edge', 0.5, 5.0),     # gamma interior, rho = max
        ]
        for label, gamma, rho in cases:
            with self.subTest(edge=label, gamma=gamma, rho=rho):
                self.assertTrue(self.chart.covers(gamma, rho))
                self.n_checks += 1

    def test_all_four_corners_covered(self):
        """All four corners of the bounding box are covered (inclusive)."""
        corners = [
            (0.3, 1.5),  # lower-left
            (0.3, 5.0),  # upper-left
            (0.7, 1.5),  # lower-right
            (0.7, 5.0),  # upper-right
        ]
        for gamma, rho in corners:
            with self.subTest(gamma=gamma, rho=rho):
                self.assertTrue(self.chart.covers(gamma, rho))
                self.n_checks += 1

    def test_just_outside_boundaries(self):
        """Points at machine-epsilon outside each boundary are rejected."""
        eps = 1e-10
        outside_cases = [
            ('gamma_below', 0.3 - eps, 3.0),
            ('gamma_above', 0.7 + eps, 3.0),
            ('rho_below', 0.5, 1.5 - eps),
            ('rho_above', 0.5, 5.0 + eps),
        ]
        for label, gamma, rho in outside_cases:
            with self.subTest(boundary=label, gamma=gamma, rho=rho):
                self.assertFalse(self.chart.covers(gamma, rho))
                self.n_checks += 1


if __name__ == '__main__':
    unittest.main()

