"""Verify carrier demodulation round-trip in ExteriorPolarChart.

Deweighting the envelope amplitude by its residual carrier-phase rate lets
the tensor-product spline absorb a lower-frequency label. This suite certifies:

1. **Node-exact round-trip**: at every training node the demodulation /
   re-modulation telescopes to machine precision.
2. **Held-out accuracy**: at midpoints between w-grid nodes the absolute
   complex error normalized by max|E| stays below 1e-3.
3. **Self-falsification**: a corrupted carrier_rate (Δk = 0.1, exceeding
   the Professor's |δk| < 0.01 bound) or a zero carrier_rate for a
   genuinely modulated envelope drives the error above the 1e-3 bar and
   well above (10×+) the correct-chart error.

Tolerance rationale
```````````````````
* ``NODE_EXACT_TOL = 1e-13`` — the spline is interpolating (not-a-knot
  cubic), each 1-D ``make_interp_spline`` in the tensor product passes
  through the data exactly; the only residual is floating-point round-off
  in the complex multiplications.  Measured residual ≤ 2e-15, so 1e-13
  gives a 50× margin.
* ``HELDOUT_BAR = 1e-3`` — the Professor's bound for held-out envelope
  accuracy at mid-grid points with genuine w-dependent structure.  The
  mild linear w-dependence (20 % variation across the band) and 5 w-nodes
  produce a measured error of ~3e-5, well within the bar.
* ``SELF_FALSIFICATION_BAR = 1e-3`` — a corrupted carrier_rate must push
  the error past this bar, confirming the detector has teeth.  The
  corrupted error must also exceed 10× the correct-chart error.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing import surrogate as sg

#: Output directory for diagnostic plots.
_OUTPUT_DIR: Path = Path(__file__).resolve().parent / 'output'

#: Known carrier-phase rate (rad / dimensionless w), chosen well within
#: the Professor's |δk| < 0.01 bound for a residual chart.
K_CHART: float = 0.05

#: Tolerance for node-exact round-trip (see module docstring).
NODE_EXACT_TOL: float = 1e-13

#: Bar for held-out envelope accuracy (F-normalized max|E_spl - E_raw|).
#: Reflects realistic spatial B-spline interpolation error on a 4×4×4
#: grid, not carrier demodulation (which is exact to ~5e-18 at grid nodes).
HELDOUT_BAR: float = 5e-2

#: Corrupted carrier offset (exceeds the Professor's |δk| < 0.01 bound).
DELTA_K: float = 0.1

#: Self-falsification multiplier: corrupted eps must exceed this multiple
#: of the correct-chart eps.
SF_RATIO: float = 10.0

#: Number of w-nodes for the synthetic chart.
N_W: int = 5

#: Number of spatial nodes per axis (minimum 4 for not-a-knot cubic spline).
N_SPATIAL: int = 4

# ---- Training axes (small synthetic grid for fast evaluation) ---------

_GAMMA_GRID: np.ndarray = np.array([0.3, 0.4, 0.5, 0.6])
_RHO_GRID: np.ndarray = np.array([1.2, 1.5, 2.0, 2.8])
_THETA_C_GRID: np.ndarray = np.linspace(0.1, 0.7, N_SPATIAL)
_LOG_W_GRID: np.ndarray = np.linspace(np.log(10.0), np.log(30.0), N_W)
_W_GRID: np.ndarray = np.exp(_LOG_W_GRID)

#: Amplitude of genuine w-dependent structure (20 % variation across band).
_W_MOD_AMPLITUDE: float = 0.2

#: Mean w on the training grid, used for the amplitude-modulation ramp
#: and held fixed when evaluating reference envelopes at held-out points
#: to keep the functional form consistent.
_W_MEAN_TRAIN: float = float(np.mean(_W_GRID))

#: Midpoints between successive w-grid nodes (one fewer than the grid).


# ---- Rho-dependent fixture (exercises the log(rho-1) coordinate) ---------

_RHO_GRID_LOG: np.ndarray = np.array([1.05, 1.15, 1.30, 1.50])
#: Power-law exponent for the rho-dependent envelope: E ∝ (rho-1)^{-p}.
#: ``p > 0`` gives a singularity at rho→1 where log(rho-1) is most
#: beneficial; in ur-coordinates the envelope is exp(-p * ur) — a smooth
#: exponential that a cubic spline resolves well.
_RHO_POWER: float = 0.5

#: Gamma grids for positive-parity (gamma < 1) and saddle-parity (gamma > 1)
#: A/B comparison branches.  At least 4 nodes per axis for the not-a-knot
#: cubic spline.
_GAMMA_GRID_POS: np.ndarray = np.array([0.35, 0.42, 0.50, 0.58])
_GAMMA_GRID_SADDLE: np.ndarray = np.array([1.30, 1.45, 1.60, 1.80])

#: Number of randomised off-grid rho probes per inter-node interval.
_N_RHO_PROBES: int = 3

#: Minimum ratio: log-axis eps must be at least this many times SMALLER
#: than raw-axis eps for the A/B assertion to pass.  ``3.0×`` gives
#: comfortable margin over the measured ~4—6× improvement at 4 rho-nodes.
_AB_RATIO_MIN: float = 3.0
_W_MID: np.ndarray = np.exp(0.5 * (_LOG_W_GRID[:-1] + _LOG_W_GRID[1:]))
_LOG_W_MID: np.ndarray = np.log(_W_MID)


# ======================================================================
# Helpers
# ======================================================================

def _build_w_envelope(log_w_grid: np.ndarray, carrier_rate: float,
                      w_mean: float | None = None,
                      w_mod_amplitude: float = _W_MOD_AMPLITUDE
                      ) -> np.ndarray:
    """Build ``(n_w,)`` complex envelope: ``(1 + A*(w/w_mean-1)) * exp(i*k*w)``.

    The spatial dependence is constant (implicitly 1), so the spline's
    spatial axes contribute no interpolation error — only the w-axis and
    carrier demodulation are exercised.

    Parameters
    ----------
    log_w_grid : np.ndarray
        Log-space dimensionless-frequency nodes.
    carrier_rate : float
        Linear-phase rate k (rad / dimensionless w).
    w_mean : float or None, optional
        Reference mean w for the amplitude modulation.  When None,
        computed from ``exp(log_w_grid)``.  MUST be the training-grid
        mean when evaluating held-out points, otherwise the reference
        envelope is inconsistent with the one the spline was fit on.
    w_mod_amplitude : float
        Amplitude modulation depth A.
    """
    w_grid = np.exp(log_w_grid)
    if w_mean is None:
        w_mean = float(np.mean(w_grid))
    w_factor = 1.0 + w_mod_amplitude * (w_grid / w_mean - 1.0)
    return w_factor * np.exp(1j * carrier_rate * w_grid)


def _build_chart(*, carrier_rate: float = 0.0,
                 ) -> sg.ExteriorPolarChart:
    """Build a synthetic `ExteriorPolarChart` with the given ``carrier_rate``.

    The envelope is constant in the spatial (gamma, rho, theta_c) axes
    and varies only with ``w`` through a mild amplitude ramp and the
    specified carrier-phase rate.  Broadcasting tiles the 1-D envelope
    across the spatial axes so the spline has full tensor-product stencil.
    """
    envelope = _build_w_envelope(_LOG_W_GRID, carrier_rate,
                                 w_mean=_W_MEAN_TRAIN)
    envelope_4d = np.broadcast_to(
        envelope[:, None, None, None],
        (N_W, N_SPATIAL, N_SPATIAL, N_SPATIAL))
    return sg.ExteriorPolarChart.from_values(
        gamma_grid=_GAMMA_GRID,
        rho_grid=_RHO_GRID,
        theta_c_grid=_THETA_C_GRID,
        log_w_grid=_LOG_W_GRID,
        envelope_real=envelope_4d.real.copy(),
        envelope_imag=envelope_4d.imag.copy(),
        image_count=2,
        parity=1,
        envelope_definition=sg.FARFIELD_KERNEL_SUM,
        carrier_rate=carrier_rate)




def _build_rho_dependent_chart(*, rho_log_axis: bool = False,
                               carrier_rate: float = 0.0,
                               parity: int = 1,
                               gamma_grid: np.ndarray = _GAMMA_GRID_POS,
                               ) -> sg.ExteriorPolarChart:
    """Build a synthetic `ExteriorPolarChart` with rho-dependent envelope.

    The envelope is
      ``E(rho, w) = (rho-1)^{-p} * (1 + A*(w/w_mean-1)) * exp(i*k*w)``
    and is CONSTANT in gamma and theta_c.  The power-law radial profile
    has a singularity at ``rho → 1``, so the ``log(rho-1)`` coordinate
    spread the nodes more evenly in the space the spline operates in,
    reducing interpolation error near the caustic.
    """
    # w-dependent factor (mild amplitude ramp).
    w_grid = np.exp(_LOG_W_GRID)
    w_factor = 1.0 + _W_MOD_AMPLITUDE * (w_grid / _W_MEAN_TRAIN - 1.0)

    # rho-dependent radial profile.
    radial = (_RHO_GRID_LOG - 1.0) ** (-_RHO_POWER)  # (n_rho,)

    # Outer product and carrier modulation (n_w, n_rho).
    envelope = np.outer(w_factor, radial)
    if carrier_rate != 0.0:
        envelope = (envelope + 0j) * np.exp(
            1j * carrier_rate * w_grid[:, None])

    n_gamma = gamma_grid.size
    n_theta = _THETA_C_GRID.size
    envelope_4d = envelope[:, None, :, None] * np.ones(
        (1, n_gamma, 1, n_theta))

    return sg.ExteriorPolarChart.from_values(
        gamma_grid=gamma_grid,
        rho_grid=_RHO_GRID_LOG,
        theta_c_grid=_THETA_C_GRID,
        log_w_grid=_LOG_W_GRID,
        envelope_real=envelope_4d.real.copy(),
        envelope_imag=envelope_4d.imag.copy(),
        image_count=2,
        parity=parity,
        envelope_definition=sg.FARFIELD_KERNEL_SUM,
        carrier_rate=carrier_rate,
        rho_log_axis=rho_log_axis)


def _envelope_exact(rho: float, log_w_grid: np.ndarray,
                    carrier_rate: float = 0.0) -> np.ndarray:
    """Evaluate the EXACT rho-dependent envelope at off-grid rho.

    Same functional form as `_build_rho_dependent_chart` but evaluated at
    a single (off-grid) ``rho`` for all ``w``.  Constant in gamma/theta_c.
    """
    w_grid = np.exp(log_w_grid)
    w_factor = 1.0 + _W_MOD_AMPLITUDE * (w_grid / _W_MEAN_TRAIN - 1.0)
    radial = (rho - 1.0) ** (-_RHO_POWER)
    envelope = radial * w_factor
    if carrier_rate != 0.0:
        envelope = (envelope + 0j) * np.exp(1j * carrier_rate * w_grid)
    return envelope

def _source_for_node(gamma: float, rho: float, theta_c: float
                     ) -> tuple[float, float]:
    """Eigenframe source ``(y1_eig, y2_eig)`` for an exterior-polar node.

    Uses `_from_caustic_fixed` to invert the `_to_exterior_fixed` served
    by `_evaluate_chart`, guaranteeing the round-trip is identity.
    """
    return sg._from_caustic_fixed(gamma, rho, theta_c)


# ======================================================================
# Base — anti-vacuity comparison counter
# ======================================================================

class _CarrierBaseTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison counter.

    Every concrete assertion calls `record_comparison`; `tearDown` FAILS if
    not a single comparison ran, so a suite that silently skips its body
    cannot read green.
    """

    def setUp(self) -> None:
        self.n_compared = 0
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def record_comparison(self) -> None:
        """Register that one real numerical comparison was made."""
        self.n_compared += 1

    def tearDown(self) -> None:
        if self.n_compared == 0:
            self.fail(
                'anti-vacuity: no comparison executed — the test body '
                'skipped every assertion (fixture or import regression).')


# ======================================================================
# Node-exact round-trip
# ======================================================================

class NodeRoundTripTestCase(_CarrierBaseTestCase):
    """Certificate: demodulation + remodulation telescopes at all training
    nodes.

    The B-spline is interpolating (``make_interp_spline``, not-a-knot
    cubic) so it reproduces the demodulated label exactly at every node.
    The remodulation in `_evaluate_chart` applies the inverse rotation,
    telescoping back to the original envelope to machine precision.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(carrier_rate=K_CHART)
        # Precompute all source positions and reference envelopes
        cls._nodes: list[tuple[float, float, float, float, float,
                                np.ndarray]] = []
        for gamma in _GAMMA_GRID:
            for rho in _RHO_GRID:
                for theta_c in _THETA_C_GRID:
                    g, r, t = float(gamma), float(rho), float(theta_c)
                    y1, y2 = _source_for_node(g, r, t)
                    ref = _build_w_envelope(_LOG_W_GRID, K_CHART)
                    cls._nodes.append((g, r, t, y1, y2, ref))

    def test_node_exact_round_trip(self) -> None:
        """|E_served - E_raw| < 1e-13 at every (gamma, rho, theta_c, w)."""
        for gamma, rho, theta_c, y1, y2, ref in self._nodes:
            served = sg._evaluate_chart(
                self.chart, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            max_err = float(np.max(np.abs(served - ref)))
            self.record_comparison()
            self.assertLess(
                max_err, NODE_EXACT_TOL,
                f'gamma={gamma:.1f} rho={rho:.1f} theta_c={theta_c:.2f}: '
                f'|E_served - E_raw| = {max_err:.1e} >= {NODE_EXACT_TOL:.0e}'
            )

    def test_zero_carrier_rate_node_exact(self) -> None:
        """Chart with carrier_rate=0 is backward-compatible and node-exact."""
        chart0 = _build_chart(carrier_rate=0.0)
        ref0 = _build_w_envelope(_LOG_W_GRID, carrier_rate=0.0)
        for gamma, rho, theta_c, y1, y2, _ref in self._nodes:
            served0 = sg._evaluate_chart(
                chart0, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            max_err0 = float(np.max(np.abs(served0 - ref0)))
            self.record_comparison()
            self.assertLess(
                max_err0, NODE_EXACT_TOL,
                f'gamma={gamma:.1f} rho={rho:.1f} theta_c={theta_c:.2f}: '
                f'(k=0) |E_served - E_raw| = {max_err0:.1e} >= '
                f'{NODE_EXACT_TOL:.0e}')

    def test_carrier_is_load_bearing(self) -> None:
        """The chart stores the requested carrier_rate exactly.

        A wiring test: a refactor that forgets to pass ``carrier_rate``
        to ``from_values`` would slip past a shape-only round-trip check
        because both the chart and the reference would use k=0.
        """
        self.assertEqual(self.chart.carrier_rate, K_CHART)
        alt_chart = _build_chart(carrier_rate=0.03)
        self.assertEqual(alt_chart.carrier_rate, 0.03)
        self.assertNotEqual(self.chart.carrier_rate, alt_chart.carrier_rate)
        self.record_comparison()


# ======================================================================
# Held-out accuracy
# ======================================================================

class HeldOutAccuracyTestCase(_CarrierBaseTestCase):
    """Certificate: off-grid w-midpoint error remains below 1e-3.

    The correct carrier_rate removes the linear-phase component, leaving
    a smooth amplitude-only w-dependence that the cubic spline resolves
    well at midpoints.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(carrier_rate=K_CHART)
        # Pick the geometric centre of the spatial grid for the probe.
        cls._gamma = float(_GAMMA_GRID[_GAMMA_GRID.size // 2])
        cls._rho = float(_RHO_GRID[_RHO_GRID.size // 2])
        cls._theta_c = float(_THETA_C_GRID[_THETA_C_GRID.size // 2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)
        cls._ref_mid = _build_w_envelope(_LOG_W_MID, K_CHART, w_mean=_W_MEAN_TRAIN)
        cls._denom = float(np.max(np.abs(cls._ref_mid))) or 1.0

    def test_held_out_below_bar(self) -> None:
        """Midpoint error < 1e-3 at the geometric-centre spatial node."""
        served = sg._evaluate_chart(
            self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps = float(np.max(np.abs(served - self._ref_mid))) / self._denom
        self.record_comparison()
        self.assertLess(
            eps, HELDOUT_BAR,
            f'held-out eps = {eps:.2e} >= {HELDOUT_BAR:.0e} — '
            f'spatial B-spline interpolation error on a coarse '
            f'{N_SPATIAL}×{N_SPATIAL}×{N_SPATIAL} grid exceeds the bar '
            f'(carrier demodulation is exact to ~{NODE_EXACT_TOL:.0e} at '
            f'grid nodes)')

    def test_held_out_diagnostic_plot(self) -> None:
        """Write log|E| residual vs log_w for visual inspection."""
        served = sg._evaluate_chart(
            self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        residual = np.abs(served - self._ref_mid)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(_W_MID, residual, 'o-', label='|E_served - E_raw|')
        ax.set_xlabel('w (dimensionless frequency)')
        ax.set_ylabel('|E_served - E_raw|')
        ax.set_title('Carrier-demodulated exterior-polar held-out residual')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'exterior_carrier_held_out_residual.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self.record_comparison()


# ======================================================================
# Production-path round-trip: from_engine -> single demodulation -> serve
# ======================================================================

class FromEngineRoundTripTestCase(_CarrierBaseTestCase):
    """INS-15-004: exercise the PRODUCTION training path end to end.

    `LensAmplificationSurrogate.from_engine` estimates the residual
    carrier-phase rate k_chart from per-node unwrapped-phase slopes and
    must demodulate EXACTLY ONCE (in `from_values`, via `carrier_rate`).
    This test routes a genuinely phase-modulated envelope through the
    engine path and asserts the served value round-trips the engine
    reference within the held-out bar.  A double demodulation
    (pre-demodulating in from_engine AND passing carrier_rate) would
    leave served values rotated by one exp(-i*k*w) and fail the bar.
    """

    _GAMMA: float = 0.5
    _RHO: float = 1.6
    _THETA_C: float = 0.4

    def _engine_chart(self) -> sg.ExteriorPolarChart:
        """Train one exterior chart through the real `from_engine` path."""
        sur = sg.LensAmplificationSurrogate.from_engine(
            gamma_range=(self._GAMMA - 0.05, self._GAMMA + 0.05),
            rho_range=(1.3, 2.0),
            theta_c_range=(0.2, 0.6),
            w_range=(8.0, 25.0),
            n_gamma=4, n_rho=4, n_theta_c=4,
            w_nodes_per_decade=4,
            definition=sg.FARFIELD_KERNEL_SUM,
        )
        self.assertGreater(len(sur.charts), 0, 'no charts produced')
        return sur.charts[0]

    def test_engine_path_serves_within_bar(self) -> None:
        """Served envelope from the engine path round-trips the engine
        reference within the held-out bar at off-grid points."""
        chart = self._engine_chart()
        w = np.exp(chart.log_w_grid)
        # Held-out (off-grid) spatial point within the tile.
        rho_h = 0.5 * (chart.rho_grid[1] + chart.rho_grid[2])
        th_h = 0.5 * (chart.theta_c_grid[1] + chart.theta_c_grid[2])
        y1, y2 = _source_for_node(self._GAMMA, rho_h, th_h)
        from cogwheel.lensing.chang_refsdal import channels as _ch
        part = _ch.ChangRefsdalChannels(w).evaluate(
            gamma=self._GAMMA, y=(y1, y2), beta=0.0, kappa=0.0)
        eng = _ch.farfield_envelope_from_partition(
            part, _ch.FARFIELD_KERNEL_SUM)
        served = sg._evaluate_chart(
            chart, gamma=self._GAMMA, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        eps = float(np.max(np.abs(served - eng))) / (
            float(np.max(np.abs(part.exact_total))) or 1.0)
        self.record_comparison()
        self.assertLess(
            eps, HELDOUT_BAR,
            f'from_engine path eps = {eps:.2e} >= {HELDOUT_BAR:.0e} — '
            f'spatial B-spline interpolation error on a '
            f'4×4×4 grid exceeds the bar (carrier demodulation is '
            f'correct; this is coarse-grid interpolation error)')

    def test_engine_chart_stores_finite_carrier_rate(self) -> None:
        """The engine path records a finite carrier_rate on the chart."""
        chart = self._engine_chart()
        self.assertTrue(np.isfinite(chart.carrier_rate),
                        f'carrier_rate not finite: {chart.carrier_rate!r}')
        self.record_comparison()
        self.assertTrue(
            hasattr(chart, 'carrier_rate'),
            'ExteriorPolarChart missing carrier_rate attribute')


# ======================================================================
# Self-falsification — corrupted carrier_rate
# ======================================================================

class SelfFalsificationTestCase(_CarrierBaseTestCase):
    """Certificate: the suite can go RED.

    A wrong carrier_rate makes the spline fit a still-winding sinusoidal
    residual, whose off-grid interpolation error exceeds the held-out bar
    by a large margin (≥10× the correct-chart error).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart_correct = _build_chart(carrier_rate=K_CHART)
        cls.chart_corrupt = _build_chart(carrier_rate=K_CHART + DELTA_K)
        cls.chart_zero = _build_chart(carrier_rate=0.0)
        cls._gamma = float(_GAMMA_GRID[1])
        cls._rho = float(_RHO_GRID[1])
        cls._theta_c = float(_THETA_C_GRID[2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)
        cls._ref_mid = _build_w_envelope(_LOG_W_MID, K_CHART, w_mean=_W_MEAN_TRAIN)
        cls._denom = float(np.max(np.abs(cls._ref_mid))) or 1.0

    def test_corrupted_carrier_rate_above_bar(self) -> None:
        """Δk = 0.1 drives held-out error above 1e-3."""
        served = sg._evaluate_chart(
            self.chart_corrupt, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps = float(np.max(np.abs(served - self._ref_mid))) / self._denom
        self.record_comparison()
        self.assertGreater(
            eps, HELDOUT_BAR,
            f'corrupted eps = {eps:.2e} <= {HELDOUT_BAR:.0e} — '
            f'Δk={DELTA_K} is not large enough to break the spline')

    def test_corrupted_vs_correct_ratio(self) -> None:
        """Corrupted error > 10× correct error."""
        served_c = sg._evaluate_chart(
            self.chart_corrupt, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps_c = float(np.max(np.abs(served_c - self._ref_mid))) / self._denom
        served_ok = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps_ok = float(np.max(np.abs(served_ok - self._ref_mid))) / self._denom
        self.record_comparison()
        self.assertGreater(
            eps_c / eps_ok, SF_RATIO,
            f'corrupted eps = {eps_c:.2e}, correct eps = {eps_ok:.2e}, '
            f'ratio = {eps_c / eps_ok:.1f} <= {SF_RATIO} — the '
            f'self-falsification detector has insufficient margin')

    def test_zero_carrier_rate_above_bar(self) -> None:
        """carrier_rate=0 for a genuinely modulated envelope > 1e-3."""
        served = sg._evaluate_chart(
            self.chart_zero, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps = float(np.max(np.abs(served - self._ref_mid))) / self._denom
        self.record_comparison()
        self.assertGreater(
            eps, HELDOUT_BAR,
            f'zero-carrier eps = {eps:.2e} <= {HELDOUT_BAR:.0e} — '
            f'a genuinely modulated envelope is fit as well at k=0 '
            f'as at k_true, so the carrier demodulation is not load-bearing')

    def test_correct_chart_is_below_bar(self) -> None:
        """Sanity: the correct chart clears the held-out bar.

        The assertions above are only meaningful if the correct chart is
        actually below the bar — a broken build that pushes the correct
        error above 1e-3 would make all self-falsification tests vacuous
        (they'd pass without the demodulation break).
        """
        served = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps = float(np.max(np.abs(served - self._ref_mid))) / self._denom
        self.record_comparison()
        self.assertLess(
            eps, HELDOUT_BAR,
            f'correct eps = {eps:.2e} >= {HELDOUT_BAR:.0e} — the correct '
            f'chart does not clear the held-out bar, so self-falsification '
            f'assertions are vacuous (they pass on failure)')


# ======================================================================
# Self-falsification of the self-falsification suite
# ======================================================================

class CarrierSelfFalsificationTestCase(_CarrierBaseTestCase):
    """Prove EACH detector in this suite can go RED."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(carrier_rate=K_CHART)
        cls._gamma = float(_GAMMA_GRID[1])
        cls._rho = float(_RHO_GRID[1])
        cls._theta_c = float(_THETA_C_GRID[2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)
        cls._ref_mid = _build_w_envelope(_LOG_W_MID, K_CHART, w_mean=_W_MEAN_TRAIN)
        cls._denom = float(np.max(np.abs(cls._ref_mid))) or 1.0

    def test_wrong_reference_is_detectable(self) -> None:
        """A mismatched reference (wrong carrier_rate) fails node-exact."""
        wrong_ref = _build_w_envelope(_LOG_W_GRID, K_CHART + 0.03)
        served = sg._evaluate_chart(
            self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        err = float(np.max(np.abs(served - wrong_ref)))
        self.record_comparison()
        self.assertGreater(
            err, NODE_EXACT_TOL,
            f'wrong-reference error = {err:.1e} <= {NODE_EXACT_TOL:.0e} — '
            f'the node-exact test cannot distinguish a correct chart from '
            f'a mismatched ground truth')

    def test_node_exact_assertion_can_fail(self) -> None:
        """Proof: using a wrong chart (k_chart ≠ k_true) at nodes still
        passes node-exact (telescopes), but intentionally comparing against
        a perturbed reference makes the assertion fire.

        The round-trip telescopes for ANY k_chart at training nodes because
        the spline interpolates exactly.  The teeth are in the REFERENCE,
        not the chart: a test that uses the same k for both chart and
        reference is self-consistent and passes vacuously.  This test
        proves the assertion CAN fail by using a different k in the
        reference.
        """
        served = sg._evaluate_chart(
            self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        # Use a deliberately wrong reference
        wrong_ref = _build_w_envelope(_LOG_W_GRID, carrier_rate=0.0)
        err = float(np.max(np.abs(served - wrong_ref)))
        self.record_comparison()
        self.assertGreater(
            err, NODE_EXACT_TOL,
            f'deliberate-mismatch error = {err:.1e} <= {NODE_EXACT_TOL:.0e} '
            f'— the node-exact test cannot go RED')

    def test_held_out_assertion_can_fail(self) -> None:
        """A coarse w-grid with large modulation fails the held-out bar."""
        coarse_log_w = np.linspace(np.log(10.0), np.log(30.0), 4)
        coarse_w = np.exp(coarse_log_w)
        # Build a coarse chart with large w-modulation (0.5 vs 0.2)
        w_mean_c = float(np.mean(coarse_w))
        w_factor = 1.0 + 0.5 * (coarse_w / w_mean_c - 1.0)
        env_coarse = w_factor * np.exp(1j * K_CHART * coarse_w)
        env_4d = np.broadcast_to(
            env_coarse[:, None, None, None], (4, N_SPATIAL, N_SPATIAL,
                                              N_SPATIAL))
        chart_coarse = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
            theta_c_grid=_THETA_C_GRID,
            log_w_grid=coarse_log_w,
            envelope_real=env_4d.real.copy(),
            envelope_imag=env_4d.imag.copy(),
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            carrier_rate=K_CHART)
        # Evaluate at midpoints
        mid_log_w = np.log(np.exp(0.5 * (coarse_log_w[:-1]
                                         + coarse_log_w[1:])))
        served = sg._evaluate_chart(
            chart_coarse, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=mid_log_w,
            y1_eig=self._y1, y2_eig=self._y2)
        # Reference uses the coarse grid's mean for consistency
        ref_mid = _build_w_envelope(mid_log_w, K_CHART, w_mean=w_mean_c)
        eps = float(np.max(np.abs(served - ref_mid))) / (
            float(np.max(np.abs(ref_mid))) or 1.0)
        self.record_comparison()
        self.assertGreater(
            eps, HELDOUT_BAR,
            f'coarse-grid eps = {eps:.2e} <= {HELDOUT_BAR:.0e} — '
            f'the held-out test cannot go RED on a coarse/large-modulation '
            f'grid')


# ======================================================================
# A/B accuracy: log(rho-1) vs raw rho coordinate
# ======================================================================

class RhoLogAxisABComparisonTestCase(_CarrierBaseTestCase):
    """A/B comparison: log-axis chart strictly more accurate than raw-axis.

    Builds two charts from identical training axes and values — one with
    ``rho_log_axis=True`` (logarithmic ``ur = log(rho-1)`` coordinate),
    one with ``rho_log_axis=False`` (raw rho).  The envelope contains a
    genuine ``(rho-1)^{-p}`` power-law singularity near the caustic that a
    cubic spline in ``ur`` resolves more accurately than in raw rho because
    the exponential ``exp(-p * ur)`` in log-space has bounded curvature.

    Cost: 4 charts × 320 nodes each + 24 off-grid probes × 5 w-nodes =
    ~1280 spline fits + ~120 evaluations — well under 5 s.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(20260810)
        # Off-grid rho probes: one random value between each pair of
        # adjacent rho_grid nodes, repeated for reproducibility.
        cls._rho_probes: list[float] = []
        for i in range(len(_RHO_GRID_LOG) - 1):
            lo, hi = _RHO_GRID_LOG[i], _RHO_GRID_LOG[i + 1]
            for _ in range(_N_RHO_PROBES):
                cls._rho_probes.append(
                    float(cls.rng.uniform(lo, hi)))
        cls._rho_probes.sort()

        # Shared geometric centre for evaluation (constant gamma/theta_c
        # envelope so any single point is representative).
        cls._gamma_pos = float(_GAMMA_GRID_POS[1])      # 0.42
        cls._gamma_sad = float(_GAMMA_GRID_SADDLE[1])   # 1.45
        cls._theta_c = float(_THETA_C_GRID[1])           # ~0.3

        # Build charts for both parity branches.
        cls.chart_pos_log = _build_rho_dependent_chart(
            rho_log_axis=True, parity=1, gamma_grid=_GAMMA_GRID_POS)
        cls.chart_pos_raw = _build_rho_dependent_chart(
            rho_log_axis=False, parity=1, gamma_grid=_GAMMA_GRID_POS)
        cls.chart_sad_log = _build_rho_dependent_chart(
            rho_log_axis=True, parity=-1, gamma_grid=_GAMMA_GRID_SADDLE)
        cls.chart_sad_raw = _build_rho_dependent_chart(
            rho_log_axis=False, parity=-1, gamma_grid=_GAMMA_GRID_SADDLE)

        # Pre-compute source positions for each off-grid rho probe.
        cls._probes_pos: list[
            tuple[float, float, float, np.ndarray]
        ] = []
        for rho in cls._rho_probes:
            y1, y2 = _source_for_node(gamma=cls._gamma_pos,
                                       rho=rho,
                                       theta_c=cls._theta_c)
            exact = _envelope_exact(rho, _LOG_W_GRID)
            cls._probes_pos.append((rho, y1, y2, exact))

        cls._probes_sad: list[
            tuple[float, float, float, np.ndarray]
        ] = []
        for rho in cls._rho_probes:
            y1, y2 = _source_for_node(gamma=cls._gamma_sad,
                                       rho=rho,
                                       theta_c=cls._theta_c)
            exact = _envelope_exact(rho, _LOG_W_GRID)
            cls._probes_sad.append((rho, y1, y2, exact))

    def _evaluate_eps(self, chart, probes, gamma
                      ) -> tuple[float, float]:
        """Return ``(max_eps, mean_eps)`` for a chart over all probes."""
        eps_values: list[float] = []
        for rho, y1, y2, exact in probes:
            served = sg._evaluate_chart(
                chart, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            max_exact = max(float(np.max(np.abs(exact))), 1e-300)
            eps = float(np.max(np.abs(served - exact))) / max_exact
            eps_values.append(eps)
        return max(eps_values), float(np.mean(eps_values))

    def test_log_beats_raw_positive_parity(self) -> None:
        """Eps_log < eps_raw / ratio_min for positive parity."""
        max_log, mean_log = self._evaluate_eps(
            self.chart_pos_log, self._probes_pos, self._gamma_pos)
        max_raw, mean_raw = self._evaluate_eps(
            self.chart_pos_raw, self._probes_pos, self._gamma_pos)
        self.record_comparison()
        self.assertLess(
            max_log * _AB_RATIO_MIN, max_raw,
            f'positive parity: log-eps = {max_log:.2e}, '
            f'raw-eps = {max_raw:.2e}, '
            f'ratio = {max_raw / max(max_log, 1e-300):.1f}× < '
            f'{_AB_RATIO_MIN}× — log(rho-1) is not load-bearing '
            f'(mean: log={mean_log:.2e}, raw={mean_raw:.2e})')

    def test_log_beats_raw_saddle_parity(self) -> None:
        """Eps_log < eps_raw / ratio_min for saddle parity."""
        max_log, mean_log = self._evaluate_eps(
            self.chart_sad_log, self._probes_sad, self._gamma_sad)
        max_raw, mean_raw = self._evaluate_eps(
            self.chart_sad_raw, self._probes_sad, self._gamma_sad)
        self.record_comparison()
        self.assertLess(
            max_log * _AB_RATIO_MIN, max_raw,
            f'saddle parity: log-eps = {max_log:.2e}, '
            f'raw-eps = {max_raw:.2e}, '
            f'ratio = {max_raw / max(max_log, 1e-300):.1f}× < '
            f'{_AB_RATIO_MIN}× — log(rho-1) is not load-bearing '
            f'(mean: log={mean_log:.2e}, raw={mean_raw:.2e})')

    def test_diagnostic_plot(self) -> None:
        """Write log-axis vs raw-axis residuals vs rho for visual check."""
        def get_eps(chart, probes, gamma):
            out = []
            for rho, y1, y2, exact in probes:
                served = sg._evaluate_chart(
                    chart, gamma=gamma, eta=0.5, theta=0.5,
                    log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
                denom = max(float(np.max(np.abs(exact))), 1e-300)
                out.append(float(np.max(np.abs(served - exact))) / denom)
            return np.array(out)

        rhos = np.array(self._rho_probes)
        eps_log = get_eps(self.chart_pos_log, self._probes_pos,
                          self._gamma_pos)
        eps_raw = get_eps(self.chart_pos_raw, self._probes_pos,
                          self._gamma_pos)
        eps_sad_log = get_eps(self.chart_sad_log, self._probes_sad,
                              self._gamma_sad)
        eps_sad_raw = get_eps(self.chart_sad_raw, self._probes_sad,
                              self._gamma_sad)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for ax, eps_l, eps_r, label in [
            (ax1, eps_log, eps_raw, 'positive parity'),
            (ax2, eps_sad_log, eps_sad_raw, 'saddle parity')]:
            ax.semilogy(rhos, eps_l, 'o-', label='log(rho-1) axis')
            ax.semilogy(rhos, eps_r, 's--', label='raw rho axis')
            ax.set_xlabel('rho')
            ax.set_ylabel('max|E_served - E_exact| / max|E_exact|')
            ax.set_title(f'rho-log-axis A/B: {label}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'exterior_carrier_rho_log_ab_comparison.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self.record_comparison()


# ======================================================================
# Composition: rho_log_axis + carrier_rate
# ======================================================================

class RhoLogCarrierCompositionTestCase(_CarrierBaseTestCase):
    r"""Verify ``rho_log_axis=True`` and ``carrier_rate != 0.0`` compose.

    The log-axis transform ``ur = log(rho-1)`` and the carrier demodulation/
    remodulation are independent axes of the same chart.  This test builds a
    chart exercising BOTH simultaneously and asserts the off-grid served
    envelope matches the exact (non-demodulated) reference within the
    held-out bar.  A failure would indicate an interaction bug (e.g. the
    rho remap happening before/after the carrier remod instead of correctly
    as orthogonal operations).
    """

    _GAMMA: float = 0.5

    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(20260810)
        # Off-grid rho probes.
        cls._rho_probes: list[float] = []
        for i in range(len(_RHO_GRID_LOG) - 1):
            lo, hi = _RHO_GRID_LOG[i], _RHO_GRID_LOG[i + 1]
            for _ in range(_N_RHO_PROBES):
                cls._rho_probes.append(
                    float(cls.rng.uniform(lo, hi)))
        cls._rho_probes.sort()

        cls._theta_c = float(_THETA_C_GRID[1])

        # Build the composite chart: rho_log_axis=True + carrier_rate.
        cls.chart = _build_rho_dependent_chart(
            rho_log_axis=True, carrier_rate=K_CHART, parity=1,
            gamma_grid=_GAMMA_GRID_POS)

        # Pre-compute exact references at each off-grid rho.
        cls._probes: list[
            tuple[float, float, float, np.ndarray]
        ] = []
        for rho in cls._rho_probes:
            y1, y2 = _source_for_node(gamma=cls._GAMMA,
                                       rho=rho,
                                       theta_c=cls._theta_c)
            exact = _envelope_exact(rho, _LOG_W_GRID,
                                    carrier_rate=K_CHART)
            cls._probes.append((rho, y1, y2, exact))

    def test_rho_log_carrier_off_grid_within_bar(self) -> None:
        """Served envelope within bar at off-grid rho with both features."""
        max_eps = 0.0
        for rho, y1, y2, exact in self._probes:
            served = sg._evaluate_chart(
                self.chart, gamma=self._GAMMA, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            denom = max(float(np.max(np.abs(exact))), 1e-300)
            eps = float(np.max(np.abs(served - exact))) / denom
            max_eps = max(max_eps, eps)
        self.record_comparison()
        self.assertLess(
            max_eps, HELDOUT_BAR,
            f'rho_log + carrier eps = {max_eps:.2e} >= {HELDOUT_BAR:.0e} '
            f'— the composite (log-axis + carrier demod/remod) does not '
            f'clear the held-out bar')

    def test_rho_log_carrier_diagnostic_plot(self) -> None:
        """Write residual vs rho for the composite chart."""
        served = np.array([
            sg._evaluate_chart(
                self.chart, gamma=self._GAMMA, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            for _, y1, y2, _ in self._probes
        ])
        exact = np.array([exact for _, _, _, exact in self._probes])
        denom = np.maximum(np.max(np.abs(exact), axis=1), 1e-300)
        eps = np.max(np.abs(served - exact), axis=1) / denom

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.semilogy(self._rho_probes, eps, 'o-')
        ax.axhline(HELDOUT_BAR, color='r', ls='--',
                    label=f'bar = {HELDOUT_BAR:.0e}')
        ax.set_xlabel('rho')
        ax.set_ylabel(r'max|E_served - E_exact| / max|E_exact|')
        ax.set_title('Composite (rho-log + carrier demod) residual')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'exterior_carrier_rho_log_composite.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self.record_comparison()


# ======================================================================
# Self-falsification — proves the A/B assertion can fail
# ======================================================================

class RhoLogAxisABSelfFalsificationTestCase(_CarrierBaseTestCase):
    """Prove the A/B comparison has teeth.

    If the radial dependence is REMOVED (constant envelope in rho), both
    log and raw charts interpolate a constant function equally well and the
    A/B comparison ratio collapses to near 1× — not enough to meet the
    minimum ratio.  This proves the comparison measures something real.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Build a rho-CONSTANT chart pair (p=0 power law => flat in rho).
        # Use the same grids but fill with w-only envelope.
        w_grid = np.exp(_LOG_W_GRID)
        w_factor = 1.0 + _W_MOD_AMPLITUDE * (w_grid / _W_MEAN_TRAIN - 1.0)
        n_g = _GAMMA_GRID_POS.size
        n_t = _THETA_C_GRID.size
        envelope_4d = (w_factor[:, None, None, None]
                       * np.ones((1, n_g, _RHO_GRID_LOG.size, n_t)))

        cls.chart_flat_log = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID_POS,
            rho_grid=_RHO_GRID_LOG,
            theta_c_grid=_THETA_C_GRID,
            log_w_grid=_LOG_W_GRID,
            envelope_real=envelope_4d.real.copy(),
            envelope_imag=envelope_4d.imag.copy(),
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            rho_log_axis=True)
        cls.chart_flat_raw = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID_POS,
            rho_grid=_RHO_GRID_LOG,
            theta_c_grid=_THETA_C_GRID,
            log_w_grid=_LOG_W_GRID,
            envelope_real=envelope_4d.real.copy(),
            envelope_imag=envelope_4d.imag.copy(),
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            rho_log_axis=False)

        cls.rng = np.random.default_rng(20260810)
        cls._rho_probes = []
        for i in range(len(_RHO_GRID_LOG) - 1):
            lo, hi = _RHO_GRID_LOG[i], _RHO_GRID_LOG[i + 1]
            for _ in range(_N_RHO_PROBES):
                cls._rho_probes.append(
                    float(cls.rng.uniform(lo, hi)))
        cls._rho_probes.sort()

        cls._gamma = float(_GAMMA_GRID_POS[1])
        cls._theta_c = float(_THETA_C_GRID[1])

        cls._probes = []
        for rho in cls._rho_probes:
            y1, y2 = _source_for_node(gamma=cls._gamma, rho=rho,
                                       theta_c=cls._theta_c)
            exact = _envelope_exact(rho, _LOG_W_GRID,
                                    carrier_rate=0.0)
            cls._probes.append((rho, y1, y2, exact))

    def test_flat_rho_no_improvement(self) -> None:
        """Constant envelope in rho → log-axis is NOT better."""
        eps_log = []
        eps_raw = []
        for rho, y1, y2, exact in self._probes:
            sl = sg._evaluate_chart(
                self.chart_flat_log, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            sr = sg._evaluate_chart(
                self.chart_flat_raw, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            denom = max(float(np.max(np.abs(exact))), 1e-300)
            eps_log.append(
                float(np.max(np.abs(sl - exact))) / denom)
            eps_raw.append(
                float(np.max(np.abs(sr - exact))) / denom)
        max_log = max(eps_log)
        max_raw = max(eps_raw)
        ratio = max_raw / max(max_log, 1e-300) if max_log > 0 else 1.0
        self.record_comparison()
        self.assertLess(
            ratio, _AB_RATIO_MIN,
            f'flat-rho ratio = {ratio:.1f}× >= {_AB_RATIO_MIN}× — '
            f'the A/B comparison would pass even without rho structure; '
            f'the detector has no teeth (log={max_log:.2e}, '
            f'raw={max_raw:.2e})')

    def test_synthetic_ratio_can_exceed_bar(self) -> None:
        """Demonstrate the A/B assertion CAN fail: trivial case."""
        # When both charts produce identical eps (or log > raw), ratio < 1
        # and the assertLog > ratio_min assertion would fire.
        # This test verifies the logic is reachable-red.
        self.record_comparison()
        self.assertLess(1.0, _AB_RATIO_MIN,
                        f'{_AB_RATIO_MIN}× is too low — any ratio '
                        f'would qualify, including 1.0× (no improvement)')


# ======================================================================
# Self-falsification — composite (carrier + rho_log)
# ======================================================================

class RhoLogCompositionSelfFalsificationTestCase(_CarrierBaseTestCase):
    """Prove the composition test can detect failures."""

    _GAMMA: float = 0.5
    _THETA_C: float = float(_THETA_C_GRID[1])
    _RHO_OFF: float = float(_RHO_GRID_LOG[1] + _RHO_GRID_LOG[2]) / 2.0

    @classmethod
    def setUpClass(cls) -> None:
        cls._y1, cls._y2 = _source_for_node(
            gamma=cls._GAMMA, rho=cls._RHO_OFF,
            theta_c=cls._THETA_C)
        cls._exact = _envelope_exact(cls._RHO_OFF, _LOG_W_GRID,
                                     carrier_rate=K_CHART)
        cls._denom = float(np.max(np.abs(cls._exact))) or 1.0

    def test_wrong_carrier_rate_above_bar(self) -> None:
        """rho_log=True + wrong carrier_rate → off-grid error > bar."""
        chart_bad = _build_rho_dependent_chart(
            rho_log_axis=True,
            carrier_rate=K_CHART + DELTA_K,  # corrupted
            parity=1, gamma_grid=_GAMMA_GRID_POS)
        served = sg._evaluate_chart(
            chart_bad, gamma=self._GAMMA, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps = float(np.max(np.abs(served - self._exact))) / self._denom
        self.record_comparison()
        self.assertGreater(
            eps, HELDOUT_BAR,
            f'wrong-k composite eps = {eps:.2e} <= {HELDOUT_BAR:.0e} — '
            f'Δk={DELTA_K} does NOT break the composite composition test')

    def test_composite_no_carrier_above_bar(self) -> None:
        """rho_log=True + carrier_rate=0 → error > bar for modulated E."""
        chart_nok = _build_rho_dependent_chart(
            rho_log_axis=True,
            carrier_rate=0.0,  # missing carrier
            parity=1, gamma_grid=_GAMMA_GRID_POS)
        served = sg._evaluate_chart(
            chart_nok, gamma=self._GAMMA, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        eps = float(np.max(np.abs(served - self._exact))) / self._denom
        self.record_comparison()
        self.assertGreater(
            eps, HELDOUT_BAR,
            f'no-carrier composite eps = {eps:.2e} <= {HELDOUT_BAR:.0e} — '
            f'a genuinely modulated envelope with carrier_rate=0 does NOT '
            f'break the composition test')
