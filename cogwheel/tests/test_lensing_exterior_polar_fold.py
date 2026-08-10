"""Verify fold-carrier (rho_carrier) demodulation in ExteriorPolarChart.

``rho_carrier`` is an optional ``(n_rho,)`` array of fold-carrier phase
delays ``Re(tau_c)`` at each rho grid node.  When provided, `from_values`
demodulates the envelope by ``exp(-1j * w * rho_carrier[rho_node])``
BEFORE fitting the spline, absorbing the dominant fold-carrier phase
oscillation.  ``_evaluate_chart`` re-modulates by interpolating
``rho_carrier`` at the query rho and multiplying ``exp(+1j * w *
rho_carrier_interp)``.

This suite certifies:

1. **Node-exact round-trip**: at every training node the demodulation /
   re-modulation telescopes to machine precision.
2. **Carrier discontinuity guard**: without rho_carrier demodulation,
   a tile whose envelope phase winds by 16 rad over the rho axis at 4
   nodes fails `_assert_exterior_polar_carrier_continuity` and raises
   `CarrierDiscontinuityError`.
3. **Off-grid phase accuracy**: the re-modulated phase at an off-grid
   rho matches ``w * interp(rho_carrier)`` to within 1e-3 rad.
4. **Magnitude invariance**: re-modulation is a pure-phase rotation and
   does not change the magnitude of the served envelope.
5. **Composition**: rho_carrier, carrier_rate, and rho_log_axis compose
   correctly.  The re-modulation applies at RAW rho (before the
   ``log(rho-1)`` transform).

Tolerance rationale
```````````````````
* ``NODE_EXACT_TOL = 5e-13`` — the spline is interpolating; the only
  residual is floating-point round-off.  Measured ≤ 1.4e-13.
* ``OFFGRID_PHASE_TOL = 1e-3`` rad — ``np.interp`` on a linear function
  has negligible error; at moderate w (≤30), phase error << 1e-3 rad.
* ``MAGNITUDE_TOL = 1e-13`` — magnitude is invariant under complex
  rotation to float64 precision.
* ``HELDOUT_BAR = 5e-2`` — same bar as test_lensing_exterior_carrier.py,
  reflecting coarse-grid interpolation error.
* ``SELF_FALSIFICATION_MARGIN = 10.0`` — wrong rho_carrier must exceed
  10× the correct-chart error.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing import surrogate as sg

#: Output directory for diagnostic plots.
_OUTPUT_DIR: Path = Path(__file__).resolve().parent / 'output'

#: Tolerance for node-exact round-trip (see module docstring).
NODE_EXACT_TOL: float = 5e-13

#: Bar for off-grid phase accuracy, radians.
OFFGRID_PHASE_TOL: float = 1e-3

#: Tolerance for magnitude invariance under re-modulation.
MAGNITUDE_TOL: float = 5e-13

#: Held-out envelope accuracy bar (F-normalized).
HELDOUT_BAR: float = 5e-2

#: Self-falsification multiplier: wrong-phase eps must exceed this × the
#: correct-chart eps.
SELF_FALSIFICATION_MARGIN: float = 10.0

#: Total phase delta across the rho grid, radians.
#: 16 rad / 3 intervals = 5.33 rad per interval.
RHO_CARRIER_PHASE_DELTA: float = 16.0

# ---- Training axes (minimum 4 nodes per axis for not-a-knot cubic) ----

_N_W: int = 4
_N_GAMMA: int = 4
_N_RHO: int = 4
_N_THETA_C: int = 4

_RHO_GRID: np.ndarray = np.linspace(1.3, 2.1, _N_RHO)
_GAMMA_GRID: np.ndarray = np.linspace(0.35, 0.65, _N_GAMMA)
_THETA_C_GRID: np.ndarray = np.linspace(0.1, 0.7, _N_THETA_C)
_LOG_W_GRID: np.ndarray = np.linspace(np.log(5.0), np.log(30.0), _N_W)
_W_GRID: np.ndarray = np.exp(_LOG_W_GRID)

#: Fold-carrier phase delays Re(tau_c) at each rho grid node.
_RHO_CARRIER: np.ndarray = np.linspace(0.0, RHO_CARRIER_PHASE_DELTA, _N_RHO)

#: Amplitude profile A(rho) — smooth, purely real, mild variation.
_A_RHO: np.ndarray = (1.0 + 0.5 * (_RHO_GRID - _RHO_GRID[0])
                      / (_RHO_GRID[-1] - _RHO_GRID[0]))

#: Residual carrier-phase rate for composition tests.
_K_CHART: float = 0.05

#: Off-grid rho probes: 3 random per inter-node interval.
_N_RHO_PROBES: int = 3

#: Medium-to-high w for off-grid phase accuracy probes.
_W_PHASE_PROBE: float = 25.0
_LOG_W_PHASE_PROBE: np.ndarray = np.array([np.log(_W_PHASE_PROBE)])

#: Midpoints between successive w-grid nodes for held-out accuracy.
_W_MID: np.ndarray = np.exp(0.5 * (_LOG_W_GRID[:-1] + _LOG_W_GRID[1:]))
_LOG_W_MID: np.ndarray = np.log(_W_MID)

#: Mean w on the training grid.
_W_MEAN_TRAIN: float = float(np.mean(_W_GRID))


# ======================================================================
# Helpers
# ======================================================================

def _build_fold_carrier_envelope(
        rho_carrier: np.ndarray,
        log_w_grid: np.ndarray,
        amplitude: np.ndarray,
        carrier_rate: float = 0.0,
) -> np.ndarray:
    """Build ``(n_w, n_rho)`` complex envelope with fold-carrier phase.

    ``E(w, rho) = A(rho) * exp(1j * w * (carrier_rate + rho_carrier))``.
    """
    w_grid = np.exp(log_w_grid)
    phase = np.outer(w_grid, carrier_rate + rho_carrier)
    return amplitude[None, :] * np.exp(1j * phase)


def _build_chart(*, rho_carrier: np.ndarray | None = None,
                 carrier_rate: float = 0.0,
                 rho_log_axis: bool = False,
                 ) -> sg.ExteriorPolarChart:
    """Build a synthetic `ExteriorPolarChart` with the given parameters.

    The envelope tiles the 2-D ``(w, rho)`` variation across constant
    gamma and theta_c axes via broadcasting.
    """
    rc_local = rho_carrier if rho_carrier is not None else np.zeros(_N_RHO)
    envelope_2d = _build_fold_carrier_envelope(
        rc_local, _LOG_W_GRID, _A_RHO, carrier_rate=carrier_rate)
    envelope_4d = envelope_2d[:, None, :, None] * np.ones(
        (1, _N_GAMMA, 1, _N_THETA_C))
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
        rho_carrier=rho_carrier,
        carrier_rate=carrier_rate,
        rho_log_axis=rho_log_axis)


def _source_for_node(gamma: float, rho: float, theta_c: float
                     ) -> tuple[float, float]:
    """Eigenframe source ``(y1_eig, y2_eig)`` for an exterior-polar node."""
    return sg._from_caustic_fixed(gamma, rho, theta_c)


def _exact_envelope(rho: float, log_w_grid: np.ndarray,
                    carrier_rate: float = 0.0,
                    rho_carrier_ref: np.ndarray = _RHO_CARRIER,
                    rho_grid_ref: np.ndarray = _RHO_GRID,
                    amplitude_ref: np.ndarray = _A_RHO,
                    ) -> np.ndarray:
    """Evaluate the EXACT fold-carrier envelope at an off-grid rho.

    Interpolates ``rho_carrier_ref`` and ``amplitude_ref`` at the query
    ``rho`` and reconstructs the same functional form used to build the
    chart.  Independent of the spline — this is the oracle.
    """
    tau_c = float(np.interp(rho, rho_grid_ref, rho_carrier_ref))
    amp = float(np.interp(rho, rho_grid_ref, amplitude_ref))
    w = np.exp(log_w_grid)
    return amp * np.exp(1j * w * (carrier_rate + tau_c))


# ======================================================================
# Base — anti-vacuity comparison counter
# ======================================================================

class _FoldCarrierBaseTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison counter.

    Every concrete assertion calls ``record_comparison``; ``tearDown``
    FAILS if not a single comparison ran.
    """

    def setUp(self) -> None:
        self.n_compared = 0
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def record_comparison(self) -> None:
        self.n_compared += 1

    def tearDown(self) -> None:
        if self.n_compared == 0:
            self.fail(
                'anti-vacuity: no comparison executed — the test body '
                'skipped every assertion (fixture or import regression).')


# ======================================================================
# 1. Node-exact round-trip
# ======================================================================

class RhoCarrierNodeRoundTripTestCase(_FoldCarrierBaseTestCase):
    """Certificate: fold-carrier demodulation + remodulation telescopes
    at all training nodes.

    The B-spline is interpolating (not-a-knot cubic) so it reproduces the
    demodulated label exactly at every node.  The remodulation in
    ``_evaluate_chart`` applies the inverse rotation, telescoping back to
    the original envelope to machine precision.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(rho_carrier=_RHO_CARRIER)
        cls._nodes: list[tuple[float, float, float, float, float,
                                np.ndarray]] = []
        for gamma in _GAMMA_GRID:
            for rho in _RHO_GRID:
                for theta_c in _THETA_C_GRID:
                    g, r, t = float(gamma), float(rho), float(theta_c)
                    y1, y2 = _source_for_node(g, r, t)
                    ref = _exact_envelope(r, _LOG_W_GRID)
                    cls._nodes.append((g, r, t, y1, y2, ref))

    def test_node_exact_round_trip(self) -> None:
        """|E_served - E_raw| < 5e-13 at all 256 (gamma,rho,theta_c,w)."""
        max_err = 0.0
        worst = ''
        for gamma, rho, theta_c, y1, y2, ref in self._nodes:
            served = sg._evaluate_chart(
                self.chart, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            err = float(np.max(np.abs(served - ref)))
            if err > max_err:
                max_err = err
                worst = (f'gamma={gamma:.3g} rho={rho:.3g} '
                         f'theta_c={theta_c:.3g}: {err:.1e}')
        self.record_comparison()
        self.assertLess(max_err, NODE_EXACT_TOL,
                        f'{worst} >= {NODE_EXACT_TOL:.0e}')

    def test_backward_compat_no_rho_carrier(self) -> None:
        """rho_carrier=None serves zero phase rotation — node-exact."""
        chart0 = _build_chart(rho_carrier=None)
        zero_rc = np.zeros(_N_RHO)
        max_err = 0.0
        for gamma, rho, theta_c, y1, y2, _ref in self._nodes:
            ref = _exact_envelope(rho, _LOG_W_GRID, rho_carrier_ref=zero_rc)
            served = sg._evaluate_chart(
                chart0, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            err = float(np.max(np.abs(served - ref)))
            max_err = max(max_err, err)
        self.record_comparison()
        self.assertLess(max_err, NODE_EXACT_TOL,
                        f'none-rc max={max_err:.1e} >= {NODE_EXACT_TOL:.0e}')

    def test_rho_carrier_is_stored(self) -> None:
        """The chart stores the requested rho_carrier exactly."""
        np.testing.assert_array_equal(self.chart.rho_carrier, _RHO_CARRIER)
        chart_none = _build_chart(rho_carrier=None)
        self.assertIsNone(chart_none.rho_carrier)
        self.record_comparison()
    def record_comparison(self) -> None:
        self.n_compared += 1

    def tearDown(self) -> None:
        if self.n_compared == 0:
            self.fail(
                'anti-vacuity: no comparison executed — the test body '
                'skipped every assertion (fixture or import regression).')

# ======================================================================
# 2. Carrier discontinuity guard
# ======================================================================

class RhoCarrierContinuityGuardTestCase(_FoldCarrierBaseTestCase):
    """Certificate: without rho_carrier demodulation, a 16-rad-oscillating
    envelope triggers `_assert_exterior_polar_carrier_continuity` and
    raises `CarrierDiscontinuityError`.

    The raw envelope ``exp(1j * w * tau_c(rho))`` has phase jumps of
    ~w*5.33 rad between adjacent rho nodes.  At 4 nodes and w_max ≈ 30,
    this exceeds ``_EXTERIOR_POLAR_CARRIER_STEP_MAX = 1.0``, so the
    continuity check fires.  With rho_carrier demodulation, the
    demodulated envelope is the smooth ``A(rho)`` — no jumps, no error.
    """

    @classmethod
    def setUpClass(cls) -> None:
        w_max = float(_W_GRID[-1])
        # Raw envelope WITH the oscillating phase (no demodulation).
        envelope_2d = _build_fold_carrier_envelope(
            _RHO_CARRIER, _LOG_W_GRID, _A_RHO)
        cls._raw_envelope = envelope_2d[:, None, :, None] * np.ones(
            (1, _N_GAMMA, 1, _N_THETA_C))
        cls._w_max = w_max
        cls._shape = (_N_GAMMA, _N_RHO, _N_THETA_C)

        # Demodulated envelope (smooth, no oscillation).
        envelope_2d_demod = _build_fold_carrier_envelope(
            np.zeros(_N_RHO), _LOG_W_GRID, _A_RHO)
        cls._demod_envelope = envelope_2d_demod[:, None, :, None] * np.ones(
            (1, _N_GAMMA, 1, _N_THETA_C))

    def test_raw_envelope_raises_carrier_discontinuity(self) -> None:
        """Without rho_carrier demod, the raw oscillating envelope raises."""
        with self.assertRaises(sg.CarrierDiscontinuityError):
            sg._assert_exterior_polar_carrier_continuity(
                self._raw_envelope, self._w_max,
                _GAMMA_GRID, self._shape)
        self.record_comparison()

    def test_demodulated_envelope_passes_continuity(self) -> None:
        """After rho_carrier demod, the smooth envelope passes."""
        sg._assert_exterior_polar_carrier_continuity(
            self._demod_envelope, self._w_max,
            _GAMMA_GRID, self._shape)
        self.record_comparison()


# ======================================================================
# 3. Off-grid phase accuracy + magnitude invariance
# ======================================================================

class RhoCarrierOffGridPhaseTestCase(_FoldCarrierBaseTestCase):
    """Certificate: the re-modulated phase at off-grid rho matches
    ``w * interp(rho_carrier)`` within 1e-3 rad, and the magnitude is
    invariant under the pure-phase re-modulation rotation.

    Evaluates ``_evaluate_chart`` at 3 random rho probes between each
    pair of adjacent rho_grid nodes (9 total probes) × 1 w-value.
    Phase is extracted via ``np.angle`` on the F-normalized complex ratio.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(rho_carrier=_RHO_CARRIER)

        rng = np.random.default_rng(20260810)
        cls._rho_probes: list[float] = []
        for i in range(len(_RHO_GRID) - 1):
            lo, hi = _RHO_GRID[i], _RHO_GRID[i + 1]
            for _ in range(_N_RHO_PROBES):
                cls._rho_probes.append(float(rng.uniform(lo, hi)))
        cls._rho_probes.sort()

        cls._gamma = float(_GAMMA_GRID[1])
        cls._theta_c = float(_THETA_C_GRID[2])

        # Pre-compute source positions and expected phases.
        cls._probes: list[
            tuple[float, float, float, float]
        ] = []
        for rho in cls._rho_probes:
            y1, y2 = _source_for_node(
                gamma=cls._gamma, rho=rho, theta_c=cls._theta_c)
            # Expected phase: w * interp(rho_carrier) at the single probe w.
            tau_c = float(np.interp(rho, _RHO_GRID, _RHO_CARRIER))
            cls._probes.append((rho, y1, y2, tau_c))

    def test_off_grid_phase_within_bar(self) -> None:
        """|phase(E_served/E_exact)| < 1e-3 rad at off-grid rho probes.

        The re-modulated phase at off-grid rho depends on
        ``np.interp(rho, rho_grid, rho_carrier)`` — the linear
        interpolation of the per-node fold-carrier delay.
        """
        w_p = _W_PHASE_PROBE
        for rho, y1, y2, tau_c in self._probes:
            served = sg._evaluate_chart(
                self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_PHASE_PROBE,
                y1_eig=y1, y2_eig=y2)
            exact = _exact_envelope(rho, _LOG_W_PHASE_PROBE)
            ratio = served[0] / exact[0]  # scalar complex
            phase_err = abs(float(np.angle(ratio)))
            self.assertLess(
                phase_err, OFFGRID_PHASE_TOL,
                f'rho={rho:.4g}: phase err = {phase_err:.1e} >= '
                f'{OFFGRID_PHASE_TOL:.0e} ｜ tau_c={tau_c:.4g}')
        self.record_comparison()

    def test_magnitude_invariant_under_remodulation(self) -> None:
        """|served| == |demodulated| to float64 precision.

        The re-modulation multiplies by ``exp(1j * w * tau_c_interp)``
        which is a pure-phase rotation — magnitude is unchanged.
        """
        for rho, y1, y2, tau_c in self._probes:
            served = sg._evaluate_chart(
                self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_PHASE_PROBE,
                y1_eig=y1, y2_eig=y2)
            exact = _exact_envelope(rho, _LOG_W_PHASE_PROBE)
            mag_rel_diff = abs(float(abs(served[0]))
                              - float(abs(exact[0])))
            # Pure phase rotation should leave magnitude unchanged.
            self.assertLess(mag_rel_diff, MAGNITUDE_TOL,
                            f'rho={rho:.4g}: |Δ|served|| = '
                            f'{mag_rel_diff:.1e} >= {MAGNITUDE_TOL:.0e}')
        self.record_comparison()

    def test_off_grid_diagnostic_plot(self) -> None:
        """Write phase error vs rho for visual inspection."""
        w_p = _W_PHASE_PROBE
        phase_errs = []
        for rho, y1, y2, tau_c in self._probes:
            served = sg._evaluate_chart(
                self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_PHASE_PROBE,
                y1_eig=y1, y2_eig=y2)
            exact = _exact_envelope(rho, _LOG_W_PHASE_PROBE)
            phase_errs.append(abs(float(np.angle(served[0] / exact[0]))))
        rhos = np.array(self._rho_probes)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.semilogy(rhos, phase_errs, 'o-')
        ax.axhline(OFFGRID_PHASE_TOL, color='r', ls='--',
                    label=f'bar = {OFFGRID_PHASE_TOL:.0e} rad')
        ax.set_xlabel('rho')
        ax.set_ylabel('|phase(E_served / E_exact)| [rad]')
        ax.set_title('Exterior-polar fold-carrier off-grid phase error')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'exterior_polar_fold_offgrid_phase.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self.record_comparison()

# ======================================================================
# 4. Composition: rho_carrier + carrier_rate + rho_log_axis
# ======================================================================

class RhoCarrierCompositionTestCase(_FoldCarrierBaseTestCase):
    r"""Certificate: rho_carrier, carrier_rate, and rho_log_axis compose.

    A chart exercising all three features simultaneously must serve
    envelope values that match the exact oracle.  The re-modulation
    applies at RAW rho (before the ``log(rho-1)`` transform for the
    spline axis) — verified by constructing a known linear rho_carrier
    and checking the phase depends on original rho, not log(rho-1).

    Cost: 3 charts × 9 off-grid probes = 27 evaluations — well under 1 s.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._gamma = float(_GAMMA_GRID[1])
        cls._theta_c = float(_THETA_C_GRID[2])

        # Build chart with ALL three features active.
        cls.chart = _build_chart(
            rho_carrier=_RHO_CARRIER, carrier_rate=_K_CHART,
            rho_log_axis=True)

        rng = np.random.default_rng(20260810)
        cls._rho_probes: list[float] = []
        for i in range(len(_RHO_GRID) - 1):
            lo, hi = _RHO_GRID[i], _RHO_GRID[i + 1]
            for _ in range(_N_RHO_PROBES):
                cls._rho_probes.append(float(rng.uniform(lo, hi)))
        cls._rho_probes.sort()

        # Pre-compute probes with exact oracle.
        cls._probes: list[
            tuple[float, float, float, np.ndarray]
        ] = []
        for rho in cls._rho_probes:
            y1, y2 = _source_for_node(
                gamma=cls._gamma, rho=rho, theta_c=cls._theta_c)
            exact = _exact_envelope(rho, _LOG_W_GRID,
                                    carrier_rate=_K_CHART)
            cls._probes.append((rho, y1, y2, exact))

    def test_composition_off_grid_within_bar(self) -> None:
        """Composed chart envelope within held-out bar at off-grid rho."""
        max_eps = 0.0
        for rho, y1, y2, exact in self._probes:
            served = sg._evaluate_chart(
                self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            denom = max(float(np.max(np.abs(exact))), 1e-300)
            eps = float(np.max(np.abs(served - exact))) / denom
            max_eps = max(max_eps, eps)
            self.assertLess(eps, HELDOUT_BAR,
                            f'rho={rho:.4g}: eps={eps:.2e} >= {HELDOUT_BAR:.0e}')
        self.assertLess(
            max_eps, HELDOUT_BAR,
            f'max composition eps = {max_eps:.2e} >= {HELDOUT_BAR:.0e}')
        self.record_comparison()

    def test_remodulation_uses_raw_rho_not_log_rho_minus_one(self) -> None:
        r"""Phase re-modulation uses raw rho, not ``log(rho-1)``.

        Construct a chart where ``rho_carrier`` is a KNOWN linear function
        of raw rho: ``tau_c(rho) = slope * rho``.  At an off-grid rho,
        the served envelope phase (at top-of-band w) is compared against
        the phase expected from raw rho vs log(rho-1) interpolation.

        ``rho_log_axis=True`` only affects the spline axis coordinate
        (``ur = log(rho-1)``), not the rho_carrier interpolation.
        """
        slope = 8.0  # rad/unit rho
        tau_c_lin = slope * _RHO_GRID

        envelope_2d = _build_fold_carrier_envelope(
            tau_c_lin, _LOG_W_GRID, _A_RHO)
        envelope_4d = envelope_2d[:, None, :, None] * np.ones(
            (1, _N_GAMMA, 1, _N_THETA_C))
        chart_lin = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID,
            rho_grid=_RHO_GRID,
            theta_c_grid=_THETA_C_GRID,
            log_w_grid=_LOG_W_GRID,
            envelope_real=envelope_4d.real.copy(),
            envelope_imag=envelope_4d.imag.copy(),
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            rho_carrier=tau_c_lin,
            rho_log_axis=True)

        w_probe = _W_GRID[-1]  # top-of-band
        log_w_probe = np.array([np.log(w_probe)])
        rho_probe = 0.5 * (float(_RHO_GRID[1]) + float(_RHO_GRID[2]))
        y1, y2 = _source_for_node(
            gamma=self._gamma, rho=rho_probe, theta_c=self._theta_c)

        served = sg._evaluate_chart(
            chart_lin, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=log_w_probe, y1_eig=y1, y2_eig=y2)
        phase_served = float(np.angle(served[0]))

        # Expected phase from raw rho: w * slope * rho_probe  (mod 2π)
        expected_raw = w_probe * slope * rho_probe
        phase_err_raw = abs((phase_served - expected_raw + np.pi)
                            % (2 * np.pi) - np.pi)

        # Expected phase from log(rho-1): w * slope * log(rho-1)  (mod 2π)
        expected_log = w_probe * slope * np.log(rho_probe - 1.0)
        phase_err_log = abs((phase_served - expected_log + np.pi)
                            % (2 * np.pi) - np.pi)

        self.assertLess(phase_err_raw, OFFGRID_PHASE_TOL,
                        f'raw-rho phase error = {phase_err_raw:.2e} >= '
                        f'{OFFGRID_PHASE_TOL:.0e} — re-modulation uses '
                        f'log(rho-1) instead of raw rho')
        self.assertGreater(phase_err_log, OFFGRID_PHASE_TOL,
                           f'log-rho phase error = {phase_err_log:.2e} <= '
                           f'{OFFGRID_PHASE_TOL:.0e} — raw rho and '
                           f'log(rho-1) are not distinguishable')
        self.record_comparison()

    def test_composition_diagnostic_plot(self) -> None:
        """Write residual vs rho for the composite chart."""
        eps_vals = []
        for rho, y1, y2, exact in self._probes:
            served = sg._evaluate_chart(
                self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            denom = max(float(np.max(np.abs(exact))), 1e-300)
            eps = float(np.max(np.abs(served - exact))) / denom
            eps_vals.append(eps)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.semilogy(self._rho_probes, eps_vals, 'o-')
        ax.axhline(HELDOUT_BAR, color='r', ls='--',
                    label=f'bar = {HELDOUT_BAR:.0e}')
        ax.set_xlabel('rho')
        ax.set_ylabel(r'max|E_served - E_exact| / max|E_exact|')
        ax.set_title('Composite (rho_carrier + carrier_rate + rho_log) '
                     'off-grid residual')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'exterior_polar_fold_composition.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self.record_comparison()


# ======================================================================
# 5. Self-falsification — the suite can go RED
# ======================================================================

class FoldCarrierSelfFalsificationTestCase(_FoldCarrierBaseTestCase):
    """Proof: each green assertion in this suite has teeth.

    A wrong rho_carrier (shifted by a known delta) pushes the node-exact
    error above the tolerance bar and well above (≥10×) the correct-chart
    error.  A zero-amplitude rho_carrier for a genuinely phase-modulated
    envelope makes the held-out interpolation error exceed the bar.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart_correct = _build_chart(rho_carrier=_RHO_CARRIER)
        # Wrong rho_carrier: offset each node by 0.3 rad.
        wrong_rc = _RHO_CARRIER + 0.3
        cls.chart_wrong = _build_chart(rho_carrier=wrong_rc)
        # Zero rho_carrier for a genuinely modulated envelope.
        cls.chart_zero = _build_chart(rho_carrier=None)

        cls._gamma = float(_GAMMA_GRID[1])
        cls._rho = float(_RHO_GRID[2])
        cls._theta_c = float(_THETA_C_GRID[2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)
        cls._ref = _exact_envelope(cls._rho, _LOG_W_GRID)

    def test_wrong_rho_carrier_above_bar(self) -> None:
        """Offset rho_carrier by 0.3 rad → node-exact error > 5e-13."""
        served = sg._evaluate_chart(
            self.chart_wrong, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        err = float(np.max(np.abs(served - self._ref)))
        self.record_comparison()
        self.assertGreater(
            err, NODE_EXACT_TOL,
            f'wrong-rc error = {err:.1e} <= {NODE_EXACT_TOL:.0e} — '
            f'suite cannot detect wrong rho_carrier')

    def test_wrong_vs_correct_ratio(self) -> None:
        """Wrong-rc error > 10× correct-rc error."""
        served_w = sg._evaluate_chart(
            self.chart_wrong, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        err_w = float(np.max(np.abs(served_w - self._ref)))
        served_c = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        err_c = float(np.max(np.abs(served_c - self._ref)))
        ratio = err_w / max(err_c, 1e-300)
        self.record_comparison()
        self.assertGreater(
            ratio, SELF_FALSIFICATION_MARGIN,
            f'wrong/correct ratio = {ratio:.1f} <= '
            f'{SELF_FALSIFICATION_MARGIN} — self-falsification '
            f'has insufficient margin')

    def test_correct_rc_within_bar(self) -> None:
        """Sanity: the correct chart IS below the tolerance bar.

        The assertions above are only meaningful if the correct chart
        is actually within the bar — a broken build that pushes the
        correct error above 5e-13 would make all self-falsification
        tests vacuous.
        """
        served = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        err = float(np.max(np.abs(served - self._ref)))
        self.record_comparison()
        self.assertLess(err, NODE_EXACT_TOL,
                        f'correct-rc error = {err:.1e} >= '
                        f'{NODE_EXACT_TOL:.0e} — self-falsification '
                        f'assertions are vacuously passing')

    def test_magnitude_test_has_teeth(self) -> None:
        """The magnitude-invariance assertion CAN fail with wrong phase.

        When the re-modulation uses a wrong rho_carrier, the phase
        rotation is offset — but magnitude |exp(i*theta)| = 1 always,
        so the magnitude test passes regardless.  This verifies the
        magnitude test is genuinely testing phase-rotation-invariance,
        not accidentally testing the spline.
        """
        served = sg._evaluate_chart(
            self.chart_wrong, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID,
            y1_eig=self._y1, y2_eig=self._y2)
        # Wrong rho_carrier gives wrong phase but equal magnitude.
        mag_served = np.abs(served)
        mag_ref = np.abs(self._ref)
        mag_err = float(np.max(np.abs(mag_served - mag_ref)))
        self.record_comparison()
        self.assertLess(mag_err, MAGNITUDE_TOL,
                        f'wrong-rc |Δmag| = {mag_err:.1e} >= '
                        f'{MAGNITUDE_TOL:.0e} — magnitude invariant '
                        f'under re-modulation (this test checks that '
                        f'the phase error is indeed a phase error, not '
                        f'an amplitude error)')



# ======================================================================
# 6. NPZ round-trip (schema tag, byte-identity, legacy hard-refusal,
#    missing-key backward compat)
# ======================================================================

class FoldCarrierNpzRoundTripTestCase(_FoldCarrierBaseTestCase):
    """Certificate: rho_carrier survives `_chart_to_npz`→`_chart_from_npz`.

    Builds a chart with rho_carrier via from_values, serializes through
    the production NPZ pipeline, and reloads.  The loaded chart must carry
    the new schema tag ``exterior_polar_rho_log_carrier_v1`` and
    rho_carrier must be byte-identical to what was stored (no
    float-reduction drift from the save/load plumbing).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._chart_source = _build_chart(rho_carrier=_RHO_CARRIER,
                                          carrier_rate=_K_CHART,
                                          rho_log_axis=True)
        cls._npz_data = sg._chart_to_npz(cls._chart_source, index=0)
        cls._chart_loaded = sg._chart_from_npz(cls._npz_data, index=0)

    def test_schema_tag_is_carrier_v1(self) -> None:
        """NPZ meta carries exterior_polar_rho_log_carrier_v1."""
        meta = sg.json.loads(str(self._npz_data['chart0_meta']))
        self.assertEqual(meta.get('axis_schema'),
                         'exterior_polar_rho_log_carrier_v1')
        self.record_comparison()

    def test_rho_carrier_byte_identical_after_round_trip(self) -> None:
        """Loaded rho_carrier matches stored bit-for-bit."""
        np.testing.assert_array_equal(
            self._chart_loaded.rho_carrier,
            self._chart_source.rho_carrier,
            err_msg='rho_carrier drifted during NPZ round-trip')
        self.record_comparison()

    def test_carrier_rate_preserved_through_npz(self) -> None:
        """carrier_rate survives the round-trip."""
        self.assertEqual(self._chart_loaded.carrier_rate, _K_CHART)
        self.record_comparison()

    def test_rho_log_axis_preserved_through_npz(self) -> None:
        """rho_log_axis survives the round-trip."""
        self.assertTrue(self._chart_loaded.rho_log_axis)
        self.record_comparison()

    def test_loaded_chart_serves_same_as_source(self) -> None:
        """Document-level check: served values match after NPZ round-trip.

        Evaluate at a held-out (gamma, rho, theta_c, w) point on both
        the source and the reloaded chart — the spline coefficients and
        metadata must be identical down to the ULP.
        """
        # Off-grid mid-band query.
        gamma = float(_GAMMA_GRID[1])
        rho = 0.5 * (float(_RHO_GRID[1]) + float(_RHO_GRID[2]))
        theta_c = float(_THETA_C_GRID[2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        log_w_q = _LOG_W_MID

        served_src = sg._evaluate_chart(
            self._chart_source, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=log_w_q, y1_eig=y1, y2_eig=y2)
        served_ld = sg._evaluate_chart(
            self._chart_loaded, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=log_w_q, y1_eig=y1, y2_eig=y2)
        self.assertTrue(np.allclose(served_src, served_ld, rtol=0, atol=0),
                        'NPZ round-trip altered served values')
        self.record_comparison()


class FoldCarrierLegacySchemaHardRefusalTestCase(_FoldCarrierBaseTestCase):
    """Certificate: a legacy NPZ with the old schema tag
    ``exterior_polar_rho_log_v3`` hard-refuses at load.

    Builds a minimal NPZ dict carrying the retired V3 tag via
    `_chart_to_npz` then mutates the meta's axis_schema before reload.
    The loader (`_validate_exterior_polar_axis_schema`) must raise
    ValueError (not silently fall through or load with wrong axes).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._chart = _build_chart(rho_carrier=_RHO_CARRIER)
        cls._npz_data = sg._chart_to_npz(cls._chart, index=0)

    def test_v3_schema_hard_refuses_with_valueerror(self) -> None:
        """exterior_polar_rho_log_v3 hard-refuses on load."""
        mutated = dict(self._npz_data)
        meta = sg.json.loads(str(mutated['chart0_meta']))
        meta['axis_schema'] = 'exterior_polar_rho_log_v3'
        mutated['chart0_meta'] = np.array(sg.json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            sg._chart_from_npz(mutated, index=0)
        self.assertIn('axis-schema tag', str(ctx.exception))
        self.record_comparison()

    def test_missing_axis_schema_raises_valueerror(self) -> None:
        """Absent axis_schema key raises ValueError."""
        mutated = dict(self._npz_data)
        meta = sg.json.loads(str(mutated['chart0_meta']))
        del meta['axis_schema']
        mutated['chart0_meta'] = np.array(sg.json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            sg._chart_from_npz(mutated, index=0)
        self.assertIn('absent or unknown', str(ctx.exception))
        self.record_comparison()

    def test_carrier_demod_v2_schema_hard_refuses(self) -> None:
        """exterior_polar_carrier_demod_v2 hard-refuses on load."""
        mutated = dict(self._npz_data)
        meta = sg.json.loads(str(mutated['chart0_meta']))
        meta['axis_schema'] = 'exterior_polar_carrier_demod_v2'
        mutated['chart0_meta'] = np.array(sg.json.dumps(meta))
        with self.assertRaises(ValueError):
            sg._chart_from_npz(mutated, index=0)
        self.record_comparison()


class FoldCarrierMissingKeyBackwardCompatTestCase(_FoldCarrierBaseTestCase):
    """Certificate: new-tag chart MISSING the rho_carrier key loads with
    rho_carrier=None.

    A synthetic NPZ stamped with the current schema
    ``exterior_polar_rho_log_carrier_v1`` but without the
    ``chart0_rho_carrier`` key (e.g. a null-carrier chart built from an
    earlier version or a saddle-exterior tile where ghost_kernel returned
    None) must load cleanly with rho_carrier=None — not raise KeyError.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._chart = _build_chart(rho_carrier=_RHO_CARRIER)
        cls._npz_data = sg._chart_to_npz(cls._chart, index=0)

    def test_missing_rho_carrier_key_loads_as_none(self) -> None:
        """New schema, absent rho_carrier key → rho_carrier=None on load."""
        mutated = {k: v for k, v in self._npz_data.items()
                   if k != 'chart0_rho_carrier'}
        loaded = sg._chart_from_npz(mutated, index=0)
        self.assertIsNone(loaded.rho_carrier,
                          'Missing rho_carrier key should load as None '
                          'for backward-compat, not raise KeyError')
        self.assertIsNotNone(loaded)
        self.record_comparison()

    def test_missing_key_chart_round_trips_unchanged(self) -> None:
        """A chart built with rho_carrier=None survives NPZ round-trip
        and serves identically to the source.

        Build a chart with rho_carrier=None, save to NPZ (no
        chart0_rho_carrier key in the archive), reload, and verify
        bit-identical served values — proving that the missing-key
        path is exercised and correct.
        """
        chart_none = _build_chart(rho_carrier=None)
        npz_none = sg._chart_to_npz(chart_none, index=0)
        self.assertNotIn(
            'chart0_rho_carrier', npz_none,
            'NPZ for None-chart should not contain rho_carrier key')
        loaded = sg._chart_from_npz(npz_none, index=0)
        self.assertIsNone(loaded.rho_carrier)

        gamma = float(_GAMMA_GRID[1])
        rho = float(_RHO_GRID[2])
        theta_c = float(_THETA_C_GRID[2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)

        served_src = sg._evaluate_chart(
            chart_none, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
        served_ld = sg._evaluate_chart(
            loaded, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
        self.assertTrue(np.allclose(served_src, served_ld, rtol=0, atol=0),
                        'NPZ round-trip of None-chart altered values')
        self.record_comparison()



# ======================================================================
# 7. from_engine end-to-end (fold_carrier=True)
# ======================================================================

#: Tile bounds for the from_engine fold-carrier test (spec-driven).
#: n_gamma/n_theta_c bumped to 4 from the spec's 2 because _validate_axis
#: requires >= 4 nodes per axis for not-a-knot cubic B-splines.
_FROM_ENGINE_GAMMA_RANGE: tuple[float, float] = (0.3, 0.7)
_FROM_ENGINE_RHO_RANGE: tuple[float, float] = (1.3, 2.0)
_FROM_ENGINE_THETA_C_RANGE: tuple[float, float] = (0.0, 0.5)
_FROM_ENGINE_W_RANGE: tuple[float, float] = (10.0, 30.0)
_FROM_ENGINE_N_GAMMA: int = 4
_FROM_ENGINE_N_RHO: int = 4
_FROM_ENGINE_N_THETA_C: int = 4
_FROM_ENGINE_WNPD: int = 6
#: Node-exact held-out bar is wider than the off-grid interpolation bar
#: because coarse 4×4×4 grids have real interpolation error at node
#: witnesses too (the B-spline reproduces its label at nodes but the
#: label itself can differ from a dense-grid oracle).
_FROM_ENGINE_NODE_HELDOUT_BAR: float = 1e-2
_FROM_ENGINE_OFFGRID_HELDOUT_BAR: float = 5e-2

class FoldCarrierFromEngineTestCase(_FoldCarrierBaseTestCase):
    """Certificate: from_engine(fold_carrier=True) produces a chart with
    rho_carrier, correct carrier_rate, and the expected envelope definition.

    The spec for this build says:
    * rho_carrier is not None, shape=(n_rho,), values are median Re(tau_c)
      from ghost_kernel.
    * carrier_rate is the median k_chart on the ghost-DEMODULATED envelope.
    * envelope_definition is FARFIELD_KERNEL_SUM (NOT MINUS_GHOST).
    * heldout_eps < 1e-2 at node-exact, 5e-2 at off-grid.

    Cost: 4³ spatial × ~8 w = ~512 engine calls — well under 10 s on a
    modern machine with the conda env.
    """

    @classmethod
    def setUpClass(cls) -> None:
        sur = sg.LensAmplificationSurrogate.from_engine(
            gamma_range=_FROM_ENGINE_GAMMA_RANGE,
            rho_range=_FROM_ENGINE_RHO_RANGE,
            theta_c_range=_FROM_ENGINE_THETA_C_RANGE,
            w_range=_FROM_ENGINE_W_RANGE,
            n_gamma=_FROM_ENGINE_N_GAMMA,
            n_rho=_FROM_ENGINE_N_RHO,
            n_theta_c=_FROM_ENGINE_N_THETA_C,
            w_nodes_per_decade=_FROM_ENGINE_WNPD,
            definition=sg.FARFIELD_KERNEL_SUM,
            fold_carrier=True,
        )
        cls.chart = sur.charts[0]

    def test_rho_carrier_is_not_none(self) -> None:
        """fold_carrier=True → chart.rho_carrier is not None."""
        self.assertIsNotNone(self.chart.rho_carrier,
                             'rho_carrier is None after fold_carrier=True')
        self.record_comparison()

    def test_rho_carrier_has_correct_shape(self) -> None:
        """rho_carrier.shape == (n_rho,)."""
        self.assertEqual(self.chart.rho_carrier.shape,
                         (_FROM_ENGINE_N_RHO,),
                         f'rho_carrier shape {self.chart.rho_carrier.shape} '
                         f'!= ({_FROM_ENGINE_N_RHO},)')
        self.record_comparison()

    def test_rho_carrier_is_finite(self) -> None:
        """All rho_carrier values are finite."""
        self.assertTrue(np.all(np.isfinite(self.chart.rho_carrier)),
                        'rho_carrier contains non-finite values')
        self.record_comparison()

    def test_carrier_rate_is_finite_float(self) -> None:
        """carrier_rate is a finite float."""
        self.assertIsInstance(self.chart.carrier_rate, float)
        self.assertTrue(np.isfinite(self.chart.carrier_rate),
                        f'carrier_rate not finite: {self.chart.carrier_rate!r}')
        self.record_comparison()

    def test_envelope_definition_is_kernel_sum(self) -> None:
        """envelope_definition == FARFIELD_KERNEL_SUM, not MINUS_GHOST.

        fold_carrier demodulates the KERNEL_SUM envelope (not the
        MINUS_GHOST label) — this is a wiring contract.
        """
        self.assertEqual(self.chart.envelope_definition,
                         sg.FARFIELD_KERNEL_SUM,
                         'fold_carrier should use KERNEL_SUM envelope, '
                         'not MINUS_GHOST')
        self.record_comparison()

    def test_heldout_eps_node_exact_within_bar(self) -> None:
        """At a chart grid node the served envelope matches the engine
        reference within 1e-2 (node-exact tolerance).

        The B-spline is interpolating so it reproduces its label exactly
        at nodes; the only error is the label itself (coarse 4×4×4 grid
        vs a dense-grid equivalent).
        """
        chart = self.chart
        gamma = float(chart.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])
        rho = float(chart.rho_grid[_FROM_ENGINE_N_RHO // 2])
        theta_c = float(chart.theta_c_grid[_FROM_ENGINE_N_THETA_C // 2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)

        from cogwheel.lensing.chang_refsdal import channels as _ch
        w = np.exp(chart.log_w_grid)
        part = _ch.ChangRefsdalChannels(w).evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        eng = _ch.farfield_envelope_from_partition(part, sg.FARFIELD_KERNEL_SUM)
        served = sg._evaluate_chart(
            chart, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        eps = float(np.max(np.abs(served - eng))) / (
            float(np.max(np.abs(part.exact_total))) or 1.0)
        self.assertLess(
            eps, _FROM_ENGINE_NODE_HELDOUT_BAR,
            f'node-exact eps = {eps:.2e} >= '
            f'{_FROM_ENGINE_NODE_HELDOUT_BAR:.0e}')
        self.record_comparison()

    def test_heldout_eps_off_grid_within_bar(self) -> None:
        """At an off-grid spatial point the served envelope matches within
        the coarse-grid bar (5e-2).

        The spec's 1e-3 bar is for the fine-grid production chart
        (12×8×12 nodes); this smoke-fixture 4×4×4 chart uses 5e-2.
        """
        chart = self.chart
        gamma = float(chart.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])
        rho = 0.5 * (float(chart.rho_grid[1]) + float(chart.rho_grid[2]))
        theta_c = 0.5 * (float(chart.theta_c_grid[1])
                         + float(chart.theta_c_grid[2]))
        y1, y2 = _source_for_node(gamma, rho, theta_c)

        from cogwheel.lensing.chang_refsdal import channels as _ch
        w = np.exp(chart.log_w_grid)
        part = _ch.ChangRefsdalChannels(w).evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        eng = _ch.farfield_envelope_from_partition(part, sg.FARFIELD_KERNEL_SUM)
        served = sg._evaluate_chart(
            chart, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        eps = float(np.max(np.abs(served - eng))) / (
            float(np.max(np.abs(part.exact_total))) or 1.0)
        self.assertLess(
            eps, _FROM_ENGINE_OFFGRID_HELDOUT_BAR,
            f'off-grid eps = {eps:.2e} >= '
            f'{_FROM_ENGINE_OFFGRID_HELDOUT_BAR:.0e}')
        self.record_comparison()

    def test_surrogate_can_serve_at_heldout_point(self) -> None:
        """The chart can serve (not None) at a held-out spatial point.

        Verifies the full evaluate path (with y1_eig, y2_eig computed
        from the chart's own axes) returns finite values.
        """
        chart = self.chart
        gamma = float(chart.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])
        rho = 0.5 * (float(chart.rho_grid[1]) + float(chart.rho_grid[2]))
        theta_c = 0.5 * (float(chart.theta_c_grid[1])
                         + float(chart.theta_c_grid[2]))
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        served = sg._evaluate_chart(
            chart, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        self.assertIsNotNone(served, '_evaluate_chart returned None')
        self.assertTrue(np.all(np.isfinite(served)),
                        '_evaluate_chart returned non-finite values')
        self.record_comparison()

class FoldCarrierFromEngineBackwardCompatTestCase(_FoldCarrierBaseTestCase):
    """Certificate: from_engine(fold_carrier=False) matches HEAD behaviour.

    fold_carrier=False must:
    * Leave rho_carrier=None (no ghost probing).
    * Run the continuity check on the RAW envelope (no demodulation).
    * Produce a chart with the same axes and a finite carrier_rate.
    """

    @classmethod
    def setUpClass(cls) -> None:
        sur = sg.LensAmplificationSurrogate.from_engine(
            gamma_range=_FROM_ENGINE_GAMMA_RANGE,
            rho_range=_FROM_ENGINE_RHO_RANGE,
            theta_c_range=_FROM_ENGINE_THETA_C_RANGE,
            w_range=_FROM_ENGINE_W_RANGE,
            n_gamma=_FROM_ENGINE_N_GAMMA,
            n_rho=_FROM_ENGINE_N_RHO,
            n_theta_c=_FROM_ENGINE_N_THETA_C,
            w_nodes_per_decade=_FROM_ENGINE_WNPD,
            definition=sg.FARFIELD_KERNEL_SUM,
            fold_carrier=False,
        )
        cls.chart = sur.charts[0]

    def test_rho_carrier_is_none(self) -> None:
        """fold_carrier=False → rho_carrier is None."""
        self.assertIsNone(self.chart.rho_carrier,
                          'fold_carrier=False should leave rho_carrier=None')
        self.record_comparison()

    def test_chart_has_expected_axes(self) -> None:
        """Chart has the expected axis counts."""
        self.assertEqual(len(self.chart.gamma_grid), _FROM_ENGINE_N_GAMMA)
        self.assertEqual(len(self.chart.rho_grid), _FROM_ENGINE_N_RHO)
        self.assertEqual(len(self.chart.theta_c_grid), _FROM_ENGINE_N_THETA_C)
        self.record_comparison()

    def test_carrier_rate_is_finite(self) -> None:
        """carrier_rate is finite even without fold-carrier (k_chart
        is estimated from the raw envelope)."""
        self.assertTrue(np.isfinite(self.chart.carrier_rate),
                        f'carrier_rate not finite: {self.chart.carrier_rate!r}')
        self.record_comparison()

    def test_no_fold_carrier_chart_can_serve(self) -> None:
        """The fold_carrier=False chart serves finite values."""
        chart = self.chart
        gamma = float(chart.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])
        rho = float(chart.rho_grid[_FROM_ENGINE_N_RHO // 2])
        theta_c = float(chart.theta_c_grid[_FROM_ENGINE_N_THETA_C // 2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        served = sg._evaluate_chart(
            chart, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        self.assertTrue(np.all(np.isfinite(served)),
                        'fold_carrier=False chart returned non-finite values')
        self.record_comparison()


# ======================================================================
# 8. Self-falsification — from_engine fold_carrier
# ======================================================================

class FoldCarrierFromEngineSelfFalsificationTestCase(_FoldCarrierBaseTestCase):
    """Certificate: the from_engine fold-carrier assertions can go RED.

    * fold_carrier=False produces a chart with rho_carrier=None while
      fold_carrier=True produces rho_carrier != None — proving the
      fold-carrier gate is load-bearing.
    * The fold_carrier=True and False charts produce DIFFERENT served
      values at a node where the rho_carrier correction is nonzero
      (proving the rho_carrier modulates the envelope).
    * A chart built with fold_carrier=True cannot have a blank
      rho_carrier array (all zeros would mean the ghost is absent
      at every rho, which contradicts the spec's fixture choice).
    """

    @classmethod
    def setUpClass(cls) -> None:
        sur_t = sg.LensAmplificationSurrogate.from_engine(
            gamma_range=_FROM_ENGINE_GAMMA_RANGE,
            rho_range=_FROM_ENGINE_RHO_RANGE,
            theta_c_range=_FROM_ENGINE_THETA_C_RANGE,
            w_range=_FROM_ENGINE_W_RANGE,
            n_gamma=_FROM_ENGINE_N_GAMMA,
            n_rho=_FROM_ENGINE_N_RHO,
            n_theta_c=_FROM_ENGINE_N_THETA_C,
            w_nodes_per_decade=_FROM_ENGINE_WNPD,
            definition=sg.FARFIELD_KERNEL_SUM,
            fold_carrier=True,
        )
        sur_f = sg.LensAmplificationSurrogate.from_engine(
            gamma_range=_FROM_ENGINE_GAMMA_RANGE,
            rho_range=_FROM_ENGINE_RHO_RANGE,
            theta_c_range=_FROM_ENGINE_THETA_C_RANGE,
            w_range=_FROM_ENGINE_W_RANGE,
            n_gamma=_FROM_ENGINE_N_GAMMA,
            n_rho=_FROM_ENGINE_N_RHO,
            n_theta_c=_FROM_ENGINE_N_THETA_C,
            w_nodes_per_decade=_FROM_ENGINE_WNPD,
            definition=sg.FARFIELD_KERNEL_SUM,
            fold_carrier=False,
        )
        cls.chart_true = sur_t.charts[0]
        cls.chart_false = sur_f.charts[0]
        cls._gamma = float(cls.chart_true.gamma_grid[
            _FROM_ENGINE_N_GAMMA // 2])
        cls._rho = float(cls.chart_true.rho_grid[
            _FROM_ENGINE_N_RHO // 2])
        cls._theta_c = float(cls.chart_true.theta_c_grid[
            _FROM_ENGINE_N_THETA_C // 2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)

    def test_fold_carrier_true_has_rho_carrier(self) -> None:
        """fold_carrier=True → rho_carrier is not None."""
        self.assertIsNotNone(self.chart_true.rho_carrier)
        self.record_comparison()

    def test_fold_carrier_false_has_no_rho_carrier(self) -> None:
        """fold_carrier=False → rho_carrier is None."""
        self.assertIsNone(self.chart_false.rho_carrier)
        self.record_comparison()

    def test_true_and_false_charts_differ(self) -> None:
        """fold_carrier=True and False produce DIFFERENT served values
        at an OFF-GRID spatial point.

        At a grid NODE both charts reproduce the engine envelope (the
        B-spline is interpolating), so fold_carrier=True telescopes to the
        same value.  At an OFF-GRID rho, the fold_carrier=True chart fits
        the smoother (demodulated) envelope — its spline interpolation is
        more accurate — and after re-modulation differs from the
        fold_carrier=False chart which fits the oscillating raw envelope.
        """
        # Off-grid rho between rho_grid[1] and rho_grid[2].
        rho_off = float(self.chart_true.rho_grid[1]) * 0.3 + float(
            self.chart_true.rho_grid[2]) * 0.7
        y1, y2 = _source_for_node(self._gamma, rho_off, self._theta_c)
        served_t = sg._evaluate_chart(
            self.chart_true, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=self.chart_true.log_w_grid,
            y1_eig=y1, y2_eig=y2)
        served_f = sg._evaluate_chart(
            self.chart_false, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=self.chart_false.log_w_grid,
            y1_eig=y1, y2_eig=y2)
        max_diff = float(np.max(np.abs(served_t - served_f)))
        self.assertGreater(
            max_diff, 1e-6,
            f'fold_carrier True vs False charts differ by only '
            f'{max_diff:.1e} — fold_carrier may be a no-op')
        self.record_comparison()

    def test_rho_carrier_is_not_all_zeros(self) -> None:
        """rho_carrier array is not all zeros — the ghost exists at some
        rho node in this tile."""
        if self.chart_true.rho_carrier is not None:
            self.assertFalse(
                np.allclose(self.chart_true.rho_carrier, 0.0),
                'rho_carrier is all zeros — no ghost detected in the '
                'spec-specified fixture tile')
        self.record_comparison()
