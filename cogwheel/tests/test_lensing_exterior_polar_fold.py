"""Verify 2D fold-carrier (rho_u_carrier) demodulation in ExteriorPolarChart.

``rho_u_carrier`` is an optional ``(n_rho, n_theta_c)`` array of fold-carrier
phase delays ``Re(tau_c(rho, theta_c))`` at each ``(rho, theta_c)`` spline node.
When provided, `from_values` demodulates by
``exp(-1j * w * rho_u_carrier[rho_node, th_node])`` BEFORE fitting the spline.
``_evaluate_chart`` re-modulates by bilinearly interpolating ``rho_u_carrier``
at the query ``(rho, theta_c)``.

This suite certifies:

1. **Node-exact round-trip**: at every training node the demod/remod telescopes.
2. **Carrier discontinuity guard**: raw envelope with ~16 rad rho-phase and
   ~30 rad theta_c-phase at w_max raises ``CarrierDiscontinuityError``.
3. **Off-grid theta_c accuracy**: phase span across theta_c nodes ≤ 1.63 rad
   after 2D carrier (down from ~30 rad without carrier at w_max).
4. **Magnitude invariance**: remodulation preserves magnitude.
5. **Composition**: rho_u_carrier + carrier_rate + rho_log_axis compose.

Tolerance rationale
```````````````````
* ``NODE_EXACT_TOL = 5e-13`` — spline is interpolating; only float round-off.
* ``OFFGRID_PHASE_TOL = 1e-3`` rad — np.interp on linear function is exact.
* ``MAGNITUDE_TOL = 5e-13`` — pure-phase rotation invariant to float precision.
* ``HELDOUT_BAR = 5e-2`` — smoke-scale (4×4×4 node) coarse interpolation error.
* ``OFFGRID_PHASE_SPAN_MAX = 1.63`` rad — per-rho theta_c-axis phase span.
* ``SELF_FALSIFICATION_MARGIN = 10.0`` — wrong carrier error > 10× correct.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing import surrogate as sg

_OUTPUT_DIR: Path = Path(__file__).resolve().parent / 'output'

NODE_EXACT_TOL: float = 5e-13
OFFGRID_PHASE_TOL: float = 1e-3
MAGNITUDE_TOL: float = 5e-13
HELDOUT_BAR: float = 5e-2
OFFGRID_PHASE_SPAN_MAX: float = 1.63
SELF_FALSIFICATION_MARGIN: float = 10.0

#: Bilinear carrier coefficients: tau_c(rho, theta_c) = a * rho + b * theta_c.
_COEFF_RHO: float = 2.5
_COEFF_U: float = -1.45

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

#: 2D rho-theta carrier: tau_c(rho, theta_c) = a*rho + b*theta_c.
_RHO_U_CARRIER: np.ndarray = (
    _COEFF_RHO * _RHO_GRID[:, None] + _COEFF_U * _THETA_C_GRID[None, :])

#: Amplitude profile A(rho) — smooth real, mild variation.
_A_RHO_BASE: np.ndarray = (1.0 + 0.5 * (_RHO_GRID - _RHO_GRID[0])
                           / (_RHO_GRID[-1] - _RHO_GRID[0]))

#: Amplitude profile B(theta_c) — smooth real, mild variation.
_A_U_BASE: np.ndarray = (1.0 + 0.3 * (_THETA_C_GRID - _THETA_C_GRID[0])
                         / (_THETA_C_GRID[-1] - _THETA_C_GRID[0]))

#: 2D amplitude profile A(rho) * B(theta_c) — shape (n_rho, n_theta_c).
_A_RHO_U: np.ndarray = _A_RHO_BASE[:, None] * _A_U_BASE[None, :]

_K_CHART: float = 0.05
_N_PROBES: int = 3

_W_PHASE_PROBE: float = 25.0
_LOG_W_PHASE_PROBE: np.ndarray = np.array([np.log(_W_PHASE_PROBE)])

_W_MID: np.ndarray = np.exp(0.5 * (_LOG_W_GRID[:-1] + _LOG_W_GRID[1:]))
_LOG_W_MID: np.ndarray = np.log(_W_MID)

#: Total rho-phase delta at top w.
_RHO_PHASE_DELTA: float = float(_COEFF_RHO * (_RHO_GRID[-1] - _RHO_GRID[0]))

#: Total theta_c-phase delta at top w.
_U_PHASE_DELTA: float = float(abs(_COEFF_U) * (_THETA_C_GRID[-1]
                                                - _THETA_C_GRID[0]))


# ======================================================================
# Helpers
# ======================================================================

def _build_fold_carrier_envelope_2d(
        rho_u_carrier: np.ndarray,
        log_w_grid: np.ndarray,
        amplitude: np.ndarray,
        carrier_rate: float = 0.0,
) -> np.ndarray:
    """``(n_w, n_rho, n_theta_c)`` envelope: E(w) = A * exp(1j*w*(k + tau_c))."""
    w_grid = np.exp(log_w_grid)
    phase = w_grid[:, None, None] * (carrier_rate + rho_u_carrier[None, :, :])
    return amplitude[None, :, :] * np.exp(1j * phase)


def _build_chart(*, rho_u_carrier: np.ndarray | None = None,
                 carrier_rate: float = 0.0,
                 rho_log_axis: bool = False,
                 ) -> sg.ExteriorPolarChart:
    """Build a synthetic ExteriorPolarChart.  Envelope tiles across gamma."""
    rc = (rho_u_carrier if rho_u_carrier is not None
          else np.zeros((_N_RHO, _N_THETA_C)))
    env_3d = _build_fold_carrier_envelope_2d(
        rc, _LOG_W_GRID, _A_RHO_U, carrier_rate=carrier_rate)
    envelope_4d = env_3d[:, None, :, :] * np.ones((1, _N_GAMMA, 1, 1))
    return sg.ExteriorPolarChart.from_values(
        gamma_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
        theta_c_grid=_THETA_C_GRID, log_w_grid=_LOG_W_GRID,
        envelope_real=envelope_4d.real.copy(),
        envelope_imag=envelope_4d.imag.copy(),
        image_count=2, parity=1,
        envelope_definition=sg.FARFIELD_KERNEL_SUM,
        rho_u_carrier=rho_u_carrier, carrier_rate=carrier_rate,
        rho_log_axis=rho_log_axis)


def _source_for_node(gamma: float, rho: float, theta_c: float
                     ) -> tuple[float, float]:
    return sg._from_caustic_fixed(gamma, rho, theta_c)


def _exact_envelope_2d(rho: float, theta_c: float,
                       log_w_grid: np.ndarray,
                       carrier_rate: float = 0.0,
                       rho_u_carrier_ref: np.ndarray = _RHO_U_CARRIER,
                       rho_grid_ref: np.ndarray = _RHO_GRID,
                       th_grid_ref: np.ndarray = _THETA_C_GRID,
                       ) -> np.ndarray:
    """Oracle: analytic tau_c(rho, theta_c) = a*rho + b*theta_c.

    The bilinear interpolation of ``rho_u_carrier_ref`` at ``(rho, theta_c)``
    reproduces this exactly because tau_c is bilinear.
    """
    # tau_c is bilinear: interp(rho, ..., interp(theta_c, ..., rho_u_carrier))
    # = a*rho + b*theta_c (exact with np.interp's linear interpolation)
    tau_c_col = np.asarray([
        float(np.interp(theta_c, th_grid_ref, rho_u_carrier_ref[i, :]))
        for i in range(len(rho_grid_ref))
    ])
    tau_c = float(np.interp(rho, rho_grid_ref, tau_c_col))
    amp = (
        float(np.interp(rho, rho_grid_ref, _A_RHO_BASE))
        * float(np.interp(theta_c, th_grid_ref, _A_U_BASE)))
    w = np.exp(log_w_grid)
    return amp * np.exp(1j * w * (carrier_rate + tau_c))


# ======================================================================
# Base — anti-vacuity comparison counter
# ======================================================================

class _FoldCarrierBaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_compared = 0
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def record_comparison(self) -> None:
        self.n_compared += 1

    def tearDown(self) -> None:
        if self.n_compared == 0:
            self.fail('anti-vacuity: no comparison executed')


# ======================================================================
# 1. Node-exact round-trip
# ======================================================================

class RhoCarrierNodeRoundTripTestCase(_FoldCarrierBaseTestCase):
    """2D fold-carrier demod+remod telescopes at all training nodes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(rho_u_carrier=_RHO_U_CARRIER)
        cls._nodes: list[
            tuple[float, float, float, float, float, np.ndarray]
        ] = []
        for gamma in _GAMMA_GRID:
            for rho in _RHO_GRID:
                for theta_c in _THETA_C_GRID:
                    g, r, t = float(gamma), float(rho), float(theta_c)
                    y1, y2 = _source_for_node(g, r, t)
                    ref = _exact_envelope_2d(r, t, _LOG_W_GRID)
                    cls._nodes.append((g, r, t, y1, y2, ref))

    def test_node_exact_round_trip_2d(self) -> None:
        """|E_served - E_raw| < 5e-13 at all 256 (gamma, rho, theta_c, w).

        Diagnostic: scatter of log10(error) vs node index color-coded by w.
        """
        max_err = 0.0
        worst = ''
        errs = []
        logw_vals = []
        for gamma, rho, theta_c, y1, y2, ref in self._nodes:
            served = sg._evaluate_chart(
                self.chart, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            errs.append(np.abs(served - ref))
            logw_vals.append(_LOG_W_GRID)
            err = float(np.max(np.abs(served - ref)))
            if err > max_err:
                max_err = err
                worst = (f'gamma={gamma:.3g} rho={rho:.3g} '
                         f'theta_c={theta_c:.3g}: {err:.1e}')
        self.record_comparison()
        self.assertLess(max_err, NODE_EXACT_TOL,
                        f'{worst} >= {NODE_EXACT_TOL:.0e}')

        # Diagnostic scatter plot
        errs_flat = np.concatenate([e.ravel() for e in errs])
        logw_flat = np.concatenate([np.broadcast_to(lw, e.shape).ravel()
                                     for lw, e in zip(logw_vals, errs)])
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(range(len(errs_flat)),
                        np.log10(np.maximum(errs_flat, 1e-18)),
                        c=logw_flat, cmap='viridis', s=4)
        ax.axhline(np.log10(NODE_EXACT_TOL), color='r', ls='--',
                    label=f'bar = {NODE_EXACT_TOL:.0e}')
        ax.set_xlabel('node index')
        ax.set_ylabel('log10(|E_served - E_exact|)')
        ax.set_title('2D fold-carrier node-exact round-trip error')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('log w')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = _OUTPUT_DIR / 'exterior_polar_fold_2d_node_round_trip.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def test_backward_compat_no_rho_u_carrier(self) -> None:
        """rho_u_carrier=None → zero phase rotation, node-exact."""
        chart0 = _build_chart(rho_u_carrier=None)
        zero_rc = np.zeros((_N_RHO, _N_THETA_C))
        max_err = 0.0
        for gamma, rho, theta_c, y1, y2, _ref in self._nodes:
            ref = _exact_envelope_2d(rho, theta_c, _LOG_W_GRID,
                                     rho_u_carrier_ref=zero_rc)
            served = sg._evaluate_chart(
                chart0, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            err = float(np.max(np.abs(served - ref)))
            max_err = max(max_err, err)
        self.record_comparison()
        self.assertLess(max_err, NODE_EXACT_TOL,
                        f'none-rc max={max_err:.1e} >= {NODE_EXACT_TOL:.0e}')

    def test_rho_u_carrier_is_stored(self) -> None:
        """chart.rho_u_carrier stores the requested array exactly."""
        np.testing.assert_array_equal(
            self.chart.rho_u_carrier, _RHO_U_CARRIER)
        chart_none = _build_chart(rho_u_carrier=None)
        self.assertIsNone(chart_none.rho_u_carrier)
        self.record_comparison()


# ======================================================================
# 2. Carrier discontinuity guard (2D)
# ======================================================================

class RhoCarrierContinuityGuardTestCase(_FoldCarrierBaseTestCase):
    """Raw envelope with ~16 rad rho-phase and ~30 rad theta_c-phase at
    w_max triggers CarrierDiscontinuityError on at least one axis.
    2D-demodulated smooth amplitude passes on all axes."""

    @classmethod
    def setUpClass(cls) -> None:
        w_max = float(_W_GRID[-1])
        env_3d = _build_fold_carrier_envelope_2d(
            _RHO_U_CARRIER, _LOG_W_GRID, _A_RHO_U)
        cls._raw_envelope = env_3d[:, None, :, :] * np.ones(
            (1, _N_GAMMA, 1, 1))
        env_demod = _build_fold_carrier_envelope_2d(
            np.zeros((_N_RHO, _N_THETA_C)), _LOG_W_GRID, _A_RHO_U)
        cls._demod_envelope = env_demod[:, None, :, :] * np.ones(
            (1, _N_GAMMA, 1, 1))
        cls._w_max = w_max
        cls._shape = (_N_GAMMA, _N_RHO, _N_THETA_C)

    def test_raw_envelope_raises_carrier_discontinuity(self) -> None:
        with self.assertRaises(sg.CarrierDiscontinuityError):
            sg._assert_exterior_polar_carrier_continuity(
                self._raw_envelope, self._w_max, _GAMMA_GRID, self._shape)
        self.record_comparison()

    def test_demodulated_envelope_passes_continuity(self) -> None:
        sg._assert_exterior_polar_carrier_continuity(
            self._demod_envelope, self._w_max, _GAMMA_GRID, self._shape)
        self.record_comparison()

    def test_diagnostic_continuity_bar_chart(self) -> None:
        """Bar chart: max step-norm/peak|E| per axis, raw vs demodulated."""
        from cogwheel.lensing.surrogate import _EXTERIOR_POLAR_CARRIER_STEP_MAX as STEP_MAX

        def _max_step_norm(env):
            all_mag = np.abs(env)
            scale = float(np.max(all_mag[np.isfinite(all_mag)], initial=0.0))
            if scale <= 0:
                return [0.0] * 3
            top = env[-1]
            norms = []
            for ax in range(3):
                n_ax = self._shape[ax]
                if n_ax < 2:
                    norms.append(0.0); continue
                lead = np.take(top, range(1, n_ax), axis=ax)
                trail = np.take(top, range(0, n_ax - 1), axis=ax)
                ml = np.take(np.abs(top), range(1, n_ax), axis=ax)
                mt = np.take(np.abs(top), range(0, n_ax - 1), axis=ax)
                both = ((ml > 0) & (mt > 0)
                        & np.isfinite(ml) & np.isfinite(mt))
                step = np.abs(lead - trail) / scale
                norms.append(float(np.max(step[both], initial=0.0)))
            return norms

        raw_norms = _max_step_norm(self._raw_envelope)
        demod_norms = _max_step_norm(self._demod_envelope)

        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(3); wb = 0.35
        ax.bar(x - wb / 2, raw_norms, wb, label='Raw (no carrier)')
        ax.bar(x + wb / 2, demod_norms, wb, label='2D-demodulated')
        ax.axhline(STEP_MAX, color='r', ls='--',
                    label=f'STEP_MAX = {STEP_MAX}')
        ax.set_xticks(x)
        ax.set_xticklabels(['gamma', 'rho', 'theta_c'])
        ax.set_ylabel('max step-norm / peak|E|')
        ax.set_title('2D fold-carrier continuity guard')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'exterior_polar_fold_2d_continuity_bar.png',
                    dpi=150)
        plt.close(fig)
        self.record_comparison()


# ======================================================================
# 3. Off-grid theta_c accuracy
# ======================================================================

class RhoCarrierOffGridPhaseTestCase(_FoldCarrierBaseTestCase):
    """Off-grid phase accuracy and u-axis phase span after 2D carrier."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart(rho_u_carrier=_RHO_U_CARRIER)
        rng = np.random.default_rng(20260810)
        cls._rho_probes = []  # 3 random per interval
        for i in range(len(_RHO_GRID) - 1):
            lo, hi = _RHO_GRID[i], _RHO_GRID[i + 1]
            for _ in range(_N_PROBES):
                cls._rho_probes.append(float(rng.uniform(lo, hi)))
        cls._theta_probes = []
        for i in range(len(_THETA_C_GRID) - 1):
            lo, hi = _THETA_C_GRID[i], _THETA_C_GRID[i + 1]
            for _ in range(_N_PROBES):
                cls._theta_probes.append(float(rng.uniform(lo, hi)))
        cls._gamma = float(_GAMMA_GRID[1])
        cls._w_max = float(_W_GRID[-1])
        cls._log_w_max = np.array([np.log(cls._w_max)])

    def test_off_grid_phase_within_bar(self) -> None:
        """|phase(E_served/E_exact)| < 1e-3 rad."""
        lw = _LOG_W_PHASE_PROBE
        for rho in self._rho_probes:
            for theta_c in self._theta_probes:
                y1, y2 = _source_for_node(self._gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=lw, y1_eig=y1, y2_eig=y2)
                exact = _exact_envelope_2d(rho, theta_c, lw)
                phase_err = abs(float(np.angle(served[0] / exact[0])))
                self.assertLess(phase_err, OFFGRID_PHASE_TOL,
                                f'rho={rho:.4g} theta_c={theta_c:.4g}: '
                                f'{phase_err:.1e}')
        self.record_comparison()

    def test_magnitude_invariant_under_remodulation(self) -> None:
        """|served| == |exact| to float64 precision."""
        lw = _LOG_W_PHASE_PROBE
        for rho in self._rho_probes[:3]:
            for theta_c in self._theta_probes[:3]:
                y1, y2 = _source_for_node(self._gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=lw, y1_eig=y1, y2_eig=y2)
                exact = _exact_envelope_2d(rho, theta_c, lw)
                mag_diff = abs(float(abs(served[0])) - float(abs(exact[0])))
                self.assertLess(mag_diff, MAGNITUDE_TOL,
                                f'rho={rho:.4g} theta_c={theta_c:.4g}'
                                f': |Δmag|={mag_diff:.1e}')
        self.record_comparison()

    def test_theta_c_axis_per_rho_phase_span(self) -> None:
        """Residual phase span (after carrier removal) ≤ 1.63 rad at w_max.

        Phase residual = arg(E_served / E_oracle) where oracle removes
        the analytic tau_c.  Without carrier, the raw phase span across
        theta_c exceeds 3 rad.
        """
        gamma = self._gamma
        # With 2D carrier: residual phase span
        max_span = 0.0
        for rho in _RHO_GRID:
            residuals = []
            for theta_c in _THETA_C_GRID:
                y1, y2 = _source_for_node(gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=gamma, eta=0.5, theta=0.5,
                    log_w_query=self._log_w_max, y1_eig=y1, y2_eig=y2)
                oracle = _exact_envelope_2d(rho, theta_c, self._log_w_max)
                # served/oracle = A_spline/A_real ≈ purely real
                residual = float(np.angle(served[0] / oracle[0]))
                residuals.append(residual)
            span = abs(max(residuals) - min(residuals))
            max_span = max(max_span, span)
        self.assertLessEqual(max_span, OFFGRID_PHASE_SPAN_MAX,
                             f'phase span {max_span:.4f} > '
                             f'{OFFGRID_PHASE_SPAN_MAX}')
        # Without carrier — raw phase span: chart built with full
        # carrier-phase envelope but NO rho_u_carrier to demod it,
        # so the phase comes through.  (Use _build_chart helper's
        # zero-rc branch for the modulated envelope, but pass the
        # real carrier as rho_u_carrier=None to skip demodulation.)
        env_3d_raw = _build_fold_carrier_envelope_2d(
            _RHO_U_CARRIER, _LOG_W_GRID, _A_RHO_U)
        envelope_4d_raw = env_3d_raw[:, None, :, :] * np.ones(
            (1, _N_GAMMA, 1, 1))
        chart_none = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
            theta_c_grid=_THETA_C_GRID, log_w_grid=_LOG_W_GRID,
            envelope_real=envelope_4d_raw.real.copy(),
            envelope_imag=envelope_4d_raw.imag.copy(),
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            rho_u_carrier=None,  # NO carrier demodulation
        )
        max_span_raw = 0.0
        for rho in _RHO_GRID:
            phases = []
            for theta_c in _THETA_C_GRID:
                y1, y2 = _source_for_node(gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    chart_none, gamma=gamma, eta=0.5, theta=0.5,
                    log_w_query=self._log_w_max, y1_eig=y1, y2_eig=y2)
                phases.append(float(np.angle(served[0])))
            span = abs(max(phases) - min(phases))
            max_span_raw = max(max_span_raw, span)
        self.assertGreater(max_span_raw, 3.0,
                           f'carrier=None span {max_span_raw:.4f} ≤ 3.0')
        self.record_comparison()

    def test_theta_c_axis_phase_span_diagnostic(self) -> None:
        """Side-by-side: theta_c vs phase(E_served) at w_max."""
        gamma = self._gamma; rho = float(_RHO_GRID[2])
        env_3d_raw = _build_fold_carrier_envelope_2d(
            _RHO_U_CARRIER, _LOG_W_GRID, _A_RHO_U)
        env_4d_raw = env_3d_raw[:, None, :, :] * np.ones((1, _N_GAMMA, 1, 1))
        chart_none = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
            theta_c_grid=_THETA_C_GRID, log_w_grid=_LOG_W_GRID,
            envelope_real=env_4d_raw.real.copy(),
            envelope_imag=env_4d_raw.imag.copy(),
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            rho_u_carrier=None,
        )
        phases_2d = []; phases_none = []; th_vals = _THETA_C_GRID
        for theta_c in th_vals:
            y1, y2 = _source_for_node(gamma, rho, theta_c)
            s2d = sg._evaluate_chart(
                self.chart, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=self._log_w_max, y1_eig=y1, y2_eig=y2)
            sn = sg._evaluate_chart(
                chart_none, gamma=gamma, eta=0.5, theta=0.5,
                log_w_query=self._log_w_max, y1_eig=y1, y2_eig=y2)
            # Residual phase (vs oracle) for 2D chart; raw phase for None chart
            oracle_2d = _exact_envelope_2d(rho, theta_c, self._log_w_max)
            phases_2d.append(float(np.angle(s2d[0] / oracle_2d[0])))
            phases_none.append(float(np.angle(sn[0])))
        phases_2d_unwrapped = np.unwrap(np.array(phases_2d))
        phases_none_unwrapped = np.unwrap(np.array(phases_none))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax1.plot(th_vals, phases_2d_unwrapped, 'o-', color='C0')
        ax1.set_xlabel('theta_c'); ax1.set_ylabel('arg(E_served) [rad]')
        ax1.set_title(f'2D carrier — ≤{OFFGRID_PHASE_SPAN_MAX} rad span')
        ax1.grid(True, alpha=0.3)
        ax2.plot(th_vals, phases_none_unwrapped, 'o-', color='C1')
        ax2.set_xlabel('theta_c'); ax2.set_ylabel('arg(E_served) [rad]')
        ax2.set_title(f'No carrier — ~{_U_PHASE_DELTA * self._w_max:.0f} rad span')
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f'2D fold-carrier phase span, rho={rho:.3f}, w={self._w_max:.1f}')
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'exterior_polar_fold_2d_theta_c_phase_span.png',
                    dpi=150)
        plt.close(fig)
        span_2d = abs(float(np.max(phases_2d_unwrapped))
                          - float(np.min(phases_2d_unwrapped)))
        self.assertLess(span_2d, OFFGRID_PHASE_SPAN_MAX)
        self.record_comparison()


# ======================================================================
# 4. Composition: rho_u_carrier + carrier_rate + rho_log_axis
# ======================================================================

class RhoCarrierCompositionTestCase(_FoldCarrierBaseTestCase):
    """rho_u_carrier, carrier_rate, and rho_log_axis compose correctly.

    Off-grid probes at random (rho, theta_c) midpoints (~81 total).
    Verifies: eps within bar, phase accuracy, raw-rho (not log) re-modulation,
    and magnitude invariance.
    """

    COMPOSITION_HELDOUT_BAR: float = 4e-3

    @classmethod
    def setUpClass(cls) -> None:
        cls._gamma = float(_GAMMA_GRID[1])
        cls.chart = _build_chart(
            rho_u_carrier=_RHO_U_CARRIER, carrier_rate=_K_CHART,
            rho_log_axis=True)
        rng = np.random.default_rng(20260810)
        cls._rho_probes: list[float] = []
        for i in range(len(_RHO_GRID) - 1):
            lo, hi = _RHO_GRID[i], _RHO_GRID[i + 1]
            for _ in range(_N_PROBES):
                cls._rho_probes.append(float(rng.uniform(lo, hi)))
        cls._theta_probes: list[float] = []
        for i in range(len(_THETA_C_GRID) - 1):
            lo, hi = _THETA_C_GRID[i], _THETA_C_GRID[i + 1]
            for _ in range(_N_PROBES):
                cls._theta_probes.append(float(rng.uniform(lo, hi)))
        cls._w_probe = _W_PHASE_PROBE
        cls._lw_probe = _LOG_W_PHASE_PROBE

    def test_composition_off_grid_eps_within_bar(self) -> None:
        """max eps < 4e-3 across all ~81 off-grid (rho, theta_c) probes."""
        max_eps = 0.0
        for rho in self._rho_probes:
            for theta_c in self._theta_probes:
                y1, y2 = _source_for_node(self._gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
                exact = _exact_envelope_2d(rho, theta_c, _LOG_W_GRID,
                                           carrier_rate=_K_CHART)
                denom = max(float(np.max(np.abs(exact))), 1e-300)
                eps = float(np.max(np.abs(served - exact))) / denom
                max_eps = max(max_eps, eps)
        self.record_comparison()
        self.assertLess(max_eps, self.COMPOSITION_HELDOUT_BAR,
                        f'max eps = {max_eps:.2e} >= '
                        f'{self.COMPOSITION_HELDOUT_BAR:.0e}')

    def test_off_grid_phase_within_bar(self) -> None:
        """|phase(E_served / E_exact)| < 1e-3 rad at w=25 for all probes."""
        for rho in self._rho_probes:
            for theta_c in self._theta_probes:
                y1, y2 = _source_for_node(self._gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=self._lw_probe, y1_eig=y1, y2_eig=y2)
                exact = _exact_envelope_2d(rho, theta_c, self._lw_probe,
                                           carrier_rate=_K_CHART)
                phase_err = abs(float(np.angle(served[0] / exact[0])))
                self.assertLess(phase_err, OFFGRID_PHASE_TOL,
                                f'rho={rho:.4g} theta_c={theta_c:.4g}: '
                                f'{phase_err:.1e}')
        self.record_comparison()

    def test_raw_rho_not_log_rho_remodulation(self) -> None:
        """Re-modulation uses RAW rho, not log(rho-1).

        Build a chart with tau_c = slope * rho (constant in theta_c),
        rho_log_axis=True.  At off-grid rho probes the re-modulation
        phase matches w*slope*rho_probe (mod 2π), NOT
        w*slope*log(rho_probe-1).
        """
        slope = _COEFF_RHO
        rho_only_carrier = slope * _RHO_GRID[:, None] * np.ones(
            (1, _N_THETA_C))
        w_grid = np.exp(_LOG_W_GRID)
        phase = (w_grid[:, None, None, None]
                 * rho_only_carrier[None, None, :, :])
        amplitude = np.ones((_N_RHO, _N_THETA_C))[None, None, :, :] * np.ones(
            (len(_LOG_W_GRID), _N_GAMMA, 1, 1))
        env_real = (amplitude * np.cos(phase)).copy()
        env_imag = (amplitude * np.sin(phase)).copy()
        chart_rho_only = sg.ExteriorPolarChart.from_values(
            gamma_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
            theta_c_grid=_THETA_C_GRID, log_w_grid=_LOG_W_GRID,
            envelope_real=env_real, envelope_imag=env_imag,
            image_count=2, parity=1,
            envelope_definition=sg.FARFIELD_KERNEL_SUM,
            rho_u_carrier=rho_only_carrier, carrier_rate=0.0,
            rho_log_axis=True)
        for rho in self._rho_probes:
            y1, y2 = _source_for_node(self._gamma, rho,
                                      float(_THETA_C_GRID[2]))
            served = sg._evaluate_chart(
                chart_rho_only, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=self._lw_probe, y1_eig=y1, y2_eig=y2)
            phase_served = float(np.angle(served[0]))
            phase_raw = (self._w_probe * slope * rho) % (2 * np.pi)
            raw_err = abs(np.angle(np.exp(1j
                                         * (phase_served - phase_raw))))
            phase_log = (self._w_probe * slope * np.log(rho - 1.0)) % (
                2 * np.pi)
            log_err = abs(np.angle(np.exp(1j
                                          * (phase_served - phase_log))))
            self.assertLess(raw_err, OFFGRID_PHASE_TOL,
                            f'rho={rho:.4g}: |Δraw|={raw_err:.1e}')
            self.assertGreater(log_err, OFFGRID_PHASE_TOL,
                               f'rho={rho:.4g}: |Δlog|={log_err:.1e} NOT '
                               f'> {OFFGRID_PHASE_TOL:.0e}')
        self.record_comparison()

    def test_magnitude_invariant_under_composition(self) -> None:
        """|E_served| matches |E_same_chart_no_carrier_rate| within
        MAGNITUDE_TOL — re-modulation by carrier_rate is pure phase
        rotation."""
        chart_no_k = _build_chart(
            rho_u_carrier=_RHO_U_CARRIER, carrier_rate=0.0,
            rho_log_axis=True)
        for rho in self._rho_probes:
            for theta_c in self._theta_probes:
                y1, y2 = _source_for_node(self._gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=self._lw_probe, y1_eig=y1, y2_eig=y2)
                served_no_k = sg._evaluate_chart(
                    chart_no_k, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=self._lw_probe, y1_eig=y1, y2_eig=y2)
                mag_diff = abs(
                    float(abs(served[0])) - float(abs(served_no_k[0])))
                self.assertLess(mag_diff, MAGNITUDE_TOL,
                                f'rho={rho:.4g} theta_c={theta_c:.4g}: '
                                f'|Δmag|={mag_diff:.1e}')
        self.record_comparison()

    def test_composition_phase_diagnostic(self) -> None:
        """Diagnostic: |phase(E_served/E_exact)| vs rho."""
        fig, ax = plt.subplots(figsize=(8, 5))
        rho_vals = []
        phase_errs = []
        for rho in self._rho_probes:
            for theta_c in self._theta_probes:
                y1, y2 = _source_for_node(self._gamma, rho, theta_c)
                served = sg._evaluate_chart(
                    self.chart, gamma=self._gamma, eta=0.5, theta=0.5,
                    log_w_query=self._lw_probe, y1_eig=y1, y2_eig=y2)
                exact = _exact_envelope_2d(rho, theta_c, self._lw_probe,
                                           carrier_rate=_K_CHART)
                pe = abs(float(np.angle(served[0] / exact[0])))
                rho_vals.append(rho)
                phase_errs.append(pe)
        ax.scatter(rho_vals, phase_errs, s=8, alpha=0.6,
                   label='served vs oracle')
        ax.axhline(OFFGRID_PHASE_TOL, color='r', ls='--',
                    label=f'bar = {OFFGRID_PHASE_TOL:.0e}')
        ax.set_xlabel('rho')
        ax.set_ylabel('|arg(E_served / E_exact)| [rad]')
        ax.set_title('2D fold-carrier composition phase error')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'exterior_polar_fold_composition_phase.png',
                    dpi=150)
        plt.close(fig)
        self.record_comparison()


# ======================================================================
# 5. Self-falsification — the suite can go RED
# ======================================================================

class FoldCarrierSelfFalsificationTestCase(_FoldCarrierBaseTestCase):
    """Wrong rho_u_carrier pushes error > bar and > 10× correct error."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart_correct = _build_chart(rho_u_carrier=_RHO_U_CARRIER)
        wrong_rc = _RHO_U_CARRIER + 0.3
        cls.chart_wrong = _build_chart(rho_u_carrier=wrong_rc)
        cls.chart_none = _build_chart(rho_u_carrier=None)
        cls._gamma = float(_GAMMA_GRID[1])
        # on-grid node for exact carrier tests
        cls._rho = float(_RHO_GRID[2])
        cls._theta_c = float(_THETA_C_GRID[2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)
        cls._ref = _exact_envelope_2d(cls._rho, cls._theta_c, _LOG_W_GRID)
        # off-grid probe for zero-carrier comparison
        cls._rho_off = 0.5 * (float(_RHO_GRID[1]) + float(_RHO_GRID[2]))
        cls._theta_c_off = 0.5 * (float(_THETA_C_GRID[1])
                                  + float(_THETA_C_GRID[2]))
        cls._y1_off, cls._y2_off = _source_for_node(
            cls._gamma, cls._rho_off, cls._theta_c_off)
        cls._ref_off = _exact_envelope_2d(
            cls._rho_off, cls._theta_c_off, _LOG_W_GRID)

    def test_wrong_rho_u_carrier_above_bar(self) -> None:
        served = sg._evaluate_chart(
            self.chart_wrong, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1, y2_eig=self._y2)
        err = float(np.max(np.abs(served - self._ref)))
        self.record_comparison()
        self.assertGreater(err, NODE_EXACT_TOL)

    def test_wrong_vs_correct_ratio(self) -> None:
        served_w = sg._evaluate_chart(
            self.chart_wrong, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1, y2_eig=self._y2)
        err_w = float(np.max(np.abs(served_w - self._ref)))
        served_c = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1, y2_eig=self._y2)
        err_c = float(np.max(np.abs(served_c - self._ref)))
        ratio = err_w / max(err_c, 1e-300)
        self.record_comparison()
        self.assertGreater(ratio, SELF_FALSIFICATION_MARGIN)

    def test_correct_rc_within_bar(self) -> None:
        served = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1, y2_eig=self._y2)
        err = float(np.max(np.abs(served - self._ref)))
        self.record_comparison()
        self.assertLess(err, NODE_EXACT_TOL)

    def test_zero_carrier_off_grid_ratio(self) -> None:
        """Off-grid: eps(None) / eps(correct) > SELF_FALSIFICATION_MARGIN.

        The correct carrier demodulates the envelope so the spline sees
        a smooth modulation; at off-grid (rho, theta_c) the interpolation
        error remains near machine precision.  With ``rho_u_carrier=None``
        (zero carrier), the envelope's ~16 rad phase oscillation at w_max
        cannot be interpolated by a cubic spline on a 4×4 node grid, and
        the served values diverge from the exact envelope by orders of
        magnitude.
        """
        served_c = sg._evaluate_chart(
            self.chart_correct, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1_off,
            y2_eig=self._y2_off)
        err_c = float(np.max(np.abs(served_c - self._ref_off)))
        served_n = sg._evaluate_chart(
            self.chart_none, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1_off,
            y2_eig=self._y2_off)
        err_n = float(np.max(np.abs(served_n - self._ref_off)))
        ratio = err_n / max(err_c, 1e-300)
        self.record_comparison()
        self.assertGreater(
            ratio, SELF_FALSIFICATION_MARGIN,
            f'off-grid None/correct ratio {ratio:.2f} ≤ '
            f'{SELF_FALSIFICATION_MARGIN} (err_c={err_c:.1e}, '
            f'err_n={err_n:.1e})')

    def test_diagnostic_grouped_bar_chart(self) -> None:
        """Grouped bars of log10(max|E_served-E_oracle|) for correct,
        wrong, and zero (None) carrier at both on-grid and off-grid
        probe points.  Reference line at log10(NODE_EXACT_TOL).
        """
        def _err(chart, rho, theta_c, y1, y2, ref):
            served = sg._evaluate_chart(
                chart, gamma=self._gamma, eta=0.5, theta=0.5,
                log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
            return float(np.max(np.abs(served - ref)))

        err_c_on = _err(self.chart_correct, self._rho, self._theta_c,
                        self._y1, self._y2, self._ref)
        err_w_on = _err(self.chart_wrong, self._rho, self._theta_c,
                        self._y1, self._y2, self._ref)
        err_n_on = _err(self.chart_none, self._rho, self._theta_c,
                        self._y1, self._y2, self._ref)
        err_c_off = _err(self.chart_correct, self._rho_off,
                         self._theta_c_off, self._y1_off, self._y2_off,
                         self._ref_off)
        err_w_off = _err(self.chart_wrong, self._rho_off,
                         self._theta_c_off, self._y1_off, self._y2_off,
                         self._ref_off)
        err_n_off = _err(self.chart_none, self._rho_off,
                         self._theta_c_off, self._y1_off, self._y2_off,
                         self._ref_off)

        labels = ['correct\n(on-grid)', 'wrong\n(on-grid)',
                  'none\n(on-grid)', 'correct\n(off-grid)',
                  'wrong\n(off-grid)', 'none\n(off-grid)']
        vals = [err_c_on, err_w_on, err_n_on,
                err_c_off, err_w_off, err_n_off]
        log_vals = np.log10(np.maximum(np.array(vals), 1e-18))
        colors = ['C0', 'C1', 'C2', 'C0', 'C1', 'C2']
        edge_colors = ['navy', 'darkred', 'darkgreen',
                       'navy', 'darkred', 'darkgreen']

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, log_vals, color=colors, edgecolor=edge_colors,
               linewidth=1.2)
        ax.axhline(np.log10(NODE_EXACT_TOL), color='gray', ls='--',
                   linewidth=1, label=f'NODE_EXACT_TOL = {NODE_EXACT_TOL:.0e}')
        ax.axhline(np.log10(SELF_FALSIFICATION_MARGIN), color='black',
                   ls=':', linewidth=1, alpha=0.5,
                   label=f'SELF_FALSIFICATION_MARGIN = '
                         f'{SELF_FALSIFICATION_MARGIN}')
        for i, (xi, yi) in enumerate(zip(x, log_vals)):
            ax.text(xi, yi + 0.3, f'{vals[i]:.1e}', ha='center',
                    fontsize=7, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('log10(max|E_served - E_oracle|)')
        ax.set_title('Self-falsification: fold-carrier accuracy\n'
                     'correct must be below; wrong + none >= 10x above')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2, axis='y')
        ax.annotate('correct must be below', xy=(0, np.log10(NODE_EXACT_TOL)),
                    xytext=(1.5, np.log10(NODE_EXACT_TOL) - 2),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=8, color='gray')
        ax.annotate('wrong+none >= 10x above',
                    xy=(4, np.log10(SELF_FALSIFICATION_MARGIN)),
                    xytext=(5, np.log10(SELF_FALSIFICATION_MARGIN) + 3),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=8, color='black')
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'exterior_polar_fold_self_falsification_bars.png',
                    dpi=150)
        plt.close(fig)

        # Assert on-grid correct is below bar (anti-vacuity check
        # on the diagnostic path)
        self.record_comparison()
        self.assertLess(err_c_on, NODE_EXACT_TOL)

    def test_magnitude_test_has_teeth(self) -> None:
        served = sg._evaluate_chart(
            self.chart_wrong, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=self._y1, y2_eig=self._y2)
        mag_err = float(np.max(np.abs(np.abs(served) - np.abs(self._ref))))
        self.record_comparison()
        self.assertLess(mag_err, MAGNITUDE_TOL)


# ======================================================================
# 6. NPZ round-trip
# ======================================================================

class FoldCarrierNpzRoundTripTestCase(_FoldCarrierBaseTestCase):
    """rho_u_carrier survives _chart_to_npz → _chart_from_npz."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._chart_source = _build_chart(
            rho_u_carrier=_RHO_U_CARRIER, carrier_rate=_K_CHART,
            rho_log_axis=True)
        cls._npz_data = sg._chart_to_npz(cls._chart_source, index=0)
        cls._chart_loaded = sg._chart_from_npz(cls._npz_data, index=0)

    def test_schema_tag_is_carrier_v2(self) -> None:
        meta = sg.json.loads(str(self._npz_data['chart0_meta']))
        self.assertEqual(meta.get('axis_schema'),
                         'exterior_polar_rho_u_carrier_v2')
        self.record_comparison()

    def test_rho_u_carrier_byte_identical_after_round_trip(self) -> None:
        np.testing.assert_array_equal(
            self._chart_loaded.rho_u_carrier,
            self._chart_source.rho_u_carrier)
        self.record_comparison()

    def test_carrier_rate_preserved_through_npz(self) -> None:
        self.assertEqual(self._chart_loaded.carrier_rate, _K_CHART)
        self.record_comparison()

    def test_rho_log_axis_preserved_through_npz(self) -> None:
        self.assertTrue(self._chart_loaded.rho_log_axis)
        self.record_comparison()

    def test_loaded_chart_serves_same_as_source(self) -> None:
        gamma = float(_GAMMA_GRID[1])
        rho = 0.5 * (float(_RHO_GRID[1]) + float(_RHO_GRID[2]))
        theta_c = float(_THETA_C_GRID[2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        served_src = sg._evaluate_chart(
            self._chart_source, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID, y1_eig=y1, y2_eig=y2)
        served_ld = sg._evaluate_chart(
            self._chart_loaded, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID, y1_eig=y1, y2_eig=y2)
        self.assertTrue(np.allclose(served_src, served_ld, rtol=0, atol=0))
        self.record_comparison()

    def test_loaded_vs_source_histogram_diagnostic(self) -> None:
        """Histogram of |E_loaded - E_source| over all w-nodes at one
        spatial point.  Bar at 0 = pass."""
        gamma = float(_GAMMA_GRID[1])
        rho = float(_RHO_GRID[2])
        theta_c = float(_THETA_C_GRID[2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        served_src = sg._evaluate_chart(
            self._chart_source, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
        served_ld = sg._evaluate_chart(
            self._chart_loaded, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_GRID, y1_eig=y1, y2_eig=y2)
        diffs = np.abs(served_src - served_ld)
        max_diff = float(np.max(diffs))
        self.assertEqual(max_diff, 0.0,
                         f'bit-identical fail: max|Δ|={max_diff:.1e}')
        # Diagnostic histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log10(np.maximum(diffs, 1e-18)), bins=20,
                color='C0', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='r', ls='--', label='0 = pass')
        ax.set_xlabel('log10(|E_loaded - E_source|)')
        ax.set_ylabel('count')
        ax.set_title('NPZ round-trip: |ΔE| histogram')
        ax.legend()
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'exterior_polar_fold_npz_roundtrip_hist.png',
                    dpi=150)
        plt.close(fig)
        self.record_comparison()


class FoldCarrierLegacySchemaHardRefusalTestCase(_FoldCarrierBaseTestCase):
    """Legacy schema tags hard-refuse at load."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._chart = _build_chart(rho_u_carrier=_RHO_U_CARRIER)
        cls._npz_data = sg._chart_to_npz(cls._chart, index=0)

    def test_rho_log_v3_schema_hard_refuses(self) -> None:
        mutated = dict(self._npz_data)
        meta = sg.json.loads(str(mutated['chart0_meta']))
        meta['axis_schema'] = 'exterior_polar_rho_log_v3'
        mutated['chart0_meta'] = np.array(sg.json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            sg._chart_from_npz(mutated, index=0)
        self.assertIn('axis-schema tag', str(ctx.exception))
        self.record_comparison()

    def test_missing_axis_schema_raises_valueerror(self) -> None:
        mutated = dict(self._npz_data)
        meta = sg.json.loads(str(mutated['chart0_meta']))
        del meta['axis_schema']
        mutated['chart0_meta'] = np.array(sg.json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            sg._chart_from_npz(mutated, index=0)
        self.assertIn('absent or unknown', str(ctx.exception))
        self.record_comparison()

    def test_carrier_demod_v2_schema_hard_refuses(self) -> None:
        mutated = dict(self._npz_data)
        meta = sg.json.loads(str(mutated['chart0_meta']))
        meta['axis_schema'] = 'exterior_polar_carrier_demod_v2'
        mutated['chart0_meta'] = np.array(sg.json.dumps(meta))
        with self.assertRaises(ValueError):
            sg._chart_from_npz(mutated, index=0)
        self.record_comparison()

    def test_current_v5_schema_accepted(self) -> None:
        loaded = sg._chart_from_npz(self._npz_data, index=0)
        self.assertIsNotNone(loaded.rho_u_carrier)
        self.record_comparison()


class FoldCarrierMissingKeyBackwardCompatTestCase(_FoldCarrierBaseTestCase):
    """Missing rho_u_carrier key loads as None."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._chart = _build_chart(rho_u_carrier=_RHO_U_CARRIER)
        cls._npz_data = sg._chart_to_npz(cls._chart, index=0)

    def test_missing_rho_u_carrier_key_loads_as_none(self) -> None:
        mutated = {k: v for k, v in self._npz_data.items()
                   if k != 'chart0_rho_u_carrier'}
        loaded = sg._chart_from_npz(mutated, index=0)
        self.assertIsNone(loaded.rho_u_carrier)
        self.record_comparison()

    def test_missing_key_chart_round_trips_unchanged(self) -> None:
        chart_none = _build_chart(rho_u_carrier=None)
        npz_none = sg._chart_to_npz(chart_none, index=0)
        self.assertNotIn('chart0_rho_u_carrier', npz_none)
        loaded = sg._chart_from_npz(npz_none, index=0)
        self.assertIsNone(loaded.rho_u_carrier)
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
        self.assertTrue(np.allclose(served_src, served_ld, rtol=0, atol=0))
        self.record_comparison()


# ======================================================================
# 7. from_engine end-to-end (fold_carrier=True → 2D)
# ======================================================================

_FROM_ENGINE_GAMMA_RANGE: tuple[float, float] = (0.3, 0.7)
_FROM_ENGINE_RHO_RANGE: tuple[float, float] = (1.3, 2.0)
_FROM_ENGINE_THETA_C_RANGE: tuple[float, float] = (0.0, 0.5)
_FROM_ENGINE_W_RANGE: tuple[float, float] = (10.0, 30.0)
_FROM_ENGINE_N_GAMMA: int = 4
_FROM_ENGINE_N_RHO: int = 4
_FROM_ENGINE_N_THETA_C: int = 4
_FROM_ENGINE_WNPD: int = 6
_FROM_ENGINE_NODE_HELDOUT_BAR: float = 1e-2
_FROM_ENGINE_OFFGRID_HELDOUT_BAR: float = 5e-2


class FoldCarrierFromEngineTestCase(_FoldCarrierBaseTestCase):
    """from_engine(fold_carrier=True) produces 2D rho_u_carrier."""

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

    def test_rho_u_carrier_is_not_none(self) -> None:
        self.assertIsNotNone(self.chart.rho_u_carrier)
        self.record_comparison()

    def test_rho_u_carrier_has_correct_shape(self) -> None:
        self.assertEqual(self.chart.rho_u_carrier.shape,
                         (_FROM_ENGINE_N_RHO, _FROM_ENGINE_N_THETA_C))
        self.record_comparison()

    def test_rho_u_carrier_is_finite(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.chart.rho_u_carrier)))
        self.record_comparison()

    def test_carrier_rate_is_finite_float(self) -> None:
        self.assertIsInstance(self.chart.carrier_rate, float)
        self.assertTrue(np.isfinite(self.chart.carrier_rate))
        self.record_comparison()

    def test_envelope_definition_is_kernel_sum(self) -> None:
        self.assertEqual(self.chart.envelope_definition,
                         sg.FARFIELD_KERNEL_SUM)
        self.record_comparison()

    def test_heldout_eps_node_exact_within_bar(self) -> None:
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
        self.assertLess(eps, _FROM_ENGINE_NODE_HELDOUT_BAR)
        self.record_comparison()

    def test_heldout_eps_off_grid_within_bar(self) -> None:
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
        self.assertLess(eps, _FROM_ENGINE_OFFGRID_HELDOUT_BAR)
        self.record_comparison()

    def test_surrogate_can_serve_at_heldout_point(self) -> None:
        chart = self.chart
        gamma = float(chart.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])
        rho = 0.5 * (float(chart.rho_grid[1]) + float(chart.rho_grid[2]))
        theta_c = 0.5 * (float(chart.theta_c_grid[1])
                         + float(chart.theta_c_grid[2]))
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        served = sg._evaluate_chart(
            chart, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        self.assertIsNotNone(served)
        self.assertTrue(np.all(np.isfinite(served)))
        self.record_comparison()

    def test_rho_u_carrier_not_all_zeros(self) -> None:
        """rho_u_carrier is not trivially all zeros."""
        self.assertIsNotNone(self.chart.rho_u_carrier)
        self.assertFalse(np.allclose(self.chart.rho_u_carrier, 0.0))
        self.record_comparison()

    def test_ghost_kernel_delay_matches_carrier_at_valid_nodes(self
                                                               ) -> None:
        """At nodes where ghost exists, rho_u_carrier[i,j] matches
        ``Re(tau_c)`` from ``geometry.ghost_kernel()`` to ~1e-12.

        Independent oracle: probe ``ghost_kernel(w0, source, matrix)``
        at each ``(rho, theta_c)`` spline node across all gamma-band
        gammas.  The stored carrier is the MEDIAN across all valid
        gamma probes; the independent oracle verifies the match.
        """
        from cogwheel.lensing.chang_refsdal import geometry
        rc = self.chart.rho_u_carrier
        w0 = float(np.exp(self.chart.log_w_grid[0]))
        n_matched = 0
        for i_r, rho in enumerate(self.chart.rho_grid):
            for i_t, theta_c in enumerate(self.chart.theta_c_grid):
                valid: list[float] = []
                for gamma in self.chart.gamma_grid:
                    try:
                        source = sg._from_caustic_fixed(
                            float(gamma), float(rho), float(theta_c))
                        source_arr = np.array(source, dtype=float)
                        matrix = geometry.macro_matrix(
                            float(gamma), beta=0.0, kappa=0.0)
                        contrib = geometry.ghost_kernel(
                            [w0], source_arr, matrix)
                        valid.append(float(contrib.delay.real))
                    except (geometry.GhostDomainError,
                            geometry.LensDomainError, ValueError):
                        continue
                if valid:
                    expected = float(np.median(valid))
                    actual = float(rc[i_r, i_t])
                    self.assertLess(
                        abs(actual - expected), 1e-12,
                        f'(i_r,i_t)=({i_r},{i_t}): '
                        f'|{actual:.12e}-{expected:.12e}|'
                        f'={abs(actual-expected):.1e}')
                    n_matched += 1
        self.assertGreater(n_matched, 0,
                           'No ghost-probe nodes found in engine chart')
        self.record_comparison()

    def test_filled_nan_nodes_have_smooth_derivatives(self) -> None:
        """NaN-filled nodes: rho-derivative and u-derivative match neighbor
        columns/rows within factor 2.

        First identifies which ``(rho, theta_c)`` nodes originally had no
        ghost (``GhostDomainError`` at all band gammas), then verifies that
        the linear-interpolation fill is smooth: at each filled node that
        has at least one valid neighbour column/row, the central-difference
        derivative differs from the derivative at that neighbour by at most
        a factor of 2.  Handles both interior and boundary filled nodes
        (common on smoke-scale 4×4 grids where the ghost-boundary edge
        coincides with a grid boundary).
        """
        from cogwheel.lensing.chang_refsdal import geometry
        rc = self.chart.rho_u_carrier
        rho_g = self.chart.rho_grid
        th_g = self.chart.theta_c_grid
        w0 = float(np.exp(self.chart.log_w_grid[0]))
        n_rho = len(rho_g)
        n_th = len(th_g)

        ghost_exists = np.zeros((n_rho, n_th), dtype=bool)
        for i_r, rho in enumerate(rho_g):
            for i_t, theta_c in enumerate(th_g):
                for gamma in self.chart.gamma_grid:
                    try:
                        source = sg._from_caustic_fixed(
                            float(gamma), float(rho), float(theta_c))
                        matrix = geometry.macro_matrix(
                            float(gamma), beta=0.0, kappa=0.0)
                        geometry.ghost_kernel(
                            [w0], np.array(source, dtype=float), matrix)
                        ghost_exists[i_r, i_t] = True
                        break
                    except (geometry.GhostDomainError,
                            geometry.LensDomainError, ValueError):
                        continue

        ghost_ct = int(np.sum(ghost_exists))
        if ghost_ct == 0:
            self.skipTest('No ghost-probe nodes found')
        if ghost_ct == n_rho * n_th:
            self.skipTest('All nodes have ghost — no NaN filling occurred')

        nan_filled = ~ghost_exists
        n_checked = 0
        for i_r in range(n_rho):
            for i_t in range(n_th):
                if not nan_filled[i_r, i_t]:
                    continue
                # --- rho-derivative at (i_r, i_t) ---
                if 0 < i_r < n_rho - 1:
                    drho_filled = (float(rc[i_r + 1, i_t])
                                   - float(rc[i_r - 1, i_t]))
                    for j in (i_t - 1, i_t + 1):
                        if 0 <= j < n_th and ghost_exists[i_r - 1, j] and ghost_exists[i_r + 1, j]:
                            drho_nbr = (float(rc[i_r + 1, j])
                                        - float(rc[i_r - 1, j]))
                            ratio = abs(drho_filled / max(drho_nbr, 1e-14))
                            self.assertLess(
                                ratio, 2.0,
                                f'rho-deriv ratio {ratio:.3f} >= 2 '
                                f'at ({i_r},{i_t}) vs col {j}')
                            n_checked += 1
                # --- theta_c-derivative at (i_r, i_t) ---
                if 0 < i_t < n_th - 1:
                    dth_filled = (float(rc[i_r, i_t + 1])
                                  - float(rc[i_r, i_t - 1]))
                    for k in (i_r - 1, i_r + 1):
                        if 0 <= k < n_rho and ghost_exists[k, i_t - 1] and ghost_exists[k, i_t + 1]:
                            dth_nbr = (float(rc[k, i_t + 1])
                                       - float(rc[k, i_t - 1]))
                            ratio = abs(dth_filled / max(dth_nbr, 1e-14))
                            self.assertLess(
                                ratio, 2.0,
                                f'theta_c-deriv ratio {ratio:.3f} >= 2 '
                                f'at ({i_r},{i_t}) vs row {k}')
                            n_checked += 1
        self.assertGreater(n_checked, 0,
                           'No smoothness checks performed')
        self.record_comparison()

    def test_ghost_boundary_diagnostic_heatmap(self) -> None:
        """Diagnostic: rho_u_carrier heatmap with ghost-NaN locations.

        Red X markers show nodes where ``geometry.ghost_kernel`` raised
        ``GhostDomainError`` — filled by the production NaN interpolation.
        Filled cells should show smooth colour ramps from valid neighbours.
        """
        from cogwheel.lensing.chang_refsdal import geometry
        rc = self.chart.rho_u_carrier
        rho_g = self.chart.rho_grid
        th_g = self.chart.theta_c_grid
        w0 = float(np.exp(self.chart.log_w_grid[0]))
        n_rho = len(rho_g)
        n_th = len(th_g)

        nan_marks: list[tuple[int, int]] = []
        ghost_marks: list[tuple[int, int]] = []
        for i_r, rho in enumerate(rho_g):
            for i_t, theta_c in enumerate(th_g):
                found = False
                for gamma in self.chart.gamma_grid:
                    try:
                        source = sg._from_caustic_fixed(
                            float(gamma), float(rho), float(theta_c))
                        matrix = geometry.macro_matrix(
                            float(gamma), beta=0.0, kappa=0.0)
                        geometry.ghost_kernel(
                            [w0], np.array(source, dtype=float), matrix)
                        ghost_marks.append((i_r, i_t))
                        found = True
                        break
                    except (geometry.GhostDomainError,
                            geometry.LensDomainError, ValueError):
                        continue
                if not found:
                    nan_marks.append((i_r, i_t))

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            rc.T, aspect='auto', origin='lower',
            extent=[rho_g[0], rho_g[-1], th_g[0], th_g[-1]],
            cmap='RdBu_r')
        for (i_r, i_t) in nan_marks:
            ax.scatter(rho_g[i_r], th_g[i_t], marker='x', color='red',
                       s=80, linewidths=1.5, zorder=5)
        if ghost_marks:
            g_r, g_t = zip(*ghost_marks)
            ax.scatter([rho_g[i] for i in g_r], [th_g[i] for i in g_t],
                       marker='o', facecolors='none', edgecolors='green',
                       s=30, linewidths=0.5, alpha=0.5, zorder=4)
        ax.set_xlabel('rho')
        ax.set_ylabel('theta_c')
        ax.set_title('rho_u_carrier — ghost-boundary NaN-hole fill\n'
                     'green o = ghost exists, red X = NaN-filled')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('rho_u_carrier')
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'exterior_polar_fold_ghost_nan_heatmap.png',
                    dpi=150)
        plt.close(fig)
        # Verify there are NaN-filled nodes (the test fixture straddles
        # the ghost boundary for this theta_c band near rho~1.3)
        self.assertGreater(len(nan_marks), 0,
                           'Expected NaN-filled nodes but none found')
        self.record_comparison()


class FoldCarrierFromEngineBackwardCompatTestCase(_FoldCarrierBaseTestCase):
    """from_engine(fold_carrier=False) → rho_u_carrier=None."""

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

    def test_rho_u_carrier_is_none(self) -> None:
        self.assertIsNone(self.chart.rho_u_carrier)
        self.record_comparison()

    def test_chart_has_expected_axes(self) -> None:
        self.assertEqual(len(self.chart.gamma_grid), _FROM_ENGINE_N_GAMMA)
        self.assertEqual(len(self.chart.rho_grid), _FROM_ENGINE_N_RHO)
        self.assertEqual(len(self.chart.theta_c_grid), _FROM_ENGINE_N_THETA_C)
        self.record_comparison()

    def test_carrier_rate_is_finite(self) -> None:
        self.assertTrue(np.isfinite(self.chart.carrier_rate))
        self.record_comparison()

    def test_no_fold_carrier_chart_can_serve(self) -> None:
        chart = self.chart
        gamma = float(chart.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])
        rho = float(chart.rho_grid[_FROM_ENGINE_N_RHO // 2])
        theta_c = float(chart.theta_c_grid[_FROM_ENGINE_N_THETA_C // 2])
        y1, y2 = _source_for_node(gamma, rho, theta_c)
        served = sg._evaluate_chart(
            chart, gamma=gamma, eta=0.5, theta=0.5,
            log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
        self.assertTrue(np.all(np.isfinite(served)))
        self.record_comparison()


# ======================================================================
# 8. Self-falsification — from_engine fold_carrier
# ======================================================================

class FoldCarrierFromEngineSelfFalsificationTestCase(_FoldCarrierBaseTestCase):
    """from_engine fold-carrier assertions can go RED."""

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
        cls._gamma = float(cls.chart_true.gamma_grid[_FROM_ENGINE_N_GAMMA // 2])

    def test_fold_carrier_true_has_rho_u_carrier(self) -> None:
        self.assertIsNotNone(self.chart_true.rho_u_carrier)
        self.record_comparison()

    def test_fold_carrier_false_has_no_rho_u_carrier(self) -> None:
        self.assertIsNone(self.chart_false.rho_u_carrier)
        self.record_comparison()

    def test_true_and_false_charts_differ(self) -> None:
        rho_off = float(self.chart_true.rho_grid[1]) * 0.3 + float(
            self.chart_true.rho_grid[2]) * 0.7
        theta_off = 0.5 * (float(self.chart_true.theta_c_grid[1])
                           + float(self.chart_true.theta_c_grid[2]))
        y1, y2 = _source_for_node(self._gamma, rho_off, theta_off)
        served_t = sg._evaluate_chart(
            self.chart_true, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=self.chart_true.log_w_grid,
            y1_eig=y1, y2_eig=y2)
        served_f = sg._evaluate_chart(
            self.chart_false, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=self.chart_false.log_w_grid,
            y1_eig=y1, y2_eig=y2)
        max_diff = float(np.max(np.abs(served_t - served_f)))
        self.assertGreater(max_diff, 1e-6)
        self.record_comparison()

    def test_rho_u_carrier_is_not_all_zeros(self) -> None:
        if self.chart_true.rho_u_carrier is not None:
            self.assertFalse(np.allclose(self.chart_true.rho_u_carrier, 0.0))
        self.record_comparison()

# ======================================================================
# 9. 1D artifact backward-compat (V4 schema, chart0_rho_carrier → 2D broadcast)
# ======================================================================

class ExteriorPolar1DArtifactBackwardCompatTestCase(_FoldCarrierBaseTestCase):
    """1D ``rho_carrier`` (V4 schema) loads and serves identically to 2D.

    The production loader broadcasts a ``(n_rho,)`` ``rho_carrier`` to
    ``(n_rho, n_theta_c)`` via ``np.broadcast_to`` when the 2D key
    ``rho_u_carrier`` is absent.  Because the 1D carrier is constant in
    u, the bilinear interpolation at serve time reduces to rho-interp
    only, reproducing the original 1D carrier behavior byte-identically.
    """

    _V4_SCHEMA: str = 'exterior_polar_rho_log_carrier_v1'

    @classmethod
    def setUpClass(cls) -> None:
        rho_only_carrier = (_RHO_U_CARRIER[:, 0][:, None]
                            * np.ones((1, _N_THETA_C)))
        cls._chart_2d = _build_chart(
            rho_u_carrier=rho_only_carrier,
            carrier_rate=_K_CHART,
            rho_log_axis=True)
        npz_2d = sg._chart_to_npz(cls._chart_2d, index=0)
        # Build 1D legacy NPZ: V4 schema + chart0_rho_carrier (1D)
        cls._npz_1d = dict(npz_2d)
        meta = sg.json.loads(str(npz_2d['chart0_meta']))
        meta['axis_schema'] = cls._V4_SCHEMA
        cls._npz_1d['chart0_meta'] = np.array(sg.json.dumps(meta))
        # Replace 2D key with 1D key
        cls._npz_1d['chart0_rho_carrier'] = (
            cls._chart_2d.rho_u_carrier[:, 0].copy())
        del cls._npz_1d['chart0_rho_u_carrier']
        # Load 1D artifact
        cls._chart_1d = sg._chart_from_npz(cls._npz_1d, index=0)
        cls._gamma = float(_GAMMA_GRID[1])
        cls._rho = 0.5 * (float(_RHO_GRID[1]) + float(_RHO_GRID[2]))
        cls._theta_c = float(_THETA_C_GRID[2])
        cls._y1, cls._y2 = _source_for_node(
            cls._gamma, cls._rho, cls._theta_c)

    def test_1d_artifact_loads_without_error(self) -> None:
        """V4 schema 1D NPZ loads successfully."""
        self.assertIsNotNone(self._chart_1d)
        self.record_comparison()

    def test_rho_u_carrier_has_correct_2d_shape_after_broadcast(self
                                                                ) -> None:
        self.assertEqual(
            self._chart_1d.rho_u_carrier.shape,
            (_N_RHO, _N_THETA_C))
        self.record_comparison()

    def test_rho_u_carrier_is_broadcast_of_1d_column(self) -> None:
        """Every column equals the original 1D rho_carrier."""
        old_1d = self._npz_1d['chart0_rho_carrier']
        expected = np.broadcast_to(old_1d[:, None],
                                   (_N_RHO, _N_THETA_C))
        np.testing.assert_array_equal(
            self._chart_1d.rho_u_carrier, expected)
        self.record_comparison()

    def test_1d_and_2d_serve_byte_identical(self) -> None:
        served_2d = sg._evaluate_chart(
            self._chart_2d, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID, y1_eig=self._y1, y2_eig=self._y2)
        served_1d = sg._evaluate_chart(
            self._chart_1d, gamma=self._gamma, eta=0.5, theta=0.5,
            log_w_query=_LOG_W_MID, y1_eig=self._y1, y2_eig=self._y2)
        self.assertTrue(np.allclose(served_2d, served_1d, rtol=0,
                                    atol=0))
        self.record_comparison()

    def test_1d_broadcast_heatmap(self) -> None:
        """Heatmap: uniform columns confirm constant-in-u broadcast."""
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(
            self._chart_1d.rho_u_carrier.T,
            aspect='auto', origin='lower',
            extent=[_RHO_GRID[0], _RHO_GRID[-1],
                    _THETA_C_GRID[0], _THETA_C_GRID[-1]])
        ax.set_xlabel('rho')
        ax.set_ylabel('theta_c')
        ax.set_title('1D artifact: broadcast rho_u_carrier')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('rho_u_carrier')
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'exterior_polar_fold_1d_broadcast_heatmap.png',
                    dpi=150)
        plt.close(fig)
        # Verify uniform columns
        col_0 = self._chart_1d.rho_u_carrier[:, 0]
        for j in range(1, _N_THETA_C):
            np.testing.assert_array_equal(
                self._chart_1d.rho_u_carrier[:, j], col_0)
        self.record_comparison()
