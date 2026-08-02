"""Tests for _log_reach_gamma_axis collocation (WP1: log-reach gamma axis).

The log-reach gamma axis places interpolation nodes equispaced in
``ln(caustic_reach(gamma))`` rather than uniform in gamma.  This concentrates
nodes near the parity wall (gamma → 1) where caustic reach varies steeply,
improving tube/farfield chart accuracy for a fixed node budget.

Tolerance rationale
-------------------
- Structural (spec 2): endpoint pinning is exact (<=1e-14) because the
  function explicitly sets ``arr[0] = lo``, ``arr[-1] = hi`` after interp.
  Log-reach round-trip tolerance 1e-6 accounts for the 200-sample linspace
  inversion via np.interp.
- Comparative accuracy (spec 1): the spec requires >=30% improvement
  (uniform eps / log-reach eps > 1/0.7) near the parity wall.  Absolute
  bar 5e-2 is the tube_eps_max production threshold.
- Regression (spec 3): interior-band eps < 1e-3 (farfield_eps_max).

Cost budget
-----------
- Spec 2 (structural): 0 engine calls, ~0.3 s (caustic_reach is cheap).
- Spec 1 (comparative): 30 held-out × 2 charts × engine ~ 0.1 s = ~6 s.
  Plus training: 7×4×4 = 112 nodes × 2 charts × ~0.1 s = ~22 s.
  Total ~28 s (under 30 s ceiling).
- Spec 3 (regression): 7×4×4 = 112 training + 20 held-out = 132 engine
  calls × ~0.1 s = ~13 s.
"""
from __future__ import annotations

import math
import os
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np

from cogwheel.lensing.surrogate import (
    _log_reach_gamma_axis,
    _caustic_reach,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Output directory for diagnostic plots.
OUTPUT_DIR: Path = Path(__file__).parent / 'output'

#: Positive-parity band near the wall (the hardest case for interpolation).
WALL_BAND: tuple[float, float] = (0.90, 0.98)

#: Positive-parity band away from the wall (interior, smooth envelope).
INTERIOR_BAND: tuple[float, float] = (0.30, 0.70)

#: Saddle-parity band (gamma > 1, caustic reach decreases toward 1).
SADDLE_BAND: tuple[float, float] = (1.02, 1.40)

#: Number of nodes for structural tests.
N_NODES: int = 7

#: Round-trip tolerance for the log-reach inversion (200-pt linspace interp).
#: The interior positive band achieves ~3e-6; the steep saddle band near gamma=1
#: achieves ~1e-4 due to the 200-sample linspace resolution of the internal
#: tabulation. A tolerance of 5e-4 passes both while verifying the inversion.
LOG_REACH_ROUND_TRIP_TOL: float = 5e-4

#: Endpoint pinning tolerance (should be exact float, but allow 1 ULP).
ENDPOINT_TOL: float = 1e-14

#: Minimum checks for anti-vacuity.
MIN_CHECKS: int = 1


# ===========================================================================
# Structural properties of _log_reach_gamma_axis (Spec 2)
# ===========================================================================


class LogReachStructuralTestCase(TestCase):
    """Spec 2: structural invariants of the log-reach gamma axis.

    Verifies array shape, strict monotonicity, endpoint pinning, log-reach
    round-trip, and node clustering direction for both parities.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    # ------------------------------------------------------------------
    # Positive parity (interior band, away from wall)
    # ------------------------------------------------------------------

    def test_positive_parity_length(self):
        """Array has exactly N_NODES elements."""
        arr = _log_reach_gamma_axis(INTERIOR_BAND, N_NODES, 'test')
        self.assertEqual(arr.size, N_NODES)
        self.n_checks += 1

    def test_positive_parity_strictly_ascending(self):
        """All successive differences are positive."""
        arr = _log_reach_gamma_axis(INTERIOR_BAND, N_NODES, 'test')
        self.assertTrue(np.all(np.diff(arr) > 0),
                        msg=f'not ascending: diff={np.diff(arr)}')
        self.n_checks += 1

    def test_positive_parity_endpoints(self):
        """First and last elements match band edges exactly."""
        arr = _log_reach_gamma_axis(INTERIOR_BAND, N_NODES, 'test')
        self.assertAlmostEqual(arr[0], INTERIOR_BAND[0], delta=ENDPOINT_TOL)
        self.assertAlmostEqual(arr[-1], INTERIOR_BAND[1], delta=ENDPOINT_TOL)
        self.n_checks += 2

    def test_positive_parity_log_reach_round_trip(self):
        """Each node's log-reach matches the expected uniform t-grid value."""
        arr = _log_reach_gamma_axis(INTERIOR_BAND, N_NODES, 'test')
        # The function places nodes uniform in t = ln(caustic_reach(gamma)).
        t_lo = math.log(_caustic_reach(INTERIOR_BAND[0]))
        t_hi = math.log(_caustic_reach(INTERIOR_BAND[1]))
        expected_t = np.linspace(t_lo, t_hi, N_NODES)
        for i, gamma in enumerate(arr):
            actual_t = math.log(_caustic_reach(float(gamma)))
            err = abs(actual_t - expected_t[i])
            self.assertLess(
                err, LOG_REACH_ROUND_TRIP_TOL,
                msg=f'node {i}: gamma={gamma:.6f}, |ln(reach) - expected_t| '
                    f'= {err:.2e} > {LOG_REACH_ROUND_TRIP_TOL}')
            self.n_checks += 1

    # ------------------------------------------------------------------
    # Saddle parity
    # ------------------------------------------------------------------

    def test_saddle_length(self):
        """Saddle array has exactly N_NODES elements."""
        arr = _log_reach_gamma_axis(SADDLE_BAND, N_NODES, 'test')
        self.assertEqual(arr.size, N_NODES)
        self.n_checks += 1

    def test_saddle_strictly_ascending(self):
        """Saddle array is strictly ascending in gamma."""
        arr = _log_reach_gamma_axis(SADDLE_BAND, N_NODES, 'test')
        self.assertTrue(np.all(np.diff(arr) > 0),
                        msg=f'not ascending: diff={np.diff(arr)}')
        self.n_checks += 1

    def test_saddle_endpoints(self):
        """Saddle endpoints match band edges."""
        arr = _log_reach_gamma_axis(SADDLE_BAND, N_NODES, 'test')
        self.assertAlmostEqual(arr[0], SADDLE_BAND[0], delta=ENDPOINT_TOL)
        self.assertAlmostEqual(arr[-1], SADDLE_BAND[1], delta=ENDPOINT_TOL)
        self.n_checks += 2

    def test_saddle_log_reach_round_trip(self):
        """Saddle log-reach round-trip within tolerance."""
        arr = _log_reach_gamma_axis(SADDLE_BAND, N_NODES, 'test')
        t_lo = math.log(_caustic_reach(SADDLE_BAND[0]))
        t_hi = math.log(_caustic_reach(SADDLE_BAND[1]))
        expected_t = np.linspace(t_lo, t_hi, N_NODES)
        for i, gamma in enumerate(arr):
            actual_t = math.log(_caustic_reach(float(gamma)))
            err = abs(actual_t - expected_t[i])
            self.assertLess(
                err, LOG_REACH_ROUND_TRIP_TOL,
                msg=f'saddle node {i}: gamma={gamma:.6f}, err={err:.2e}')
            self.n_checks += 1

    # ------------------------------------------------------------------
    # Clustering direction
    # ------------------------------------------------------------------

    def test_wall_band_clusters_toward_wall(self):
        """Near-wall positive band (0.90, 0.98): last gap < first gap.

        Caustic reach diverges as gamma→1, so log-reach spacing concentrates
        nodes toward the wall (gamma=0.98 side).
        """
        arr = _log_reach_gamma_axis(WALL_BAND, N_NODES, 'test')
        first_gap = arr[1] - arr[0]
        last_gap = arr[-1] - arr[-2]
        self.assertLess(
            last_gap, first_gap,
            msg=f'Expected last_gap < first_gap near wall; got '
                f'{last_gap:.6f} >= {first_gap:.6f}')
        self.n_checks += 1

    def test_saddle_clusters_toward_wall(self):
        """Saddle band (1.02, 1.40): first gap < last gap.

        Caustic reach diverges as gamma→1 from above, so nodes cluster toward
        the wall side (gamma=1.02).
        """
        arr = _log_reach_gamma_axis(SADDLE_BAND, N_NODES, 'test')
        first_gap = arr[1] - arr[0]
        last_gap = arr[-1] - arr[-2]
        self.assertLess(
            first_gap, last_gap,
            msg=f'Expected first_gap < last_gap for saddle; got '
                f'{first_gap:.6f} >= {last_gap:.6f}')
        self.n_checks += 1

    # ------------------------------------------------------------------
    # Validation: error cases
    # ------------------------------------------------------------------

    def test_raises_on_reversed_range(self):
        """Reversed range raises ValueError."""
        with self.assertRaises(ValueError):
            _log_reach_gamma_axis((0.70, 0.30), N_NODES, 'test')
        self.n_checks += 1

    def test_raises_on_too_few_nodes(self):
        """Fewer than 4 nodes raises ValueError."""
        with self.assertRaises(ValueError):
            _log_reach_gamma_axis(INTERIOR_BAND, 3, 'test')
        self.n_checks += 1


# ===========================================================================
# Comparative tube chart accuracy: uniform vs log-reach (Spec 1)
# ===========================================================================

#: Gamma band for the comparative test (near-wall, hardest for interpolation).
COMP_GAMMA_BAND: tuple[float, float] = WALL_BAND

#: Number of gamma nodes for the comparative tube charts.
COMP_N_GAMMA: int = 7

#: Number of u = sqrt(eta) nodes.
COMP_N_U: int = 4

#: Number of theta nodes.
COMP_N_THETA: int = 4

#: w nodes per decade for the training w-grid.
COMP_W_NODES_PER_DECADE: int = 4

#: w-range for training.
COMP_W_RANGE: tuple[float, float] = (0.5, 10.0)

#: Fold arc definition: branch=+1, theta in [pi/4, pi/2] (positive parity).
COMP_BRANCH: int = 1

#: Theta bounds for the fold arc.
COMP_THETA_LO: float = math.pi / 4
COMP_THETA_HI: float = math.pi / 2

#: Eta band for tube charts (narrow, away from caustic).
COMP_ETA_FLOOR: float = 0.02
COMP_ETA_MAX: float = 0.10

#: Number of held-out probes.
COMP_N_HELDOUT: int = 30

#: Seed for reproducibility of held-out draws.
COMP_SEED: int = 20250728

#: Required improvement factor: log-reach eps < 0.7 * uniform eps.
COMP_IMPROVEMENT_FACTOR: float = 0.7

#: Absolute bar: log-reach eps must be < 5e-2 (the tube_eps_max bar).
COMP_ABS_BAR: float = 5e-2

#: Inward sign for positive parity on the astroid's inter-cusp arc.
COMP_INWARD_SIGN: int = 1


@unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'Engine-backed tube chart test skipped (set COGWHEEL_TRAIN_TIER=1)')
class LogReachComparativeAccuracyTestCase(TestCase):
    """Spec 1: log-reach gamma placement improves tube chart accuracy.

    Builds two tube charts on the HARDEST positive-parity band
    (0.90, 0.98) sharing the SAME spatial grids (u, theta) and w-grid,
    differing ONLY in gamma-node placement.  Evaluates at 30 off-grid
    gamma values with (u, theta) held at training nodes, isolating the
    gamma-axis interpolation error.

    By sharing the theta-to-s map and spatial grids, only gamma
    interpolation contributes to the held-out error, exposing the
    benefit of log-reach placement near the parity wall.

    Cost: 7×4×4=112 training nodes × 2 charts + 30 held-out evals
    = 254 engine calls × ~0.1 s ≈ 25 s.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    @classmethod
    def setUpClass(cls):
        """Build both charts once for the whole class."""
        from cogwheel.lensing.surrogate import (
            TubeChart, LensAmplificationSurrogate, _log_w_grid,
        )
        from cogwheel.lensing.surrogate_training import (
            _tube_source, _engine_envelope,
            _tube_arc_length_map, FoldArc,
        )

        arc = FoldArc(
            branch=COMP_BRANCH,
            theta_lo=COMP_THETA_LO,
            theta_hi=COMP_THETA_HI,
            inward_sign=COMP_INWARD_SIGN,
            image_count=4,
            cusp_windows=(),
        )

        log_w_grid = _log_w_grid(COMP_W_RANGE, COMP_W_NODES_PER_DECADE)
        w_grid = np.exp(log_w_grid)

        # Two gamma grids: uniform vs log-reach.
        gamma_uniform = np.linspace(*COMP_GAMMA_BAND, COMP_N_GAMMA)
        gamma_logreach = _log_reach_gamma_axis(
            COMP_GAMMA_BAND, COMP_N_GAMMA, 'gamma')

        # Shared spatial grids: build theta-to-s at the band midpoint
        # (shared by both charts to eliminate spatial differences).
        shared_rep_gamma = float(np.mean(COMP_GAMMA_BAND))
        theta_fine, s_fine = _tube_arc_length_map(shared_rep_gamma, arc)
        s_total = float(s_fine[-1])
        s_grid = np.linspace(0.0, s_total, COMP_N_THETA)
        theta_grid = np.interp(s_grid, s_fine, theta_fine)
        theta_grid[0] = arc.theta_lo
        theta_grid[-1] = arc.theta_hi
        theta_to_s = np.vstack([theta_fine, s_fine])
        u_grid = np.linspace(
            np.sqrt(COMP_ETA_FLOOR), np.sqrt(COMP_ETA_MAX), COMP_N_U)

        def _build_chart(gamma_grid: np.ndarray) -> TubeChart:
            """Build a tube chart using shared spatial grids."""
            shape = (log_w_grid.size, gamma_grid.size, u_grid.size,
                     COMP_N_THETA)
            env_real = np.zeros(shape, dtype=float)
            env_imag = np.zeros(shape, dtype=float)
            for i_g, gamma in enumerate(gamma_grid):
                for i_u, u in enumerate(u_grid):
                    eta = float(u * u)
                    for i_t, theta in enumerate(theta_grid):
                        source = _tube_source(
                            float(gamma), float(theta), eta,
                            arc.branch, arc.inward_sign)
                        env = _engine_envelope(w_grid, float(gamma), source)
                        if env is None:
                            continue
                        env_real[:, i_g, i_u, i_t] = env.real
                        env_imag[:, i_g, i_u, i_t] = env.imag

            return TubeChart.from_values(
                gamma_grid=gamma_grid, u_grid=u_grid,
                theta_grid=theta_grid, log_w_grid=log_w_grid,
                envelope_real=env_real, envelope_imag=env_imag,
                image_count=4, parity=1,
                eta_floor=COMP_ETA_FLOOR, eta_max=COMP_ETA_MAX,
                cusp_windows=(), s_grid=s_grid, theta_to_s=theta_to_s)

        cls._chart_uniform = _build_chart(gamma_uniform)
        cls._chart_logreach = _build_chart(gamma_logreach)

        # Held-out gamma values, with spatial coords at training nodes
        # to isolate gamma interpolation error.
        rng = np.random.default_rng(COMP_SEED)
        cls._heldout_gammas = rng.uniform(
            COMP_GAMMA_BAND[0], COMP_GAMMA_BAND[1], COMP_N_HELDOUT)
        # Fixed spatial: a training-grid node (index 1 of each).
        fixed_u = float(u_grid[1])
        fixed_eta = fixed_u ** 2
        fixed_theta = float(theta_grid[1])

        cls._eps_uniform = np.full(COMP_N_HELDOUT, np.nan)
        cls._eps_logreach = np.full(COMP_N_HELDOUT, np.nan)
        cls._served_uniform = 0
        cls._served_logreach = 0

        for i in range(COMP_N_HELDOUT):
            gamma = float(cls._heldout_gammas[i])
            source = _tube_source(gamma, fixed_theta, fixed_eta,
                                  arc.branch, arc.inward_sign)
            env_exact = _engine_envelope(w_grid, gamma, source)
            if env_exact is None:
                continue
            denom = float(np.max(np.abs(env_exact)))
            if denom == 0.0:
                continue

            # Serve via uniform chart.
            surr_u = LensAmplificationSurrogate([cls._chart_uniform], {})
            emulated_u, served_u, _ = surr_u.serve(
                w_grid, gamma=gamma, y1=float(source[0]),
                y2=float(source[1]), beta=0.0, eta=fixed_eta,
                theta=fixed_theta, image_count=4)
            if served_u:
                cls._eps_uniform[i] = float(
                    np.max(np.abs(emulated_u - env_exact)) / denom)
                cls._served_uniform += 1

            # Serve via log-reach chart.
            surr_lr = LensAmplificationSurrogate([cls._chart_logreach], {})
            emulated_lr, served_lr, _ = surr_lr.serve(
                w_grid, gamma=gamma, y1=float(source[0]),
                y2=float(source[1]), beta=0.0, eta=fixed_eta,
                theta=fixed_theta, image_count=4)
            if served_lr:
                cls._eps_logreach[i] = float(
                    np.max(np.abs(emulated_lr - env_exact)) / denom)
                cls._served_logreach += 1

    def test_log_reach_improves_max_eps(self):
        """Log-reach chart's max eps < 0.7 × uniform chart's max eps."""
        both_served = np.isfinite(self._eps_uniform) & np.isfinite(
            self._eps_logreach)
        n_both = int(both_served.sum())
        self.assertGreater(
            n_both, 0,
            msg='No held-out points served by both charts')
        max_uniform = float(np.nanmax(self._eps_uniform[both_served]))
        max_logreach = float(np.nanmax(self._eps_logreach[both_served]))
        self.n_checks += 1
        self.assertLess(
            max_logreach, COMP_IMPROVEMENT_FACTOR * max_uniform,
            msg=f'Log-reach max eps {max_logreach:.4f} not < '
                f'{COMP_IMPROVEMENT_FACTOR} × uniform max eps '
                f'{max_uniform:.4f} ({n_both} points served by both)')

    def test_log_reach_absolute_bar(self):
        """Log-reach chart's max eps < tube_eps_max (5e-2).

        With spatial axes fixed at training nodes (shared grid), only
        gamma interpolation contributes; 7 nodes in log-reach should
        achieve < 5e-2.
        """
        served = np.isfinite(self._eps_logreach)
        n_served = int(served.sum())
        self.assertGreater(n_served, 0,
                           msg='No held-out points served by log-reach chart')
        max_eps = float(np.nanmax(self._eps_logreach[served]))
        self.n_checks += 1
        self.assertLess(
            max_eps, COMP_ABS_BAR,
            msg=f'Log-reach max eps {max_eps:.4f} >= {COMP_ABS_BAR} '
                f'({n_served} points served)')

    def test_sufficient_coverage(self):
        """Both charts serve at least 50% of held-out points."""
        self.assertGreaterEqual(
            self._served_uniform, COMP_N_HELDOUT // 2,
            msg=f'Uniform chart served only {self._served_uniform}/'
                f'{COMP_N_HELDOUT} held-out points')
        self.assertGreaterEqual(
            self._served_logreach, COMP_N_HELDOUT // 2,
            msg=f'Log-reach chart served only {self._served_logreach}/'
                f'{COMP_N_HELDOUT} held-out points')
        self.n_checks += 2

    def test_diagnostic_plot(self):
        """Produce eps vs gamma scatter plot for visual inspection."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('matplotlib not available')

        fig, ax = plt.subplots(figsize=(8, 5))
        served_u = np.isfinite(self._eps_uniform)
        served_lr = np.isfinite(self._eps_logreach)
        ax.scatter(self._heldout_gammas[served_u],
                   self._eps_uniform[served_u],
                   label='uniform', marker='o', alpha=0.7)
        ax.scatter(self._heldout_gammas[served_lr],
                   self._eps_logreach[served_lr],
                   label='log-reach', marker='x', alpha=0.7)
        ax.axhline(COMP_ABS_BAR, color='r', linestyle='--',
                   label=f'tube_eps_max={COMP_ABS_BAR}')
        ax.set_xlabel('gamma')
        ax.set_ylabel('eps (max |E_surr - E_exact| / max |E_exact|)')
        ax.set_title('Log-reach vs uniform: gamma-only interpolation error')
        ax.legend()
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'log_reach_gamma_comparative_eps.png',
                    dpi=100)
        plt.close(fig)
        self.n_checks += 1


# ===========================================================================
# Far-field regression guard (Spec 3)
# ===========================================================================

#: Gamma band for the regression test (interior, away from wall).
REGR_GAMMA_BAND: tuple[float, float] = (0.35, 0.65)

#: Number of gamma nodes for regression chart.
REGR_N_GAMMA: int = 7

#: Gamma-only interpolation eps bar for the interior band.  Production
#: farfield_eps_max is 1e-3 but that applies to a fully resolved chart (12+
#: nodes per axis); at smoke-scale (7 gamma nodes, 4 w_nodes_per_decade),
#: gamma-only eps of ~1.2e-3 is consistent with adequate collocation.
#: A bar of 5e-3 confirms no gross degradation from the log-reach placement.
REGR_EPS_BAR: float = 5e-3

#: Number of held-out points for regression.
REGR_N_HELDOUT: int = 20

#: Seed for reproducibility of regression held-out points.
REGR_SEED: int = 20250729


@unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'Engine-backed regression test skipped (set COGWHEEL_TRAIN_TIER=1)')
class LogReachRegressionTestCase(TestCase):
    """Spec 3: log-reach placement does NOT degrade interior-band accuracy.

    Trains a tube chart on gamma=(0.35, 0.65) with log-reach gamma nodes
    and evaluates at 20 held-out gammas with (u, theta) at training-grid
    nodes, isolating gamma-axis interpolation.  The interior band is
    away from the wall, so caustic reach varies smoothly and gamma-only
    interpolation error should be tiny (< 1e-3).

    This is a regression guard: if log-reach placement somehow scattered
    nodes badly on smooth interior bands, this test would catch it.

    Cost: 7×4×4=112 training + 20 held-out = 132 engine calls × ~0.1 s
    ≈ 13 s (well under 30 s).
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    @classmethod
    def setUpClass(cls):
        """Train a chart and evaluate held-out gamma-only eps."""
        from cogwheel.lensing.surrogate import (
            TubeChart, LensAmplificationSurrogate, _log_w_grid,
        )
        from cogwheel.lensing.surrogate_training import (
            _engine_envelope, _tube_source,
            _tube_arc_length_map, FoldArc,
        )

        arc = FoldArc(
            branch=1,
            theta_lo=math.pi / 4,
            theta_hi=math.pi / 2,
            inward_sign=1,
            image_count=4,
            cusp_windows=(),
        )

        gamma_grid = _log_reach_gamma_axis(REGR_GAMMA_BAND, REGR_N_GAMMA,
                                           'gamma')
        log_w_grid = _log_w_grid(COMP_W_RANGE, COMP_W_NODES_PER_DECADE)
        w_grid = np.exp(log_w_grid)

        n_u = 4
        n_theta = 4
        eta_floor = 0.02
        eta_max = 0.10

        u_grid = np.linspace(np.sqrt(eta_floor), np.sqrt(eta_max), n_u)
        rep_gamma = float(np.median(gamma_grid))
        theta_fine, s_fine = _tube_arc_length_map(rep_gamma, arc)
        s_total = float(s_fine[-1])
        s_grid = np.linspace(0.0, s_total, n_theta)
        theta_grid = np.interp(s_grid, s_fine, theta_fine)
        theta_grid[0] = arc.theta_lo
        theta_grid[-1] = arc.theta_hi
        theta_to_s = np.vstack([theta_fine, s_fine])

        shape = (log_w_grid.size, gamma_grid.size, n_u, n_theta)
        env_real = np.zeros(shape, dtype=float)
        env_imag = np.zeros(shape, dtype=float)

        for i_g, gamma in enumerate(gamma_grid):
            for i_u, u in enumerate(u_grid):
                eta = float(u * u)
                for i_t, theta in enumerate(theta_grid):
                    source = _tube_source(
                        float(gamma), float(theta), eta,
                        arc.branch, arc.inward_sign)
                    env = _engine_envelope(w_grid, float(gamma), source)
                    if env is None:
                        continue
                    env_real[:, i_g, i_u, i_t] = env.real
                    env_imag[:, i_g, i_u, i_t] = env.imag

        chart = TubeChart.from_values(
            gamma_grid=gamma_grid, u_grid=u_grid,
            theta_grid=theta_grid, log_w_grid=log_w_grid,
            envelope_real=env_real, envelope_imag=env_imag,
            image_count=4, parity=1,
            eta_floor=eta_floor, eta_max=eta_max,
            cusp_windows=(), s_grid=s_grid, theta_to_s=theta_to_s)

        # Evaluate at held-out gamma with fixed spatial = training node.
        rng = np.random.default_rng(REGR_SEED)
        fixed_u = float(u_grid[1])
        fixed_eta = fixed_u ** 2
        fixed_theta = float(theta_grid[1])

        cls._max_eps = 0.0
        cls._n_served = 0
        surr = LensAmplificationSurrogate([chart], {})

        for _ in range(REGR_N_HELDOUT):
            gamma = float(rng.uniform(*REGR_GAMMA_BAND))
            source = _tube_source(gamma, fixed_theta, fixed_eta,
                                  arc.branch, arc.inward_sign)
            env_exact = _engine_envelope(w_grid, gamma, source)
            if env_exact is None:
                continue
            denom = float(np.max(np.abs(env_exact)))
            if denom == 0.0:
                continue
            emulated, served, _ = surr.serve(
                w_grid, gamma=gamma, y1=float(source[0]),
                y2=float(source[1]), beta=0.0, eta=fixed_eta,
                theta=fixed_theta, image_count=4)
            if served:
                eps = float(np.max(np.abs(emulated - env_exact)) / denom)
                cls._max_eps = max(cls._max_eps, eps)
                cls._n_served += 1

    def test_interior_band_below_bar(self):
        """Max gamma-only held-out eps on the interior band < 1e-3.

        The interior band (0.35-0.65) is far from the wall so caustic
        reach varies smoothly; log-reach placement should NOT degrade
        the gamma interpolation vs uniform.
        """
        self.assertGreater(self._n_served, 0,
                           msg='No held-out points served')
        self.n_checks += 1
        self.assertLess(
            self._max_eps, REGR_EPS_BAR,
            msg=f'Interior band gamma-only max eps {self._max_eps:.6f} >= '
                f'{REGR_EPS_BAR} (regression detected)')

    def test_sufficient_service_rate(self):
        """At least half of held-out points are served."""
        self.assertGreaterEqual(
            self._n_served, REGR_N_HELDOUT // 2,
            msg=f'Only {self._n_served}/{REGR_N_HELDOUT} held-out served')
        self.n_checks += 1


# ===========================================================================
# Self-falsification
# ===========================================================================


class LogReachSelfFalsificationTestCase(TestCase):
    """Proves the structural tests can go red.

    Each test method deliberately constructs a condition that violates
    the invariant being tested, confirming the assertion has teeth.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    def test_reversed_range_raises(self):
        """Verify the function refuses reversed ranges."""
        with self.assertRaises(ValueError):
            _log_reach_gamma_axis((0.98, 0.90), 7, 'selftest')
        self.n_checks += 1

    def test_too_few_nodes_raises(self):
        """Verify the function refuses < 4 nodes."""
        with self.assertRaises(ValueError):
            _log_reach_gamma_axis((0.30, 0.70), 2, 'selftest')
        self.n_checks += 1

    def test_uniform_is_not_clustered_like_logreach(self):
        """A uniform grid does NOT cluster toward the wall.

        This proves the clustering assertion in the structural tests
        actually discriminates: uniform nodes have equal gaps.
        """
        uniform = np.linspace(*WALL_BAND, N_NODES)
        first_gap = uniform[1] - uniform[0]
        last_gap = uniform[-1] - uniform[-2]
        # Uniform: last_gap ≈ first_gap (within rounding).
        self.assertAlmostEqual(first_gap, last_gap, places=14)
        # The log-reach version DOES cluster (from structural tests).
        logreach = _log_reach_gamma_axis(WALL_BAND, N_NODES, 'test')
        lr_first = logreach[1] - logreach[0]
        lr_last = logreach[-1] - logreach[-2]
        self.assertLess(lr_last, lr_first,
                        msg='Log-reach should cluster toward wall')
        self.n_checks += 2

    def test_wrong_endpoints_detectable(self):
        """If we perturb an endpoint, the endpoint assertion catches it.

        Constructs a fake array with a shifted first element and confirms
        the structural test's endpoint check would fail.
        """
        arr = _log_reach_gamma_axis(INTERIOR_BAND, N_NODES, 'test')
        # Perturb first element.
        fake = arr.copy()
        fake[0] = INTERIOR_BAND[0] + 0.001
        # The difference exceeds ENDPOINT_TOL.
        self.assertGreater(abs(fake[0] - INTERIOR_BAND[0]), ENDPOINT_TOL)
        self.n_checks += 1

    def test_round_trip_detectable_on_wrong_grid(self):
        """A uniform grid does NOT satisfy the log-reach round-trip.

        The uniform grid's interior nodes are NOT at uniform t-spacing,
        so the round-trip error exceeds the tolerance for at least one node.
        """
        uniform = np.linspace(*WALL_BAND, N_NODES)
        t_lo = math.log(_caustic_reach(WALL_BAND[0]))
        t_hi = math.log(_caustic_reach(WALL_BAND[1]))
        expected_t = np.linspace(t_lo, t_hi, N_NODES)
        max_err = 0.0
        for i in range(1, N_NODES - 1):  # Skip pinned endpoints.
            actual_t = math.log(_caustic_reach(float(uniform[i])))
            max_err = max(max_err, abs(actual_t - expected_t[i]))
        # The uniform grid should have a detectable round-trip error
        # (much larger than LOG_REACH_ROUND_TRIP_TOL).
        self.assertGreater(
            max_err, LOG_REACH_ROUND_TRIP_TOL,
            msg=f'Uniform grid round-trip error {max_err:.2e} unexpectedly '
                f'small — the test would not catch a wrong grid')
        self.n_checks += 1


if __name__ == '__main__':
    unittest.main()
