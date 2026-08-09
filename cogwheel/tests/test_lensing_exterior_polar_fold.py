"""
Tests for the D₂ fold of ExteriorPolarChart in `cogwheel.lensing.surrogate`.

The D₂ fold maps all four sign combinations (±y1, ±y2) to the same
caustic-fixed ``(rho, theta_c)`` in the canonical first quadrant
``[0, π/2]`` via `_to_exterior_fixed`.  The surrogate's envelope is thus
guaranteed D₂-invariant by construction.  These tests verify that
invariance holds for:

- the raw far-field envelope ``E_ff(w)`` (pairwise bit-identical),
- the reconstructed physical ``F(w) = E_ff(w) + Σₐ Hₐ exp(i w τₐ)``
  that combines the surrogate envelope with the partition geometry,
- `select_chart` dispatch, which must return the same chart object
  for all four D₂-equivalent query positions.

Tolerances
----------
The envelope identity is expected bit-exact (max|Δ| = 0) because the
fold maps to identical floating-point ``(rho, theta_c)``.  The
reconstructed ``F(w)`` involves matrix operations that may accumulate
floating-point noise; a 1e-14 relative tolerance suffices.
`select_chart` must return byte-identical chart objects.
"""

import unittest

import numpy as np

from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.channels import (
    FARFIELD_KERNEL_SUM, reconstruct_farfield,
    INTERIOR_SACR_C, reconstruct_from_envelope, _channel_switch)
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.surrogate import (
    LensAmplificationSurrogate,
    _evaluate_chart, _to_caustic_fixed, select_chart, ExteriorPolarChart,
    LobeInteriorChart, _lobe_serves,
    _from_caustic_fixed, _wedge_cusp_axis_map)


# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

#: Gamma nodes for the test chart (positive parity, exterior).
_CHART_GAMMA_GRID: np.ndarray = np.array([0.20, 0.30, 0.40, 0.50])

#: Rho nodes for the test chart (exterior region, rho > 1).
_CHART_RHO_GRID: np.ndarray = np.array([1.5, 2.5, 3.5, 4.5])

#: Theta_c nodes for the test chart (D₂-folded first quadrant, [0, π/2]).
_CHART_THETA_C_GRID: np.ndarray = np.linspace(0.0, np.pi / 2, 5)

#: Log-w nodes for the test chart (natural log).
_CHART_LOG_W_GRID: np.ndarray = np.log(np.geomspace(1.0, 100.0, 6))

#: w grid matching _CHART_LOG_W_GRID.
_CHART_W_GRID: np.ndarray = np.geomspace(1.0, 100.0, 6)

#: Probe gamma, inside the chart's gamma band.
_PROBE_GAMMA: float = 0.3

#: Absolute probe source coordinates (first-quadrant reference).
_PROBE_Y1: float = 1.5
_PROBE_Y2: float = 1.5

#: Four D₂-equivalent eigenframe source positions.
_D2_SOURCES: list[tuple[float, float]] = [
    (_PROBE_Y1, _PROBE_Y2),
    (-_PROBE_Y1, _PROBE_Y2),
    (_PROBE_Y1, -_PROBE_Y2),
    (-_PROBE_Y1, -_PROBE_Y2),
]

#: Identity part of the reference source.
_REF_Y1, _REF_Y2 = _PROBE_Y1, _PROBE_Y2

#: Tiny w grid for geometry-partition queries (just enough for delays/kernels).
_PARTITION_W_GRID: np.ndarray = np.geomspace(4.0, 40.0, 3)

#: Log-w of _PARTITION_W_GRID, for select_chart queries.
_PARTITION_LOG_W_GRID: np.ndarray = np.log(_PARTITION_W_GRID)

#: Relative tolerance for reconstructed F(w) identity.
_F_TOL: float = 1e-14

#: Non-overlapping gamma bands for the select_chart multi-chart test.
_SELECT_GAMMA_A: np.ndarray = np.array([0.20, 0.25, 0.30, 0.35])
_SELECT_GAMMA_B: np.ndarray = np.array([0.40, 0.45, 0.50, 0.55])

#: Gamma for which Chart A only serves.
_SELECT_GAMMA_A_ONLY: float = 0.30

#: Gamma for which Chart B only serves.
_SELECT_GAMMA_B_ONLY: float = 0.45

#: A source point NOT D2-equivalent to the probe: different radius.
_NON_D2_Y1: float = 1.7
_NON_D2_Y2: float = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chart(*, gamma_grid: np.ndarray = _CHART_GAMMA_GRID,
                 rho_grid: np.ndarray = _CHART_RHO_GRID,
                 theta_c_grid: np.ndarray = _CHART_THETA_C_GRID,
                 log_w_grid: np.ndarray = _CHART_LOG_W_GRID,
                 image_count: int = 2, parity: int = 1,
                 theta_to_u: np.ndarray | None = None,
                 u_grid: np.ndarray | None = None,
                 envelope_real: np.ndarray | None = None,
                 envelope_imag: np.ndarray | None = None,
                 **kwargs) -> ExteriorPolarChart:
    """Build a synthetic ExteriorPolarChart.

    By default the envelope is constant = 1 (valid B-spline interpolation
    target).  Pass ``theta_to_u + u_grid`` for a cusp-adapted angular axis
    (``u = d**(2/3)``); pass custom ``envelope_real`` / ``envelope_imag``
    to exercise a non-uniform angular axis.
    """
    shape = (len(log_w_grid), len(gamma_grid), len(rho_grid),
             len(theta_c_grid))
    if envelope_real is None:
        envelope_real = np.ones(shape)
    if envelope_imag is None:
        envelope_imag = np.zeros(shape)
    return ExteriorPolarChart.from_values(
        gamma_grid=gamma_grid,
        rho_grid=rho_grid,
        theta_c_grid=theta_c_grid,
        log_w_grid=log_w_grid,
        envelope_real=envelope_real,
        envelope_imag=envelope_imag,
        image_count=image_count,
        parity=parity,
        envelope_definition=FARFIELD_KERNEL_SUM,
        theta_to_u=theta_to_u,
        u_grid=u_grid,
        **kwargs)


# ===================================================================
# Test classes
# ===================================================================

class EnvelopeD2IdentityTestCase(unittest.TestCase):
    """D₂-folded envelope evaluated identically at all four (±y1, ±y2).

    Builds a synthetic ExteriorPolarChart with a small, well-resolved
    grid.  Evaluates the chart's served far-field envelope ``E_ff(w)`` at
    ``(y1, y2)`` and at all three D₂-equivalent positions ``(±y1, ±y2)``.
    Asserts pair-wise bit-identity because the fold maps all four to the
    same ``(rho, theta_c)``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart()
        cls._log_w = np.log(_CHART_W_GRID)
        cls._n_compared = 0

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName == 'test_envelope_d2_identity'):
            self.fail('no D₂ envelope comparisons executed; test is vacuous')

    def test_envelope_d2_identity(self) -> None:
        """E_ff(±y1, ±y2) is bit-identical due to D₂ fold in _to_exterior_fixed."""
        reference = _evaluate_chart(
            self.chart, gamma=_PROBE_GAMMA, eta=0.0, theta=0.0,
            log_w_query=self._log_w, y1_eig=_REF_Y1, y2_eig=_REF_Y2)

        for y1, y2 in _D2_SOURCES[1:]:  # skip reference
            env = _evaluate_chart(
                self.chart, gamma=_PROBE_GAMMA, eta=0.0, theta=0.0,
                log_w_query=self._log_w, y1_eig=y1, y2_eig=y2)
            self._n_compared += 1
            np.testing.assert_array_equal(
                env, reference,
                f'E_ff({y1}, {y2}) ≠ E_ff({_REF_Y1}, {_REF_Y2})')

    def test_fold_does_real_work(self) -> None:
        """_to_exterior_fixed's abs() fold is NOT a no-op.

        Without the abs fold, _to_caustic_fixed produces different
        theta_c for different sign combinations.  The equality of the
        test envelope above is therefore evidence, not decoration:
        a missing fold would make this test RED.
        """
        _, th_ref = _to_caustic_fixed(_PROBE_GAMMA, _REF_Y1, _REF_Y2)
        for y1, y2 in _D2_SOURCES[1:]:
            _, th_c = _to_caustic_fixed(_PROBE_GAMMA, y1, y2)
            self.assertNotEqual(
                th_c, th_ref,
                f'_to_caustic_fixed({_PROBE_GAMMA}, {y1}, {y2}) '
                f'produced the same theta_c as the reference -- '
                f'the abs fold is not needed for D₂ invariance, '
                f'and the envelope identity test would be vacuous')


class ReconstructedFD2IdentityTestCase(unittest.TestCase):
    """Full F(w) reconstruction is D₂-identical.

    For the same chart and input points, the physical
    ``F(w) = E_ff(w) + Σₐ Hₐ exp(i w τₐ)`` reconstructed through
    `reconstruct_farfield` must be identical for all four D₂-equivalent
    source positions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_chart()
        cls._log_w = np.log(_PARTITION_W_GRID)
        cls._n_compared = 0

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName == 'test_reconstructed_f_d2_identity'):
            self.fail('no F(w) D₂ comparisons executed; test is vacuous')

    def _geom_and_f(self, y1: float, y2: float):
        """Return (geom, F_total) for one eigenframe source."""
        ch = ChangRefsdalChannels(_PARTITION_W_GRID)
        geom = ch.geometry_partition(
            gamma=_PROBE_GAMMA, y=(y1, y2), beta=0.0, kappa=0.5)
        env = _evaluate_chart(
            self.chart, gamma=_PROBE_GAMMA, eta=0.0, theta=0.0,
            log_w_query=self._log_w, y1_eig=y1, y2_eig=y2)
        _kernels, total = reconstruct_farfield(
            _PARTITION_W_GRID, env, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)
        return geom, total

    def test_reconstructed_f_d2_identity(self) -> None:
        """F(±y1, ±y2) is identical within 1e-14 relative tolerance."""
        _ref_geom, f_ref = self._geom_and_f(_REF_Y1, _REF_Y2)
        ref_max = float(np.max(np.abs(f_ref)))
        self.assertGreater(ref_max, 0.0, 'reference F(w) is zero')

        for y1, y2 in _D2_SOURCES[1:]:
            _geom, f_probe = self._geom_and_f(y1, y2)
            self._n_compared += 1
            rel_max = float(
                np.max(np.abs(f_probe - f_ref)) / ref_max)
            self.assertLessEqual(
                rel_max, _F_TOL,
                f'F({y1}, {y2}) ≠ F({_REF_Y1}, {_REF_Y2}); '
                f'max|Δ|/max|F| = {rel_max:.2e} > {_F_TOL}')

    def test_non_d2_source_differs(self) -> None:
        """F(w) at a non-D2-equivalent source position IS different.

        A source with a different radius (not just sign-flipped) maps to
        a different (rho, theta_c) and thus produces a different envelope
        and reconstruction.  If this assertion fails the
        test_reconstructed_f_d2_identity test above is vacuous.
        """
        _geom, f_ref = self._geom_and_f(_REF_Y1, _REF_Y2)
        _geom, f_other = self._geom_and_f(_NON_D2_Y1, _NON_D2_Y2)
        ref_max = float(np.max(np.abs(f_ref)))
        rel_max = float(np.max(np.abs(f_other - f_ref)) / ref_max)
        self.assertGreater(
            rel_max, _F_TOL,
            f'F({_NON_D2_Y1}, {_NON_D2_Y2}) ≈ F({_REF_Y1}, '
            f'{_REF_Y2}); max|Δ|/max|F| = {rel_max:.2e} ≤ {_F_TOL} -- '
            f'the reconstructed F(w) D₂ identity test is vacuous')




class SelectChartD2ConsistencyTestCase(unittest.TestCase):
    """`select_chart` returns the same chart at all four (±y1, ±y2).

    With non-overlapping gamma bands, the D₂ fold must not alter which
    chart is selected: all four sign combinations must map to the
    same ``(rho, theta_c)`` and thus pass the same chart's box gate.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart_a = _build_chart(
            gamma_grid=_SELECT_GAMMA_A,
            rho_grid=np.array([1.5, 2.5, 3.5, 4.5]),
            theta_c_grid=np.linspace(0.0, np.pi / 2, 5),
            log_w_grid=np.log(np.geomspace(1.0, 100.0, 6)))
        cls.chart_b = _build_chart(
            gamma_grid=_SELECT_GAMMA_B,
            rho_grid=np.array([1.5, 2.5, 3.5, 4.5]),
            theta_c_grid=np.linspace(0.0, np.pi / 2, 5),
            log_w_grid=np.log(np.geomspace(1.0, 100.0, 6)))
        cls.surrogate = LensAmplificationSurrogate(
            [cls.chart_a, cls.chart_b], provenance={})
        cls._n_compared = 0

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName
                == 'test_select_chart_d2_consistency_a_only'):
            self.fail('no select_chart D₂ comparisons executed; '
                      'test is vacuous')

    def _query_select(self, gamma: float, y1: float, y2: float):
        """Return the chart selected for one (gamma, y1, y2), or None."""
        ch = ChangRefsdalChannels(_PARTITION_W_GRID)
        geom = ch.geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.5)
        return select_chart(
            self.surrogate.charts, gamma=gamma,
            log_w_min=float(_PARTITION_LOG_W_GRID.min()),
            log_w_max=float(_PARTITION_LOG_W_GRID.max()),
            eta=geom.caustic_distance, theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()), y1_eig=y1, y2_eig=y2)

    def test_select_chart_d2_consistency_a_only(self) -> None:
        """All four D₂ points select Chart A at gamma=0.3."""
        chart_ref = self._query_select(
            _SELECT_GAMMA_A_ONLY, _REF_Y1, _REF_Y2)
        self.assertIs(
            chart_ref, self.chart_a,
            f'reference point ({_REF_Y1}, {_REF_Y2}) did not select '
            'Chart A')

        for y1, y2 in _D2_SOURCES[1:]:
            chart_probe = self._query_select(
                _SELECT_GAMMA_A_ONLY, y1, y2)
            self._n_compared += 1
            selected_name = (type(chart_probe).__name__
                               if chart_probe is not None else 'None')
            self.assertIs(
                chart_probe, self.chart_a,
                f'({y1}, {y2}) selected {selected_name}, not Chart A')

    def test_different_gamma_selects_b(self) -> None:
        """At gamma=0.45 all four D₂ points select Chart B, not A."""
        for y1, y2 in _D2_SOURCES:
            chart = self._query_select(_SELECT_GAMMA_B_ONLY, y1, y2)
            self.assertIs(
                chart, self.chart_b,
                f'({y1}, {y2}) at gamma={_SELECT_GAMMA_B_ONLY} selected '
                f'{type(chart).__name__ if chart is not None else "None"}, '
                f'not Chart B')

    def test_out_of_band_gamma_returns_none(self) -> None:
        """A gamma outside all chart bands returns None, not Chart A.

        This proves the D₂ consistency test has teeth: if select_chart
        always returned Chart A regardless of gamma, the test would
        not discriminate.
        """
        chart = self._query_select(0.60, _REF_Y1, _REF_Y2)
        self.assertIsNone(
            chart,
            f'gamma=0.60 selected '
            f'{type(chart).__name__ if chart is not None else "None"}, '
            f'not None -- the D₂ consistency test would be vacuous')


# ===========================================================================
# Exterior-polar D₂ delay identity
# ===========================================================================

#: A fixed gamma for delay-identity testing (positive parity, exterior).
_DELAY_GAMMA: float = 0.3

#: Source radius for delay-identity testing (exterior region).
_DELAY_RADIUS: float = 2.0

#: Source angle relative to the shear axis, radians.
_DELAY_THETA: float = 0.6

#: The four sign combinations for the delay-identity test.
_DELAY_SOURCES: list[tuple[float, float]] = [
    (+_DELAY_RADIUS * np.cos(_DELAY_THETA),
     +_DELAY_RADIUS * np.sin(_DELAY_THETA)),
    (-_DELAY_RADIUS * np.cos(_DELAY_THETA),
     +_DELAY_RADIUS * np.sin(_DELAY_THETA)),
    (+_DELAY_RADIUS * np.cos(_DELAY_THETA),
     -_DELAY_RADIUS * np.sin(_DELAY_THETA)),
    (-_DELAY_RADIUS * np.cos(_DELAY_THETA),
     -_DELAY_RADIUS * np.sin(_DELAY_THETA)),
]

#: Absolute tolerance for delay identity.  The Fermat potential has
#: exact D₂ symmetry, so the images and their delays are identical
#: up to machine precision.  We allow 1e-14 -- twice the float64 ULP
#: margin for the quartic root-find and a dot-product accumulation.
_DELAY_TOL: float = 1e-14

#: Tiny w grid for the delay-identity geometry partition queries.
_DELAY_W_GRID: np.ndarray = np.geomspace(4.0, 40.0, 3)


class ExteriorPolarDelayIdentityTestCase(unittest.TestCase):
    """Image delays are D₂-identical at all four (+-y1, +-y2).

    For a fixed gamma and source radius |y|, the D₂ symmetry of the
    Fermat potential forces the image positions -- and therefore the
    per-image delays computed via `geometry.delay` -- to be identical
    across all four sign combinations.  This test queries the geometry
    partition (which resolves the images via the quartic root-find)
    and then independently computes the delay of each image through
    the authoritative `delay` closure.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._geoms: list[object] = []
        cls._n_compared = 0
        for y1, y2 in _DELAY_SOURCES:
            ch = ChangRefsdalChannels(_DELAY_W_GRID)
            geom = ch.geometry_partition(
                gamma=_DELAY_GAMMA, y=(y1, y2), beta=0.0, kappa=0.5)
            cls._geoms.append(geom)

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName == 'test_d2_delay_identity'):
            self.fail('no D₂ delay comparisons executed; test is vacuous')

    def _per_image_delays(self, geom, y1: float, y2: float) -> list[float]:
        """Compute the per-image delay for each image (real + ghost).

        The macro matrix is recomputed from (gamma, kappa=0.5, beta=0.0)
        since `ChangRefsdalGeometryPartition` does not carry ``matrix``
        or ``source``.  ``y1, y2`` is the eigenframe source passed to
        `geometry_partition`.
        """
        source = np.array([float(y1), float(y2)])
        gamma = _DELAY_GAMMA
        kappa = 0.5
        beta = 0.0
        cos2b = float(np.cos(2.0 * beta))
        sin2b = float(np.sin(2.0 * beta))
        lam = 1.0 - kappa
        matrix = np.array([
            [lam - gamma * cos2b, -gamma * sin2b],
            [-gamma * sin2b, lam + gamma * cos2b]])
        return [geometry.delay(img, source, matrix)
                for img in geom.images]

    def test_d2_delay_identity(self) -> None:
        """Delays of ref source match the 3 D₂-equivalent copies."""
        y1_ref, y2_ref = _DELAY_SOURCES[0]
        ref_delays = self._per_image_delays(
            self._geoms[0], y1_ref, y2_ref)
        self.assertGreater(len(ref_delays), 0,
                           'No images found for reference source')

        for i in range(1, len(self._geoms)):
            y1_i, y2_i = _DELAY_SOURCES[i]
            probe_delays = self._per_image_delays(
                self._geoms[i], y1_i, y2_i)
            self.assertEqual(len(probe_delays), len(ref_delays),
                             f'Image count mismatch for D₂ source {i}')
            self._n_compared += 1
            for j, (d_ref, d_probe) in enumerate(
                    zip(ref_delays, probe_delays)):
                self.assertLessEqual(
                    abs(d_probe - d_ref), _DELAY_TOL,
                    f'Delay of image {j} at D₂ source {i} '
                    f'({d_probe:.16e}) != reference ({d_ref:.16e})')


# ===========================================================================
# Lobe-interior D₂ fold fixture constants
# ===========================================================================

#: Gamma grid for the synthetic lobe charts (macro-saddle, gamma > 1).
_LOBE_GAMMA_GRID: np.ndarray = np.array([1.5, 2.0, 2.5, 3.0])

#: Rho_lobe grid (interior region, 0 < rho_lobe < 1).
_LOBE_RHO_GRID: np.ndarray = np.linspace(0.1, 0.9, 5)

#: Theta_local grid (lobe-local polar angle, full circle).
_LOBE_THETA_GRID: np.ndarray = np.linspace(-np.pi + 0.1, np.pi - 0.1, 9)

#: Log-w grid for the synthetic lobe charts.
_LOBE_LOG_W_GRID: np.ndarray = np.log(np.geomspace(1.0, 30.0, 8))

#: Canonical lobe centroid (positive y1 quadrant, lobe 0 chart).
_LOBE_CENTROID_A: np.ndarray = np.array([1.5, 0.0])

#: Other lobe centroid (negative y1 quadrant, lobe B source region).
_LOBE_CENTROID_B: np.ndarray = np.array([-1.5, 0.0])

#: Inter-lobe corridor half-width (dimensionless y units).
_LOBE_CORRIDOR_HALF: float = 0.3

#: Dense angular nodes for the directional lobe boundary.
_LOBE_BOUNDARY_THETA: np.ndarray = np.linspace(-np.pi, np.pi, 361)

#: Constant boundary radius (circular lobe approximation, r = 1.2 y-units).
_LOBE_BOUNDARY_R: np.ndarray = np.full(361, 1.2)

#: Four D₂-equivalent lobe sources for the fold-identity test.
_LOBE_D2_SOURCES: list[tuple[float, float]] = [
    (+1.2, +0.3),
    (+1.2, -0.3),
    (-1.2, +0.3),
    (-1.2, -0.3),
]

#: Corridor test source: on the y-axis (inter-lobe corridor).
_CORRIDOR_Y1: float = 0.0
_CORRIDOR_Y2: float = 0.8

#: Eta for the corridor source (interior eta above the floor).
_CORRIDOR_ETA: float = 0.04

#: Image count for the interior region (macro-saddle = 2 real images).
_CORRIDOR_IMAGE_COUNT: int = 2

#: Clear lobe-B source (y1 < 0, well inside the lobe for lobe 0 after fold).
_CLEAR_LOBE_B_Y1: float = -1.2
_CLEAR_LOBE_B_Y2: float = 0.3

#: Relative tolerance for lobe interior F(w) identity.
#  Relaxed vs 1e-14 for the exterior-polar case: the lobe interior
#  reconstruction introduces a tau_c carrier-phase multiplication and
#  summation that adds one extra FP operation per frequency point.
_LOBE_F_TOL: float = 1e-12

#: Tiny w grid for lobe geometry partition / reconstruction queries.
_LOBE_W_GRID: np.ndarray = np.geomspace(4.0, 20.0, 5)

#: Log-w matching _LOBE_W_GRID (for select_chart / lobe_serves).
_LOBE_LOG_W_GRID_QUERY: np.ndarray = np.log(_LOBE_W_GRID)

#: Eta for the lobe interior (above floor, interior region).
_LOBE_ETA: float = 0.05


# ---------------------------------------------------------------------------
# Lobe chart builder helper
# ---------------------------------------------------------------------------

def _build_lobe_chart(*,
                      centroid: np.ndarray,
                      other_centroid: np.ndarray,
                      corridor_half: float = _LOBE_CORRIDOR_HALF,
                      eta_overlap_min: float = 0.01,
                      ) -> LobeInteriorChart:
    """Build a synthetic LobeInteriorChart with constant envelope = 1.

    A constant unit envelope is a valid B-spline interpolation target;
    the D₂ fold is tested via the coordinate mapping -- any deviation
    would register as a mismatch in the reconstructed F(w).
    """
    shape = (len(_LOBE_LOG_W_GRID), len(_LOBE_GAMMA_GRID),
             len(_LOBE_RHO_GRID), len(_LOBE_THETA_GRID))
    return LobeInteriorChart.from_lobe_values(
        gamma_grid=_LOBE_GAMMA_GRID,
        rho_lobe_grid=_LOBE_RHO_GRID,
        theta_local_grid=_LOBE_THETA_GRID,
        log_w_grid=_LOBE_LOG_W_GRID,
        envelope_real=np.ones(shape),
        envelope_imag=np.zeros(shape),
        image_count=_CORRIDOR_IMAGE_COUNT,
        parity=-1,
        centroid=centroid,
        other_centroid=other_centroid,
        corridor_half=corridor_half,
        boundary_theta=_LOBE_BOUNDARY_THETA,
        boundary_r=_LOBE_BOUNDARY_R,
        eta_overlap_min=eta_overlap_min,
        envelope_definition=INTERIOR_SACR_C)




# ===========================================================================
# Lobe interior D₂ fold -- select_chart consistency fixtures
# ===========================================================================

#: Lobe-B source for the select_chart consistency test.
_SELECT_LOBE_B_Y1: float = _CLEAR_LOBE_B_Y1  # -1.2
_SELECT_LOBE_B_Y2: float = _CLEAR_LOBE_B_Y2  #  0.3

#: Gamma for the lobe select_chart consistency query (macro-saddle).
_SELECT_LOBE_GAMMA: float = _LOBE_GAMMA_GRID[1]  # 2.0

#: D₂-folded coordinates of the same source (|y1|, |y2|).
_SELECT_LOBE_D2_Y1: float = abs(_SELECT_LOBE_B_Y1)  # 1.2
_SELECT_LOBE_D2_Y2: float = abs(_SELECT_LOBE_B_Y2)  # 0.3

#: Out-of-lobe source (y1 > 0, large y2 -- corridor region).
_SELECT_LOBE_OUTSIDE_Y1: float = 0.0
_SELECT_LOBE_OUTSIDE_Y2: float = 2.0

def _lobe_reconstruct_f(
        chart: LobeInteriorChart, y1_eig: float, y2_eig: float,
        gamma: float = _LOBE_GAMMA_GRID[1],
        w: np.ndarray = _LOBE_W_GRID,
        eta: float = _LOBE_ETA
        ) -> np.ndarray:
    """Full F(w) reconstruction through a lobe-interior chart.

    Evaluates the chart envelope at ``(y1_eig, y2_eig)`` and then
    reconstructs F(w) via `reconstruct_from_envelope` with the same
    source's geometry partition.  ``y1_eig, y2_eig`` are passed straight
    to `_evaluate_chart` (which applies the internal abs() fold for
    lobe charts).
    """
    log_w = np.log(np.atleast_1d(w).ravel())
    env = _evaluate_chart(
        chart, gamma=gamma, eta=eta, theta=0.0,
        log_w_query=log_w, y1_eig=y1_eig, y2_eig=y2_eig)
    ch = ChangRefsdalChannels(w)
    geom = ch.geometry_partition(
        gamma=gamma, y=(y1_eig, y2_eig), beta=0.0, kappa=0.5)
    switch = _channel_switch(
        w, geom.delays, geom.real_mask, geom.critical_delay)
    _kernels, total = reconstruct_from_envelope(
        w, env, geom.delays, geom.saddle_kernels, switch,
        geom.critical_delay)
    return total


# ===================================================================
# Lobe-interior test classes
# ===================================================================

class LobeInteriorFD2FoldIdentityTestCase(unittest.TestCase):
    """Lobe interior F(w) is D₂-identical after the coordinate fold.

    Builds a single canonical lobe-interior chart (centroid in +y1).
    The D₂ quad fold (abs(y1), abs(y2)) maps all four sign
    combinations to the same lobe-local coordinates, so the
    reconstructed F(w) must be identical for all four (±y1, ±y2).

    A second lobe chart (centroid in -y1, the unnormalised lobe B)
    is built for the reference assertion: a lobe B source
    (y1 < 0) served through the canonical chart with D₂-folded
    coordinates must produce the same F(w) as an equivalent
    lobe-A source (|y1|, |y2|) at the same chart.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart_a = _build_lobe_chart(
            centroid=_LOBE_CENTROID_A,
            other_centroid=_LOBE_CENTROID_B)
        cls._n_compared = 0

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName
                == 'test_lobe_f_d2_fold_identity'):
            self.fail('no lobe F(w) D₂ comparisons executed; '
                      'test is vacuous')

    def test_lobe_f_d2_fold_identity(self) -> None:
        """F(w) at (±y1, ±y2) is identical after D₂ fold in lobe chart.

        3+ comparisons with expected tolerance of 1e-12,
        budgeted at ~50 ms per comparison (5 geo partitions + 3
        evaluations), for a test total of ~0.2 s.
        """
        w = _LOBE_W_GRID
        f_ref = _lobe_reconstruct_f(
            self.chart_a, _LOBE_D2_SOURCES[0][0], _LOBE_D2_SOURCES[0][1])
        ref_max = float(np.max(np.abs(f_ref)))
        self.assertGreater(ref_max, 0.0, 'reference F(w) is zero')

        for y1, y2 in _LOBE_D2_SOURCES[1:]:
            f_probe = _lobe_reconstruct_f(self.chart_a, y1, y2)
            self._n_compared += 1
            rel_max = float(np.max(np.abs(f_probe - f_ref)) / ref_max)
            self.assertLessEqual(
                rel_max, _LOBE_F_TOL,
                f'F({y1}, {y2}) != F({_LOBE_D2_SOURCES[0]}); '
                f'max|Δ|/max|F| = {rel_max:.2e} > {_LOBE_F_TOL}')


class LobeInteriorCorridorNonDegenerateTestCase(unittest.TestCase):
    """Lobe D₂ fold corridor gate is non-degenerate.

    The inter-lobe corridor test in `_lobe_serves` must:
    (a) DECLINE for a source on the inter-lobe corridor (y1 ≈ 0)
        for ALL lobe charts, and
    (b) ACCEPT for a source clearly inside a lobe (y1 < 0) served
        through the canonical lobe-0 chart.

    These boolean assertions verify the gate has teeth: it must not
    collapse to always-true or always-false after the D₂ fold.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart_0 = _build_lobe_chart(
            centroid=_LOBE_CENTROID_A,
            other_centroid=_LOBE_CENTROID_B)
        cls.chart_b = _build_lobe_chart(
            centroid=_LOBE_CENTROID_B,
            other_centroid=_LOBE_CENTROID_A)
        cls._charts = [cls.chart_0, cls.chart_b]
        cls._n_compared = 0

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName == 'test_corridor_source_declined'):
            self.fail('no lobe corridor comparisons executed; '
                      'test is vacuous')

    def test_corridor_source_declined(self) -> None:
        """A corridor source (y1=0, y2=0.8) is declined by ALL lobe charts.

        2 lobe chart comparisons, budgeted at ~1 ms each (< 1 s total).
        """
        log_w_min = float(_LOBE_LOG_W_GRID_QUERY.min())
        log_w_max = float(_LOBE_LOG_W_GRID_QUERY.max())
        for chart in self._charts:
            self._n_compared += 1
            served = _lobe_serves(
                chart, gamma=_LOBE_GAMMA_GRID[1],
                log_w_min=log_w_min, log_w_max=log_w_max,
                eta=_CORRIDOR_ETA, image_count=_CORRIDOR_IMAGE_COUNT,
                y1_eig=_CORRIDOR_Y1, y2_eig=_CORRIDOR_Y2)
            chart_name = 'lobe_0' if chart is self.chart_0 else 'lobe_B'
            self.assertFalse(
                served,
                f'{chart_name} chart served corridor source ({_CORRIDOR_Y1}, '
                f'{_CORRIDOR_Y2}) -- corridor gate has no teeth')

    def test_lobe_b_source_served_by_lobe_0(self) -> None:
        """A lobe-B source (y1 < 0) is served by the lobe-0 chart.

        After D₂ fold, lobe-0 chart's corridor test places the folded
        (|y1|, |y2|) closer to centroid_A (+1.5, 0) than to
        other_centroid (-1.5, 0), so the gate admits it.
        1 comparison, budgeted at < 10 ms.
        """
        log_w_min = float(_LOBE_LOG_W_GRID_QUERY.min())
        log_w_max = float(_LOBE_LOG_W_GRID_QUERY.max())
        served = _lobe_serves(
            self.chart_0, gamma=_LOBE_GAMMA_GRID[1],
            log_w_min=log_w_min, log_w_max=log_w_max,
            eta=_CORRIDOR_ETA, image_count=_CORRIDOR_IMAGE_COUNT,
            y1_eig=_CLEAR_LOBE_B_Y1, y2_eig=_CLEAR_LOBE_B_Y2)
        self.assertTrue(
            served,
            f'lobe_0 chart did NOT serve lobe-B source ({_CLEAR_LOBE_B_Y1}, '
            f'{_CLEAR_LOBE_B_Y2}) after D₂ fold')

    def test_lobe_b_source_declined_by_lobe_b_chart(self) -> None:
        """Lobe-B chart's own gate DECLINES a lobe-B source after fold.

        After D₂-fold to (|y1|, |y2|), the corridor test for the
        -y1-centroid chart finds the source closer to the OTHER
        centroid (+y1), so it declines.  This proves that the
        fold genuinely assigns all four quadrants to the canonical
        lobe-0 chart -- lobe B's chart is not a parallel path.
        """
        log_w_min = float(_LOBE_LOG_W_GRID_QUERY.min())
        log_w_max = float(_LOBE_LOG_W_GRID_QUERY.max())
        served = _lobe_serves(
            self.chart_b, gamma=_LOBE_GAMMA_GRID[1],
            log_w_min=log_w_min, log_w_max=log_w_max,
            eta=_CORRIDOR_ETA, image_count=_CORRIDOR_IMAGE_COUNT,
            y1_eig=_CLEAR_LOBE_B_Y1, y2_eig=_CLEAR_LOBE_B_Y2)
        self.assertFalse(
            served,
            f'lobe_B chart served a lobe-B source ({_CLEAR_LOBE_B_Y1}, '
            f'{_CLEAR_LOBE_B_Y2}) after D₂ fold -- the fold should '
            f'assign all quadrants to the canonical lobe-0 chart')



class LobeSelectChartD2ConsistencyTestCase(unittest.TestCase):
    """`select_chart` returns the same chart for lobe-B and D₂-folded sources.

    For a macro-saddle surrogate, the D₂ coordinate fold (abs(y1), abs(y2))
    inside `_lobe_serves` must produce the same chart selection at
    all four sign combinations.  This test verifies that both the
    lobe-B source (y1<0) and its D₂-folded equivalent (|y1|,|y2|)
    pass through `select_chart` and return the same chart object.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_lobe_chart(
            centroid=_LOBE_CENTROID_A,
            other_centroid=_LOBE_CENTROID_B)
        cls.surrogate = LensAmplificationSurrogate(
            [cls.chart], provenance={})
        cls._log_w_min = float(_LOBE_LOG_W_GRID_QUERY.min())
        cls._log_w_max = float(_LOBE_LOG_W_GRID_QUERY.max())
        cls._n_compared = 0

    def tearDown(self) -> None:
        if (self._n_compared == 0
                and self._testMethodName
                == 'test_select_chart_lobe_b_returns_chart'):
            self.fail('no select_chart lobe D₂ comparisons executed; '
                      'test is vacuous')

    def _query_select(self, gamma, y1_eig, y2_eig):
        """Return the chart selected by `select_chart` for one source."""
        ch = ChangRefsdalChannels(_LOBE_W_GRID)
        geom = ch.geometry_partition(
            gamma=gamma, y=(y1_eig, y2_eig), beta=0.0, kappa=0.5)
        return select_chart(
            self.surrogate.charts, gamma=gamma,
            log_w_min=self._log_w_min, log_w_max=self._log_w_max,
            eta=geom.caustic_distance, theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()),
            y1_eig=y1_eig, y2_eig=y2_eig)

    def test_select_chart_lobe_b_returns_chart(self) -> None:
        """A lobe-B source (y1<0) selects the lobe chart after D₂ fold.

        1 comparison, budgeted at < 50 ms.
        """
        chart = self._query_select(
            _SELECT_LOBE_GAMMA, _SELECT_LOBE_B_Y1, _SELECT_LOBE_B_Y2)
        self._n_compared += 1
        self.assertIsNotNone(
            chart,
            f'select_chart returned None for lobe-B source '
            f'({_SELECT_LOBE_B_Y1}, {_SELECT_LOBE_B_Y2})')
        self.assertIsInstance(
            chart, LobeInteriorChart,
            f'select_chart returned {type(chart).__name__}, not '
            f'LobeInteriorChart for lobe-B source')

    def test_select_chart_d2_folded_returns_chart(self) -> None:
        """The D₂-folded source (|y1|,|y2|) selects the SAME chart object.

        1 comparison, budgeted at < 50 ms.
        3+2 comparisons total, budget at ~0.3 s for the class.
        """
        chart_lobe_b = self._query_select(
            _SELECT_LOBE_GAMMA, _SELECT_LOBE_B_Y1, _SELECT_LOBE_B_Y2)
        chart_d2 = self._query_select(
            _SELECT_LOBE_GAMMA, _SELECT_LOBE_D2_Y1, _SELECT_LOBE_D2_Y2)
        self.assertIsNotNone(chart_d2)
        self.assertIsInstance(chart_d2, LobeInteriorChart)
        self.assertIs(
            chart_d2, chart_lobe_b,
            f'D₂-folded source ({_SELECT_LOBE_D2_Y1}, {_SELECT_LOBE_D2_Y2}) '
            f'selected a DIFFERENT chart than lobe-B source '
            f'({_SELECT_LOBE_B_Y1}, {_SELECT_LOBE_B_Y2})')

    def test_outside_lobe_returns_none(self) -> None:
        """A source outside all lobe charts returns None -- proves teeth.

        If select_chart always returned the same chart regardless of
        source position, the consistency test above would be vacuous.
        This test independently verifies that a source in the corridor
        region (y1=0, |y2|=2.0) is declined.
        """
        chart = self._query_select(
            _SELECT_LOBE_GAMMA,
            _SELECT_LOBE_OUTSIDE_Y1, _SELECT_LOBE_OUTSIDE_Y2)
        self.assertIsNone(
            chart,
            f'select_chart served source ({_SELECT_LOBE_OUTSIDE_Y1}, '
            f'{_SELECT_LOBE_OUTSIDE_Y2}), not None -- the D₂ '
            f'consistency test would be vacuous')



class D2FoldRegressionTestCase(unittest.TestCase):
    """Tube and wedge chart test suites are unaffected by the D₂ fold.

    The D₂ fold changes only the exterior-polar and lobe serve paths
    through `_to_exterior_fixed` and `_lobe_serves`.  Tube charts
    (`_tube_serves`) and wedge charts (`_wedge_serves`) do NOT take
    ``y1_eig, y2_eig`` arguments -- their serve paths are structurally
    unchanged.  This test runs the existing tube and wedge test suites
    to verify zero regressions.

    Budget: ~45 s combined (tube surrogate 62 tests, ~30 s; wedge
    chart 40 tests, ~15 s).  Within the 60 s per-test ceiling.
    """

    def test_tube_surrogate_suite_passes(self) -> None:
        """All tube-surrogate tests pass (exterior-polar gate OFF).

        Runs `cogwheel/tests/test_lensing_surrogate.py` via subprocess.
        """
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-m', 'pytest',
             'cogwheel/tests/test_lensing_surrogate.py',
             '-q', '--tb=short'],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(
            result.returncode, 0,
            f'Tube surrogate suite FAILED:\n'
            f'STDOUT:\n{result.stdout[-2000:]}\n'
            f'STDERR:\n{result.stderr[-2000:]}')

    def test_wedge_chart_suite_passes(self) -> None:
        """All wedge-chart tests pass (interior wedge gate OFF).

        Runs `cogwheel/tests/test_lensing_interior_wedge_chart.py`
        via subprocess.
        """
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-m', 'pytest',
             'cogwheel/tests/test_lensing_interior_wedge_chart.py',
             '-q', '--tb=short'],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(
            result.returncode, 0,
            f'Wedge chart suite FAILED:\n'
            f'STDOUT:\n{result.stdout[-2000:]}\n'
            f'STDERR:\n{result.stderr[-2000:]}')

# ===========================================================================
# Exterior-polar Cusp-Adapted Axis — fixtures
# ===========================================================================

#: Tile bounds for a low-origin (theta_c near 0 cusp) chart.
_CUSP_LOW_THETA_LO: float = 0.02
_CUSP_LOW_THETA_HI: float = 0.55

#: Tile bounds for a high-origin (theta_c near pi/2 cusp) chart.
_CUSP_HIGH_THETA_LO: float = 1.02  # pi/2 − 0.55
_CUSP_HIGH_THETA_HI: float = np.pi / 2.0 - 0.02

#: Gamma probe value for cusp-adapted axis queries.
_CUSP_GAMMA: float = 0.30

#: Rho mid-tile probe coordinate.
_CUSP_RHO: float = 3.0

#: Theta-c grid size for the cusp-adapted test charts.
#  Odd count preserves an exact symmetry point.
_CUSP_N_THETA_C: int = 7

#: Log-w query band (scalar band, one log-w point).
_CUSP_LOG_W: np.ndarray = np.log(np.array([10.0]))

#: Eta for the exterior query (must be > caustic floor).
_CUSP_ETA: float = 0.1

#: Tolerance for B-spline node-exact reproduction.
#  Budget: ~6e-9 interp error at 2001 nodes (mem:test_dev_knowledge).
_CUSP_NODE_EXACT_TOL: float = 1e-7

#: Tolerance for off-grid interp vs exact u(theta_c).
#  Relaxed: the spline axis is u, not theta_c; the query theta_c
#  maps to a u-coordinate that may be between spline nodes.
_CUSP_OFFGRID_TOL: float = 2e-2


def _angular_envelope(log_w_size: int, gamma_size: int,
                      rho_size: int, angular_size: int,
                      angular_values: np.ndarray) -> np.ndarray:
    """Create a (n_w, n_gamma, n_rho, n_theta_c) envelope
    varying only along the angular (4th) axis."""
    return np.broadcast_to(
        angular_values.reshape(1, 1, 1, -1),
        (log_w_size, gamma_size, rho_size, angular_size)).copy()


def _eigenframe_source(gamma: float, rho: float,
                       theta_c: float) -> tuple[float, float]:
    """Convenience wrapper: (y1_eig, y2_eig) for a caustic-fixed point."""
    return _from_caustic_fixed(gamma, rho, theta_c)


# ===========================================================================
# Exterior-polar Cusp-Adapted Axis — test classes
# ===========================================================================

class ExteriorPolarCuspAdaptedAxisTestCase(unittest.TestCase):
    """ExteriorPolarChart ``theta_to_u`` remap resolves correctly.

    Builds two synthetic exterior-polar charts with a cusp-adapted
    ``u = d**(2/3)`` angular axis via ``_wedge_cusp_axis_map``:
    one nearest the ``theta_c = 0`` cusp (origin ``'low'``) and one
    nearest the ``theta_c = π/2`` cusp (origin ``'high'``).

    The envelope is set to the ``u_grid`` ordinate values, so the
    served envelope magnitude carries the angular-coordinate
    information.  A correct theta_to_u remap means: querying at a
    theta_c value returns an envelope ≈ u(theta_c), where u is the
    cusp-adapted coordinate computed via np.interp.
    """

    _low_chart: ExteriorPolarChart | None = None
    _high_chart: ExteriorPolarChart | None = None
    _low_theta_map: np.ndarray | None = None  # (2, N) theta_to_u
    _high_theta_map: np.ndarray | None = None
    _n_compared: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        # --- low-origin chart (theta_c near 0) ---
        theta_lo_low = _CUSP_LOW_THETA_LO
        theta_hi_low = _CUSP_LOW_THETA_HI
        theta_c_grid_low = np.linspace(theta_lo_low, theta_hi_low,
                                       _CUSP_N_THETA_C)
        theta_fine_low, u_fine_low = _wedge_cusp_axis_map(
            theta_lo_low, theta_hi_low, 'low')
        theta_to_u_low = np.vstack([theta_fine_low, u_fine_low])
        u_grid_low = np.interp(theta_c_grid_low, theta_fine_low, u_fine_low)

        ni = len(_CHART_LOG_W_GRID)
        ng = len(_CHART_GAMMA_GRID)
        nr = len(_CHART_RHO_GRID)
        nc = _CUSP_N_THETA_C
        env_low = _angular_envelope(ni, ng, nr, nc, u_grid_low)

        cls._low_chart = _build_chart(
            theta_c_grid=theta_c_grid_low,
            theta_to_u=theta_to_u_low,
            u_grid=u_grid_low,
            envelope_real=env_low)
        cls._low_theta_map = theta_to_u_low

        # --- high-origin chart (theta_c near pi/2) ---
        theta_lo_high = _CUSP_HIGH_THETA_LO
        theta_hi_high = _CUSP_HIGH_THETA_HI
        theta_c_grid_high = np.linspace(theta_lo_high, theta_hi_high,
                                        _CUSP_N_THETA_C)
        theta_fine_high, u_fine_high = _wedge_cusp_axis_map(
            theta_lo_high, theta_hi_high, 'high')
        theta_to_u_high = np.vstack([theta_fine_high, u_fine_high])
        u_grid_high = np.interp(theta_c_grid_high,
                                theta_fine_high, u_fine_high)

        env_high = _angular_envelope(ni, ng, nr, nc, u_grid_high)

        cls._high_chart = _build_chart(
            theta_c_grid=theta_c_grid_high,
            theta_to_u=theta_to_u_high,
            u_grid=u_grid_high,
            envelope_real=env_high)
        cls._high_theta_map = theta_to_u_high

    def tearDown(self) -> None:
        if self._n_compared == 0:
            self.fail(f'no comparisons executed in {self._testMethodName}; '
                      f'test is vacuous')

    def _evaluate_at_theta_c(self, chart: ExteriorPolarChart,
                             theta_c: float) -> float:
        """Evaluate the chart envelope at one (rho, theta_c)."""
        y1, y2 = _eigenframe_source(_CUSP_GAMMA, _CUSP_RHO, theta_c)
        env = _evaluate_chart(chart, gamma=_CUSP_GAMMA, eta=_CUSP_ETA,
                              theta=0.0, log_w_query=_CUSP_LOG_W,
                              y1_eig=y1, y2_eig=y2)
        return float(env[0].real)

    def test_low_cusp_on_grid_nodes_reproduce(self) -> None:
        """On-grid theta_c nodes reproduce u_grid values to ~1e-7.

        7 queries, budgeted at ~1 ms each (< 1 s total).
        """
        chart = self._low_chart
        theta_c_grid = chart.theta_c_grid
        for tc in theta_c_grid:
            with self.subTest(theta_c=tc):
                served = self._evaluate_at_theta_c(chart, tc)
                expected_u = float(np.interp(
                    tc, self._low_theta_map[0], self._low_theta_map[1]))
                self._n_compared += 1
                self.assertLessEqual(
                    abs(served - expected_u), _CUSP_NODE_EXACT_TOL,
                    f'served={served:.6e} != expected_u={expected_u:.6e} '
                    f'at theta_c={tc:.4f}')

    def test_low_cusp_off_grid_remaps_correctly(self) -> None:
        """Off-grid theta_c queries map correctly via np.interp.

        3 comparisons, budgeted at ~1 ms each (< 1 s total).
        """
        chart = self._low_chart
        lo, hi = chart.theta_c_grid[0], chart.theta_c_grid[-1]
        test_tc = np.linspace(lo * 1.3, hi * 0.7, 3)
        for tc in test_tc:
            with self.subTest(theta_c=tc):
                served = self._evaluate_at_theta_c(chart, tc)
                expected_u = float(np.interp(
                    tc, self._low_theta_map[0], self._low_theta_map[1]))
                self._n_compared += 1
                self.assertLessEqual(
                    abs(served - expected_u), _CUSP_OFFGRID_TOL,
                    f'served={served:.6e} != expected_u={expected_u:.6e} '
                    f'at theta_c={tc:.4f}')

    def test_high_cusp_on_grid_nodes_reproduce(self) -> None:
        """On-grid theta_c nodes reproduce u_grid values to ~1e-7
        for the high-origin (theta_c near pi/2) tile.

        7 queries, budgeted at ~1 ms each (< 1 s total).
        """
        chart = self._high_chart
        theta_c_grid = chart.theta_c_grid
        for tc in theta_c_grid:
            with self.subTest(theta_c=tc):
                served = self._evaluate_at_theta_c(chart, tc)
                expected_u = float(np.interp(
                    tc, self._high_theta_map[0], self._high_theta_map[1]))
                self._n_compared += 1
                self.assertLessEqual(
                    abs(served - expected_u), _CUSP_NODE_EXACT_TOL,
                    f'served={served:.6e} != expected_u={expected_u:.6e} '
                    f'at theta_c={tc:.4f}')

    def test_high_cusp_off_grid_remaps_correctly(self) -> None:
        """Off-grid theta_c queries map correctly via np.interp
        for the high-origin (theta_c near pi/2) tile.

        3 comparisons, budgeted at ~1 ms each (< 1 s total).
        """
        chart = self._high_chart
        lo, hi = chart.theta_c_grid[0], chart.theta_c_grid[-1]
        test_tc = np.linspace(lo * 1.05, hi * 0.95, 3)
        for tc in test_tc:
            with self.subTest(theta_c=tc):
                served = self._evaluate_at_theta_c(chart, tc)
                expected_u = float(np.interp(
                    tc, self._high_theta_map[0], self._high_theta_map[1]))
                self._n_compared += 1
                self.assertLessEqual(
                    abs(served - expected_u), _CUSP_OFFGRID_TOL,
                    f'served={served:.6e} != expected_u={expected_u:.6e} '
                    f'at theta_c={tc:.4f}')

    def test_theta_to_u_is_strictly_increasing(self) -> None:
        """Both rows of theta_to_u are strictly increasing."""
        for label, ttu in [('low', self._low_theta_map),
                           ('high', self._high_theta_map)]:
            with self.subTest(origin=label):
                self._n_compared += 1
                self.assertTrue(np.all(np.diff(ttu[0]) > 0),
                                f'{label}: theta row not increasing')
                self.assertTrue(np.all(np.diff(ttu[1]) > 0),
                                f'{label}: u row not increasing')

    def test_theta_to_u_starts_at_zero(self) -> None:
        """u_row[0] ≈ 0 for both origins."""
        for label, ttu in [('low', self._low_theta_map),
                           ('high', self._high_theta_map)]:
            with self.subTest(origin=label):
                self._n_compared += 1
                self.assertLessEqual(
                    abs(float(ttu[1, 0])), 1e-15,
                    f'{label}: u_row[0]={ttu[1,0]:.2e} != 0')

    def test_theta_row_matches_chart_tile_bounds(self) -> None:
        """theta_to_u[0] starts at the chart's theta_c_grid[0]."""
        for label, chart, ttu in [
                ('low', self._low_chart, self._low_theta_map),
                ('high', self._high_chart, self._high_theta_map)]:
            with self.subTest(origin=label):
                self._n_compared += 1
                self.assertEqual(float(ttu[0, 0]),
                                 float(chart.theta_c_grid[0]))


# ===========================================================================
# Self-falsification: flat theta_to_u
# ===========================================================================

class ExteriorPolarCuspAdaptedSelfFalsification(unittest.TestCase):
    """Replacing theta_to_u with a flat map degrades served values.

    If ``theta_to_u`` maps all theta_c to the same u value,
    the served envelope loses all angular-coordinate dependence.
    This proves the cusp-adapted axis is load-bearing: a neutered
    map gives a vacuously constant output regardless of theta_c.
    """

    _chart: ExteriorPolarChart | None = None
    _theta_to_u: np.ndarray | None = None
    _n_compared: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        theta_lo = _CUSP_LOW_THETA_LO
        theta_hi = _CUSP_LOW_THETA_HI
        theta_c_grid = np.linspace(theta_lo, theta_hi, _CUSP_N_THETA_C)
        theta_fine, u_fine = _wedge_cusp_axis_map(
            theta_lo, theta_hi, 'low')
        theta_to_u = np.vstack([theta_fine, u_fine])
        u_grid = np.interp(theta_c_grid, theta_fine, u_fine)

        ni = len(_CHART_LOG_W_GRID)
        ng = len(_CHART_GAMMA_GRID)
        nr = len(_CHART_RHO_GRID)
        nc = _CUSP_N_THETA_C
        env = _angular_envelope(ni, ng, nr, nc, u_grid)

        cls._chart = _build_chart(
            theta_c_grid=theta_c_grid,
            theta_to_u=theta_to_u,
            u_grid=u_grid,
            envelope_real=env)
        cls._theta_to_u = theta_to_u

    def tearDown(self) -> None:
        if self._n_compared == 0:
            self.fail(f'no comparisons executed in {self._testMethodName}; '
                      f'test is vacuous')

    def _eval(self, chart: ExteriorPolarChart, theta_c: float) -> float:
        y1, y2 = _eigenframe_source(_CUSP_GAMMA, _CUSP_RHO, theta_c)
        env = _evaluate_chart(chart, gamma=_CUSP_GAMMA, eta=_CUSP_ETA,
                              theta=0.0, log_w_query=_CUSP_LOG_W,
                              y1_eig=y1, y2_eig=y2)
        return float(env[0].real)

    def test_correct_map_spread_exceeds_flat_map_spread(self) -> None:
        """The angular spread with the correct map >> flat-map spread.

        Query two theta_c values far apart within the tile.  With the
        correct theta_to_u the two queries resolve to different u
        values → different envelope magnitudes.  With a flat map both
        queries resolve to the same u → nearly identical envelopes.

        2+2 comparisons, budgeted at ~1 ms each (< 1 s total).
        """
        chart = self._chart
        tc_lo = float(chart.theta_c_grid[1])   # near low edge
        tc_hi = float(chart.theta_c_grid[-2])  # near high edge

        spread_correct = abs(self._eval(chart, tc_hi)
                             - self._eval(chart, tc_lo))

        # Build a flat theta_to_u: all theta -> u_min.
        flat_map = chart.theta_to_u.copy()
        flat_map[1, :] = flat_map[1, 0]  # all u = u_min

        # Mutate the frozen dataclass to inject the flat map.
        object.__setattr__(chart, 'theta_to_u', flat_map)

        spread_flat = abs(self._eval(chart, tc_hi)
                          - self._eval(chart, tc_lo))

        # Restore correct map for any subsequent tests.
        object.__setattr__(chart, 'theta_to_u', self._theta_to_u)

        self._n_compared = 2
        self.assertGreater(
            spread_correct, 0.01,
            f'correct map spread = {spread_correct:.6e}; expected > 0.01. '
            f'The test fixture does not exercise the angular axis.')
        self.assertLess(
            spread_flat, 0.01,
            f'flat map spread = {spread_flat:.6e}; expected < 0.01. '
            f'A flat theta_to_u should collapse the angular axis.')
        self.assertGreater(
            spread_correct / max(spread_flat, 1e-30), 10.0,
            f'correct/flat spread ratio = '
            f'{spread_correct / max(spread_flat, 1e-30):.1f}; '
            f'expected > 10.  The theta_to_u remap is not load-bearing.')


# ===========================================================================
# Self-falsification: the cusp-adapted suite goes red when it should
# ===========================================================================

class CuspAdaptedAxisSelfFalsification(unittest.TestCase):
    """Prove ``ExteriorPolarCuspAdaptedAxisTestCase`` has teeth.

    If the correct-map tests could pass vacuously (e.g. envelope
    constant), this class demonstrates that flipping theta_to_u
    rows MAKES the suite go red — the assertions above are not
    trivially satisfied.
    """

    def test_flipped_theta_to_u_breaks_node_exact(self) -> None:
        """With u-row replaced by theta-row, on-grid assertion breaks.

        Swapping the u-row for the theta-row yields an envelope
        that varies as theta_c (not u), so the served value at a
        theta_c node no longer matches the expected u(theta_c).
        """
        theta_lo = _CUSP_LOW_THETA_LO
        theta_hi = _CUSP_LOW_THETA_HI
        theta_c_grid = np.linspace(theta_lo, theta_hi, _CUSP_N_THETA_C)
        theta_fine, u_fine = _wedge_cusp_axis_map(theta_lo, theta_hi, 'low')
        theta_to_u = np.vstack([theta_fine, u_fine])
        u_grid = np.interp(theta_c_grid, theta_fine, u_fine)

        ni = len(_CHART_LOG_W_GRID)
        ng = len(_CHART_GAMMA_GRID)
        nr = len(_CHART_RHO_GRID)
        nc = _CUSP_N_THETA_C
        env = _angular_envelope(ni, ng, nr, nc, u_grid)

        chart = _build_chart(
            theta_c_grid=theta_c_grid,
            theta_to_u=theta_to_u,
            u_grid=u_grid,
            envelope_real=env)

        # Replace u-row with theta-row: now u ≈ theta_c (wrong).
        bad_map = theta_to_u.copy()
        bad_map[1] = bad_map[0].copy()
        object.__setattr__(chart, 'theta_to_u', bad_map)

        # Pick a mid-grid node where theta_c ≠ u(theta_c).
        tc = float(theta_c_grid[_CUSP_N_THETA_C // 2])
        y1, y2 = _eigenframe_source(_CUSP_GAMMA, _CUSP_RHO, tc)
        env_val = _evaluate_chart(chart, gamma=_CUSP_GAMMA, eta=_CUSP_ETA,
                                  theta=0.0, log_w_query=_CUSP_LOG_W,
                                  y1_eig=y1, y2_eig=y2)
        served = float(env_val[0].real)
        expected_u = float(np.interp(tc, theta_fine, u_fine))

        # With correct map: served ≈ expected_u (within tolerance).
        # With flipped map (u ← theta): served ≈ tc, which is NOT ≈
        # expected_u unless the power law happens to agree (it doesn't
        # for this tile: tc=0.285 whereas u(tc)=0.285**(2/3)≈0.35).
        err = abs(served - expected_u)
        self.assertGreater(
            err, _CUSP_NODE_EXACT_TOL,
            f'flipped theta_to_u error = {err:.6e} ≤ '
            f'{_CUSP_NODE_EXACT_TOL}; the suite would pass vacuously.')


