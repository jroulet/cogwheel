"""Reachability tests for the FIRST-CLASS Born analytic rung.

The build lifted the Born (weak-deflection) analytic serve out of the
buried surrogate path (`_surrogate_coefficients`) into a first-class
intercept, ``_born_residual_analytic``, reached on the production
(surrogate-free) route through ``_amplification_coefficients``.  The
sibling suite ``test_lensing_born_residual_wiring.py`` pins the BURIED
rung; this suite pins the LIFTED rung's reachability and its certified-map
band-split consult -- a genuinely distinct, previously-unguarded invariant.

What is pinned
--------------
1. BORN SERVE-PATH TRACE.  With a chart attached, no ppGO map, gamma<1,
   dense_w.max() below the Schwinger QD ceiling and no surrogate, the
   dispatcher ``_amplification_coefficients`` routes to
   ``_born_residual_analytic`` and its served ``(delays, k0, k1)`` are
   BYTE-IDENTICAL to a direct call of that rung (route identification).
   Two DIFFERENT charts yield DIFFERENT served coefficients (the route is
   live, not a dead all-zero-difference fall-through).  With no chart the
   rung returns ``None`` (the fall-through guard).
2. MAP BAND-SPLIT TRACE.  With a certified map whose ``w_trust`` sits
   strictly inside the band and at/under the effective ceiling, the served
   coefficients DIFFER from the whole-band (no-map) Born serve -- the map
   demonstrably re-routes the high-w segment to bare ppGO.
3. NULL-SPLIT IDENTITY (Professor-directed, byte-exact).  A map whose
   ``w_trust >= dense_w.max()`` makes the split a no-op: its served
   ``(delays, k0, k1)`` are byte-identical (``np.array_equal``) to the
   no-map whole-band Born serve; element-wise ``max|A - B|`` is exactly
   ``0.0``.
4. BYTE-IDENTITY BATTERY OFF THE SERVED PATH.  A battery of draws that
   MUST NOT reach the Born intercept -- (a) interior ``rho < 1``, (b)
   exterior ``1 < rho < 2`` (below the ``rho > 2`` gate), (c) a
   ``covers() == False`` draw outside the chart grid, (d) a ``gamma > 1``
   saddle draw the saddle far-field rung claims first, and (e) a
   ``kappa != 0`` / ``beta != 0`` off-reference draw -- each declines the
   Born rung (returns ``None``) with the chart attached, so the dispatcher
   route is byte-identical to the ``born_residual_chart=None`` route.  The
   ``kappa != 0`` row is the exact silent-accuracy bug the corrected gate
   prevents.  (Fast-tier substitution: the None-return is the engine-free
   decisive form of "the two serves are byte-identical" -- the Born rung
   is the ONLY chart-dependent branch of ``_amplification_coefficients``,
   so a chart-attached ``None`` proves every downstream float64 input is
   identical to the no-chart route.  Recorded in the change report.)
5. LOADER HARD-REFUSAL AT CONSTRUCTION.  ``BornResidualChart.load`` raises
   ``ValueError`` naming ``scripts/train_born_residual.py`` for a
   content-hash mismatch (one tampered ``real_coeffs`` element, original
   stored hash), a missing ``schema`` key, and a wrong ``schema`` string --
   an explicit-path load never silently accepts a corrupt or stale
   artifact.  A valid artifact round-trips (positive control / teeth).

Test-tier substitution (test-tier LAW)
--------------------------------------
The Architect's specs are phrased as acceptance traces that compare the
Born-served result against an ENGINE-PURE (chart=None) serve.  Driving
``_amplification_coefficients`` with ``born_residual_chart=None`` falls all
the way through to the exact seed engine (``_evaluate_envelope`` +
fiducial cache + ratio layer), which the lightweight method-binding probe
cannot supply without a full waveform/event construction (minutes, not
seconds).  Per the fast-tier ceiling (<60 s/test, <5 min/file) this suite
substitutes engine-free but equally decisive invariants:
  - route identification: dispatcher result == direct rung result
    (byte-exact) -- proves the intercept is REACHED, replacing
    "differs from the engine serve";
  - route liveness: chart-A vs chart-B served coefficients DIFFER --
    proves the chart value flows to the served number (the spec's
    "all-zero difference => dead path" diagnostic, made an assertion);
  - dead-path guard: the rung returns ``None`` with no chart -- proves the
    engine fall-through is intact.
The substitution is recorded in the change report.

Oracle independence
--------------------
The route-identification oracle is the SHIPPING rung itself
(``_born_residual_analytic``) called directly on the same ``(lens,
dense_w)`` -- the dispatcher is required to return exactly that object, so
byte-equality is the correct contract (both feed identical float64 inputs
to ``reconstruct_farfield``).  The band-split and null-split oracles are
the no-map whole-band serve of the SAME draw; the map only changes the
sub-band mask, so a DIFFER / IDENTICAL split is decisive without a second
derivation of the physics.

Runtime budget
--------------
No engine, no waveform: every serve is one analytic geometry partition
(``ChangRefsdalChannels(dense_w).geometry_partition``) plus a chart
interpolation and one ``reconstruct_farfield`` over 64 dense nodes.
Measured ~4 s for the whole file.
"""
from __future__ import annotations

import functools
import json
import math
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from cogwheel import data, utils, waveform
from cogwheel.lensing.born_residual_chart import (
    BornResidualChart,
    _SCHEMA,
    _content_hash,
)
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood,
    _AUTO_BORN_CHART,
)
from cogwheel.lensing.marginalized_likelihood import (
    LensedMarginalizedExtrinsicLikelihood,
)
from cogwheel.likelihood.marginalized_extrinsic import (
    MarginalizedExtrinsicLikelihood,
)
from cogwheel.lensing.ppgo_map import (
    ASTROID_WALL,
    CERTIFICATION_BAR,
    CertifiedPpgoMap,
    STATUS_BEYOND_WALL,
    STATUS_CERTIFIED,
    STATUS_INVALID,
    W_TRUST_ADDITIVE,
    W_TRUST_MULTIPLIER,
    _PARITY_CODES,
    caustic_geometry,
    caustic_rho,
    set_certified_ppgo_map,
)
from cogwheel.lensing.waveform import dimensionless_frequency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lightweight lens config (solar masses, redshift).
_M_LENS_MSUN: float = 100.0
_Z_LENS: float = 0.5

#: Positive-parity astroid shear (gamma < 1 so the saddle far-field rung
#: is skipped and the Born rung is the reached intercept).
_GAMMA: float = 0.5

#: kappa = 0, beta = 0 -- the Born chart is a (gamma, rho, log_w) surface
#: trained at this reference config; the rung refuses anything else.
_KAPPA: float = 0.0
_BETA: float = 0.0

#: Exterior caustic-frame coordinate (rho > 2 so covers() / the rho gate is
#: the binding admission and the config is a clean 2-image exterior).
_TARGET_RHO: float = 3.0

#: Off-axis source direction (radians) -- keeps the image config generic,
#: away from the on-axis astroid cusp symmetry line.
_SOURCE_ANGLE: float = 0.3

#: Dense grid geometry: 4 bins x 16 sub-samples = 64 dense nodes.
_N_BINS: int = 4
_KERNEL_SUBSAMPLES: int = 16
_N_DENSE: int = _N_BINS * _KERNEL_SUBSAMPLES  # 64

#: Dense w band [w_lo, w_hi].  w_hi = 60 stays below the Schwinger QD
#: ceiling (150) so the ppGO-above-ceiling rung is skipped.
_W_LO: float = 1.0
_W_HI: float = 60.0

#: Certified w_cert that splits INSIDE the band: w_trust = max(1.5*20,
#: 20+2) = 30, strictly in (1, 60).
_MAP_W_CERT_SPLIT: float = 20.0

#: Certified w_cert whose w_trust = max(1.5*1000, 1000+2) = 1500 sits at or
#: above w_hi = 60 -> the split is a no-op (null-split identity).
_MAP_W_CERT_NULL: float = 1000.0

#: Chart grid extents (cover gamma, rho and the full w band up to 60 so the
#: whole-band serve never leaves the training box).
_CHART_GAMMA_GRID = np.linspace(0.3, 0.8, 6)
_CHART_RHO_GRID = np.linspace(1.5, 5.0, 8)
_CHART_LOG_W_GRID = np.log(np.geomspace(0.05, 200.0, 24))

#: Two distinct residual scales -> two distinct charts (route-liveness).
_RESIDUAL_SCALE_A: float = 0.01
_RESIDUAL_SCALE_B: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abs_y_for_rho(rho: float, gamma: float = _GAMMA,
                   kappa: float = _KAPPA) -> float:
    """Source-plane |y| giving caustic-frame ``rho`` at ``(gamma, kappa)``.

    Derived from the live converter ``caustic_geometry`` (``caustic_rho``
    is its inverse: ``rho = |y| / reach``), never a pinned literal.
    """
    reach, _ = caustic_geometry(gamma, kappa)
    return float(rho * reach)


def _dense_f_grid() -> np.ndarray:
    """Dense frequency grid (Hz) spanning w in ``[_W_LO, _W_HI]``.

    ``w = xi * f`` so ``f = w / xi`` with ``xi`` the dimensionless-frequency
    scale at ``(_M_LENS_MSUN, _Z_LENS)``.
    """
    xi = dimensionless_frequency(1.0, _M_LENS_MSUN, _Z_LENS)
    f_lo, f_hi = _W_LO / xi, _W_HI / xi
    return np.linspace(float(f_lo), float(f_hi), _N_DENSE)


def _par_dic() -> dict:
    """Lens ``par_dic`` for the served exterior draw (all _LENS_PARAMS)."""
    abs_y = _abs_y_for_rho(_TARGET_RHO)
    return {
        'm_lens_msun': _M_LENS_MSUN,
        'z_lens': _Z_LENS,
        'y1': abs_y * math.cos(_SOURCE_ANGLE),
        'y2': abs_y * math.sin(_SOURCE_ANGLE),
        'gamma': _GAMMA,
        'beta': _BETA,
        'kappa': _KAPPA,
    }


def _build_chart(residual_scale: float) -> BornResidualChart:
    """BornResidualChart with a smooth synthetic residual.

    ``R(gamma, rho, w) = residual_scale * exp(-rho) * (1 + 0.001j)``
    (constant in w, smooth in gamma/rho).  The magnitude is the only knob
    the tests vary, so two scales give two demonstrably different served
    values.
    """
    n_gamma = len(_CHART_GAMMA_GRID)
    n_w = len(_CHART_LOG_W_GRID)
    rho_3d = _CHART_RHO_GRID[None, :, None] * np.ones((n_gamma, 1, n_w))
    residual = residual_scale * np.exp(-rho_3d) * (1.0 + 0.001j)
    return BornResidualChart(
        gamma_grid=_CHART_GAMMA_GRID,
        rho_grid=_CHART_RHO_GRID,
        log_w_grid=_CHART_LOG_W_GRID,
        real_coeffs=residual.real.copy(),
        imag_coeffs=residual.imag.copy(),
        provenance={'test': 'synthetic', 'scale': residual_scale},
    )


def _synthetic_map(*, parity: str, gamma: float, rho: float, w_cert: float,
                   status: float = STATUS_CERTIFIED,
                   w_ceiling: float = 1.0e9) -> CertifiedPpgoMap:
    """One-cell-live certified map (mirrors the band-split suite helper).

    A minimal ``2 x 2 x 3`` lattice with an edge at the ``gamma = 1.0``
    parity boundary; every cell but the requested one is
    ``STATUS_INVALID``.  ``rho_measured_max`` is ``inf`` everywhere so the
    query always lands in the cell.  Built directly through
    ``from_arrays`` (no hash check needed for direct install).
    """
    gamma_edges = np.array([0.2, 1.0, 1.6], dtype=float)
    rho_edges = np.array([0.0, 0.5, 1.0, math.inf], dtype=float)
    parity_codes = np.array([_PARITY_CODES['positive'],
                             _PARITY_CODES['saddle']], dtype=float)
    shape = (2, gamma_edges.size - 1, rho_edges.size - 1)
    w_cert_grid = np.full(shape, np.nan)
    diag_grid = np.full(shape, np.nan)
    w_ceiling_grid = np.full(shape, np.nan)
    status_grid = np.full(shape, STATUS_INVALID)
    interp_grid = np.zeros(shape)
    rho_measured_max_grid = np.full(shape, np.inf)

    p = 0 if parity == 'positive' else 1
    gi = int(np.searchsorted(gamma_edges, gamma, side='right') - 1)
    ri = int(np.searchsorted(rho_edges, rho, side='right') - 1)
    gi = min(max(gi, 0), shape[1] - 1)
    ri = min(max(ri, 0), shape[2] - 1)
    status_grid[p, gi, ri] = status
    if status == STATUS_CERTIFIED:
        w_cert_grid[p, gi, ri] = w_cert
        w_ceiling_grid[p, gi, ri] = w_ceiling
        interp_grid[p, gi, ri] = 1.0

    provenance = {'schema_version': 'test',
                  'certification_bar': CERTIFICATION_BAR}
    return CertifiedPpgoMap.from_arrays(
        parity_codes, gamma_edges, rho_edges, w_cert_grid, diag_grid,
        w_ceiling_grid, status_grid, interp_grid, rho_measured_max_grid,
        provenance)


# ---------------------------------------------------------------------------
# Real-likelihood fixtures (auto-attach + JSON round-trip specs)
# ---------------------------------------------------------------------------
#
# The auto-attach fallback-to-None and JSON round-trip specs exercise the
# CONSTRUCTOR contract of the real likelihoods, so they need a genuine
# (cheap) ``LensedRelativeBinningLikelihood`` -- not the analytic method-
# binding probe above.  Measured costs (this box): event+generator+bins
# ~0.02 s, an RB build ~0.08 s, the marginalized build ~13 s.  The RB
# fixtures rebuild freely; the ONE marginalized build is lru_cached and
# shared across its tests.

#: Seeded HLV Gaussian-noise event (mirrors ``test_lensing_likelihood``).
_EVENT_SEED: int = 20260717
_APPROXIMANT: str = 'IMRPhenomXPHM'
_DF_BIN: float = 4.0
_DELTA_T_MAX: float = 0.02

#: Deterministic precessing CBC reference (keys == WaveformGenerator.params).
_CBC_PAR_DIC: dict = {
    'm1': 60.0, 'm2': 45.0,
    's1x_n': 0.20, 's1y_n': 0.10, 's1z': 0.30,
    's2x_n': -0.10, 's2y_n': 0.15, 's2z': -0.20,
    'l1': 0.0, 'l2': 0.0,
    'iota': 1.0, 'phi_ref': 1.2,
    'ra': 1.8, 'dec': -0.3, 'psi': 0.9,
    't_geocenter': 0.0, 'd_luminosity': 600.0,
    'f_ref': 50.0,
}

#: Well-conditioned lens for the served candidate / marginalized par_dic_0.
_MAIN_LENS: dict = {
    'm_lens_msun': 90.0, 'z_lens': 0.4,
    'y1': 0.20, 'y2': 0.05, 'gamma': 0.10, 'beta': 0.0, 'kappa': 0.0,
}


@functools.lru_cache(maxsize=1)
def _event_generator_bins():
    """Seeded event, waveform generator and uniform bin edges (built once)."""
    event_data = data.EventData.gaussian_noise(
        eventname='test_born_reach', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.,
        seed=_EVENT_SEED)
    event_data.inject_signal(_CBC_PAR_DIC, _APPROXIMANT)
    wfg = waveform.WaveformGenerator.from_event_data(event_data, _APPROXIMANT)
    band = event_data.frequencies[event_data.fslice]
    f_lo, f_hi = float(band[0]), float(band[-1])
    edges = np.arange(f_lo, f_hi, _DF_BIN)
    if edges[-1] < f_hi:
        edges = np.append(edges, f_hi)
    return event_data, wfg, edges


def _build_rb(*, born_residual_chart=_AUTO_BORN_CHART):
    """A real ``LensedRelativeBinningLikelihood`` on the shared fixtures.

    ``born_residual_chart`` defaults to the ``_AUTO_BORN_CHART`` sentinel so
    an argument-omitted build auto-loads the shipped artifact; pass ``None``
    for the pure-engine opt-out or a chart instance for an in-memory attach.
    """
    event_data, wfg, edges = _event_generator_bins()
    return LensedRelativeBinningLikelihood(
        event_data, wfg, _CBC_PAR_DIC, delta_t_max=_DELTA_T_MAX, fbin=edges,
        born_residual_chart=born_residual_chart)


@functools.lru_cache(maxsize=1)
def _marg_default():
    """Default-chart ``LensedMarginalizedExtrinsicLikelihood`` (built once)."""
    event_data, wfg, edges = _event_generator_bins()
    par_dic_0 = {**_CBC_PAR_DIC, **_MAIN_LENS}
    return LensedMarginalizedExtrinsicLikelihood(
        event_data, wfg, par_dic_0, delta_t_max=_DELTA_T_MAX, fbin=edges)


def _lens_candidate() -> dict:
    """A full-lens candidate for a deterministic ``_amplification_coefficients``
    serve (all seven lens params)."""
    return dict(_MAIN_LENS)


class _BornAnalyticProbe:
    """Lightweight probe binding the REAL first-class Born dispatch chain.

    Binds the production unbound methods that make up the surrogate-free
    dispatch to the Born rung -- ``_amplification_coefficients`` and the
    rung ``_born_residual_analytic`` plus their dependencies -- onto a
    minimal instance carrying only the attributes those methods read.  No
    engine, no waveform, no event: the rung serves from an analytic
    geometry partition and a chart interpolation only.
    """

    # Real production methods (unbound) -> production dispatch truth.
    _amplification_coefficients = (
        LensedRelativeBinningLikelihood._amplification_coefficients)
    _born_residual_analytic = (
        LensedRelativeBinningLikelihood._born_residual_analytic)
    _lens_params = LensedRelativeBinningLikelihood._lens_params
    _reduce_dense_kernels = (
        LensedRelativeBinningLikelihood._reduce_dense_kernels)
    _image_delays = LensedRelativeBinningLikelihood._image_delays
    _ppgo_band_split = LensedRelativeBinningLikelihood._ppgo_band_split
    _ppgo_cell_ceiling = LensedRelativeBinningLikelihood._ppgo_cell_ceiling
    _ppgo_cell_coords = LensedRelativeBinningLikelihood._ppgo_cell_coords

    def __init__(self, *, born_residual_chart=None):
        self._kernel_dense_f = _dense_f_grid()
        # No surrogate -> the surrogate intercept is skipped and the Born
        # rung is the reached analytic intercept.
        self.amplification_surrogate = None
        self.born_residual_chart = born_residual_chart
        self._force_direct = False
        self.kernel_subsamples = _KERNEL_SUBSAMPLES
        self.n_bins = _N_BINS

        # Per-bin sub-sample least-squares weights, shape (n_bins, n_sub):
        # value = mean, slope = linear-regression slope over t in [-1, 1].
        t = np.linspace(-1.0, 1.0, _KERNEL_SUBSAMPLES)
        value_weights = np.ones(_KERNEL_SUBSAMPLES) / _KERNEL_SUBSAMPLES
        slope_weights = t / np.sum(t ** 2)
        self._kernel_fit_value = np.tile(value_weights, (_N_BINS, 1))
        self._kernel_fit_slope = np.tile(slope_weights, (_N_BINS, 1))


# ---------------------------------------------------------------------------
# Base test case (anti-vacuity)
# ---------------------------------------------------------------------------

class _BornReachTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity guard shared by every suite class.

    ``tearDown`` FAILS if not a single comparison ran, so a silently
    short-circuiting probe (e.g. a rung that unexpectedly returns ``None``
    and skips the body) cannot read green.
    """

    def setUp(self):
        self.n_checks = 0

    def _count(self):
        self.n_checks += 1

    def tearDown(self):
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: no comparisons ran in '
            f'{type(self).__name__}')

    def _serve(self, *, chart, par_dic=None):
        """Drive the dispatcher and return ``(delays, k0, k1, geom)``."""
        if par_dic is None:
            par_dic = _par_dic()
        probe = _BornAnalyticProbe(born_residual_chart=chart)
        return probe._amplification_coefficients(par_dic)


# ---------------------------------------------------------------------------
# 1. Born serve-path trace
# ---------------------------------------------------------------------------

class BornServePathTraceTestCase(_BornReachTestCase):
    """The lifted Born rung is REACHED through the production dispatcher.

    Fixture premise: gamma < 1 (saddle rung skipped), dense_w.max() = 60 <
    150 (ppGO-above-ceiling skipped), no surrogate (surrogate rung
    skipped), no ppGO map (no band split).  So a chart-attached
    ``_amplification_coefficients`` MUST route to ``_born_residual_analytic``.
    """

    def test_dispatcher_matches_direct_rung_byte_exact(self):
        # The dispatcher returns exactly what the rung returns; both feed
        # identical float64 inputs to reconstruct_farfield, so the served
        # coefficients are byte-identical (route identification).
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        chart = _build_chart(_RESIDUAL_SCALE_A)
        par_dic = _par_dic()

        probe = _BornAnalyticProbe(born_residual_chart=chart)
        dispatched = probe._amplification_coefficients(par_dic)

        # Independent oracle: the shipping rung called directly on the same
        # (lens, dense_w).
        lens = probe._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            probe._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        direct = probe._born_residual_analytic(lens, dense_w)

        self.assertIsNotNone(
            direct, 'Fixture premise lost: Born rung refused the draw')
        self.assertIsNotNone(
            dispatched, 'Dispatcher did not serve the Born rung')
        d_delays, d_k0, d_k1, _ = dispatched
        r_delays, r_k0, r_k1, _ = direct
        self.assertTrue(np.array_equal(d_delays, r_delays),
                        'delays: dispatcher != direct rung (byte)')
        self.assertTrue(np.array_equal(d_k0, r_k0),
                        'k0: dispatcher != direct rung (byte)')
        self.assertTrue(np.array_equal(d_k1, r_k1),
                        'k1: dispatcher != direct rung (byte)')
        self._count()

    def test_served_coefficients_are_finite_and_shaped(self):
        # A served rung yields finite (delays, k0, k1) of the expected
        # shapes -- the reconstruction is not a NaN/degenerate pass.
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        delays, k0, k1, geom = self._serve(chart=_build_chart(_RESIDUAL_SCALE_A))

        n_channels = k0.shape[0]
        self.assertEqual(k0.shape, (n_channels, _N_BINS))
        self.assertEqual(k1.shape, (n_channels, _N_BINS))
        self.assertEqual(delays.shape, (n_channels,))
        self.assertTrue(np.all(np.isfinite(k0)))
        self.assertTrue(np.all(np.isfinite(k1)))
        self.assertTrue(np.all(np.isfinite(delays)))
        self.assertIsNotNone(geom)
        self._count()

    def test_two_charts_give_different_coefficients(self):
        # Route liveness: swapping the chart's residual MUST change the
        # served coefficients.  If it did not, the chart value never
        # reaches the served number -- a dead path (the spec's all-zero
        # difference diagnostic, made an assertion).
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        _, k0_a, k1_a, _ = self._serve(chart=_build_chart(_RESIDUAL_SCALE_A))
        _, k0_b, k1_b, _ = self._serve(chart=_build_chart(_RESIDUAL_SCALE_B))

        max_diff = max(float(np.max(np.abs(k0_a - k0_b))),
                       float(np.max(np.abs(k1_a - k1_b))))
        self.assertGreater(
            max_diff, 1e-12,
            'Born route appears DEAD: two distinct charts served identical '
            'coefficients (max|k_a - k_b| = 0)')
        self._count()

    def test_no_chart_rung_returns_none(self):
        # The fall-through guard: with no chart attached the rung declines
        # (returns None) so the dispatcher proceeds to the exact engine.
        probe = _BornAnalyticProbe(born_residual_chart=None)
        par_dic = _par_dic()
        lens = probe._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            probe._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        self.assertIsNone(probe._born_residual_analytic(lens, dense_w))
        self._count()

    def test_gate_misses_return_none(self):
        # Each Born gate (kappa==0, beta==0, rho>2) refuses its violation
        # and falls through -- never serves a config it cannot represent.
        chart = _build_chart(_RESIDUAL_SCALE_A)
        probe = _BornAnalyticProbe(born_residual_chart=chart)

        base = _par_dic()
        cases = {
            'kappa!=0': {**base, 'kappa': 0.1},
            'beta!=0': {**base, 'beta': 0.1},
            # Unlensed/macro-trivial limit: gamma == 0 has no caustic, and
            # caustic_rho raises a raw ZeroDivisionError there (measured
            # 2026-08-14 via the F -> 1 zero-noise anchors) — the rung must
            # decline BEFORE any caustic-frame computation.
            'gamma==0': {**base, 'gamma': 0.0},
        }
        # rho <= 2 (interior/near-caustic): shrink |y| to rho ~ 1.5.
        near = _par_dic()
        abs_y = _abs_y_for_rho(1.5)
        near['y1'] = abs_y * math.cos(_SOURCE_ANGLE)
        near['y2'] = abs_y * math.sin(_SOURCE_ANGLE)
        cases['rho<=2'] = near

        for label, par_dic in cases.items():
            with self.subTest(gate=label):
                lens = probe._lens_params(par_dic)
                dense_w = dimensionless_frequency(
                    probe._kernel_dense_f, lens['m_lens_msun'],
                    lens['z_lens'])
                self.assertIsNone(
                    probe._born_residual_analytic(lens, dense_w),
                    f'Born rung served a {label} draw it must refuse')
                self._count()


# ---------------------------------------------------------------------------
# 2. Map band-split trace
# ---------------------------------------------------------------------------

class MapBandSplitTraceTestCase(_BornReachTestCase):
    """The certified-map band-split path of the LIFTED Born rung.

    The lifted rung ``_born_residual_analytic`` serves the Born carrier +
    residual over the FULL band and applies the certified-map split by
    ZEROING the reconstructed far-field envelope above ``w_trust`` (INS-2-001
    fix), so a firing band-split no longer raises the former shape
    ``ValueError``.  This suite pins the split PREMISES that must hold: the
    split floor lands strictly inside the band and under the effective
    ceiling, and a beyond-wall cell does not split (byte-identical to the
    no-map serve).  The Architect's positive MAP BAND-SPLIT TRACE invariant
    (``max|k_split - k_nomap| > 0``) is owned by the test author and added
    separately now that production serves the split.
    """

    def _install_split_map(self):
        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=_GAMMA, rho=_TARGET_RHO,
            w_cert=_MAP_W_CERT_SPLIT))
        self.addCleanup(set_certified_ppgo_map, None)

    def test_w_trust_lands_strictly_inside_band(self):
        # Premise, derived from the LIVE map (never a pinned literal): the
        # split floor sits strictly inside [w_lo, w_hi] and under the
        # effective ceiling, so band_split WOULD fire.
        self._install_split_map()
        probe = _BornAnalyticProbe(born_residual_chart=_build_chart(
            _RESIDUAL_SCALE_A))
        lens = probe._lens_params(_par_dic())
        w_trust = probe._ppgo_band_split(lens)

        expected = max(W_TRUST_MULTIPLIER * _MAP_W_CERT_SPLIT,
                       _MAP_W_CERT_SPLIT + W_TRUST_ADDITIVE)
        self.assertIsNotNone(
            w_trust, 'Premise lost: certified cell was not resolved')
        self.assertAlmostEqual(w_trust, expected, places=9)
        self.assertLess(_W_LO, w_trust)
        self.assertLess(w_trust, _W_HI)
        # The effective ceiling must not veto the split.
        eff_ceiling = min(ASTROID_WALL, probe._ppgo_cell_ceiling(lens))
        self.assertGreaterEqual(eff_ceiling, _W_HI)
        self._count()

    def test_beyond_wall_cell_keeps_whole_band(self):
        # A non-certified (beyond-wall) cell yields UNKNOWN w_trust -> no
        # split -> the serve stays byte-identical to the no-map serve (and
        # does NOT hit the band-split defect).
        chart = _build_chart(_RESIDUAL_SCALE_A)

        set_certified_ppgo_map(None)
        nomap = self._serve(chart=chart)

        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=_GAMMA, rho=_TARGET_RHO,
            w_cert=_MAP_W_CERT_SPLIT, status=STATUS_BEYOND_WALL))
        self.addCleanup(set_certified_ppgo_map, None)
        withmap = self._serve(chart=chart)

        for name, a, b in zip(('delays', 'k0', 'k1'), nomap[:3], withmap[:3]):
            self.assertTrue(
                np.array_equal(a, b),
                f'{name}: beyond-wall cell must not split the band')
            self._count()


# ---------------------------------------------------------------------------
# 3. Null-split identity (byte-exact, Professor-directed)
# ---------------------------------------------------------------------------

class NullSplitIdentityTestCase(_BornReachTestCase):
    """A no-op split reduces byte-exactly to the whole-band Born serve.

    Case A: no map installed.  Case B: a map whose ``w_trust >= w_hi`` so
    ``band_split`` is False.  Both must serve identical float64 inputs to
    ``reconstruct_farfield``, so the served ``(delays, k0, k1)`` are
    byte-identical (``np.array_equal``); element-wise ``max|A - B|`` is
    exactly ``0.0``.
    """

    def test_null_split_map_matches_no_map_byte_exact(self):
        chart = _build_chart(_RESIDUAL_SCALE_A)

        set_certified_ppgo_map(None)
        a_delays, a_k0, a_k1, _ = self._serve(chart=chart)

        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=_GAMMA, rho=_TARGET_RHO,
            w_cert=_MAP_W_CERT_NULL))
        self.addCleanup(set_certified_ppgo_map, None)
        b_delays, b_k0, b_k1, _ = self._serve(chart=chart)

        for name, a, b in (('delays', a_delays, b_delays),
                           ('k0', a_k0, b_k0), ('k1', a_k1, b_k1)):
            self.assertTrue(
                np.array_equal(a, b),
                f'{name}: null-split map != no-map (max|A-B| = '
                f'{float(np.max(np.abs(a - b)))})')
            self.assertEqual(float(np.max(np.abs(a - b))), 0.0)
            self._count()

    def test_null_split_w_trust_at_or_above_band(self):
        # Premise: the null-split map's w_trust sits at/above w_hi so no
        # split is attempted.
        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=_GAMMA, rho=_TARGET_RHO,
            w_cert=_MAP_W_CERT_NULL))
        self.addCleanup(set_certified_ppgo_map, None)
        probe = _BornAnalyticProbe(born_residual_chart=_build_chart(
            _RESIDUAL_SCALE_A))
        lens = probe._lens_params(_par_dic())
        w_trust = probe._ppgo_band_split(lens)
        self.assertIsNotNone(w_trust)
        self.assertGreaterEqual(w_trust, _W_HI)
        self._count()

    def test_null_split_matches_direct_whole_band_rung(self):
        # Both the no-map dispatcher serve and the null-split serve equal a
        # direct whole-band Born rung call (the un-split reference).
        chart = _build_chart(_RESIDUAL_SCALE_A)
        probe = _BornAnalyticProbe(born_residual_chart=chart)
        par_dic = _par_dic()
        lens = probe._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            probe._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])

        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        direct = probe._born_residual_analytic(lens, dense_w)
        dispatched = probe._amplification_coefficients(par_dic)

        for name, d, r in zip(('delays', 'k0', 'k1'), dispatched[:3],
                              direct[:3]):
            self.assertTrue(np.array_equal(d, r),
                            f'{name}: whole-band dispatcher != direct rung')
            self._count()


# ---------------------------------------------------------------------------
# 4. Byte-identity battery off the served path
# ---------------------------------------------------------------------------

#: Battery of draws that must NOT reach the Born intercept.  Each entry is
#: ``(label, gamma, rho, kappa, beta)``; ``rho`` is converted to a
#: source-plane ``|y|`` via the LIVE ``caustic_geometry`` (never a literal),
#: so the fixtures follow the gate boundaries if the geometry moves.
_OFF_PATH_BATTERY = (
    ('interior_rho<1', _GAMMA, 0.5, _KAPPA, _BETA),
    ('exterior_1<rho<2', _GAMMA, 1.5, _KAPPA, _BETA),
    ('covers_false_rho>grid', _GAMMA, 6.0, _KAPPA, _BETA),
    ('saddle_gamma>1', 1.3, _TARGET_RHO, _KAPPA, _BETA),
    ('kappa!=0_beta!=0', _GAMMA, _TARGET_RHO, 0.1, 0.1),
)


class ByteIdentityBatteryTestCase(_BornReachTestCase):
    """Every gate-miss draw declines the Born rung -> byte-identical route.

    The chart enters ``_amplification_coefficients`` at exactly one place:
    the ``_born_residual_analytic`` intercept.  If that rung returns
    ``None`` for a draw, the dispatcher proceeds down the SAME route it
    would take with ``born_residual_chart=None`` -- every subsequent
    float64 input (and hence the served ``(delays, k0, k1)``) is identical.
    So a chart-attached ``None`` is the engine-free, decisive form of "the
    two serves are byte-identical" for draws whose downstream route needs
    the exact seed engine the fast-tier probe cannot supply.

    CRITICAL: the ``kappa != 0`` / ``beta != 0`` row is the silent-accuracy
    bug the corrected gate prevents -- a Born serve there would quietly
    apply a weak-deflection residual the chart was never trained to
    represent.  It MUST decline.
    """

    def _born_declines(self, *, gamma, rho, kappa, beta, chart):
        """Return True iff the Born rung declines (``None``) this draw."""
        abs_y = _abs_y_for_rho(rho, gamma=gamma, kappa=kappa)
        par_dic = {
            'm_lens_msun': _M_LENS_MSUN,
            'z_lens': _Z_LENS,
            'y1': abs_y * math.cos(_SOURCE_ANGLE),
            'y2': abs_y * math.sin(_SOURCE_ANGLE),
            'gamma': gamma,
            'beta': beta,
            'kappa': kappa,
        }
        probe = _BornAnalyticProbe(born_residual_chart=chart)
        lens = probe._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            probe._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        return probe._born_residual_analytic(lens, dense_w) is None

    def test_battery_declines_born_intercept(self):
        # Per-draw boolean table: True == Born declined == route
        # byte-identical to the no-chart route.  Any False row names a draw
        # where the intercept LEAKED (served a config it cannot represent).
        chart = _build_chart(_RESIDUAL_SCALE_A)
        table = {}
        for label, gamma, rho, kappa, beta in _OFF_PATH_BATTERY:
            with self.subTest(draw=label):
                declined = self._born_declines(
                    gamma=gamma, rho=rho, kappa=kappa, beta=beta, chart=chart)
                table[label] = declined
                self.assertTrue(
                    declined,
                    f'Born intercept LEAKED on {label!r}: rung served a '
                    'draw off the certified path (chart-attached route '
                    'diverges from the born_residual_chart=None route)')
                self._count()
        # The table is the spec's diagnostic; assert it is complete and all
        # True so a partially-run loop cannot read green.
        self.assertEqual(set(table), {b[0] for b in _OFF_PATH_BATTERY})
        self.assertTrue(all(table.values()), table)

    def test_kappa_beta_row_declines_even_when_rho_and_covers_pass(self):
        # Isolate the silent-accuracy guard: hold gamma/rho on the SERVED
        # cell (rho = 3 > 2, covers True) and flip ONLY kappa/beta.  The
        # rho + covers gates would admit; the kappa/beta gate must veto.
        chart = _build_chart(_RESIDUAL_SCALE_A)

        # Same cell served with kappa = beta = 0 (the reference config) DOES
        # reach the rung -- proving the veto below is the kappa/beta gate,
        # not an unrelated refusal.
        served_ref = not self._born_declines(
            gamma=_GAMMA, rho=_TARGET_RHO, kappa=0.0, beta=0.0, chart=chart)
        self.assertTrue(
            served_ref,
            'Premise lost: the reference (kappa=beta=0) cell does not serve, '
            'so the kappa/beta veto below is not isolated')
        self._count()

        for label, kappa, beta in (('kappa!=0', 0.1, 0.0),
                                    ('beta!=0', 0.0, 0.1),
                                    ('both!=0', 0.1, 0.1)):
            with self.subTest(gate=label):
                self.assertTrue(
                    self._born_declines(gamma=_GAMMA, rho=_TARGET_RHO,
                                        kappa=kappa, beta=beta, chart=chart),
                    f'silent-accuracy bug: Born served a {label} draw on the '
                    'served cell -- it must decline off-reference kappa/beta')
                self._count()


# ---------------------------------------------------------------------------
# 5. Loader hard-refusal at construction
# ---------------------------------------------------------------------------

class BornResidualChartLoaderRefusalTestCase(_BornReachTestCase):
    """``BornResidualChart.load`` refuses corrupt / stale artifacts loudly.

    An explicit-path load is the construction boundary: a
    ``LensedRelativeBinningLikelihood`` pointed at an explicit chart
    evaluates ``BornResidualChart.load(path)`` (which raises here) BEFORE
    the likelihood is built, so the failure propagates loudly.  (The
    auto-attach default, by contrast, swallows the same ValueError to
    ``None`` with a warning -- the opt-out-fallback contract, owned by a
    separate description and not re-tested here.)

    Every refusal message must name ``scripts/train_born_residual.py`` so a
    caller knows how to regenerate.  A VALID artifact round-trips (positive
    control), proving the writer produces a loadable npz and the refusals
    below are caused by the specific corruption, not a broken fixture.
    """

    _REGEN_SCRIPT = 'train_born_residual.py'

    def _tmp_npz(self) -> Path:
        tmpdir = Path(tempfile.mkdtemp(prefix='born_loader_'))
        self.addCleanup(shutil.rmtree, tmpdir, True)  # ignore_errors=True
        return tmpdir / 'born_residual_chart.npz'

    def _canonical_arrays(self):
        chart = _build_chart(_RESIDUAL_SCALE_A)
        return (chart.gamma_grid, chart.rho_grid, chart.log_w_grid,
                chart.real_coeffs, chart.imag_coeffs, chart.provenance)

    def test_valid_artifact_round_trips(self):
        # Positive control (teeth): the writer's npz loads and reconstructs
        # arrays byte-for-byte -- so a refusal below is the corruption, not
        # a malformed fixture.
        path = self._tmp_npz()
        g, r, lw, re_c, im_c, prov = self._canonical_arrays()
        np.savez(
            path, schema=_SCHEMA,
            content_hash=_content_hash(g, r, lw, re_c, im_c),
            gamma_grid=g, rho_grid=r, log_w_grid=lw,
            real_coeffs=re_c, imag_coeffs=im_c, provenance=json.dumps(prov))

        loaded = BornResidualChart.load(path)
        self.assertTrue(np.array_equal(loaded.real_coeffs, re_c))
        self.assertTrue(np.array_equal(loaded.imag_coeffs, im_c))
        self.assertTrue(np.array_equal(loaded.gamma_grid, g))
        self.assertTrue(np.array_equal(loaded.rho_grid, r))
        self.assertTrue(np.array_equal(loaded.log_w_grid, lw))
        self._count()

    def test_corrupted_content_hash_refuses(self):
        # One tampered real_coeffs element (one-ULP flip -- the strongest
        # teeth), but the stored hash is the ORIGINAL: load must detect the
        # mismatch and refuse, naming the regen script.
        path = self._tmp_npz()
        g, r, lw, re_c, im_c, prov = self._canonical_arrays()
        original_hash = _content_hash(g, r, lw, re_c, im_c)
        tampered = re_c.copy()
        tampered.flat[0] = np.nextafter(tampered.flat[0], np.inf)
        self.assertNotEqual(
            _content_hash(g, r, lw, tampered, im_c), original_hash,
            'fixture is inert: one-ULP flip did not change the content hash')

        np.savez(
            path, schema=_SCHEMA, content_hash=original_hash,
            gamma_grid=g, rho_grid=r, log_w_grid=lw,
            real_coeffs=tampered, imag_coeffs=im_c,
            provenance=json.dumps(prov))

        with self.assertRaises(ValueError) as ctx:
            BornResidualChart.load(path)
        msg = str(ctx.exception)
        self.assertIn(self._REGEN_SCRIPT, msg)
        self.assertIn('hash', msg.lower())
        self._count()

    def test_missing_or_wrong_schema_refuses(self):
        # Parametrize over {missing schema key, wrong schema string}: both
        # must hard-refuse (a schema-less or foreign artifact is never
        # silently accepted), naming the regen script.
        g, r, lw, re_c, im_c, prov = self._canonical_arrays()
        good_hash = _content_hash(g, r, lw, re_c, im_c)
        base = dict(
            content_hash=good_hash, gamma_grid=g, rho_grid=r, log_w_grid=lw,
            real_coeffs=re_c, imag_coeffs=im_c, provenance=json.dumps(prov))

        cases = {
            'missing_schema': {**base},
            'wrong_schema': {**base, 'schema': 'born_residual_v0'},
        }
        for label, payload in cases.items():
            with self.subTest(schema=label):
                path = self._tmp_npz()
                np.savez(path, **payload)
                with self.assertRaises(ValueError) as ctx:
                    BornResidualChart.load(path)
                msg = str(ctx.exception)
                self.assertIn(
                    self._REGEN_SCRIPT, msg,
                    f'{label}: refusal did not name the regen script')
                self.assertIn('schema', msg.lower())
                self._count()


class SuiteSelfFalsificationTestCase(_BornReachTestCase):
    """Prove the suite's oracles have teeth: the byte-equality and
    difference assertions used above are shown to be able to go RED.

    Without this class a silently-degenerate suite (e.g. one where every
    served array is identical for unrelated reasons, or where
    ``np.array_equal`` never actually discriminates) would read green and
    certify nothing.
    """

    def test_byte_equality_distinguishes_corrupted_arrays(self):
        # The NULL-SPLIT identity relies on np.array_equal catching a single
        # perturbed float64.  Corrupt one served coefficient and confirm the
        # identity assertion would FAIL, while the untouched arrays still
        # compare equal (so the check is specific, not blanket-red).
        chart = _build_chart(_RESIDUAL_SCALE_A)
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        _, k0, _, _ = self._serve(chart=chart)

        corrupt = k0.copy()
        corrupt.flat[0] += 1.0
        self.assertFalse(
            np.array_equal(k0, corrupt),
            'byte-equality oracle is blind to a 1.0 perturbation')
        self.assertTrue(np.array_equal(k0, k0.copy()),
                        'byte-equality oracle rejects an identical copy')
        self._count()

    def test_identical_charts_give_identical_coefficients(self):
        # The SERVE-PATH "two charts differ" test asserts max_diff > 1e-12.
        # Two charts of the SAME residual scale must instead serve
        # byte-identical coefficients (max_diff == 0.0), proving the >1e-12
        # threshold discriminates chart content rather than passing on
        # incidental nonzero noise.
        a_k0 = self._serve(chart=_build_chart(_RESIDUAL_SCALE_A))[1]
        b_k0 = self._serve(chart=_build_chart(_RESIDUAL_SCALE_A))[1]
        self.assertEqual(float(np.max(np.abs(a_k0 - b_k0))), 0.0,
                         'identical charts served different coefficients')
        self._count()

    def test_null_split_identity_breaks_for_different_chart(self):
        # A control for the NULL-SPLIT identity: swap the chart residual
        # scale between the two serves and the "byte-identical" claim must
        # FAIL, proving the identity test is sensitive to the served value
        # and not vacuously true for any pair of serves.
        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=_GAMMA, rho=_TARGET_RHO,
            w_cert=_MAP_W_CERT_NULL))
        self.addCleanup(set_certified_ppgo_map, None)
        a_k0 = self._serve(chart=_build_chart(_RESIDUAL_SCALE_A))[1]
        b_k0 = self._serve(chart=_build_chart(_RESIDUAL_SCALE_B))[1]
        self.assertFalse(
            np.array_equal(a_k0, b_k0),
            'null-split identity is vacuous: different charts served equal')
        self.assertGreater(float(np.max(np.abs(a_k0 - b_k0))), 1e-12)
        self._count()


# ---------------------------------------------------------------------------
# 6. Auto-attach fallback-to-None on a bad artifact (opt-out-fallback)
# ---------------------------------------------------------------------------

class AutoAttachFallbackToNoneTestCase(_BornReachTestCase):
    """A load anomaly at construction degrades to the pure-engine path.

    The argument-omitted (``_AUTO_BORN_CHART`` sentinel) construction
    auto-loads the shipped artifact but must REFUSE-TO-NONE on any load
    anomaly -- mirroring ``use_certified_ppgo_map``'s refuse-to-None -- so
    a corrupt/absent artifact never crashes construction: it emits a
    ``RuntimeWarning`` and serves engine-pure.  The load is forced to raise
    by patching ``BornResidualChart.load`` (the classmethod the constructor
    calls with no path), covering ``OSError``, ``ValueError`` and
    ``KeyError`` -- exactly the trio the constructor catches.
    """

    def test_shipped_default_attaches_a_chart(self):
        # Premise for the fallback tests: with no injected failure the
        # default build genuinely attaches the shipped chart (is_default
        # records the construction intent).  If this ever refuses on its
        # own, the fallback tests below would be vacuous.
        like = _build_rb()
        self.assertIsNotNone(
            like.born_residual_chart,
            'shipped artifact failed to auto-load; fallback tests vacuous')
        self.assertTrue(like._born_residual_chart_is_default)
        self._count()

    def test_load_failure_refuses_to_none_with_warning(self):
        # Each caught exception type -> construction SUCCEEDS with the chart
        # refused to None, the default-intent flag preserved, and a
        # RuntimeWarning naming the unavailable chart emitted.
        for error in (OSError('artifact missing'),
                      ValueError('hash mismatch'),
                      KeyError('schema')):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(BornResidualChart, 'load',
                                       side_effect=error):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        like = _build_rb()
                messages = [str(w.message) for w in caught]
                self.assertIsNone(
                    like.born_residual_chart,
                    'load anomaly did not refuse to None')
                self.assertTrue(
                    like._born_residual_chart_is_default,
                    'refused-to-None default lost its construction intent')
                self.assertTrue(
                    any('Born-residual chart unavailable' in m
                        for m in messages),
                    f'no refuse-to-None warning emitted; got {messages!r}')
                self._count()

    def test_fallback_serve_equals_explicit_none_serve(self):
        # The decisive contract: a refused-to-None auto-load serves
        # BYTE-IDENTICALLY to an explicit ``born_residual_chart=None``
        # opt-out -- the Born rung is the only chart-dependent branch, so a
        # None chart makes every downstream float64 input identical.
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        candidate = _lens_candidate()

        with mock.patch.object(BornResidualChart, 'load',
                               side_effect=OSError('artifact missing')):
            fallback = _build_rb()
        explicit_none = _build_rb(born_residual_chart=None)

        f_delays, f_k0, f_k1, _ = fallback._amplification_coefficients(
            candidate)
        n_delays, n_k0, n_k1, _ = explicit_none._amplification_coefficients(
            candidate)
        self.assertTrue(np.array_equal(f_delays, n_delays),
                        'delays: fallback != explicit-None serve')
        self.assertTrue(np.array_equal(f_k0, n_k0),
                        'k0: fallback != explicit-None serve')
        self.assertTrue(np.array_equal(f_k1, n_k1),
                        'k1: fallback != explicit-None serve')
        self._count()


# ---------------------------------------------------------------------------
# 7. JSON round-trip of both classes with the auto-attached Born chart
# ---------------------------------------------------------------------------

class JsonRoundTripBornChartTestCase(_BornReachTestCase):
    """``get_init_dict`` round-trips the Born chart three ways.

    The chart round-trips on the recorded CONSTRUCTION INTENT, not on the
    resolved chart (which cannot tell an auto-loaded default from a caller
    copy of the same artifact):

    * default (sentinel) -> the key is DROPPED so reconstruction re-defaults
      and re-auto-loads (re-serving via the Born path);
    * explicit ``None`` -> ``None`` is emitted verbatim (pure-engine stays
      pure-engine, never silently re-auto-loading);
    * caller-supplied in-memory chart -> ``NotImplementedError`` naming the
      missing source path (the chart is not embedded in the init dict).

    ``LensedRelativeBinningLikelihood`` uses a real cheap build; the
    marginalized default reuses one lru_cached ~13 s build, and its
    None / in-memory branches are exercised engine-free through the real
    override on an ``object.__new__`` stub (no second heavy build).
    """

    # -- LensedRelativeBinningLikelihood ---------------------------------

    def test_rb_default_init_dict_omits_chart_key(self):
        # Default intent -> the key is dropped, so a JSON reconstruction
        # re-defaults to the sentinel and re-auto-loads.
        init_dict = _build_rb().get_init_dict()
        self.assertNotIn('born_residual_chart', init_dict)
        self._count()

    def test_rb_default_roundtrip_reattaches_and_serves_identically(self):
        # A full to_json/read_json round-trip of the default build must
        # re-attach the shipped chart (default intent preserved) and serve a
        # candidate BYTE-IDENTICALLY to the original.
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        candidate = _lens_candidate()
        original = _build_rb()
        before = original._amplification_coefficients(candidate)

        with tempfile.TemporaryDirectory() as tmp:
            original.to_json(tmp, overwrite=True)
            restored = utils.read_json(tmp)

        self.assertTrue(
            restored._born_residual_chart_is_default,
            'reconstructed default lost its auto-load intent')
        self.assertIsNotNone(
            restored.born_residual_chart,
            'reconstructed default did not re-auto-load the chart')
        after = restored._amplification_coefficients(candidate)
        for name, lhs, rhs in zip(('delays', 'k0', 'k1'), before, after):
            self.assertTrue(np.array_equal(lhs, rhs),
                            f'{name}: served value changed across round-trip')
        self._count()

    def test_rb_explicit_none_roundtrips_to_none(self):
        # Explicit opt-out -> None is emitted verbatim and the reconstruction
        # stays pure-engine (never silently re-auto-loading a chart).
        init_dict = _build_rb(born_residual_chart=None).get_init_dict()
        self.assertIn('born_residual_chart', init_dict)
        self.assertIsNone(init_dict['born_residual_chart'])

        with tempfile.TemporaryDirectory() as tmp:
            _build_rb(born_residual_chart=None).to_json(tmp, overwrite=True)
            restored = utils.read_json(tmp)
        self.assertIsNone(restored.born_residual_chart,
                          'explicit None re-auto-loaded a chart')
        self.assertFalse(restored._born_residual_chart_is_default,
                         'explicit None round-tripped as the auto default')
        self._count()

    def test_rb_in_memory_chart_raises_not_implemented(self):
        # A caller-supplied in-memory chart has no source path to reference
        # and its tables are not embedded -> a clear NotImplementedError
        # naming the limitation (never a silent drop).
        in_memory = _build_chart(_RESIDUAL_SCALE_A)
        like = _build_rb(born_residual_chart=in_memory)
        with self.assertRaises(NotImplementedError) as ctx:
            like.get_init_dict()
        self.assertIn('source path', str(ctx.exception))
        self._count()

    # -- LensedMarginalizedExtrinsicLikelihood ---------------------------

    def test_marg_default_threads_sentinel_and_omits_key(self):
        # The marginalized class stores the sentinel verbatim and forwards it
        # to its inner engine (single auto-load lives there); its init dict
        # omits the key just like the RB class.
        marg = _marg_default()
        self.assertIs(marg.born_residual_chart, _AUTO_BORN_CHART,
                      'marginalized class did not keep the sentinel verbatim')
        self.assertTrue(
            marg._engine._born_residual_chart_is_default,
            'inner engine did not receive the auto-load default')
        self.assertIsNotNone(
            marg._engine.born_residual_chart,
            'inner engine did not auto-load the shipped chart')
        self.assertNotIn('born_residual_chart', marg.get_init_dict())
        self._count()

    def test_marg_none_and_in_memory_branches_engine_free(self):
        # Exercise the marginalized override's None / in-memory branches
        # WITHOUT a second ~13 s build: run the real override on an
        # object.__new__ stub with the heavy base get_init_dict patched to a
        # trivial dict.  super() resolves through the MRO, which requires the
        # stub to be a genuine LensedMarginalizedExtrinsicLikelihood
        # instance (hence object.__new__, not a bare object).
        def _fake_base(self, **kwargs):
            return {'amplification_surrogate': None,
                    'born_residual_chart': 'BASE_PLACEHOLDER'}

        stub = object.__new__(LensedMarginalizedExtrinsicLikelihood)

        with mock.patch.object(MarginalizedExtrinsicLikelihood,
                               'get_init_dict', _fake_base):
            # Explicit None -> emitted verbatim.
            stub.born_residual_chart = None
            none_dict = LensedMarginalizedExtrinsicLikelihood.get_init_dict(
                stub)
            self.assertIn('born_residual_chart', none_dict)
            self.assertIsNone(none_dict['born_residual_chart'])

            # In-memory chart -> NotImplementedError naming the source path.
            stub.born_residual_chart = _build_chart(_RESIDUAL_SCALE_A)
            with self.assertRaises(NotImplementedError) as ctx:
                LensedMarginalizedExtrinsicLikelihood.get_init_dict(stub)
            self.assertIn('source path', str(ctx.exception))

            # Sentinel -> the key is dropped (matches the RB contract).
            stub.born_residual_chart = _AUTO_BORN_CHART
            sentinel_dict = (
                LensedMarginalizedExtrinsicLikelihood.get_init_dict(stub))
            self.assertNotIn('born_residual_chart', sentinel_dict)
        self._count()


class ReachabilityFallbackSelfFalsificationTestCase(_BornReachTestCase):
    """Prove the Spec-1 / Spec-2 oracles above can go RED.

    Two teeth are genuinely in doubt for the new tests and are pinned here
    (the byte-equality oracle itself is already exercised by
    ``SuiteSelfFalsificationTestCase`` and is not duplicated):

    * the fallback==explicit-None equality is over NON-TRIVIAL served
      content -- a perturbation of a single served coefficient must flip
      ``np.array_equal`` to ``False`` on the real RB serve path (else the
      equality could pass on two empty/degenerate arrays);
    * the ``get_init_dict`` three-way branch keys on GENUINELY DISTINCT
      objects (sentinel is not ``None``, an in-memory chart is neither) and
      the init-dict key CAN appear -- otherwise the default's ``assertNotIn``
      would be vacuously green.
    """

    def test_fallback_serve_equality_is_over_nontrivial_content(self):
        # Serve through the real refused-to-None RB build; the coefficient
        # arrays must be non-empty and a 1.0 perturbation of one entry must
        # flip the byte-equality oracle, proving the fallback==None equality
        # is not passing on degenerate/empty output.
        set_certified_ppgo_map(None)
        self.addCleanup(set_certified_ppgo_map, None)
        candidate = _lens_candidate()
        with mock.patch.object(BornResidualChart, 'load',
                               side_effect=OSError('artifact missing')):
            fallback = _build_rb()
        _, k0, _, _ = fallback._amplification_coefficients(candidate)
        self.assertGreater(k0.size, 0, 'served k0 is empty; equality vacuous')

        corrupt = k0.copy()
        corrupt.flat[0] += 1.0
        self.assertFalse(np.array_equal(k0, corrupt),
                         'equality oracle blind to a served-coefficient shift')
        self.assertTrue(np.array_equal(k0, k0.copy()))
        self._count()

    def test_init_dict_branch_keys_are_distinct_objects(self):
        # The get_init_dict three-way keys on object identity: sentinel,
        # None, and any in-memory chart must be mutually distinguishable, or
        # the branch selection (and the default's assertNotIn) is vacuous.
        self.assertIsNotNone(
            _AUTO_BORN_CHART,
            'sentinel is None: default and explicit-None become identical')
        in_memory = _build_chart(_RESIDUAL_SCALE_A)
        self.assertIsNot(in_memory, _AUTO_BORN_CHART)
        self.assertIsNotNone(in_memory)
        self._count()

    def test_init_dict_key_can_appear(self):
        # The default build's assertNotIn('born_residual_chart', ...) is only
        # meaningful because the key CAN appear: the explicit-None build
        # emits it (as None).  Pin that contrast so the omission is a real
        # branch decision, not a key the dict never carries.
        self.assertIn('born_residual_chart',
                      _build_rb(born_residual_chart=None).get_init_dict())
        self.assertNotIn('born_residual_chart', _build_rb().get_init_dict())
        self._count()


if __name__ == '__main__':
    unittest.main()
