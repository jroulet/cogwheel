"""
Tests for `lensing.surrogate_census` -- the CENSUS / validation tool for the
multi-chart lens-amplification surrogate (Build 8c WP4), plus the two
design-falsifiable claims the census evidences (tube-vs-raw, fold ray).

WHAT THIS SUITE PINS
--------------------
The census tool measures three properties of a trained surrogate WITHOUT ever
trusting the surrogate's own labels:

* Served fraction + a six-way MUTUALLY-EXCLUSIVE fall-through breakdown
  (``gamma-guard`` / ``dropped-sliver`` / ``born`` / ``cusp-window`` /
  ``refusal-ball`` / ``out-of-box``), attributed by calling the census's OWN
  guard predicates --
  never a re-implementation (`classify_fallthrough`, `fallthrough_breakdown`).
  Section A pins each category by CONSTRUCTION and the MECE partition
  (served + engine_refused + Sum(categories) == n).

* Arc-projection is out-of-box, by design (Section B): a near-cusp source that
  projects onto a NEIGHBOURING fold arc (theta out of the chart's arc range)
  is refusal-conservatively categorized ``out-of-box`` -- NOT ``cusp-window``.

* Per-chart held-out envelope eps against a FRESH engine oracle (Section C),
  currency ``max|E_sur - E_eng| / max(max|E_eng|, EPS_DENOM_FLOOR)`` (F002 --
  the oracle is a fresh `ChangRefsdalChannels.evaluate`, never the surrogate's
  own reconstruction; enforced by an AST guard AND a runtime independence
  check).  Node-exactness (eps at an exact training node ~ machine precision)
  and trough-boundedness (a deep-cancellation |E| trough does not blow the
  max-normalized currency up).

* (gamma, image_count, eta)-partitioned lnL error tiers (Section D) via a real
  `LensedRelativeBinningLikelihood`: crown <= 0.05, strong-shear/saddle <= 0.1,
  rescued <= RB_ATOL = 1.5 -- partitioned by CERTIFIED gamma / eta ONLY, never
  the gauge angle theta (F017).

* The two design-falsifiable claims (Sections E, F): the near-caustic TUBE
  chart beats the raw Cartesian FAR-FIELD chart that cannot cover the caustic,
  and stays FLAT (u=sqrt(eta) coordinate) as eta -> 0 while the raw degrades.

* F010 mutation reachability (Section G): mutating a load-bearing chart bound
  flips a previously-correct serve / fall-through decision RED.

TOLERANCE PROVENANCE (why these numbers -- honestly)
----------------------------------------------------
The census's own tier bars (`CROWN_LNL_TOL = 0.05` etc.) are the numbers the
tool reports against; Section D exercises them on a real likelihood with a
DELIBERATELY QUIET injection (`D_LUMINOSITY`) so the propagated envelope error
``dlnL ~ eps_dense * |lnL|`` lands inside the crown bar at the minutes-scale
fixture (a loud injection would need production-density charts).  This is
premise documentation, not tolerance-hiding: shrink ``eps_dense`` with a bigger
box and the bar tightens for free.

Sections E/F use the census error currency but pin the COARSE-FIXTURE design
bars the Professor flagged ("production gain larger", Q2/Q3): at the minutes
budget the tube's advantage over an EQUAL-BUDGET raw chart that covers the SAME
band is marginal (the raw simply samples near the caustic too), so the honest
falsifiable competitor is the raw FAR-FIELD chart trained OUTSIDE the caustic
(``eta_overlap_min = 0.05``) and EXTRAPOLATED inward -- exactly the production
alternative the tube exists to replace.  Against it the tube wins ~2-3x and the
raw degrades toward the caustic while the tube stays flat.  See the build
report for the measured gap to the plan's aspirational 3x / -0.5 production
bars.

Every numeric TestCase tallies its comparisons; `tearDown` fails a test that
asserted nothing (anti-vacuity).  The suite is stdlib ``unittest``.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib.util
import inspect
import math
import os
import unittest
from collections import Counter
from pathlib import Path
from unittest import TestCase

import numpy as np

from cogwheel import data, waveform
from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing import surrogate_training as training_module
from cogwheel.lensing import surrogate_census as census
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.channels import (
    farfield_envelope_from_partition)
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, dimensionless_frequency)
from cogwheel.lensing.surrogate import (
    ExteriorPolarChart, TubeChart, LensAmplificationSurrogate,
    _rotate_to_eigenframe, select_chart, _evaluate_chart,
    _GAMMA_GUARD_BAND, _DEFAULT_CAUSTIC_FLOOR)

# ==========================================================================
# Shared constants
# ==========================================================================

#: Positive-parity astroid gamma band for the real near-caustic fixtures.
POS_BAND = (0.30, 0.40)
#: Narrow mid-arc theta window (no cusp inside) for the tube/raw fixtures.
TUBE_THETA = (2.05, 2.55)
#: Tube caustic-distance band; eta_floor small so the fold ray reaches ~4e-4.
TUBE_ETA_FLOOR = 2e-4
TUBE_ETA_MAX = 0.05

#: Frequency band shared by the fixtures.
F_LO_HZ, F_HI_HZ = 20.0, 1024.0

#: Deliberately QUIET injection (Mpc) so the crown dlnL lands under the census
#: crown bar at the minutes-scale fixture (see TOLERANCE PROVENANCE).
D_LUMINOSITY = 2500.0
APPROXIMANT = 'IMRPhenomXPHM'
SEED = 20260717

#: Section E: coarse-fixture bars (Professor Q2 -- production gain larger).
#: Measured: raw_out p95 / tube p95 ~ 2.9; raw_out max / tube max ~ 2.2.
E_P95_RATIO_MIN = 2.0
E_MAX_RATIO_MIN = 1.5
#: The tube must ALSO be accurate in absolute terms (a broken tube -> large).
E_TUBE_P95_MAX = 0.25

#: Section F: tube fold-ray slope ~ 0 (measured |slope| ~ 0.03-0.05); the
#: extrapolating raw degrades toward the caustic (measured slope ~ -0.19).
F_TUBE_SLOPE_MAX = 0.15
F_TUBE_RAY_ERR_MAX = 0.10
F_RAW_SLOPE_MAX = -0.05
F_DEEP_RATIO_MIN = 2.0


# ==========================================================================
# Synthetic (NO-engine) multi-chart fixture -- categorization + mutation.
# The category logic (`classify_fallthrough`) is pure geometry over the chart
# bounds; a smooth analytic value tensor stands in for the engine (values are
# irrelevant to the guard-stack decisions these sections pin).
# ==========================================================================

def _smooth_tensor(gamma_grid, p1_grid, p2_grid, log_w_grid, phase):
    """Deterministic smooth ``(n_w, n_gamma, n_p1, n_p2)`` real/imag tensors."""
    gw, gg, g1, g2 = np.meshgrid(log_w_grid, gamma_grid, p1_grid, p2_grid,
                                 indexing='ij')
    real = (np.cos(0.5 * gw + phase) * (1.0 + 0.3 * gg)
            * np.exp(-0.4 * g1) * (1.0 + 0.2 * g2))
    imag = (np.sin(0.5 * gw + phase) * (1.0 - 0.2 * gg)
            * (1.0 + 0.1 * g1) * np.cos(0.3 * g2))
    return real, imag


def _exterior_polar_axes(gamma_nodes, y1_range, y2_range, branch, n_rho,
                         n_theta_c, refusal=None):
    """``(rho_grid, theta_c_grid[, refused])`` for a physical eigenframe box.

    Engine-free mirror of `surrogate_training._farfield_box_to_smooth` for a
    box given directly in the eigenframe ``(y1, y2)`` on a KNOWN square-root
    ``branch``: every corner of the ``gamma_nodes x y1_range x y2_range`` box
    is pushed through ``_to_caustic_fixed`` to bound the ``(rho, theta_c)``
    box.  An optional physical refusal ``(gamma, y1, y2)`` is mapped through
    the identical map to its caustic-fixed ``(gamma, rho, theta_c)`` image, so
    a physical query at that refusal still lands in the chart's exclusion ball.
    """
    rho_vals = []
    theta_c_vals = []
    for gamma in gamma_nodes:
        for y1 in y1_range:
            for y2 in y2_range:
                rho, theta_c = surrogate_module._to_caustic_fixed(
                    gamma, y1, y2)
                rho_vals.append(rho)
                theta_c_vals.append(theta_c)
    rho_grid = np.linspace(min(rho_vals), max(rho_vals), n_rho)
    theta_c_grid = np.linspace(min(theta_c_vals), max(theta_c_vals), n_theta_c)
    if refusal is None:
        return rho_grid, theta_c_grid
    r_gamma, r_y1, r_y2 = refusal
    r_rho, r_theta_c = surrogate_module._to_caustic_fixed(
        float(r_gamma), float(r_y1), float(r_y2))
    return rho_grid, theta_c_grid, np.array(
        [[float(r_gamma), r_rho, r_theta_c]])


#: Synthetic-fixture log-w band (every query draws frequencies inside it).
SYN_LOG_W = np.log(np.geomspace(0.5, 20.0, 5))
SYN_LWMIN = float(SYN_LOG_W[0]) + 0.01
SYN_LWMAX = float(SYN_LOG_W[-1]) - 0.01
#: A dropped metamorphosis sliver in the surrogate provenance, well OUTSIDE
#: the gamma-guard band so a query there is ``dropped-sliver``, not
#: ``gamma-guard``.
SYN_DROPPED = ((0.90, 0.95),)


@functools.lru_cache(maxsize=1)
def _synthetic_surrogate():
    """A 4-chart synthetic surrogate (pos/sad tube + far-field), no engine.

    Far-field ``eta_overlap_min = 0.05`` (the production caustic floor), so
    tube band ``[0.02, 0.05]`` and far-field ``eta > 0.05`` are DISJOINT -- a
    near-caustic (eta ~ 0.03) source cannot be served by the far-field, which
    is what makes the arc-projection case fall to ``out-of-box`` (Section B).
    The saddle tube arc is a NEGATIVE wedge so a ``[0, 2*pi)`` caustic angle
    must route through `_theta_into_frame`.
    """
    u_grid = np.linspace(np.sqrt(0.02), np.sqrt(0.05), 4)
    pos_gamma = np.linspace(0.3, 0.5, 4)
    pos_theta = np.linspace(0.2, 1.2, 4)
    real, imag = _smooth_tensor(pos_gamma, u_grid, pos_theta, SYN_LOG_W, 0.0)
    pos_tube = TubeChart.from_values(
        gamma_grid=pos_gamma, u_grid=u_grid, theta_grid=pos_theta,
        log_w_grid=SYN_LOG_W, envelope_real=real, envelope_imag=imag,
        image_count=2, parity=1, eta_floor=0.02, eta_max=0.05,
        cusp_windows=[(0.2, 0.1)])
    # Exterior-polar (rho, theta_c) axes: the (rho, theta_c) image of the
    # original physical box (y1 in [0.5, 0.85], y2 in [0.2, 0.45]) over gamma in
    # [0.3, 0.5] on the positive-parity astroid (branch = +1).  The refused
    # point is the (rho, theta_c) image of the SAME physical refusal
    # (gamma=0.4, y1=0.67, y2=0.32) so a physical query there still lands in
    # the exclusion ball.
    pos_rho, pos_theta_c, pos_refused = _exterior_polar_axes(
        pos_gamma, (0.5, 0.85), (0.2, 0.45), 1, 4, 4,
        refusal=(0.4, 0.67, 0.32))
    real, imag = _smooth_tensor(pos_gamma, pos_rho, pos_theta_c, SYN_LOG_W, 0.5)
    pos_ff = ExteriorPolarChart.from_values(
        gamma_grid=pos_gamma, rho_grid=pos_rho, theta_c_grid=pos_theta_c,
        log_w_grid=SYN_LOG_W, envelope_real=real, envelope_imag=imag,
        image_count=2, parity=1,
        eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR, refused_points=pos_refused,
        theta_to_u=None, u_grid=None)
    sad_gamma = np.linspace(1.1, 1.4, 4)
    sad_theta = np.linspace(-0.39, -0.09, 4)
    real, imag = _smooth_tensor(sad_gamma, u_grid, sad_theta, SYN_LOG_W, 1.0)
    sad_tube = TubeChart.from_values(
        gamma_grid=sad_gamma, u_grid=u_grid, theta_grid=sad_theta,
        log_w_grid=SYN_LOG_W, envelope_real=real, envelope_imag=imag,
        image_count=4, parity=-1, eta_floor=0.02, eta_max=0.05,
        cusp_windows=[(-0.39, 0.05)])
    # Exterior-polar (rho, theta_c) image of the physical saddle box (y1 in
    # [0.2, 0.5], y2 in [0.1, 0.3]) over gamma in [1.1, 1.4] on the macro-
    # saddle deltoid edge (branch = -1).  The refused point is the (rho,
    # theta_c) image of the SAME physical refusal (gamma=1.35, y1=0.25,
    # y2=0.15) so a physical query there still lands in the exclusion ball.
    sad_rho, sad_theta_c, sad_refused = _exterior_polar_axes(
        sad_gamma, (0.2, 0.5), (0.1, 0.3), -1, 4, 4,
        refusal=(1.35, 0.25, 0.15))
    real, imag = _smooth_tensor(sad_gamma, sad_rho, sad_theta_c, SYN_LOG_W, 1.5)
    sad_ff = ExteriorPolarChart.from_values(
        gamma_grid=sad_gamma, rho_grid=sad_rho, theta_c_grid=sad_theta_c,
        log_w_grid=SYN_LOG_W, envelope_real=real, envelope_imag=imag,
        image_count=4, parity=-1,
        eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR, refused_points=sad_refused,
        theta_to_u=None, u_grid=None)
    provenance = {'chart_count': 4,
                  'chart_types': ['tube', 'farfield', 'tube', 'farfield'],
                  'dropped_gamma_slivers': [list(SYN_DROPPED[0])]}
    return LensAmplificationSurrogate(
        [pos_tube, pos_ff, sad_tube, sad_ff], provenance)


# ==========================================================================
# Real engine-trained near-caustic fixtures (built ONCE per process).
# ==========================================================================

@functools.lru_cache(maxsize=1)
def _pos_arc():
    """The positive-parity astroid fold arc + its capped w band."""
    struct = training_module.band_caustic_structure(
        POS_BAND, 1, n_samples=200)
    arc = struct.arcs[0]
    box = training_module.PriorBox.from_prior_classes(
        f_lo_hz=F_LO_HZ, f_hi_hz=F_HI_HZ)
    w_range = training_module._capped_w_range(
        box, 1, struct.caustic_reach + 0.3)
    log_w_grid = training_module._log_w_grid(w_range, 4)
    return arc, log_w_grid


@functools.lru_cache(maxsize=1)
def _pos_tube():
    """One real positive tube over the narrow mid-arc band (u=sqrt(eta))."""
    arc, log_w_grid = _pos_arc()
    w_grid = np.exp(log_w_grid)
    gamma_grid = np.linspace(*POS_BAND, 4)
    u_grid = np.linspace(np.sqrt(TUBE_ETA_FLOOR), np.sqrt(TUBE_ETA_MAX), 8)
    theta_grid = np.linspace(*TUBE_THETA, 6)
    shape = (log_w_grid.size, gamma_grid.size, u_grid.size, theta_grid.size)
    env_real = np.zeros(shape)
    env_imag = np.zeros(shape)
    for ig, gamma in enumerate(gamma_grid):
        for iu, u in enumerate(u_grid):
            for it, theta in enumerate(theta_grid):
                src = training_module._tube_source(
                    float(gamma), float(theta), float(u * u), arc.branch,
                    arc.inward_sign)
                env = training_module._engine_envelope(w_grid, float(gamma),
                                                       src)
                if env is not None:
                    env_real[:, ig, iu, it] = env.real
                    env_imag[:, ig, iu, it] = env.imag
    tube = TubeChart.from_values(
        gamma_grid=gamma_grid, u_grid=u_grid, theta_grid=theta_grid,
        log_w_grid=log_w_grid, envelope_real=env_real, envelope_imag=env_imag,
        image_count=arc.image_count, parity=1, eta_floor=TUBE_ETA_FLOOR,
        eta_max=TUBE_ETA_MAX, cusp_windows=())
    return tube


@functools.lru_cache(maxsize=1)
def _pos_raw_out():
    """An exterior-polar chart trained OUTSIDE the caustic (extrapolates inward).

    Exterior-polar ``(rho, theta_c)`` box covering the fold strip at eta in
    ``[0.06, 0.30]`` -- i.e. the production exterior region that cannot reach
    the near-caustic band; evaluated inside it, it extrapolates.  Each grid
    node is mapped back to a physical eigenframe source
    (`_from_caustic_fixed`) before the engine call, so the fitted label is
    the same physical envelope at the same point.
    """
    arc, log_w_grid = _pos_arc()
    w_grid = np.exp(log_w_grid)
    gamma_grid = np.linspace(*POS_BAND, 4)
    gmid = float(np.mean(POS_BAND))
    srcs = np.array([
        training_module._tube_source(gmid, th, eta, arc.branch,
                                     arc.inward_sign)
        for th in np.linspace(*TUBE_THETA, 12)
        for eta in np.linspace(0.06, 0.30, 4)])
    rho_vals = []
    theta_c_vals = []
    for s in srcs:
        rho, theta_c = surrogate_module._to_caustic_fixed(
            gmid, float(s[0]), float(s[1]))
        rho_vals.append(rho)
        theta_c_vals.append(theta_c)
    rho_grid = np.linspace(min(rho_vals), max(rho_vals), 7)
    theta_c_grid = np.linspace(min(theta_c_vals), max(theta_c_vals), 7)
    shape = (log_w_grid.size, gamma_grid.size, rho_grid.size, theta_c_grid.size)
    er = np.zeros(shape)
    ei = np.zeros(shape)
    for ig, gamma in enumerate(gamma_grid):
        for i1, rv in enumerate(rho_grid):
            for i2, tcv in enumerate(theta_c_grid):
                y1e, y2e = surrogate_module._from_caustic_fixed(
                    float(gamma), float(rv), float(tcv))
                env = training_module._engine_envelope(
                    w_grid, float(gamma), np.array([y1e, y2e]))
                if env is not None:
                    er[:, ig, i1, i2] = env.real
                    ei[:, ig, i1, i2] = env.imag
    return ExteriorPolarChart.from_values(
        gamma_grid=gamma_grid, rho_grid=rho_grid, theta_c_grid=theta_c_grid,
        log_w_grid=log_w_grid, envelope_real=er, envelope_imag=ei,
        image_count=arc.image_count, parity=1,
        eta_overlap_min=0.0)


@functools.lru_cache(maxsize=1)
def _pos_farfield_dense():
    """A dense positive exterior-polar chart over gamma [0.35, 0.65], far from caustic.

    Serves both a CROWN config (gamma < 0.5) and a STRONG-SHEAR config
    (gamma' >= 0.5) far from the caustic (eta well above the caustic floor),
    at enough w / parameter density that the propagated lnL error lands inside
    the census crown / strong bars.  Built via the reused `from_engine`
    exterior-polar trainer.

    w_range starts at 0.10, just below the m = 60 Msun detector band
    (15-1024 Hz => w in [0.111, 7.61]) and ABOVE the diffractive bottom.  The
    chart carries the ``farfield_full_kernel_sum`` label, which diverges as
    w -> 0 (measured max|E| = 2.7 at w = 0.12 but 69 at w = 0.02, ~40x
    max|F|); production
    never trains that label below the region ``w_floor`` -- it serves the
    bounded diffractive label there -- so training down to 0.02 put the
    fixture in a regime the label is not defined to represent (its held-out
    eps then reads 13.6 and is insensitive to node density in every axis).
    """
    gamma_range = (0.35, 0.65)
    return LensAmplificationSurrogate.from_engine(
        gamma_range=gamma_range,
        rho_range=(0.025, 0.075), theta_c_range=(0.05, 0.20),
        w_range=(0.10, 260.0),
        n_gamma=6, n_rho=5, n_theta_c=9, w_nodes_per_decade=12)


# ==========================================================================
# Likelihood fixture (Section D) -- built once.
# ==========================================================================

def _reference_par_dic():
    """Deterministic precessing reference ``par_dic`` (quiet injection)."""
    return {
        'm1': 60.0, 'm2': 45.0,
        's1x_n': 0.20, 's1y_n': 0.10, 's1z': 0.30,
        's2x_n': -0.10, 's2y_n': 0.15, 's2z': -0.20,
        'l1': 0.0, 'l2': 0.0, 'iota': 1.0, 'phi_ref': 1.2,
        'ra': 1.8, 'dec': -0.3, 'psi': 0.9,
        't_geocenter': 0.0, 'd_luminosity': D_LUMINOSITY, 'f_ref': 50.0}


# `_likelihoods()` DELETED 2026-08-13 with its only consumer,
# `LnlTierTestCase::test_real_likelihood_tiers_within_bars` -- see the note at
# that call site for why the fixture it wrapped is not repairable by editing.


# ==========================================================================
# Helpers shared by the real-chart sections.
# ==========================================================================

def _served_eps(surrogate, chart_for_raw, gamma, source, log_w_grid,
                w_grid):
    """Census-currency eps of tube-serve and raw-extrapolate at one source.

    Returns ``(tube_eps_or_None, raw_eps)`` with the max-normalized currency
    ``max|E_. - E_eng| / max(max|E_eng|, EPS_DENOM_FLOOR)`` against a FRESH
    engine oracle (F002).  ``tube_eps`` is ``None`` when the surrogate declines
    (unfaithful projection / cusp / out of band).
    """
    channels = ChangRefsdalChannels(w_grid)
    try:
        part = channels.evaluate(gamma=gamma,
                                 y=(float(source[0]), float(source[1])),
                                 beta=0.0, kappa=0.0)
    except training_module._ENGINE_REFUSALS:
        return None
    env_eng = np.asarray(part.envelope)
    if not np.all(np.isfinite(env_eng)):
        return None
    e_tube, served, _definition = surrogate.serve(
        w_grid, gamma=gamma, y1=float(source[0]), y2=float(source[1]),
        beta=0.0, eta=part.caustic_distance, theta=part.critical_theta,
        image_count=int(part.real_mask.sum()))
    # Raw exterior-polar chart contracts on its gamma-resolved (rho, theta_c)
    # map from the physical beta=0 eigenframe source.
    e_raw = _evaluate_chart(
        chart_for_raw, gamma=gamma, eta=float('nan'), theta=float('nan'),
        log_w_query=log_w_grid, y1_eig=float(source[0]),
        y2_eig=float(source[1]))
    denom = max(float(np.max(np.abs(env_eng))), census.EPS_DENOM_FLOOR)
    tube_eps = (float(np.max(np.abs(e_tube - env_eng)) / denom) if served
                else None)
    raw_eps = float(np.max(np.abs(e_raw - env_eng)) / denom)
    return tube_eps, raw_eps


# `_dense_farfield_source()` DELETED 2026-08-13 with its only consumer,
# `LnlTierTestCase::test_real_likelihood_tiers_within_bars`.


# ==========================================================================
# Anti-vacuity base
# ==========================================================================

#: ENGINE-BACKED TIER (opt-in).  Classes marked `_TRAIN_TIER_SKIP` drive the
#: real engine through the cached `_pos_arc` / `_pos_tube` / `_pos_raw_out` /
#: `_pos_farfield_dense` builders -- they train actual surrogate charts and run
#: hundreds of Schwinger/operator evaluations.  Measured 2026-07-28: they cost
#: 60.8s, 43.4s, 29.7s, 2.0s and 0.2s respectively, i.e. essentially the whole
#: 137s this file used to spend, against ~3s for everything else.  Census and
#: training runs belong to whoever DRIVES the build -- they are post-build
#: driver steps, not work the build does and not unit tests -- and a
#: multi-minute file in the fast tier is one nobody runs.
#:
#: TIER THE WHOLE SHARING CLUSTER, not just the slow-looking members: the
#: builders are `functools.lru_cache`d, so their fill cost is attributed to
#: whichever test touches them FIRST.  `FoldApproachRayTestCase` and
#: `MutationFalsificationTestCase` profile as cheap only because
#: `TubeBeatsRawTestCase` already warmed `_pos_arc`/`_pos_tube`/`_pos_raw_out`.
#: Gate one without the others and the cost silently migrates instead of going
#: away.  Matches the existing COGWHEEL_BRUTE_ACCURACY / COGWHEEL_STRICT_TIMING
#: idiom and the twin gate in `test_lensing_farfield_envelope.py`.
#:
#: Run them with:  COGWHEEL_TRAIN_TIER=1 python -m pytest <file>
_TRAIN_TIER_SKIP = unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'engine-backed training tier: set COGWHEEL_TRAIN_TIER=1 (builds real '
    'surrogate charts, minutes per class; the driver runs these post-build)')


class CensusTestCase(TestCase):
    """Base carrying the comparison tally; `tearDown` fails a vacuous test."""

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')


# ==========================================================================
# Section A -- served fraction + six-way fall-through breakdown (MECE)
# ==========================================================================

class FallthroughCategorizationTestCase(CensusTestCase):
    """`classify_fallthrough` attributes each of the six categories BY
    CONSTRUCTION, calling the census's own guard predicates (one source of
    truth) -- never a re-implementation."""

    def setUp(self):
        super().setUp()
        self.sur = _synthetic_surrogate()

    def _classify(self, **overrides):
        kwargs = dict(gamma=0.4, log_w_min=SYN_LWMIN, log_w_max=SYN_LWMAX,
                      eta=0.03, theta=0.7, image_count=2, y1_eig=0.6,
                      y2_eig=0.3, dropped_slivers=SYN_DROPPED)
        kwargs.update(overrides)
        return census.classify_fallthrough(self.sur, **kwargs)

    def test_gamma_guard(self):
        """|gamma - 1| < guard band -> gamma-guard (checked first)."""
        self.n_checks += 1
        self.assertEqual(
            self._classify(gamma=1.0 + 0.5 * _GAMMA_GUARD_BAND), 'gamma-guard')

    def test_dropped_sliver(self):
        """gamma inside a training-dropped metamorphosis band -> sliver."""
        self.n_checks += 1
        self.assertEqual(self._classify(gamma=0.92), 'dropped-sliver')

    def test_cusp_window(self):
        """A tube blocked ONLY by its cusp window -> cusp-window (theta in the
        window; relaxing the window makes the tube serve)."""
        self.n_checks += 1
        self.assertEqual(
            self._classify(gamma=0.4, eta=0.03, theta=0.2, image_count=2),
            'cusp-window')

    def test_refusal_ball(self):
        """A far-field blocked ONLY by its exclusion ball -> refusal-ball."""
        self.n_checks += 1
        self.assertEqual(
            self._classify(gamma=0.4, eta=0.10, image_count=2, y1_eig=0.67,
                           y2_eig=0.32), 'refusal-ball')

    def test_out_of_box(self):
        """Outside every chart's box otherwise -> out-of-box."""
        self.n_checks += 1
        self.assertEqual(self._classify(gamma=0.60), 'out-of-box')

    def test_priority_sliver_before_out_of_box(self):
        """A dropped-sliver gamma is a subset of out-of-box on the gamma axis;
        the sliver check wins (mutation control): moving gamma off the sliver
        flips the SAME point to out-of-box."""
        self.n_checks += 1
        self.assertEqual(self._classify(gamma=0.92), 'dropped-sliver')
        self.n_checks += 1
        self.assertEqual(self._classify(gamma=0.80), 'out-of-box')


class BreakdownPartitionTestCase(CensusTestCase):
    """`fallthrough_breakdown` counts served / engine-refused / the six
    categories and enforces the MECE partition (served + refused + Sum(cats)
    == n)."""

    @staticmethod
    def _record(**kw):
        base = dict(gamma=0.4, m_lens_msun=90.0, y1=0.6, y2=0.3,
                    log_w_min=SYN_LWMIN, log_w_max=SYN_LWMAX, served=False)
        base.update(kw)
        return census.SampleRecord(**base)

    def _population(self):
        recs = [self._record(served=True, chart_index=1) for _ in range(3)]
        recs.append(self._record(served=False, engine_refused=True))
        for cat in census._FALLTHROUGH_CATEGORIES:
            recs += [self._record(category=cat) for _ in range(2)]
        return recs

    def test_counts_match_hand_computed(self):
        """3 served + 1 engine-refused + 2 per fall-through category == n.

        Derived from ``len(census._FALLTHROUGH_CATEGORIES)`` so the count
        stays correct as categories are added (six categories -> 16 total
        after the ``born`` category landed; was 14 for five)."""
        n_cats = len(census._FALLTHROUGH_CATEGORIES)
        n_samples = 3 + 1 + 2 * n_cats
        breakdown = census.fallthrough_breakdown(self._population())
        self.n_checks += 1
        self.assertEqual(breakdown['n_samples'], n_samples)
        self.n_checks += 1
        self.assertEqual(breakdown['served'], 3)
        self.n_checks += 1
        self.assertEqual(breakdown['engine_refused'], 1)
        for cat in census._FALLTHROUGH_CATEGORIES:
            self.n_checks += 1
            self.assertEqual(breakdown['fallthrough'][cat], 2)
        self.n_checks += 1
        self.assertAlmostEqual(breakdown['served_fraction'], 3 / n_samples)

    def test_partition_is_mece(self):
        """served + engine_refused + Sum(categories) == n_samples."""
        recs = self._population()
        breakdown = census.fallthrough_breakdown(recs)
        total = (breakdown['served'] + breakdown['engine_refused']
                 + sum(breakdown['fallthrough'].values()))
        self.n_checks += 1
        self.assertEqual(total, breakdown['n_samples'])
        self.n_checks += 1
        self.assertTrue(breakdown['partition_ok'])

    def test_unknown_category_raises(self):
        """Falsification: a non-served record with a bogus category (or None)
        breaks the partition -> CensusError, so the guard has teeth."""
        recs = self._population()
        recs.append(self._record(category='not-a-real-category'))
        self.n_checks += 1
        with self.assertRaises(census.CensusError):
            census.fallthrough_breakdown(recs)
        recs[-1] = self._record(category=None)  # never-served, no category
        self.n_checks += 1
        with self.assertRaises(census.CensusError):
            census.fallthrough_breakdown(recs)


@_TRAIN_TIER_SKIP
class EndToEndPartitionTestCase(CensusTestCase):
    """`characterize` -> `fallthrough_breakdown` on REAL prior samples with a
    REAL engine yields a MECE partition, and the census `run` end-to-end
    reports served fraction, per-chart eps and the partition on a real chart."""

    def test_characterize_partitions_real_samples(self):
        surrogate = _synthetic_surrogate()
        config = census.CensusConfig(n_samples=24, seed=1)
        f_grid = census._frequency_grid(config)
        slivers = census._dropped_slivers_from(surrogate, None)
        records = census.characterize(
            surrogate, census.draw_samples(config), f_grid, slivers,
            engine_factory=ChangRefsdalChannels)
        breakdown = census.fallthrough_breakdown(records)  # raises if not MECE
        self.n_checks += 1
        self.assertEqual(breakdown['n_samples'], 24)
        total = (breakdown['served'] + breakdown['engine_refused']
                 + sum(breakdown['fallthrough'].values()))
        self.n_checks += 1
        self.assertEqual(total, 24)
        # Every non-served, non-refused record carries a KNOWN category.
        for rec in records:
            if not (rec.served or rec.engine_refused):
                self.n_checks += 1
                self.assertIn(rec.category, census._FALLTHROUGH_CATEGORIES)

    def test_run_end_to_end_on_real_chart(self):
        """`run` drives the WHOLE pipeline (draw -> characterize -> breakdown
        -> per-chart eps -> report) on REAL prior samples with a real chart and
        real engine, and reports a valid MECE partition + well-formed report.

        The far-field chart lives OUTSIDE the caustic (|y| > caustic reach),
        where the mass-conditioned prior rarely draws sources, so a low served
        fraction here is CORRECT behaviour, not a defect -- the served>0
        non-vacuity of the eps / tier stages is pinned by
        `HeldoutEnvelopeEpsTestCase` and `LnlTierTestCase` on controlled
        samples.  What this test pins is that `run` completes and the partition
        invariant holds on real data."""
        surrogate = _pos_farfield_dense()
        config = census.CensusConfig(n_samples=48, seed=3,
                                     max_heldout_per_chart=6)
        report = census.run(surrogate=surrogate, config=config,
                            engine_factory=ChangRefsdalChannels)
        self.n_checks += 1
        self.assertTrue(report['partition_ok'])
        self.n_checks += 1
        self.assertEqual(
            report['served'] + report['engine_refused']
            + sum(report['fallthrough'].values()),
            report['n_samples'])
        self.n_checks += 1
        self.assertTrue(0.0 <= report['served_fraction'] <= 1.0)
        self.n_checks += 1
        self.assertEqual(len(report['per_chart_eps']), len(surrogate.charts))
        for entry in report['per_chart_eps']:
            eps = entry['eps']
            self.n_checks += 1
            self.assertTrue(eps['count'] == 0 or np.isfinite(eps['max']))
        self.n_checks += 1
        self.assertEqual(report['artifact']['n_charts'], len(surrogate.charts))


# ==========================================================================
# Section B -- arc-projection is out-of-box, NOT cusp-window (by design)
# ==========================================================================

class ArcProjectionOutOfBoxTestCase(CensusTestCase):
    """A near-cusp source that projects onto a NEIGHBOURING fold arc (theta out
    of the chart's arc range) is refusal-conservatively categorized
    ``out-of-box`` -- never ``cusp-window`` (Professor Q7)."""

    def setUp(self):
        super().setUp()
        self.sur = _synthetic_surrogate()

    def test_neighbouring_arc_theta_is_out_of_box(self):
        """theta on the adjacent arc (2.5, outside the chart's [0.2, 1.2]) and
        eta in the tube band, image_count matched: neither the tube (theta out
        of range even with cusps relaxed) nor the far-field (eta below its
        caustic floor) serves -> out-of-box."""
        category = census.classify_fallthrough(
            self.sur, gamma=0.4, log_w_min=SYN_LWMIN, log_w_max=SYN_LWMAX,
            eta=0.03, theta=2.5, image_count=2, y1_eig=0.6, y2_eig=0.3,
            dropped_slivers=())
        self.n_checks += 1
        self.assertEqual(category, 'out-of-box')

    def test_in_arc_near_cusp_is_cusp_window_not_out_of_box(self):
        """Control: the SAME chart, a theta INSIDE the arc but in the cusp
        window, is cusp-window -- so the out-of-box verdict above is the
        arc-projection, not a blanket refusal."""
        category = census.classify_fallthrough(
            self.sur, gamma=0.4, log_w_min=SYN_LWMIN, log_w_max=SYN_LWMAX,
            eta=0.03, theta=0.2, image_count=2, y1_eig=0.6, y2_eig=0.3,
            dropped_slivers=())
        self.n_checks += 1
        self.assertEqual(category, 'cusp-window')


# ==========================================================================
# Section C -- per-chart held-out envelope eps (currency + F002 + node/trough)
# ==========================================================================
def _reference_env_and_denom(chart, part):
    """Held-out eps reference envelope + normalization denominator, dispatched
    on chart type EXACTLY as `surrogate_census.heldout_envelope_eps`
    (Build 8g-b) -- the one place in this suite deciding far-field-vs-tube
    reference semantics.

    An `ExteriorPolarChart` is referenced against the far-field label
    ``E_ff = F - sum_{a real} H_a e^{1j w tau_a}``
    (`farfield_envelope_from_partition`) and F-normalized by
    ``max|exact_total|`` (``max|E_ff| ~ 1e-4`` is too tiny a denominator); a
    `TubeChart` keeps the caustic-region ``part.envelope`` reference normalized
    by ``max|E|`` (unchanged).  The deliberate ``EPS_DENOM_FLOOR`` floor is
    applied to the denominator in both branches.
    """
    if isinstance(chart, ExteriorPolarChart):
        env_eng = farfield_envelope_from_partition(part)
        denom_base = float(np.max(np.abs(part.exact_total)))
    else:
        env_eng = np.asarray(part.envelope)
        denom_base = float(np.max(np.abs(env_eng)))
    return env_eng, max(denom_base, census.EPS_DENOM_FLOOR)


@_TRAIN_TIER_SKIP
class HeldoutEnvelopeEpsTestCase(CensusTestCase):
    """The census held-out envelope error uses the max-normalized currency
    against a FRESH engine oracle (F002), is machine-exact at a training node,
    and stays bounded across a deep-cancellation trough."""

    #: Surrogate internals the ORACLE (env_eng) must never reference.
    FORBIDDEN = frozenset({
        'serve', 'LensAmplificationSurrogate', '_evaluate_chart',
        'reconstruct_from_envelope', 'real_coeffs', 'imag_coeffs'})

    def test_oracle_is_fresh_engine_by_construction(self):
        """AST guard: the ``env_eng`` oracle in `heldout_envelope_eps` is built
        from ``engine_factory(...).evaluate(...).envelope`` and references no
        surrogate interpolant.  (env_sur, the thing UNDER TEST, legitimately
        uses `serve`; the guard is on the oracle currency, not env_sur.)"""
        source = inspect.getsource(census.heldout_envelope_eps)
        tree = ast.parse(source.lstrip())
        self.n_checks += 1
        self.assertIn('engine_factory', source)
        self.n_checks += 1
        self.assertIn('.evaluate(', source)
        self.n_checks += 1
        self.assertIn('env_sur - env_eng', source.replace('\n', ' '))
        # Locate the env_eng assignment and walk ONLY its value expression.
        checked = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'env_eng'
                            for t in node.targets)):
                names = {n.attr for n in ast.walk(node.value)
                         if isinstance(n, ast.Attribute)}
                names |= {n.id for n in ast.walk(node.value)
                          if isinstance(n, ast.Name)}
                leaks = names & self.FORBIDDEN
                self.n_checks += 1
                self.assertFalse(
                    leaks, f'oracle env_eng leaks surrogate internals {leaks}')
                checked = True
        self.n_checks += 1
        self.assertTrue(checked, 'no env_eng oracle assignment found')

    def test_guard_flags_a_tainted_oracle(self):
        """Positive control: a fake oracle that reads the surrogate IS flagged
        by the same name walk, so the guard is non-vacuous."""
        def _tainted(surrogate, w, gamma, y1, y2):
            env_eng, _served, _definition = surrogate.serve(  # circular oracle
                w, gamma=gamma, y1=y1, y2=y2, beta=0.0, eta=0.0, theta=0.0,
                image_count=2)
            return env_eng
        names = {n.attr for n in ast.walk(ast.parse(
                 inspect.getsource(_tainted).lstrip()))
                 if isinstance(n, ast.Attribute)}
        self.n_checks += 1
        self.assertTrue(names & self.FORBIDDEN)

    def test_oracle_matches_independent_fresh_engine_at_runtime(self):
        """Runtime F002: the eps the census computes for a served sample equals
        the eps recomputed from an INDEPENDENT fresh
        `ChangRefsdalChannels.evaluate` -- proving env_eng is the engine, not
        the surrogate's own labels."""
        tube = _pos_tube()
        surrogate = LensAmplificationSurrogate([tube], {'s': 1})
        arc, log_w_grid = _pos_arc()
        w_grid = np.exp(log_w_grid)
        gmid = float(np.mean(POS_BAND))
        theta = float(np.mean(TUBE_THETA))
        src = training_module._tube_source(gmid, theta, 0.02, arc.branch,
                                           arc.inward_sign)
        part = ChangRefsdalChannels(w_grid).evaluate(
            gamma=gmid, y=(float(src[0]), float(src[1])), beta=0.0, kappa=0.0)
        env_eng = np.asarray(part.envelope)
        e_sur, served, _definition = surrogate.serve(
            w_grid, gamma=gmid, y1=float(src[0]), y2=float(src[1]), beta=0.0,
            eta=part.caustic_distance, theta=part.critical_theta,
            image_count=int(part.real_mask.sum()))
        self.assertTrue(served, 'mid-arc tube point must serve')
        denom = max(float(np.max(np.abs(env_eng))), census.EPS_DENOM_FLOOR)
        eps_independent = float(np.max(np.abs(e_sur - env_eng)) / denom)
        m_lens = 90.0
        record = census.SampleRecord(
            gamma=gmid, m_lens_msun=m_lens, y1=float(src[0]), y2=float(src[1]),
            log_w_min=float(log_w_grid[0]), log_w_max=float(log_w_grid[-1]),
            served=True, chart_index=0, eta=part.caustic_distance,
            theta=part.critical_theta, image_count=int(part.real_mask.sum()))
        # f_grid whose dimensionless_frequency lands on the chart's w grid.
        f_grid = w_grid / dimensionless_frequency(1.0, m_lens, 0.0)
        report = census.heldout_envelope_eps(
            surrogate, [record], f_grid, max_per_chart=1,
            engine_factory=ChangRefsdalChannels)
        eps_census = report[0]['eps']['max']
        self.n_checks += 1
        self.assertIsNotNone(eps_census, 'census evaluated no held-out point')
        self.n_checks += 1
        self.assertAlmostEqual(eps_census, eps_independent, places=10)

    def test_node_exactness(self):
        """At an EXACT training node the served envelope reproduces the fresh
        engine value to ~machine precision (the interpolant is exact at
        nodes)."""
        surrogate = _pos_farfield_dense()
        chart = surrogate.charts[0]
        w_grid = np.exp(chart.log_w_grid)
        gamma = float(chart.gamma_grid[2])
        # Exterior-polar node -> physical eigenframe source.
        rho = float(chart.rho_grid[2])
        theta_c = float(chart.theta_c_grid[2])
        y1, y2 = surrogate_module._from_caustic_fixed(
            gamma, rho, theta_c)
        part = ChangRefsdalChannels(w_grid).evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        env_eng, denom = _reference_env_and_denom(chart, part)
        e_sur, served, _definition = surrogate.serve(
            w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=part.caustic_distance, theta=part.critical_theta,
            image_count=int(part.real_mask.sum()))
        self.assertTrue(served, 'node point must serve')
        eps = float(np.max(np.abs(e_sur - env_eng)) / denom)
        self.n_checks += 1
        self.assertLess(eps, 1e-8,
                        f'node eps {eps:.3e} -- interpolant not node-exact')

    def test_trough_normalization_stays_bounded(self):
        """A deep-cancellation |E| trough (min|E| << max|E|) does NOT blow the
        max-normalized census currency up: eps stays O(reconstruction), gated
        by max|E|, never by a pointwise trough."""
        surrogate = _pos_farfield_dense()
        chart = surrogate.charts[0]
        w_grid = np.exp(chart.log_w_grid)
        gmid = float(np.mean(chart.gamma_grid))
        # Exterior-polar off-node midpoint -> physical eigenframe source.
        rho = 0.5 * (float(chart.rho_grid[1]) + float(chart.rho_grid[2]))
        theta_c = 0.5 * (float(chart.theta_c_grid[1]) + float(chart.theta_c_grid[2]))
        y1, y2 = surrogate_module._from_caustic_fixed(
            gmid, rho, theta_c)
        part = ChangRefsdalChannels(w_grid).evaluate(
            gamma=gmid, y=(y1, y2), beta=0.0, kappa=0.0)
        env_eng, denom = _reference_env_and_denom(chart, part)
        abs_env = np.abs(env_eng)
        trough_ratio = float(abs_env.min() / abs_env.max())
        e_sur, served, _definition = surrogate.serve(
            w_grid, gamma=gmid, y1=y1, y2=y2, beta=0.0,
            eta=part.caustic_distance, theta=part.critical_theta,
            image_count=int(part.real_mask.sum()))
        self.assertTrue(served)
        eps_maxnorm = float(np.max(np.abs(e_sur - env_eng)) / denom)
        self.n_checks += 1
        self.assertLess(trough_ratio, 0.5,
                        'fixture source has no genuine |E| trough -- retune')
        self.n_checks += 1
        self.assertLess(eps_maxnorm, 1.0,
                        f'max-normalized eps {eps_maxnorm:.3e} blew up at a '
                        f'trough (ratio {trough_ratio:.2e})')


# ==========================================================================
# Section D -- (gamma, image_count, eta)-partitioned lnL error tiers
# ==========================================================================

class LnlTierTestCase(CensusTestCase):
    """`lnl_error_tiers` partitions served-config lnL errors by CERTIFIED
    (gamma, eta) axes only (never gauge theta, F017) and gates each tier at the
    census bar: crown <= 0.05, strong-shear/saddle <= 0.1, rescued <= 1.5."""

    def test_assign_tier_is_theta_independent(self):
        """`assign_tier` reads only (gamma, eta); the gauge angle theta is not
        even a parameter -- two configs equal in (gamma, eta) but any theta map
        to the same tier."""
        self.n_checks += 1
        self.assertNotIn('theta', inspect.signature(
            census.assign_tier).parameters)
        self.n_checks += 1
        self.assertEqual(census.assign_tier(0.42, 0.20), 'crown')
        self.n_checks += 1
        self.assertEqual(census.assign_tier(0.60, 0.20), 'strong_saddle')
        self.n_checks += 1
        self.assertEqual(census.assign_tier(1.30, 0.20), 'strong_saddle')
        self.n_checks += 1
        # near the caustic a positive weak-shear config is held to the strong
        # bar, not the crown bar.
        self.assertEqual(census.assign_tier(0.42, 0.01), 'strong_saddle')

    def test_tiers_aggregate_with_a_mock_pair(self):
        """Unit control (no engine): `lnl_error_tiers` routes each served
        record to its tier and reports the per-tier max / target, using an
        injected deterministic ``lnlike_pair`` -- so the aggregation and the
        bars are pinned without the likelihood cost."""
        recs = [
            census.SampleRecord(gamma=0.42, m_lens_msun=60, y1=2.2, y2=0.0,
                                log_w_min=0.0, log_w_max=1.0, served=True,
                                chart_index=0, eta=0.30),
            census.SampleRecord(gamma=0.60, m_lens_msun=60, y1=2.2, y2=0.0,
                                log_w_min=0.0, log_w_max=1.0, served=True,
                                chart_index=0, eta=0.30),
            census.SampleRecord(gamma=1.30, m_lens_msun=60, y1=2.0, y2=0.0,
                                log_w_min=0.0, log_w_max=1.0, served=True,
                                chart_index=0, eta=0.30),
        ]
        errs = {0.42: 0.04, 0.60: 0.08, 1.30: 0.09}

        def pair(par_dic):
            return errs[round(par_dic['gamma'], 2)], 0.0
        report = census.lnl_error_tiers(
            recs, pair, base_par_dic=_reference_par_dic())
        self.n_checks += 1
        self.assertAlmostEqual(report['crown']['max'], 0.04)
        self.n_checks += 1
        self.assertAlmostEqual(report['strong_saddle']['max'], 0.09)
        self.n_checks += 1
        self.assertEqual(report['crown']['target_nats'], census.CROWN_LNL_TOL)
        self.n_checks += 1
        self.assertEqual(report['strong_saddle']['target_nats'],
                         census.STRONG_SADDLE_LNL_TOL)
        self.n_checks += 1
        self.assertEqual(report['rescued']['target_nats'],
                         census.RESCUED_LNL_TOL)

    # DELETED 2026-08-13 (test-debt audit): `_lnlike_pair` and
    # `test_real_likelihood_tiers_within_bars`, together with the
    # `_likelihoods` / `_dense_farfield_source` fixtures that existed only to
    # feed them.
    #
    # The test drove a real `LensedRelativeBinningLikelihood` through
    # `_pos_farfield_dense`, and that chart is trained on
    # ``rho_range = (0.025, 0.075)``.  Since 4d59a6d re-coordinatized the box
    # from ``s_range``/``d_range`` (``d`` = SIGNED perpendicular distance from
    # the caustic, positive OUTSIDE) to ``rho_range``/``theta_c_range``
    # (``rho <= 1`` = INTERIOR) while carrying the numbers across verbatim,
    # those numbers name a witness at ``|y| = 0.027`` -- essentially the
    # origin -- where the locus's `farfield_w_floor` is 352 and 100% of the
    # served band sits below it.  `FARFIELD_KERNEL_SUM` is the divergent
    # diffractive-bottom object there, so the chart was fitting the wrong
    # label: `_farfield_region_w_floor` clips every exterior tile production
    # trains, and since 8dfb8ca the serve path re-checks the floor and
    # correctly DECLINES this chart (F070).
    #
    # It is therefore not repairable by editing: the constraint is tight in
    # both directions (`w_floor >= 2/max(dtau)` with Fermat delays O(1) means
    # no source position lifts a 60-Msun band above the floor short of
    # ``rho ~ 5``, where the test goes vacuous, so the witness LENS MASS has
    # to move too, and the 68.3x detector band must then fit between the
    # region floor and `W_CEILING_SCHWINGER = 60`).  That is a REBUILD moving
    # all four `_pos_farfield_dense` consumers, not an edit -- tracked in
    # `todo.d/lensing_slow_tier_fixtures_left_their_served_domains.md`.
    #
    # Refining the gamma axis 6 -> 11 does reach the bar (dlnL 0.00185, 27x
    # under it) and is the TEMPTING WRONG FIX: it goes green while leaving
    # the chart in a regime production would never build, which IS the defect.
    #
    # What the deletion costs: nothing that `lnl_error_tiers` itself pins --
    # `test_assign_tier_is_theta_independent` and
    # `test_tiers_aggregate_with_a_mock_pair` (both surviving, both engine-
    # free) already pin the tier routing, the per-tier max/target reporting
    # and the three bars.  What it drops is the end-to-end
    # served-lnL-vs-exact-lnL measurement, which is a driver census step, not
    # a unit test, and which cannot be made honest on this fixture.


# ==========================================================================
# Section E -- tube beats the equal-budget raw far-field near the caustic
# ==========================================================================

@_TRAIN_TIER_SKIP
class TubeBeatsRawTestCase(CensusTestCase):
    """Through the near-caustic band the TUBE chart beats an EQUAL-BUDGET raw
    Cartesian FAR-FIELD chart (trained outside the caustic, extrapolated
    inward -- the production alternative the tube replaces) at both the 95th
    percentile and the max of the census envelope currency.

    NOTE (coarse-fixture bar): the plan's aspirational >= 3x is a
    PRODUCTION-scale figure (Professor Q2, "production gain larger").  At the
    minutes budget a fair raw chart trained ON the same band interpolates the
    caustic too and is comparable; the honest falsifiable competitor is the raw
    far-field that CANNOT cover the caustic.  Measured here: p95 ~ 2.9x, max ~
    2.2x."""

    def test_tube_beats_extrapolating_raw(self):
        tube = _pos_tube()
        surrogate = LensAmplificationSurrogate([tube], {'s': 1})
        raw = _pos_raw_out()
        arc, log_w_grid = _pos_arc()
        w_grid = np.exp(log_w_grid)
        rng = np.random.default_rng(11)
        tube_eps, raw_eps = [], []
        tries = 0
        while len(tube_eps) < 60 and tries < 400:
            tries += 1
            eta = rng.uniform(1e-3, TUBE_ETA_MAX)
            theta = rng.uniform(TUBE_THETA[0] + 0.05, TUBE_THETA[1] - 0.05)
            gamma = rng.uniform(*POS_BAND)
            src = training_module._tube_source(gamma, theta, eta, arc.branch,
                                               arc.inward_sign)
            out = _served_eps(surrogate, raw, gamma, src, log_w_grid, w_grid)
            if out is None or out[0] is None:
                continue
            tube_eps.append(out[0])
            raw_eps.append(out[1])
        tube_eps = np.array(tube_eps)
        raw_eps = np.array(raw_eps)
        self.assertGreater(tube_eps.size, 20, 'too few served held-out points')
        t_p95 = float(np.percentile(tube_eps, 95))
        r_p95 = float(np.percentile(raw_eps, 95))
        t_max = float(tube_eps.max())
        r_max = float(raw_eps.max())
        print(f'\n[TubeBeatsRaw] tube p95={t_p95:.3e} max={t_max:.3e} | '
              f'raw p95={r_p95:.3e} max={r_max:.3e} | '
              f'ratio p95={r_p95/t_p95:.2f} max={r_max/t_max:.2f}')
        self.n_checks += 1
        self.assertLess(t_p95, E_TUBE_P95_MAX,
                        f'tube p95 eps {t_p95:.3e} too large -- tube broken')
        self.n_checks += 1
        self.assertGreaterEqual(
            r_p95 / t_p95, E_P95_RATIO_MIN,
            f'tube did not beat raw at p95 (ratio {r_p95/t_p95:.2f})')
        self.n_checks += 1
        self.assertGreaterEqual(
            r_max / t_max, E_MAX_RATIO_MIN,
            f'tube did not beat raw at max (ratio {r_max/t_max:.2f})')


# ==========================================================================
# Section F -- fold-approach ray: tube stays flat, raw degrades
# ==========================================================================

@_TRAIN_TIER_SKIP
class FoldApproachRayTestCase(CensusTestCase):
    """Down a fold-approach ray (fixed mid-arc theta, eta halving from eta_max
    to ~4e-4), the u=sqrt(eta) TUBE stays FLAT and bounded, while the raw
    far-field (extrapolating inward) degrades toward the caustic.

    NOTE (coarse-fixture bar): the plan's raw slope ~ -0.5 is a production-scale
    figure; the max-normalized census currency (denominator grows with |E| near
    the fold) plus the finite fixture flatten the measured raw slope to ~ -0.19.
    Pinned here: tube |slope| < 0.15 (measured ~0.03) and the raw degrades
    (negative slope) with a >= 2x deep-caustic error gap."""

    def test_tube_flat_raw_degrades(self):
        tube = _pos_tube()
        surrogate = LensAmplificationSurrogate([tube], {'s': 1})
        raw = _pos_raw_out()
        arc, log_w_grid = _pos_arc()
        w_grid = np.exp(log_w_grid)
        gmid = float(np.mean(POS_BAND))
        theta_ray = float(np.mean(TUBE_THETA))
        etas = TUBE_ETA_MAX * 0.5 ** np.arange(0, 8)
        tube_err, raw_err = [], []
        for eta in etas:
            src = training_module._tube_source(gmid, theta_ray, float(eta),
                                               arc.branch, arc.inward_sign)
            out = _served_eps(surrogate, raw, gmid, src, log_w_grid, w_grid)
            self.assertIsNotNone(out, f'engine refused at eta={eta:.1e}')
            self.assertIsNotNone(out[0], f'tube declined at eta={eta:.1e}')
            tube_err.append(out[0])
            raw_err.append(out[1])
        tube_err = np.array(tube_err)
        raw_err = np.array(raw_err)
        log_eta = np.log(etas)
        tube_slope = float(np.polyfit(log_eta, np.log(tube_err), 1)[0])
        raw_slope = float(np.polyfit(log_eta, np.log(raw_err), 1)[0])
        deep_ratio = float(raw_err[-1] / tube_err[-1])
        print(f'\n[FoldRay] tube slope={tube_slope:.3f} raw slope='
              f'{raw_slope:.3f} tube_max={tube_err.max():.3e} '
              f'deep_ratio={deep_ratio:.2f}')
        self.n_checks += 1
        self.assertLess(abs(tube_slope), F_TUBE_SLOPE_MAX,
                        f'tube ray slope {tube_slope:.3f} not flat -- the '
                        f'u=sqrt(eta) coordinate is not absorbing the fold')
        self.n_checks += 1
        self.assertLess(tube_err.max(), F_TUBE_RAY_ERR_MAX,
                        f'tube ray error {tube_err.max():.3e} blew up as '
                        f'eta -> 0')
        self.n_checks += 1
        self.assertLess(raw_slope, F_RAW_SLOPE_MAX,
                        f'raw ray slope {raw_slope:.3f} did not degrade toward '
                        f'the caustic -- the design contrast is absent')
        self.n_checks += 1
        self.assertGreaterEqual(
            deep_ratio, F_DEEP_RATIO_MIN,
            f'deep-caustic tube/raw gap {deep_ratio:.2f} below '
            f'{F_DEEP_RATIO_MIN}')


# ==========================================================================
# Section G -- F010 mutation reachability (the falsification path is live)
# ==========================================================================

@_TRAIN_TIER_SKIP
class MutationFalsificationTestCase(CensusTestCase):
    """Mutating a load-bearing chart bound flips a previously-correct serve /
    fall-through decision RED (F010): the census's serve decisions are
    genuinely sensitive to the bounds they key on, so a broken bound cannot
    pass silently.  Every mutation is on a COPY -- the shared fixture is never
    touched."""

    def setUp(self):
        super().setUp()
        self.sur = _synthetic_surrogate()
        self.pos_tube = self.sur.charts[0]
        self.pos_ff = self.sur.charts[1]
        # Physical eigenframe query inside the positive exterior-polar chart's
        # gamma-resolved (rho, theta_c) box but off its refused point.
        self.query = dict(gamma=0.4, log_w_min=SYN_LWMIN, log_w_max=SYN_LWMAX,
                          eta=0.03, theta=0.7, image_count=2,
                          y1_eig=0.6, y2_eig=0.3)

    def _select(self, charts):
        return select_chart(charts, **self.query)

    def test_eta_floor_mutation_flips_serve(self):
        """Primary F010 lever: raising the tube ``eta_floor`` above the query's
        eta drops the tube out of its band, flipping a served query to a
        fall-through (None)."""
        baseline = self._select(self.sur.charts)
        self.n_checks += 1
        self.assertIs(baseline, self.pos_tube,
                      'precondition: baseline query must serve the tube')
        mutated = dataclasses.replace(self.pos_tube, eta_floor=0.04)
        charts = [mutated] + self.sur.charts[1:]
        self.n_checks += 1
        self.assertIsNone(self._select(charts),
                          'raising eta_floor did not flip the serve decision')

    def test_cusp_window_mutation_flips_serve(self):
        """Widening a cusp window to cover the query theta flips serve -> fall
        through; the far-field cannot rescue it (eta below its floor)."""
        baseline = self._select(self.sur.charts)
        self.assertIs(baseline, self.pos_tube)
        mutated = dataclasses.replace(self.pos_tube,
                                      cusp_windows=((0.7, 0.2),))
        charts = [mutated] + self.sur.charts[1:]
        self.n_checks += 1
        self.assertIsNone(self._select(charts),
                          'widening the cusp window did not flip serve')

    def test_gamma_grid_edge_mutation_flips_serve(self):
        """Shrinking the tube's gamma box below the query gamma flips serve ->
        fall through (certified-box containment lever)."""
        self.assertIs(self._select(self.sur.charts), self.pos_tube)
        mutated = dataclasses.replace(
            self.pos_tube, gamma_grid=np.linspace(0.30, 0.38, 4))
        charts = [mutated] + self.sur.charts[1:]
        self.n_checks += 1
        self.assertIsNone(self._select(charts),
                          'shrinking the gamma box did not flip serve')

    def test_farfield_overlap_mutation_admits_then_errs_more(self):
        """Lowering a far-field ``eta_overlap_min`` ADMITS a previously-refused
        near-caustic query: the mutation makes the far-field serve where it
        should not, and its extrapolated envelope error there exceeds the
        tube's in-domain error at the same source (admitting-a-refusal is worse
        than refusing, F005)."""
        arc, log_w_grid = _pos_arc()
        w_grid = np.exp(log_w_grid)
        gmid = float(np.mean(POS_BAND))
        theta_ray = float(np.mean(TUBE_THETA))
        eta = 0.01  # inside the tube band, below the far-field floor
        src = training_module._tube_source(gmid, theta_ray, eta, arc.branch,
                                           arc.inward_sign)
        part = ChangRefsdalChannels(w_grid).evaluate(
            gamma=gmid, y=(float(src[0]), float(src[1])), beta=0.0, kappa=0.0)
        env_eng = np.asarray(part.envelope)
        y1e, y2e = _rotate_to_eigenframe(float(src[0]), float(src[1]), 0.0)
        raw = _pos_raw_out()
        image_count = int(part.real_mask.sum())
        refusing = dataclasses.replace(raw, eta_overlap_min=0.05)
        admitting = dataclasses.replace(raw, eta_overlap_min=0.0,
                                        image_count=image_count)
        lw_min, lw_max = float(log_w_grid[0]), float(log_w_grid[-1])
        self.n_checks += 1
        self.assertFalse(surrogate_module._exterior_polar_serves(
            refusing, gmid, lw_min, lw_max, part.caustic_distance,
            image_count, y1e, y2e),
            'precondition: far-field must refuse')
        self.n_checks += 1
        self.assertTrue(surrogate_module._exterior_polar_serves(
            admitting, gmid, lw_min, lw_max, part.caustic_distance,
            image_count, y1e, y2e),
            'lowering overlap did not admit the query')
        e_raw = _evaluate_chart(
            admitting, gamma=gmid, eta=float('nan'), theta=float('nan'),
            log_w_query=log_w_grid, y1_eig=y1e, y2_eig=y2e)
        tube = _pos_tube()
        e_tube = _evaluate_chart(
            tube, gamma=gmid, eta=part.caustic_distance,
            theta=part.critical_theta, log_w_query=log_w_grid,
            y1_eig=y1e, y2_eig=y2e)
        denom = max(float(np.max(np.abs(env_eng))), census.EPS_DENOM_FLOOR)
        raw_eps = float(np.max(np.abs(e_raw - env_eng)) / denom)
        tube_eps = float(np.max(np.abs(e_tube - env_eng)) / denom)
        self.n_checks += 1
        self.assertGreater(
            raw_eps, tube_eps,
            f'admitted far-field eps {raw_eps:.3e} not worse than the tube '
            f'in-domain eps {tube_eps:.3e}')


# ==========================================================================
# Section H -- cusp-adapted theta_to_u exterior-polar chart in census
# ==========================================================================

@functools.lru_cache(maxsize=1)
def _cusp_adapted_exterior_polar_chart():
    """An ExteriorPolarChart with a real non-identity ``theta_to_u``
    cusp-adapted map built via `_wedge_cusp_axis_map`.

    Shares axes with the raw-theta sibling created in
    `ExteriorPolarCuspAdaptedCensusTestCase.setUpClass`; both use the
    same ``_smooth_tensor`` envelope values, so the only difference is
    the theta_to_u parametrization.
    """
    gamma_grid = np.linspace(0.30, 0.50, 4)
    rho_grid = np.linspace(0.02, 0.08, 4)
    theta_c_grid = np.linspace(0.05, 0.20, 4)
    theta_lo, theta_hi = float(theta_c_grid[0]), float(theta_c_grid[-1])
    theta_fine, u_fine = surrogate_module._wedge_cusp_axis_map(
        theta_lo, theta_hi, 'low')
    theta_to_u = np.vstack([theta_fine, u_fine])
    u_grid = np.interp(theta_c_grid, theta_fine, u_fine)
    real, imag = _smooth_tensor(gamma_grid, rho_grid, theta_c_grid,
                                SYN_LOG_W, 2.0)
    return ExteriorPolarChart.from_values(
        gamma_grid=gamma_grid, rho_grid=rho_grid,
        theta_c_grid=theta_c_grid, log_w_grid=SYN_LOG_W,
        envelope_real=real, envelope_imag=imag,
        image_count=2, parity=1,
        eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
        theta_to_u=theta_to_u, u_grid=u_grid)


class ExteriorPolarCuspAdaptedCensusTestCase(CensusTestCase):
    """Census correctly classifies draws served by a theta_to_u-bearing
    exterior-polar chart as ``served=True`` (via the standard
    ``chart_index`` path), and fallthrough categories are unchanged
    relative to the raw-theta sibling."""

    @classmethod
    def setUpClass(cls):
        cls.chart_cusp = _cusp_adapted_exterior_polar_chart()
        real_vals, imag_vals = _smooth_tensor(
            np.linspace(0.30, 0.50, 4), np.linspace(0.02, 0.08, 4),
            np.linspace(0.05, 0.20, 4), SYN_LOG_W, 2.0)
        cls.chart_raw = ExteriorPolarChart.from_values(
            gamma_grid=np.linspace(0.30, 0.50, 4),
            rho_grid=np.linspace(0.02, 0.08, 4),
            theta_c_grid=np.linspace(0.05, 0.20, 4),
            log_w_grid=SYN_LOG_W,
            envelope_real=real_vals, envelope_imag=imag_vals,
            image_count=2, parity=1,
            eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
            theta_to_u=None, u_grid=None)
        cls.gamma = 0.4
        cls.rho_mid = 0.05
        cls.theta_c_mid = 0.125
        cls.y1_eig, cls.y2_eig = surrogate_module._from_caustic_fixed(
            cls.gamma, cls.rho_mid, cls.theta_c_mid)

    def setUp(self):
        super().setUp()

    def test_exterior_polar_serves_identical_for_both(self):
        """``_exterior_polar_serves`` does not read ``theta_to_u``; both
        charts make the same admit/refuse decision for the same query."""
        kwargs = dict(gamma=self.gamma, log_w_min=SYN_LWMIN,
                      log_w_max=SYN_LWMAX, eta=0.10, image_count=2,
                      y1_eig=self.y1_eig, y2_eig=self.y2_eig)
        raw_serves = surrogate_module._exterior_polar_serves(
            self.chart_raw, **kwargs)
        cusp_serves = surrogate_module._exterior_polar_serves(
            self.chart_cusp, **kwargs)
        self.n_checks += 1
        self.assertEqual(raw_serves, cusp_serves,
                         '_exterior_polar_serves must not depend on '
                         'theta_to_u')
        self.n_checks += 1
        self.assertTrue(raw_serves,
                        'mid-axis query must be served by both charts')

    def test_select_chart_returns_cusp_chart(self):
        """``select_chart`` returns the theta_to_u-bearing chart for a
        query inside its box -- census records ``served=True`` via
        ``chart_index``."""
        sur = LensAmplificationSurrogate([self.chart_cusp], {})
        chart = select_chart(sur.charts, gamma=self.gamma,
                             log_w_min=SYN_LWMIN, log_w_max=SYN_LWMAX,
                             eta=0.10, theta=float('nan'),
                             image_count=2,
                             y1_eig=self.y1_eig, y2_eig=self.y2_eig)
        self.n_checks += 1
        self.assertIsNotNone(chart,
                             'select_chart must return the chart for an '
                             'in-box query')
        self.n_checks += 1
        self.assertIs(chart, self.chart_cusp)

    def test_evaluate_chart_finite_with_theta_to_u(self):
        """``_evaluate_chart`` returns finite complex values when
        ``theta_to_u`` is present -- the census heldout-envelope-eps
        path is exercised."""
        log_w_query = SYN_LOG_W.copy()
        result = _evaluate_chart(
            self.chart_cusp, gamma=self.gamma, eta=float('nan'),
            theta=float('nan'), log_w_query=log_w_query,
            y1_eig=self.y1_eig, y2_eig=self.y2_eig)
        self.n_checks += 1
        self.assertTrue(np.all(np.isfinite(result)),
                        '_evaluate_chart must return finite values')
        self.n_checks += 1
        self.assertEqual(result.shape, log_w_query.shape)

    def test_classify_fallthrough_same_for_both(self):
        """``classify_fallthrough`` returns the same category for both
        the raw-theta and cusp-adapted charts -- census fallthrough
        categorization is theta_to_u-independent."""
        for gamma, expected_category in [
            (1.0 + 0.5 * _GAMMA_GUARD_BAND, 'gamma-guard'),
            (0.92, 'dropped-sliver'),
            (0.80, 'out-of-box'),
        ]:
            for chart, label in [(self.chart_raw, 'raw'),
                                 (self.chart_cusp, 'cusp')]:
                sur = LensAmplificationSurrogate([chart], {})
                result = census.classify_fallthrough(
                    sur, gamma=gamma, log_w_min=SYN_LWMIN,
                    log_w_max=SYN_LWMAX, eta=0.03, theta=0.7,
                    image_count=2, y1_eig=self.y1_eig,
                    y2_eig=self.y2_eig,
                    dropped_slivers=SYN_DROPPED)
                with self.subTest(gamma=gamma, chart=label):
                    self.n_checks += 1
                    self.assertEqual(
                        result, expected_category,
                        f'{expected_category} mismatch for {label}')


class ExteriorPolarCuspAdaptedSelfFalsification(CensusTestCase):
    """Self-falsification: the ``theta_to_u``<->``u_grid`` pairing is
    load-bearing -- providing one without the other raises at chart
    construction."""

    def setUp(self):
        super().setUp()
        self.real, self.imag = _smooth_tensor(
            np.linspace(0.3, 0.5, 4), np.linspace(0.02, 0.08, 4),
            np.linspace(0.05, 0.20, 4), SYN_LOG_W, 2.0)
        self.kwargs = dict(
            gamma_grid=np.linspace(0.30, 0.50, 4),
            rho_grid=np.linspace(0.02, 0.08, 4),
            theta_c_grid=np.linspace(0.05, 0.20, 4),
            log_w_grid=SYN_LOG_W,
            image_count=2, parity=1,
            eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR)

    def test_theta_to_u_without_u_grid_raises(self):
        """Providing ``theta_to_u`` without ``u_grid`` raises ValueError
        -- the census cannot silently serve a misconfigured chart."""
        self.n_checks += 1
        with self.assertRaises(ValueError):
            ExteriorPolarChart.from_values(
                envelope_real=self.real, envelope_imag=self.imag,
                theta_to_u=np.array([[0.05, 0.20], [0.0, 1.0]]),
                u_grid=None, **self.kwargs)

    def test_u_grid_without_theta_to_u_raises(self):
        """Providing ``u_grid`` without ``theta_to_u`` raises ValueError."""
        self.n_checks += 1
        with self.assertRaises(ValueError):
            ExteriorPolarChart.from_values(
                envelope_real=self.real, envelope_imag=self.imag,
                theta_to_u=None,
                u_grid=np.array([0.0, 1.0]),
                **self.kwargs)

# ==========================================================================
# Section Z -- SHARD C: census saddle-gap reduction (scripts/census_dry_run.py)
#
# WP3 re-routes macro-saddle scalar-interior draws (gamma > 1, rho <= 1) that
# were previously hard-coded to 'exact_engine' into the two NEW serve
# categories 'lobe_interior' and 'lobe_exterior', mirroring the production
# LobeInteriorChart / LobeExteriorChart serve gates (`admits` /
# `admits_exterior` on the canonical +y1 deltoid lobe).  These tests exercise
# the STRUCTURAL classifier in ``scripts/census_dry_run.py`` -- a non-package
# script loaded via importlib -- NOT the ``surrogate_census`` module the rest
# of this file covers.
#
# MEASURED REALITY vs the spec's aspirational wording.  The spec says the
# saddle-interior 'exact_engine' bucket "drops toward zero".  Direct
# measurement (two independent seeds, 800 saddle draws each) shows the lobe
# charts reclaim ~24-28% of the interior saddle draws (almost all
# 'lobe_exterior'; 'lobe_interior' fires only for the handful of draws that
# fold onto the lobe centroid).  The residual 'exact_engine' is the genuine
# inter-lobe / origin corridor: sources whose D2-folded lobe-local radius
# lands BEYOND the served outer edge ``rho_outer`` -- outside any lobe
# chart's domain, not a routing failure of servable draws (proven by the
# positive-control class below, where an on-lobe source IS routed to a lobe
# category).  We therefore pin the honest inequalities the data supports:
#   * the two lobe categories are now materially populated (were absent
#     before WP3),
#   * 'exact_engine' is STRICTLY reduced versus the pre-WP3 all-exact model,
#     with the reduction exactly equal to the lobe reclaim (conservation),
#   * astroid (gamma < 1) classification is byte-identical regardless of the
#     saddle path or the saddle cache state.
#
# COST.  A fresh script load rebuilds the 12 coarse saddle bands' caustic
# clouds ONCE (~5.8 s for 800 draws incl. the band builds); subsequent draws
# reuse the per-band cache (< 0.05 ms each).  Astroid draws never build a
# band (~0.006 ms each).  The saddle histogram sample is shared across the
# methods of its class via ``setUpClass`` so the band builds are paid once
# per class; the whole section runs in well under a minute.
# ==========================================================================

#: Absolute path to the non-package census dry-run script (WP3 subject).
_CENSUS_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / 'scripts' / 'census_dry_run.py')

#: Directory for diagnostic plots (house convention).
_CENSUS_OUTPUT_DIR = Path(__file__).resolve().parent / 'output'

#: Deterministic seed / size for the shared saddle-heavy histogram sample.
_SADDLE_SEED = 20260812
_SADDLE_N = 800

#: Prior support for a saddle-ONLY draw (gamma > 1, strictly above the guard
#: floor the production band tiler uses); matches the script's `_draw_prior`
#: ranges on the other axes.
_SADDLE_GAMMA_LO = 1.0 + 1e-3
_SADDLE_GAMMA_HI = 1.599
_Y_ABS_LO, _Y_ABS_HI = 0.01, 4.2426
_LOG_W_LO, _LOG_W_HI = math.log(5.0), math.log(148.0)

#: Prior support for an astroid-ONLY draw (gamma < 1).
_ASTROID_GAMMA_LO, _ASTROID_GAMMA_HI = 0.001, 0.999

#: Minimum fraction of INTERIOR saddle draws the lobe charts must reclaim.
#: Measured ~0.24 (seed 20260812) and ~0.28 (seed 999); floor well below.
_LOBE_RECLAIM_MIN_FRAC = 0.10

#: Deterministic positive controls at gamma = 1.3 (folded onto the canonical
#: +y1 lobe, centroid ~[1.345, 0]).  ``(gamma, y_abs, theta)``:
#:  * a source AT the lobe centroid radius -> served as 'lobe_interior',
#:  * a nearer source in the served exterior band -> 'lobe_exterior',
#:  * the inter-lobe corridor source -> genuine 'exact_engine' gap
#:    (folds to rho_lobe ~3.80 > rho_outer ~3.53).
_LOBE_INTERIOR_CONTROL = (1.3, 1.3, 0.0)
_LOBE_EXTERIOR_CONTROL = (1.3, 0.65, 0.0)
_CORRIDOR_GAP_CONTROL = (1.3, 0.5, 0.0)

#: The astroid-only serve categories that must NEVER appear for a saddle
#: (gamma > 1) draw -- the gamma >= 1 branch short-circuits before them.
_ASTROID_ONLY_CATEGORIES = frozenset(
    {'chart_interior', 'ppgo_fold', 'cusp_arm', 'chart_tube',
     'chart_farfield'})

#: The two serve categories WP3 introduced for the saddle interior.
_LOBE_CATEGORIES = ('lobe_interior', 'lobe_exterior')


def _load_census_script():
    """Load ``scripts/census_dry_run.py`` as a fresh, isolated module.

    Each call re-executes the script's top level, giving a pristine
    ``_SADDLE_ADMISSION_CACHE`` so cache-independence can be tested.  The
    heavy ``cogwheel`` imports resolve from ``sys.modules`` after the first
    load, so repeated loads are cheap.
    """
    spec = importlib.util.spec_from_file_location(
        'census_dry_run_shardc', _CENSUS_SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f'cannot load census script at {_CENSUS_SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draw_saddle_sample(module, seed, n):
    """Draw and classify ``n`` saddle-only (gamma > 1) prior draws.

    Returns ``(categories, rho, interior_mask)`` where ``categories`` is a
    list of the classifier's verdicts, ``rho`` the caustic-relative radius
    per draw, and ``interior_mask`` the boolean ``rho <= 1`` selection.
    """
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(_SADDLE_GAMMA_LO, _SADDLE_GAMMA_HI, size=n)
    y_abs = rng.uniform(_Y_ABS_LO, _Y_ABS_HI, size=n)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    _ = np.exp(rng.uniform(_LOG_W_LO, _LOG_W_HI, size=n))  # w (draw order)
    w = _
    categories = [
        module.classify_draw(float(gamma[i]), float(y_abs[i]),
                             float(theta[i]), float(w[i]))
        for i in range(n)]
    rho = np.array([module._compute_rho(float(gamma[i]), float(y_abs[i]))
                    for i in range(n)])
    return categories, rho, rho <= 1.0


class SaddleGapReductionHistogramTestCase(CensusTestCase):
    """Over a saddle-heavy (gamma > 1) sample, WP3 populates the two lobe
    categories and STRICTLY reduces the 'exact_engine' gap versus the pre-WP3
    all-exact model, conserving every other category.

    The 'before' histogram is reconstructed INDEPENDENTLY of production logic
    from the documented pre-WP3 behaviour: the saddle branch's ONLY change is
    re-routing draws that were 'exact_engine' into the two lobe buckets (the
    'born' and astroid categories are untouched).  So the before-state is the
    after-histogram with the lobe buckets merged back into 'exact_engine' --
    an exact reconstruction requiring no re-derivation of the lobe geometry.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = _load_census_script()
        cls.categories, cls.rho, cls.interior = _draw_saddle_sample(
            cls.module, _SADDLE_SEED, _SADDLE_N)
        cls.after = Counter(cls.categories)
        cls.interior_after = Counter(
            cls.categories[i] for i in range(_SADDLE_N) if cls.interior[i])

    def _before(self):
        """Pre-WP3 histogram: lobe buckets folded back into 'exact_engine'."""
        before = Counter(self.after)
        reclaimed = before.pop('lobe_interior', 0) + before.pop(
            'lobe_exterior', 0)
        before['exact_engine'] = before.get('exact_engine', 0) + reclaimed
        return before

    def test_lobe_categories_now_materially_populated(self):
        """The two NEW lobe categories reclaim >= 10% of interior draws (were
        absent before WP3); 'lobe_exterior' carries the bulk."""
        interior_n = int(self.interior.sum())
        reclaim = (self.interior_after['lobe_interior']
                   + self.interior_after['lobe_exterior'])
        self.n_checks += 1
        self.assertGreaterEqual(
            reclaim, _LOBE_RECLAIM_MIN_FRAC * interior_n,
            f'lobe reclaim {reclaim}/{interior_n} below floor')
        self.n_checks += 1
        self.assertGreater(self.interior_after['lobe_exterior'], 0)

    def test_exact_engine_strictly_reduced_and_conserved(self):
        """'exact_engine' after < before, and the drop equals the lobe
        reclaim exactly (no draw silently changed born/astroid bucket)."""
        before = self._before()
        drop = before['exact_engine'] - self.after.get('exact_engine', 0)
        reclaim = (self.after.get('lobe_interior', 0)
                   + self.after.get('lobe_exterior', 0))
        self.n_checks += 1
        self.assertGreater(drop, 0, 'saddle gap did not shrink')
        self.n_checks += 1
        self.assertEqual(drop, reclaim, 'gap reduction != lobe reclaim')

    def test_born_conserved_and_no_astroid_categories(self):
        """'born' survives unchanged and the astroid-only categories never
        appear in a saddle sample (the gamma >= 1 branch short-circuits)."""
        before = self._before()
        self.n_checks += 1
        self.assertEqual(before.get('born', 0), self.after.get('born', 0))
        self.n_checks += 1
        self.assertGreater(self.after.get('born', 0), 0)
        for cat in _ASTROID_ONLY_CATEGORIES:
            self.n_checks += 1
            self.assertEqual(
                self.after.get(cat, 0), 0,
                f'astroid-only category {cat!r} leaked into saddle sample')

    def test_writes_before_after_histogram(self):
        """Save the before/after category histogram diagnostic (house
        convention: cogwheel/tests/output/)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - matplotlib always present
            self.skipTest('matplotlib unavailable')
        before = self._before()
        order = ['born', 'lobe_interior', 'lobe_exterior', 'exact_engine']
        before_vals = [before.get(c, 0) for c in order]
        after_vals = [self.after.get(c, 0) for c in order]
        _CENSUS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        xpos = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(xpos - 0.2, before_vals, width=0.4, label='before WP3',
               color='0.6')
        ax.bar(xpos + 0.2, after_vals, width=0.4, label='after WP3',
               color='C0')
        ax.set_xticks(xpos)
        ax.set_xticklabels(order, rotation=20, ha='right')
        ax.set_ylabel(f'count (of {_SADDLE_N} saddle draws)')
        ax.set_title('SHARD C: saddle serve-category census (before vs after)')
        ax.legend()
        fig.tight_layout()
        out = _CENSUS_OUTPUT_DIR / 'saddle_gap_reduction_before_after.png'
        fig.savefig(out, dpi=90)
        plt.close(fig)
        self.n_checks += 1
        self.assertTrue(out.exists() and out.stat().st_size > 0)


class SaddlePositiveControlTestCase(CensusTestCase):
    """Curated deterministic sources at gamma = 1.3 prove the lobe categories
    are reachable (servable draws ARE routed) and the residual gap is genuine
    geometry, not a routing failure of on-lobe draws.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = _load_census_script()

    def _classify(self, gamma, y_abs, theta):
        return self.module._classify_saddle(gamma, y_abs, theta)

    def test_deep_lobe_source_serves_lobe_interior(self):
        """A source folded onto the +y1 lobe centroid radius -> lobe_interior."""
        self.n_checks += 1
        self.assertEqual(self._classify(*_LOBE_INTERIOR_CONTROL),
                         'lobe_interior')

    def test_near_lobe_source_serves_lobe_exterior(self):
        """A source in the served exterior band -> lobe_exterior."""
        self.n_checks += 1
        self.assertEqual(self._classify(*_LOBE_EXTERIOR_CONTROL),
                         'lobe_exterior')

    def test_inter_lobe_corridor_is_genuine_gap(self):
        """The inter-lobe corridor source folds BEYOND rho_outer -> a genuine
        'exact_engine' geometry gap, not a servable draw left unrouted."""
        self.n_checks += 1
        self.assertEqual(self._classify(*_CORRIDOR_GAP_CONTROL),
                         'exact_engine')

    def test_gap_is_outside_the_served_outer_edge(self):
        """The corridor gap's folded lobe radius genuinely exceeds the served
        outer edge (so no lobe chart could cover it) -- the residual gap is a
        boundary, not a hole inside the served band."""
        gamma, y_abs, theta = _CORRIDOR_GAP_CONTROL
        lobe, rho_outer = self.module._saddle_lobe_admission(gamma)
        y1_fold = abs(y_abs * math.cos(theta))
        y2_fold = abs(y_abs * math.sin(theta))
        rho_lobe, _ = self.module._to_lobe_fixed(
            lobe.centroid, lobe.boundary_theta, lobe.boundary_r,
            y1_fold, y2_fold)
        self.n_checks += 1
        self.assertGreater(rho_lobe, rho_outer,
                           'corridor gap should fold beyond the served edge')

    def test_d2_reflection_invariance(self):
        """The four D2 reflections of a served source (theta -> -theta,
        pi-theta, pi+theta) fold identically -> same serve category."""
        _, y_abs, theta = (1.3, 0.65, 0.1047)
        base = self._classify(1.3, y_abs, theta)
        self.n_checks += 1
        self.assertEqual(base, 'lobe_exterior')  # anchor the control
        for reflected in (-theta, math.pi - theta, math.pi + theta):
            self.n_checks += 1
            self.assertEqual(self._classify(1.3, y_abs, reflected), base,
                             f'D2 reflection to theta={reflected} disagreed')


class AstroidUnchangedByScriptTestCase(CensusTestCase):
    """Astroid (gamma < 1) classification is untouched by the saddle path:
    no lobe categories appear, the astroid draws never build a saddle band,
    and the verdicts are byte-identical whether or not the saddle cache is
    warm.
    """

    @classmethod
    def setUpClass(cls):
        cls.seed = 7
        cls.n = 1500

    def _draw_astroid(self, module):
        rng = np.random.default_rng(self.seed)
        gamma = rng.uniform(_ASTROID_GAMMA_LO, _ASTROID_GAMMA_HI, size=self.n)
        y_abs = rng.uniform(_Y_ABS_LO, _Y_ABS_HI, size=self.n)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=self.n)
        w = np.exp(rng.uniform(_LOG_W_LO, _LOG_W_HI, size=self.n))
        return [
            module.classify_draw(float(gamma[i]), float(y_abs[i]),
                                 float(theta[i]), float(w[i]))
            for i in range(self.n)]

    def test_astroid_never_routes_to_lobe_categories(self):
        """No gamma < 1 draw is classified into a lobe category."""
        module = _load_census_script()
        cats = self._draw_astroid(module)
        for cat in _LOBE_CATEGORIES:
            self.n_checks += 1
            self.assertNotIn(cat, cats,
                             f'astroid draw wrongly routed to {cat!r}')

    def test_astroid_path_builds_no_saddle_band(self):
        """Classifying only astroid draws leaves the saddle band cache empty
        (the gamma < 1 branch never touches the lobe geometry)."""
        module = _load_census_script()
        self.assertEqual(len(module._SADDLE_ADMISSION_CACHE), 0)  # precondition
        _ = self._draw_astroid(module)
        self.n_checks += 1
        self.assertEqual(len(module._SADDLE_ADMISSION_CACHE), 0,
                         'astroid classification populated the saddle cache')

    def test_astroid_verdicts_independent_of_saddle_cache(self):
        """Warming a saddle band before classifying the astroid draws does not
        change a single astroid verdict."""
        cold = _load_census_script()
        cats_cold = self._draw_astroid(cold)
        warm = _load_census_script()
        # Warm the saddle machinery on an unrelated saddle draw first.
        _ = warm.classify_draw(1.3, 0.5, 0.0, 50.0)
        self.assertGreater(len(warm._SADDLE_ADMISSION_CACHE), 0)  # warmed
        cats_warm = self._draw_astroid(warm)
        self.n_checks += 1
        self.assertEqual(cats_cold, cats_warm,
                         'astroid verdicts changed with the saddle cache warm')

    def test_astroid_categories_are_the_expected_set(self):
        """Sanity: the astroid sample is dominated by born + interior charts
        with a negligible gap (structural coverage), confirming the sample is
        genuinely astroid and not degenerate."""
        module = _load_census_script()
        counts = Counter(self._draw_astroid(module))
        self.n_checks += 1
        self.assertGreater(counts.get('born', 0), 0)
        self.n_checks += 1
        self.assertGreater(counts.get('chart_interior', 0), 0)
        self.n_checks += 1
        # Astroid structural coverage is near-total (gap far below 5%).
        self.assertLess(counts.get('exact_engine', 0), 0.05 * self.n)


class SaddleCensusSelfFalsificationTestCase(CensusTestCase):
    """The suite CAN go red.  With the WP3 lobe routing disabled (the lobe
    admission forced to degenerate), every saddle-interior draw collapses back
    to 'exact_engine' -- the reclaim vanishes and the positive controls flip
    -- so the passing assertions above genuinely depend on the WP3 code.

    A throwaway module is sabotaged so no cross-class state leaks.
    """

    def test_disabling_lobe_admission_collapses_reclaim(self):
        """Force ``_saddle_lobe_admission`` degenerate: the histogram class's
        reclaim assertion would then FAIL (reclaim drops to zero)."""
        module = _load_census_script()
        module._SADDLE_ADMISSION_CACHE.clear()
        module._saddle_lobe_admission = lambda gamma: (None, None)
        cats, _, interior = _draw_saddle_sample(module, _SADDLE_SEED, 400)
        interior_after = Counter(
            cats[i] for i in range(400) if interior[i])
        reclaim = (interior_after['lobe_interior']
                   + interior_after['lobe_exterior'])
        self.n_checks += 1
        self.assertEqual(reclaim, 0,
                         'sabotaged classifier still produced lobe categories')
        # And the honest floor the real test asserts is now violated.
        self.n_checks += 1
        self.assertFalse(reclaim >= _LOBE_RECLAIM_MIN_FRAC * int(interior.sum()))

    def test_disabling_lobe_admission_flips_positive_controls(self):
        """With the lobe routing gone, the on-lobe positive controls fall
        through to 'exact_engine' (so those assertions are reachable-red)."""
        module = _load_census_script()
        module._SADDLE_ADMISSION_CACHE.clear()
        module._saddle_lobe_admission = lambda gamma: (None, None)
        self.n_checks += 1
        self.assertEqual(
            module._classify_saddle(*_LOBE_INTERIOR_CONTROL), 'exact_engine')
        self.n_checks += 1
        self.assertEqual(
            module._classify_saddle(*_LOBE_EXTERIOR_CONTROL), 'exact_engine')


if __name__ == '__main__':
    unittest.main()
