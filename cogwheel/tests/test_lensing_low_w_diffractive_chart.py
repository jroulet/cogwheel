"""
Tests for `lensing.low_w_diffractive_chart` + its likelihood serve.

WHAT THIS SUITE ADJUDICATES
---------------------------
`LowWDiffractiveChart` holds a trained 4-D tensor-product interpolation of the
low-w diffractive residual

    r_pure(w; gamma', rho, theta) = f_pure / (sqrt(mu_pure) * prefactor_c(w))

and `LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve`
re-modulates it back to the full amplitude

    F_serve = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full * r_pure.

The serve is the Rung-P replacement for the exact Schwinger engine over the
near-fold shell (`w_low_fit` declines there -> None) and the wall band
(`gamma' > 0.5`).  Three invariants:

  1. DC ANCHOR (`DcAnchorRemodulationTestCase`).  As ``w -> 0`` the exact
     engine's amplitude tends to ``sqrt(mu_macro) = 1/sqrt(|(1-kappa)**2 -
     gamma**2|)`` with Morse-0 phase (positive parity), NOT 1.  With a
     CONSTANT-residual chart the re-modulation factors are isolated from any
     residual accuracy, so |F_serve|/sqrt(mu_macro) -> 1 and arg -> 0 pins
     that the anchor is sqrt(mu_macro) and that ``prefactor_c`` and the
     mass-sheet phase are each applied exactly once (a magnitude offset ~
     1/lam or a phase ramp ~ w ln(w) is a missed/doubled factor).

  2. SERVE-VS-ENGINE (`ServeEngineNodeExactTestCase`).  With an EXACT-residual
     chart the re-modulated F_serve must reproduce the exact engine to
     ~1e-14 at grid NODES (the residual is node-exact by construction, so the
     only thing left is the re-modulation + normalization consistency).  This
     is the fast-tier form of the accuracy invariant -- STRONGER than the
     shipped chart's 1e-4 off-grid bar, and it runs in-build because it needs
     only a few dozen engine calls, not the dense full-bake artifact.  The
     shipped chart's 1e-4 off-grid interpolation accuracy is certified by
     ``scripts/train_low_w_diffractive_chart.py``'s off-grid margin report
     (a DRIVER post-build step, not a unit test).

  3. ONE-SIDED CONSERVATIVENESS (`ConservativenessTestCase`).  Cubic
     interpolation overshoots by construction (measured up to ~1.6x here on
     off-grid theta midpoints); the scalar ``derate`` is the SOLE margin.  The
     serve must apply it multiplicatively (node-exact ratio == derate) and a
     derate of ``1/max_overshoot`` must make the serve one-sided conservative
     (|F_serve| <= |F_engine| everywhere on the fixture set).

ORACLE INDEPENDENCE
-------------------
The engine oracle is ``f_schwinger`` (exact Schwinger double-double), the
``kappa = 0`` form evaluated in the eigenframe and the ``kappa > 0`` form
reconstructed through the shipped ``operator._mass_sheet_map`` (the same
recipe as ``test_lensing_diffractive._engine_reference_kappa``).  The chart/
serve path never calls ``f_schwinger`` (the chart is the sole serve-time
source), so agreeing with it is exactly the serve-consistency the rung needs.
The DC-anchor oracle is the CLOSED FORM of the mass-sheet phase + prefactor
(no engine), which is independent of the serve's own expression of those
quantities.  ``w`` stays <= 8 so the engine runs on its exact double-double
path (mpmath only above ~60).

TOLERANCES
----------
* `NODE_EXACT_TOL` = 1e-10: the node-exact re-modulation reconstructs
  F_engine to ~1.6e-14 (measured); 1e-10 leaves ~6000x float64 margin.
* `_DC_SHARP_TOL` = 1e-12 (relative): the closed-form DC anchor matches the
  serve to ~1e-15 (the two expressions of the mass-sheet phase agree to
  float64); 1e-12 leaves ~1000x margin.
* `_DC_MAG_TOL` = 0.05 / `_DC_ARG_TOL` = 0.15: the loose O(w) asymptotic the
  spec demands.  |prefactor_c(w)| - 1 ~ 0.79*w (0.008 at w=1e-2) and the
  phase ~ 0.024-0.027 rad at w=1e-2, so these are 5-6x above the correction
  yet far below the failure signatures (anchor->1 gives ratio 0.6 at gamma'=0.8).
"""

from __future__ import annotations

import cmath
import dataclasses
import functools
import json
import math
import tempfile
import types
from pathlib import Path
from unittest import TestCase, main, mock

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cogwheel.lensing import likelihood as _likelihood_mod
from cogwheel.lensing import low_w_diffractive_chart as _lwd_module
from cogwheel.lensing import ppgo_map as _ppgo_map
from cogwheel.lensing import serve_route_census as _census
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.low_w_diffractive_chart import (
    RHO_HI, RHO_LO, _WALL_GAMMA_PRIME, _SCHEMA, _content_hash,
    LowWDiffractiveChart)
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import operator as _operator
from cogwheel.lensing.chang_refsdal._diffractive import _caustic_rho
from cogwheel.lensing.chang_refsdal._hyp1f1 import prefactor_c
from cogwheel.lensing.chang_refsdal._schwinger import f_schwinger


#: Reduced shears spanning the two chart-owned regions: the near-fold shell
#: (``gamma' = 0.3``, where ``w_low_fit`` declines) and the wall band
#: (``gamma' = 0.8``, ``gamma' > _WALL_GAMMA_PRIME = 0.5``).
NEAR_FOLD_GAMMA_PRIME = 0.3
WALL_GAMMA_PRIME = 0.8

#: Caustic-relative distances paired with the regions above: rho = 1.0 sits
#: inside the near-fold shell fence ``[RHO_LO, RHO_HI] = [0.6, 1.4]``; rho =
#: 2.0 is a wall-band exterior source (the wall band covers all rho).
NEAR_FOLD_RHO = 1.0
WALL_RHO = 2.0

#: (gamma', rho, label) fixtures, one per chart-owned region.
FIXTURES = (
    (NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO, 'near_fold_shell'),
    (WALL_GAMMA_PRIME, WALL_RHO, 'wall_band'),
)

#: Convergence and shear-angle values exercised (the serve handles kappa /
#: beta verbatim through the reduced-shear map, so the oracle must too).
KAPPAS = (0.0, 0.2)
BETAS = (0.0, 0.3)

#: Eigenframe source angle used at grid NODES (index 1 of `_THETA_GRID`).
THETA_NODE = 0.6

#: Off-grid theta MIDPOINTS of `_THETA_GRID` (cubic interpolation is least
#: constrained between nodes -- where overshoot lives).
THETA_MIDPOINTS = (0.4, 0.8, 1.2)

#: Frequencies probing the DC anchor (w -> 0).
ANCHOR_WS = (1e-3, 3e-3, 1e-2)

#: Frequencies (grid NODES) exercising the serve-vs-engine / conservativeness
#: accuracy on the exact-residual chart.
SERVE_WS = (0.5, 2.0, 8.0)

#: Exact-residual chart grids (>= 4 nodes per axis for scipy cubic).  The
#: fixture coordinates (gamma' = 0.3/0.8, rho = 1.0/2.0, theta = 0.6,
#: w = 0.5/2/8) are INTERIOR nodes, so a float64 round-off in the serve's
#: coordinate reconstruction (~1e-16) cannot push them outside the grid and
#: trip `covers` (which uses inclusive <= on the grid edges).
_GAMMA_GRID = np.array([0.2, 0.3, 0.8, 0.9])
_RHO_GRID = np.array([0.5, 1.0, 2.0, 2.5])
_THETA_GRID = np.array([0.2, 0.6, 1.0, 1.4])
_W_GRID = np.array([0.3, 0.5, 2.0, 8.0])

#: Serve-vs-engine accuracy bar at grid nodes (see module docstring).
NODE_EXACT_TOL = 1e-10

#: Sharp (relative) tolerance of the DC-anchor closed-form pin.
_DC_SHARP_TOL = 1e-12

#: Loose O(w) magnitude / phase tolerances of the DC-anchor asymptotic.
_DC_MAG_TOL = 0.05
_DC_ARG_TOL = 0.15

#: Scalar derate used to prove the serve applies it multiplicatively.
_CONSERVATIVE_DERATE = 0.5

#: --- Coverage-union fixtures (DERIVED from the live gate constants) ---
#: The union band is ``(RHO_LO <= rho <= RHO_HI) or (gamma' > _WALL_GAMMA_PRIME)``.
#: These witnesses sit strictly inside/outside each clause so a gate move
#: fails loudly (premise assertions) instead of stranding a literal.  Note
#: ``WALL_GAMMA_PRIME`` (0.8) is the TEST fixture's reduced shear, while
#: ``_WALL_GAMMA_PRIME`` (0.5) is the PRODUCTION gate constant -- they are
#: deliberately different.
GAMMA_PRIME_BELOW_WALL = _WALL_GAMMA_PRIME - 0.3   #: 0.2 -- not in wall band
RHO_BELOW_SHELL = RHO_LO - 0.3                     #: 0.3 -- interior of shell
RHO_ABOVE_SHELL = RHO_HI + 0.6                     #: 2.0 -- exterior of shell


def _rot_minus_beta(beta: float) -> np.ndarray:
    """Return the eigenframe rotation ``R(-beta)`` (2x2)."""
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    return np.array([[cos_b, sin_b], [-sin_b, cos_b]])


def _sqrt_mu_macro(gamma: float, kappa: float) -> float:
    """Macro amplitude ``sqrt(mu_macro) = 1/sqrt(|(1-kappa)**2 - gamma**2|)``."""
    lam = 1.0 - kappa
    return 1.0 / math.sqrt(abs(lam * lam - gamma * gamma))


def _engine_reference_kappa(w: float, y, gamma: float, beta: float,
                            kappa: float) -> complex:
    """Exact engine amplitude at ``kappa >= 0`` via the mass-sheet map.

    Reuses the shipped ``operator._mass_sheet_map`` reduction plus the same
    ``f_schwinger`` engine, mirroring
    ``test_lensing_diffractive._engine_reference_kappa``.  This shares no code
    with the chart/serve path (which never calls the engine), so it is a
    genuine second derivation.  At ``kappa = 0`` it collapses to the eigenframe
    form ``f_schwinger(w, R(-beta) y, gamma)`` exactly.
    """
    lam, y_scaled, gamma_prime = _operator._mass_sheet_map(
        np.asarray(y, dtype=float), gamma, kappa)
    s = float(y_scaled @ y_scaled)
    y_eig = _rot_minus_beta(beta) @ y_scaled
    f_pure = f_schwinger(w, y_eig, gamma_prime)
    mass_sheet_phase = cmath.exp(
        0.5j * w * math.log(lam) - 0.5j * w * kappa * s)
    return mass_sheet_phase * f_pure / lam


def _make_lens(gamma_prime: float, rho: float, theta: float,
               kappa: float, beta: float) -> dict:
    """Reconstruct a lens dict from reduced ``(gamma', rho, theta)``.

    Inverts the serve's coordinate flow: the reduced eigenframe source is
    ``|y'| = rho * |y_c(theta)|`` along ``theta`` (the same
    `geometry.caustic_point` the serve and training script use), rotated back
    by ``beta``, rescaled by ``sqrt(1 - kappa)``; ``gamma = gamma' * (1 -
    kappa)``.  The serve reconstructs ``gamma', rho, theta`` from this lens,
    so a fixture here is a node-exact round trip through the serve.
    """
    lam = 1.0 - kappa
    gamma = gamma_prime * lam
    caustic = geometry.caustic_point(gamma_prime, theta)
    y_c = math.hypot(caustic[0], caustic[1])
    r_prime = rho * y_c
    y_eig = np.array([r_prime * math.cos(theta), r_prime * math.sin(theta)])
    y_p = np.array([
        y_eig[0] * math.cos(beta) - y_eig[1] * math.sin(beta),
        y_eig[0] * math.sin(beta) + y_eig[1] * math.cos(beta)])
    y = math.sqrt(lam) * y_p
    return {'gamma': gamma, 'beta': beta, 'kappa': kappa,
            'y1': float(y[0]), 'y2': float(y[1])}


def _residual_at(w: float, gamma_prime: float, rho: float,
                 theta: float) -> complex:
    """Engine residual ``r_pure = f_pure * sqrt(1 - gamma'**2) / C(w)``.

    Mirrors ``scripts/train_low_w_diffractive_chart._residual_at``: rebuilds
    the reduced eigenframe source from the chart coordinates and evaluates
    the exact engine, so the stored coefficient is the quantity the serve's
    re-modulation must invert.
    """
    caustic = geometry.caustic_point(gamma_prime, theta)
    y_c = math.hypot(caustic[0], caustic[1])
    r_prime = rho * y_c
    y_eig = np.array([r_prime * math.cos(theta), r_prime * math.sin(theta)])
    f_pure = f_schwinger(w, y_eig, gamma_prime)
    return f_pure * math.sqrt(1.0 - gamma_prime * gamma_prime) / prefactor_c(w)


def _serve_farfield(chart: LowWDiffractiveChart, lens: dict,
                    dense_w: np.ndarray) -> np.ndarray | None:
    """Return the re-modulated ``farfield`` (F_serve) from the REAL serve.

    Calls the shipped ``LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve``
    bound to a bare namespace, intercepting ``reconstruct_farfield`` (the
    shared, already-tested reconstruction tail) to capture its ``envelope``
    argument.  With ``geom.t_min = 0`` the frame-demodulation phase vanishes,
    so the captured envelope IS the re-modulated farfield -- the quantity
    under test.  Returns ``None`` when the serve declines (chart absent / out
    of coverage / reduced-shear refusal).
    """
    captured: dict[str, np.ndarray] = {}

    def _capture(w, envelope, delays, saddle_kernels, real_mask, definition,
                 t_min):
        captured['envelope'] = np.asarray(envelope).copy()
        return (np.zeros((w.shape[0], 4), dtype=complex),
                np.zeros(w.shape[0], dtype=complex))

    instance = types.SimpleNamespace(low_w_diffractive_chart=chart)
    instance._reduce_dense_kernels = lambda kernels: (np.zeros(1), np.zeros(1))
    instance._image_delays = lambda lens, geom: None
    geom = types.SimpleNamespace(
        t_min=0.0, delays=np.zeros(4), saddle_kernels=np.zeros((1, 4)),
        real_mask=np.array([True, True, False, False]))
    with mock.patch.object(_likelihood_mod, 'reconstruct_farfield', _capture):
        result = LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve(
            instance, lens, dense_w, geom)
    if result is None:
        return None
    return captured['envelope']


@functools.lru_cache(maxsize=1)
def _build_unit_chart() -> LowWDiffractiveChart:
    """Constant residual ``r_pure = 1`` chart covering the fixture domain.

    With residual identically 1 the served amplitude is the re-modulation
    factors in isolation (``mass_sheet_phase * prefactor_c * sqrt_mu_full``),
    so the DC anchor does not depend on any residual accuracy.  No engine
    calls.  Grids need >= 4 nodes per axis (scipy cubic) and COVER the
    fixtures with interior margin.
    """
    gamma_grid = np.array([0.2, 0.4, 0.6, 0.9])
    rho_grid = np.array([0.3, 1.0, 2.0, 3.0])
    theta_grid = np.array([0.0, 0.3, 0.9, 1.4])
    log_w_grid = np.log(np.geomspace(5e-4, 2e-2, 4))
    shape = (4, 4, 4, 4)
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_grid, rho_grid=rho_grid, theta_grid=theta_grid,
        log_w_grid=log_w_grid, real_coeffs=np.ones(shape),
        imag_coeffs=np.zeros(shape), derate=1.0)


@functools.lru_cache(maxsize=1)
def _build_exact_chart() -> LowWDiffractiveChart:
    """Exact-residual chart over `_GAMMA_GRID` x `_RHO_GRID` x `_THETA_GRID`
    x `_W_GRID` (256 engine calls, ~8 s), at unit de-rate.

    Cached so every test class shares one build.  The residual is the exact
    engine value in the chart's reduced frame, so `_serve_farfield` on a grid
    NODE reconstructs `_engine_reference_kappa` to ~1e-14.
    """
    real = np.zeros((4, 4, 4, 4), dtype=float)
    imag = np.zeros((4, 4, 4, 4), dtype=float)
    for i, gp in enumerate(_GAMMA_GRID):
        for j, rho in enumerate(_RHO_GRID):
            for k, theta in enumerate(_THETA_GRID):
                for ell, w in enumerate(_W_GRID):
                    r = _residual_at(float(w), float(gp), float(rho),
                                     float(theta))
                    real[i, j, k, ell] = r.real
                    imag[i, j, k, ell] = r.imag
    return LowWDiffractiveChart(
        gamma_prime_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
        theta_grid=_THETA_GRID, log_w_grid=np.log(_W_GRID),
        real_coeffs=real, imag_coeffs=imag, derate=1.0)


class _BaseChartTestCase(TestCase):
    """Anti-vacuity base: fail if a test makes no comparison."""

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('no comparisons were made; the test asserted nothing')


class DcAnchorRemodulationTestCase(_BaseChartTestCase):
    """DC anchor: the serve re-modulates to ``sqrt(mu_macro)``, never 1.

    With a CONSTANT-residual chart the served amplitude is the re-modulation
    factors in isolation (``mass_sheet_phase * prefactor_c(w) * sqrt_mu_full``)
    and must match the CLOSED FORM
    ``exp(0.5j w (log lam - kappa s)) * C(w) / sqrt(|lam^2 - gamma^2|)`` to
    float64.  That pins (a) the anchor is ``sqrt(mu_macro)`` (never ``F -> 1``
    -- a missed 1/lam mass-sheet amplitude shows up as a magnitude offset),
    and (b) ``prefactor_c`` and the mass-sheet phase are each applied exactly
    once (a doubled/missed prefactor leaves a phase ramp ~ w ln(w)).  The
    loose asymptotic is then |F_serve|/sqrt(mu_macro) -> 1 and arg -> 0 within
    O(w).
    """

    def _sweep(self):
        """Yield ``(label, kappa, beta, w, f_serve, lam, s, sqrt_mu_macro)``."""
        chart = _build_unit_chart()
        for (gp, rho, label), kappa, beta in (
                (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
            lens = _make_lens(gp, rho, THETA_NODE, kappa, beta)
            f_serve = _serve_farfield(chart, lens, np.array(ANCHOR_WS))
            if f_serve is None:
                self.fail(f'serve declined fixture {label} '
                          f'(gamma_prime={gp}, rho={rho}, kappa={kappa}, '
                          f'beta={beta}); the unit chart must cover it')
            lam = 1.0 - kappa
            root = math.sqrt(lam)
            yp0, yp1 = lens['y1'] / root, lens['y2'] / root
            s = yp0 * yp0 + yp1 * yp1
            sqrt_mu_macro = _sqrt_mu_macro(lens['gamma'], kappa)
            for i, w in enumerate(ANCHOR_WS):
                yield (label, kappa, beta, w, f_serve[i], lam, s,
                       sqrt_mu_macro)

    def test_served_amplitude_anchors_at_sqrt_mu_macro(self):
        """|F_serve|/sqrt(mu_macro) -> 1 and arg -> 0 as w -> 0 (loose O(w))."""
        for (label, kappa, beta, w, f_serve, _lam, _s,
             sqrt_mu_macro) in self._sweep():
            with self.subTest(label=label, kappa=kappa, beta=beta, w=w):
                ratio = abs(f_serve) / sqrt_mu_macro
                self.assertLess(
                    abs(ratio - 1.0), _DC_MAG_TOL,
                    f'|F_serve|/sqrt(mu_macro) = {ratio:.6f} is not within '
                    f'{_DC_MAG_TOL} of 1 at w={w:g} (the anchor is '
                    'sqrt(mu_macro), not 1)')
                self.assertLess(
                    abs(cmath.phase(f_serve)), _DC_ARG_TOL,
                    f'arg(F_serve) = {cmath.phase(f_serve):.6f} rad is not '
                    f'within {_DC_ARG_TOL} of 0 at w={w:g} (Morse-0 DC phase)')
                self.n_checks += 1

    def test_remodulation_matches_closed_form(self):
        """F_serve == exp(0.5j w (log lam - kappa s)) * C(w) * sqrt(mu_macro).

        The sharp pin: prefactor_c and the mass-sheet phase are each applied
        exactly once, and the anchor is sqrt(mu_macro).
        """
        for (label, kappa, beta, w, f_serve, lam, s,
             sqrt_mu_macro) in self._sweep():
            with self.subTest(label=label, kappa=kappa, beta=beta, w=w):
                mass_sheet = cmath.exp(0.5j * w * (math.log(lam) - kappa * s))
                expected = mass_sheet * prefactor_c(w) * sqrt_mu_macro
                rel = abs(f_serve - expected) / abs(expected)
                self.assertLess(
                    rel, _DC_SHARP_TOL,
                    f're-modulation off the closed form by {rel:.3e} at '
                    f'w={w:g}: a missed/doubled prefactor_c or mass-sheet '
                    'phase, or a wrong anchor')
                self.n_checks += 1


class ServeEngineNodeExactTestCase(_BaseChartTestCase):
    """Serve-vs-engine accuracy: the re-modulated F_serve == the engine.

    With the EXACT-residual chart, evaluating at grid NODES must reproduce
    `_engine_reference_kappa` to ~1e-14: the residual is node-exact by
    construction, so any residual mismatch is a re-modulation or
    normalization defect (missed/doubled prefactor_c or mass-sheet phase,
    wrong anchor, or an inconsistency between the training residual and the
    serve's reconstruction).  This is the fast-tier form of the accuracy
    invariant -- STRONGER than the shipped chart's 1e-4 off-grid bar, and it
    needs no full-bake artifact.
    """

    def test_remodulated_serve_matches_engine_at_nodes(self):
        """|F_serve - F_engine| / |F_engine| <= NODE_EXACT_TOL at grid nodes."""
        chart = _build_exact_chart()
        for (gp, rho, label), kappa, beta in (
                (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
            lens = _make_lens(gp, rho, THETA_NODE, kappa, beta)
            dense_w = np.array(SERVE_WS)
            f_serve = _serve_farfield(chart, lens, dense_w)
            if f_serve is None:
                self.fail(f'serve declined fixture {label} '
                          f'(gamma_prime={gp}, rho={rho}, kappa={kappa}, '
                          f'beta={beta}); the exact chart must cover it')
            y = (lens['y1'], lens['y2'])
            for i, w in enumerate(SERVE_WS):
                with self.subTest(label=label, kappa=kappa, beta=beta, w=w):
                    f_engine = _engine_reference_kappa(
                        w, y, lens['gamma'], beta, kappa)
                    rel = abs(f_serve[i] - f_engine) / abs(f_engine)
                    self.assertLess(
                        rel, NODE_EXACT_TOL,
                        f'F_serve disagrees with the engine by {rel:.3e} at '
                        f'w={w:g} ({label}); the re-modulation or residual '
                        'normalization is inconsistent')
                    self.n_checks += 1


class ConservativenessTestCase(_BaseChartTestCase):
    """One-sided conservativeness: the scalar de-rate is the sole margin.

    Cubic interpolation overshoots off-grid by construction (measured up to
    ~1.6x here on theta midpoints with a unit de-rate), so the serve must
    apply the chart's ``derate`` multiplicatively and a derate of
    ``1 / max_overshoot`` must make |F_serve| <= |F_engine| everywhere on the
    fixture set.
    """

    def _max_overshoot(self):
        """Sup over the fixture set of |F_serve|/|F_engine| at unit de-rate.

        ``F_serve`` here is the UN-de-rated re-modulation (derate = 1.0), so
        the ratio is the raw cubic-interpolation overshoot at grid nodes and
        off-grid theta midpoints.
        """
        chart = _build_exact_chart()
        worst = 1.0
        for (gp, rho, _label), kappa, beta in (
                (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
            dense_w = np.array(SERVE_WS)
            for theta in (THETA_NODE,) + THETA_MIDPOINTS:
                lens = _make_lens(gp, rho, theta, kappa, beta)
                f_serve = _serve_farfield(chart, lens, dense_w)
                if f_serve is None:
                    self.fail(f'serve declined (gamma_prime={gp}, rho={rho}, '
                              f'theta={theta}, kappa={kappa}, beta={beta})')
                for i, w in enumerate(SERVE_WS):
                    f_engine = _engine_reference_kappa(
                        w, (lens['y1'], lens['y2']), lens['gamma'], beta,
                        kappa)
                    worst = max(worst, abs(f_serve[i]) / abs(f_engine))
        return worst

    def test_unit_derate_overshoots_off_grid(self):
        """Cubic interpolation over-serves at unit de-rate (premise)."""
        worst = self._max_overshoot()
        self.assertGreater(
            worst, 1.0,
            f'max un-de-rated overshoot = {worst:.4f}; cubic interpolation '
            'did not overshoot, so the de-rate would have nothing to do')
        self.n_checks += 1

    def test_derated_serve_is_one_sided_conservative(self):
        """With derate = 1/max_overshoot, |F_serve| <= |F_engine| everywhere.

        A 10% safety margin (derate = (1/max_overshoot) / 1.1) keeps the
        worst-case ratio comfortably below 1.0 despite float round-off.
        """
        max_overshoot = self._max_overshoot()
        derate = (1.0 / max_overshoot) / 1.1
        chart = dataclasses.replace(_build_exact_chart(), derate=derate)
        for (gp, rho, _label), kappa, beta in (
                (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
            dense_w = np.array(SERVE_WS)
            for theta in (THETA_NODE,) + THETA_MIDPOINTS:
                lens = _make_lens(gp, rho, theta, kappa, beta)
                f_serve = _serve_farfield(chart, lens, dense_w)
                if f_serve is None:
                    self.fail(f'serve declined (gamma_prime={gp}, rho={rho}, '
                              f'theta={theta}, kappa={kappa}, beta={beta})')
                for i, w in enumerate(SERVE_WS):
                    with self.subTest(gp=gp, rho=rho, theta=theta,
                                      kappa=kappa, beta=beta, w=w):
                        f_engine = _engine_reference_kappa(
                            w, (lens['y1'], lens['y2']), lens['gamma'], beta,
                            kappa)
                        ratio = abs(f_serve[i]) / abs(f_engine)
                        self.assertLessEqual(
                            ratio, 1.0,
                            f'|F_serve|/|F_engine| = {ratio:.4f} > 1 at '
                            f'w={w:g}: the de-rate over-serves')
                        self.n_checks += 1

    def test_derate_scales_served_amplitude_at_nodes(self):
        """The serve applies the derate multiplicatively (node ratio == derate).

        At a grid node the un-de-rated re-modulation is node-exact
        (F == F_engine), so |F_serve|/|F_engine| must equal the derate.
        """
        chart = dataclasses.replace(_build_exact_chart(),
                                    derate=_CONSERVATIVE_DERATE)
        for (gp, rho, _label), kappa, beta in (
                (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
            lens = _make_lens(gp, rho, THETA_NODE, kappa, beta)
            y = (lens['y1'], lens['y2'])
            dense_w = np.array(SERVE_WS)
            f_serve = _serve_farfield(chart, lens, dense_w)
            if f_serve is None:
                self.fail('serve declined a node fixture it must cover')
            for i, w in enumerate(SERVE_WS):
                with self.subTest(gp=gp, rho=rho, kappa=kappa, beta=beta,
                                  w=w):
                    f_engine = _engine_reference_kappa(
                        w, y, lens['gamma'], beta, kappa)
                    ratio = abs(f_serve[i]) / abs(f_engine)
                    self.assertAlmostEqual(
                        ratio, _CONSERVATIVE_DERATE, delta=1e-12,
                        msg=f'|F_serve|/|F_engine| = {ratio:.4f} != derate '
                            f'{_CONSERVATIVE_DERATE} at w={w:g}: the derate '
                            'is not applied multiplicatively')
                    self.n_checks += 1


def _worst_serve_engine_ratio(derate: float) -> float:
    """Worst ``|F_serve|/|F_engine|`` over the fixture sweep at ``derate``.

    The same fixture sweep as `ConservativenessTestCase._max_overshoot`, but
    parameterized by the chart's de-rate so the self-falsification can compare
    the conservative derate against the derate removed.  Fails loudly if the
    serve declines any fixture (a coverage/coordinate defect, not a ratio).
    """
    chart = dataclasses.replace(_build_exact_chart(), derate=derate)
    worst = 0.0
    for (gp, rho, _label), kappa, beta in (
            (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
        dense_w = np.array(SERVE_WS)
        for theta in (THETA_NODE,) + THETA_MIDPOINTS:
            lens = _make_lens(gp, rho, theta, kappa, beta)
            f_serve = _serve_farfield(chart, lens, dense_w)
            if f_serve is None:
                raise AssertionError(
                    f'serve declined (gamma_prime={gp}, rho={rho}, '
                    f'theta={theta}, kappa={kappa}, beta={beta})')
            for i, w in enumerate(SERVE_WS):
                f_engine = _engine_reference_kappa(
                    w, (lens['y1'], lens['y2']), lens['gamma'], beta, kappa)
                worst = max(worst, abs(f_serve[i]) / abs(f_engine))
    return worst


class ConservativenessSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the de-rate is load-bearing (the conservativeness pin has teeth).

    `ConservativenessTestCase` is green at ``derate = (1/max_overshoot)/1.1``
    and asserts ``|F_serve| <= |F_engine|`` everywhere.  Removing the de-rate
    (``derate = 1.0``) re-runs the SAME witness sweep and must expose
    ``|F_serve| > |F_engine|`` at an over-serve witness -- the one-sided pin
    flips red the moment the margin is dropped, so the de-rate is doing real
    work rather than decorating an already-conservative serve.
    """

    def test_removing_derate_flips_one_sided_red(self):
        unit_worst = _worst_serve_engine_ratio(1.0)
        self.assertGreater(
            unit_worst, 1.0,
            f'unit-derate worst ratio = {unit_worst:.4f}: no over-serve '
            'appears when the de-rate is removed, so the de-rate would be a '
            'no-op and the conservativeness pin would be vacuous')
        conservative = (1.0 / unit_worst) / 1.1
        cons_worst = _worst_serve_engine_ratio(conservative)
        self.assertLessEqual(
            cons_worst, 1.0,
            f'conservative derate {conservative:.4f} worst ratio = '
            f'{cons_worst:.4f} > 1: the derate does not restore '
            'one-sidedness (should be impossible for the same sweep)')
        self.n_checks += 1

class RemodulationSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the re-modulation suite can go red.

    A re-modulation defect (the exact bug class the DC anchor and
    serve-vs-engine tests guard) must actually flip them.  Two mutations,
    each a single-factor corruption on the serve's module namespace, are
    asserted to break the node-exact agreement -- demonstrating the suite has
    teeth rather than passing vacuously.
    """

    def _node_rel_error_under(self, patch_target, patch_value) -> float:
        """Worst node-exact rel error under a patched serve factor."""
        chart = _build_exact_chart()
        worst = 0.0
        for (gp, rho, _label), kappa, beta in (
                (f, k, b) for f in FIXTURES for k in KAPPAS for b in BETAS):
            lens = _make_lens(gp, rho, THETA_NODE, kappa, beta)
            y = (lens['y1'], lens['y2'])
            dense_w = np.array(SERVE_WS)
            with mock.patch.object(_likelihood_mod, patch_target, patch_value):
                f_serve = _serve_farfield(chart, lens, dense_w)
            if f_serve is None:
                self.fail('serve declined under the mutation')
            for i, w in enumerate(SERVE_WS):
                f_engine = _engine_reference_kappa(
                    w, y, lens['gamma'], beta, kappa)
                worst = max(worst, abs(f_serve[i] - f_engine) / abs(f_engine))
        return worst

    def test_doubled_prefactor_breaks_node_exactness(self):
        """Applying prefactor_c twice blows the node-exact agreement."""
        real_prefactor = _likelihood_mod.prefactor_c
        rel = self._node_rel_error_under(
            'prefactor_c', lambda w: real_prefactor(w) ** 2)
        self.assertGreater(
            rel, NODE_EXACT_TOL,
            f'doubled prefactor left rel err {rel:.3e} <= {NODE_EXACT_TOL}; '
            'the re-modulation suite would not catch a doubled prefactor')
        self.n_checks += 1

    def test_unit_anchor_breaks_node_exactness(self):
        """Replacing sqrt_mu_full with 1 (anchor -> 1) breaks exactness."""
        real_born = _likelihood_mod._born_factors

        def _unit_sqrt_mu(y1, y2, gamma, beta, kappa):
            phi = real_born(y1, y2, gamma, beta, kappa)
            return (1.0,) + tuple(phi[1:])

        rel = self._node_rel_error_under('_born_factors', _unit_sqrt_mu)
        self.assertGreater(
            rel, NODE_EXACT_TOL,
            f'unit anchor left rel err {rel:.3e} <= {NODE_EXACT_TOL}; the '
            're-modulation suite would not catch an anchor=1 regression')
        self.n_checks += 1

def _save_chart_artifact(path: Path, chart: LowWDiffractiveChart,
                         schema: str = _SCHEMA,
                         content_hash: str | None = None,
                         drop_keys: tuple[str, ...] = ()) -> None:
    """Write a chart npz in the training-script's save format.

    ``content_hash`` defaults to the correct `_content_hash` over the chart's
    stored fields -- the 4 grids, both coefficient arrays, ``derate``, and
    ``declined_mask`` (the exact float64 bytes, matching
    ``scripts/train_low_w_diffractive_chart.py``); ``drop_keys`` removes
    named keys before writing to exercise the missing-key refusal paths.
    """
    arrays = {
        'gamma_prime_grid': chart.gamma_prime_grid,
        'rho_grid': chart.rho_grid,
        'theta_grid': chart.theta_grid,
        'log_w_grid': chart.log_w_grid,
        'real_coeffs': chart.real_coeffs,
        'imag_coeffs': chart.imag_coeffs,
        'derate': np.array(chart.derate),
        'declined_mask': chart.declined_mask,
        'provenance': np.array(json.dumps(chart.provenance)),
        'schema': np.array(schema),
    }
    if content_hash is None:
        content_hash = _content_hash(
            chart.gamma_prime_grid, chart.rho_grid, chart.theta_grid,
            chart.log_w_grid, chart.real_coeffs, chart.imag_coeffs,
            chart.derate, chart.declined_mask)
    arrays['content_hash'] = np.array(content_hash)
    for key in drop_keys:
        arrays.pop(key, None)
    np.savez(path, **arrays)


def _tamper_artifact(path: Path, key: str, mutation) -> None:
    """Rewrite ``key`` in an already-saved artifact, keeping ``content_hash``.

    Loads the arrays, applies ``mutation`` to the named array, and re-saves
    the whole artifact with the ORIGINAL content hash -- the exact "stale
    hash" corruption the loader must refuse.
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays[key] = mutation(np.asarray(arrays[key]))
    np.savez(path, **arrays)


def _round_trip_chart() -> LowWDiffractiveChart:
    """Small synthetic chart (non-trivial derate + provenance) for load tests."""
    gamma_prime_grid = np.array([0.2, 0.3, 0.8, 0.9])
    rho_grid = np.array([0.5, 1.0, 2.0, 2.5])
    theta_grid = np.array([0.2, 0.6, 1.0, 1.4])
    log_w_grid = np.log(np.array([0.3, 0.5, 2.0, 8.0]))
    shape = (4, 4, 4, 4)
    rng = np.random.default_rng(0)
    # Non-trivial mask: the middle two rho columns are declined (near-fold
    # resonance-limited cells), so an all-False tamper CHANGES the stored
    # bytes -- the exact stale-mask corruption the hash must detect.
    declined_mask = np.zeros(shape[:3], dtype=bool)
    declined_mask[:, 1:3, :] = True
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, log_w_grid=log_w_grid,
        real_coeffs=rng.standard_normal(shape),
        imag_coeffs=rng.standard_normal(shape), derate=0.85,
        declined_mask=declined_mask,
        provenance={'scale': 'test', 'derate': 0.85})


def _build_coverage_chart() -> LowWDiffractiveChart:
    """Synthetic chart whose grid BOX contains every coverage witness.

    The grid edges bracket the witnesses with interior margin so the band
    predicate (shell OR wall), not box containment, is what ``covers``
    decides.  Coefficients are never evaluated (zeros); each axis still has
    >= 4 nodes so the chart remains evaluable if reused.
    """
    gamma_prime_grid = np.array([0.1, 0.2, 0.6, 0.95])
    rho_grid = np.array([0.2, 0.3, 2.0, 3.0])
    theta_grid = np.linspace(0.0, math.pi / 2.0, 4)
    log_w_grid = np.log(np.array([0.02, 0.1, 1.0, 8.0]))
    shape = (4, 4, 4, 4)
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, log_w_grid=log_w_grid,
        real_coeffs=np.zeros(shape), imag_coeffs=np.zeros(shape), derate=1.0)


def _build_fold_chart() -> LowWDiffractiveChart:
    """D2-symmetric residual chart (``r = cos(2 theta)``), 8 theta nodes.

    ``cos(2 theta)`` is even in theta and pi-periodic -- the exact symmetry
    the fold relies on -- so `evaluate` at any of the four D2 octants must
    return the same value.  The coefficient genuinely varies in theta (8
    nodes resolve the even harmonics), so the octant equality is a statement
    about the FOLD (a broken fold would interpolate off-grid and diverge),
    not about a flat chart.
    """
    n_gp = n_rho = n_w = 4
    theta_grid = np.linspace(0.0, math.pi / 2.0, 8)
    gamma_prime_grid = np.linspace(0.2, 0.9, n_gp)
    rho_grid = np.linspace(0.5, 2.5, n_rho)
    log_w_grid = np.log(np.geomspace(0.05, 8.0, n_w))
    theta_2d = theta_grid.reshape(1, 1, -1, 1)
    real = np.broadcast_to(np.cos(2.0 * theta_2d),
                           (n_gp, n_rho, 8, n_w)).astype(float).copy()
    imag = np.zeros((n_gp, n_rho, 8, n_w), dtype=float)
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, log_w_grid=log_w_grid,
        real_coeffs=real, imag_coeffs=imag, derate=1.0)


class LoadContractTestCase(_BaseChartTestCase):
    """Schema + content-hash npz round-trip and hard-refusal contract.

    A schema-tagged, content-hashed artifact round-trips bit-identically
    through ``LowWDiffractiveChart.load``; a missing/foreign schema, a
    missing ``content_hash``, or a tampered grid/coefficient/derate value
    hard-refuses with a ``ValueError`` naming
    ``scripts/train_low_w_diffractive_chart.py``.  The loader can never
    silently deserialize a stale/corrupt chart.
    """

    def test_round_trip_is_bit_identical(self):
        """Every stored field survives a save -> load round-trip exactly."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart)
            loaded = LowWDiffractiveChart.load(path)
        for field in ('gamma_prime_grid', 'rho_grid', 'theta_grid',
                      'log_w_grid', 'real_coeffs', 'imag_coeffs',
                      'declined_mask'):
            np.testing.assert_array_equal(
                getattr(loaded, field), getattr(chart, field),
                err_msg=f'{field} did not round-trip bit-identically')
            self.n_checks += 1
        self.assertEqual(loaded.derate, chart.derate)
        self.n_checks += 1
        self.assertEqual(loaded.provenance, chart.provenance)
        self.n_checks += 1

    def test_missing_schema_hard_refuses(self):
        """An artifact without a ``schema`` key raises ValueError naming the script."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart, drop_keys=('schema',))
            with self.assertRaises(ValueError) as ctx:
                LowWDiffractiveChart.load(path)
        self.assertIn('train_low_w_diffractive_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_foreign_schema_hard_refuses(self):
        """A foreign schema tag raises ValueError naming the script."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart, schema='foreign_schema_v0')
            with self.assertRaises(ValueError) as ctx:
                LowWDiffractiveChart.load(path)
        self.assertIn('train_low_w_diffractive_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_missing_content_hash_hard_refuses(self):
        """An artifact without a ``content_hash`` key raises ValueError naming the script."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart, drop_keys=('content_hash',))
            with self.assertRaises(ValueError) as ctx:
                LowWDiffractiveChart.load(path)
        self.assertIn('train_low_w_diffractive_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_tampered_value_hard_refuses(self):
        """A tampered grid / coefficient / derate value raises ValueError.

        Each mutation rewrites a stored field while KEEPING the original
        content hash -- the exact "stale hash" corruption the loader must
        detect.  Sweeps the three tamper families (grid, coefficient, derate).
        """
        chart = _round_trip_chart()
        mutations = {
            'gamma_prime_grid': lambda a: a + 1.0,
            'real_coeffs': lambda a: a + 1.0,
            'derate': lambda a: a + 1.0,
        }
        for key, mutation in mutations.items():
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'chart.npz'
                    _save_chart_artifact(path, chart)
                    _tamper_artifact(path, key, mutation)
                    with self.assertRaises(ValueError) as ctx:
                        LowWDiffractiveChart.load(path)
                self.assertIn('train_low_w_diffractive_chart.py',
                              str(ctx.exception))
                self.n_checks += 1

    def test_tampered_declined_mask_hard_refuses(self):
        """An all-False ``declined_mask`` under a stale hash raises ValueError.

        The decline mask is the correctness-critical protection INS-1-001
        added: it flags the near-fold resonance-limited cells the chart must
        never serve, because an amplitude served from a declined cell exceeds
        the two-sided certification bar.  Un-declining those cells by
        tampering the mask to all-False is therefore a silent correctness
        corruption the content hash MUST detect (INS-2-002).  The mutation
        keeps the ORIGINAL stored hash -- the stale-artifact corruption -- so
        this pins the load-time refusal, not a format/schema guard.

        The fixture mask is derived from ``_round_trip_chart``'s non-trivial
        mask (middle two rho columns declined); the premise assertions guard
        against a future fixture edit collapsing it to all-False, which would
        make the tamper a no-op and the test vacuous.

        NOTE (authoring-time state): this test is RED until the INS-2-002
        production fix lands -- ``LowWDiffractiveChart.load`` must add
        ``declined_mask`` to its recomputed ``_content_hash`` call
        (cogwheel/lensing/low_w_diffractive_chart.py, ``actual =`` line;
        HEAD 3a56e97 hashes only the 7 pre-mask fields).  It flips green with
        zero further edits once that call covers the mask, because both the
        test helper and the loader then hash the identical bytes.
        """
        chart = _round_trip_chart()
        self.assertTrue(chart.declined_mask.any(),
                        'premise lost: fixture mask is all-False')
        self.assertFalse(chart.declined_mask.all(),
                         'premise lost: fixture mask is all-True')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart)
            _tamper_artifact(path, 'declined_mask',
                             lambda a: np.zeros_like(a, dtype=bool))
            with self.assertRaises(ValueError) as ctx:
                LowWDiffractiveChart.load(path)
        self.assertIn('train_low_w_diffractive_chart.py', str(ctx.exception))
        self.n_checks += 1


class LoadContractSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the load-contract refusals are the content hash's doing.

    A tampered value with a STALE hash refuses (``LoadContractTestCase``); a
    tampered value with a FRESH hash loads cleanly.  That pair is what shows
    the hash -- not a format/schema guard -- discriminates the refusal.
    """

    def test_rehashed_tamper_loads_cleanly(self):
        """Tampered derate with a matching hash round-trips (positive control)."""
        chart = _round_trip_chart()
        tampered = dataclasses.replace(chart, derate=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, tampered)
            loaded = LowWDiffractiveChart.load(path)
        self.assertEqual(loaded.derate, 0.5)
        self.n_checks += 1

    def test_rehashed_declined_mask_tamper_loads_cleanly(self):
        """All-False mask with a matching hash round-trips (positive control).

        The all-False mask is the INS-2-002 tamper of
        ``LoadContractTestCase.test_tampered_declined_mask_hard_refuses``;
        RE-hashing it must load cleanly.  This is what pins the refusal there
        to the hash BYTES (changed mask -> changed hash), not to a shape or
        dtype guard on the mask -- a non-hash refusal path would trip here
        too, and the pair would both be green for the wrong reason.
        """
        chart = _round_trip_chart()
        tampered = dataclasses.replace(
            chart, declined_mask=np.zeros_like(chart.declined_mask))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, tampered)
            loaded = LowWDiffractiveChart.load(path)
        np.testing.assert_array_equal(loaded.declined_mask,
                                      tampered.declined_mask)
        self.n_checks += 1

class CoverageUnionPredicateTestCase(_BaseChartTestCase):
    """Coverage union predicate: ``(shell) or (wall band)``.

    The exact serve/decline boundary is the union
    ``RHO_LO <= rho <= RHO_HI`` (inclusive, matching the ``w_low_fit`` fence)
    OR ``gamma' > _WALL_GAMMA_PRIME``, intersected with the grid box and
    (optionally) the trained log-w range.  The witnesses are DERIVED from the
    live gate constants, so a gate move fails the premise assertion loudly
    rather than stranding a literal.
    """

    def test_fixture_premises_hold(self):
        """The derived witnesses sit on the intended side of each gate clause."""
        self.assertGreaterEqual(NEAR_FOLD_RHO, RHO_LO)
        self.assertLessEqual(NEAR_FOLD_RHO, RHO_HI)
        self.assertLess(RHO_BELOW_SHELL, RHO_LO)
        self.assertGreater(RHO_ABOVE_SHELL, RHO_HI)
        self.assertLess(GAMMA_PRIME_BELOW_WALL, _WALL_GAMMA_PRIME)
        self.assertGreater(WALL_GAMMA_PRIME, _WALL_GAMMA_PRIME)
        self.assertLess(NEAR_FOLD_GAMMA_PRIME, _WALL_GAMMA_PRIME)
        self.n_checks += 1

    def test_near_fold_shell_low_gamma_prime_covers(self):
        """A near-fold-shell draw at low gamma' is covered (shell clause)."""
        chart = _build_coverage_chart()
        self.assertTrue(chart.covers(NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO))
        self.n_checks += 1

    def test_wall_band_exterior_covers(self):
        """A wall-band exterior draw (rho above the shell) is covered."""
        chart = _build_coverage_chart()
        self.assertTrue(chart.covers(WALL_GAMMA_PRIME, WALL_RHO))
        self.n_checks += 1

    def test_wall_band_interior_covers(self):
        """A wall-band interior draw (rho below the shell) is covered."""
        chart = _build_coverage_chart()
        self.assertTrue(chart.covers(WALL_GAMMA_PRIME, RHO_BELOW_SHELL))
        self.n_checks += 1

    def test_deep_interior_low_gamma_prime_declines(self):
        """A deep-interior draw at low gamma' (neither clause) is declined."""
        chart = _build_coverage_chart()
        self.assertFalse(chart.covers(GAMMA_PRIME_BELOW_WALL, RHO_BELOW_SHELL))
        self.n_checks += 1

    def test_smooth_exterior_low_gamma_prime_declines(self):
        """A smooth-exterior draw at low gamma' (neither clause) is declined."""
        chart = _build_coverage_chart()
        self.assertFalse(chart.covers(GAMMA_PRIME_BELOW_WALL, RHO_ABOVE_SHELL))
        self.n_checks += 1

    def test_shell_boundary_is_inclusive(self):
        """rho == RHO_LO and rho == RHO_HI are covered (inclusive fence)."""
        chart = _build_coverage_chart()
        for rho in (RHO_LO, RHO_HI):
            with self.subTest(rho=rho):
                self.assertTrue(chart.covers(NEAR_FOLD_GAMMA_PRIME, rho))
                self.n_checks += 1

    def test_w_beyond_log_w_ceiling_declines(self):
        """A band whose w extends beyond the trained log-w ceiling is declined."""
        chart = _build_coverage_chart()
        w_hi = float(np.exp(chart.log_w_grid[-1] + 0.1))
        self.assertFalse(chart.covers(NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO,
                                      np.array([w_hi])))
        self.n_checks += 1

    def test_w_within_log_w_range_covers(self):
        """A band at the trained log-w ceiling (inclusive) is covered."""
        chart = _build_coverage_chart()
        w_hi = float(np.exp(chart.log_w_grid[-1]))
        self.assertTrue(chart.covers(NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO,
                                     np.array([w_hi])))
        self.n_checks += 1


class CoverageUnionSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the coverage union is reading the live gate constants.

    Lowering the wall-band constant (or raising the shell floor) must flip a
    declined witness to covered -- demonstrating ``covers`` consults the
    module globals, not a hardcoded copy.
    """

    def test_wall_band_constant_is_load_bearing(self):
        """Lowering _WALL_GAMMA_PRIME admits the smooth-exterior witness."""
        chart = _build_coverage_chart()
        self.assertFalse(chart.covers(GAMMA_PRIME_BELOW_WALL, RHO_ABOVE_SHELL))
        with mock.patch.object(_lwd_module, '_WALL_GAMMA_PRIME',
                               GAMMA_PRIME_BELOW_WALL - 0.1):
            self.assertTrue(chart.covers(GAMMA_PRIME_BELOW_WALL,
                                         RHO_ABOVE_SHELL))
        self.n_checks += 1

    def test_shell_floor_constant_is_load_bearing(self):
        """Raising RHO_LO above the near-fold witness declines it."""
        chart = _build_coverage_chart()
        self.assertTrue(chart.covers(NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO))
        with mock.patch.object(_lwd_module, 'RHO_LO', NEAR_FOLD_RHO + 0.5):
            self.assertFalse(chart.covers(NEAR_FOLD_GAMMA_PRIME,
                                          NEAR_FOLD_RHO))
        self.n_checks += 1

class ThetaD2FoldTestCase(_BaseChartTestCase):
    """D2 theta fold: ``evaluate`` collapses the four octants to one value.

    The residual is even in theta and pi-periodic, so the four D2 octants
    ``{theta, -theta, pi-theta, pi+theta}`` all fold to ``[0, pi/2]`` and
    must return the same interpolated value.  The chart's residual genuinely
    varies in theta, so this equality is the fold's doing -- a broken fold
    would interpolate off the ``[0, pi/2]`` grid (extrapolation) and diverge.
    """

    _W = np.array([0.1, 1.0, 4.0])

    def test_four_d2_octants_agree(self):
        """evaluate(theta) == evaluate(-theta) == evaluate(pi ± theta)."""
        chart = _build_fold_chart()
        for theta in (0.6, 1.0, 1.4):
            octants = (theta, -theta, math.pi - theta, math.pi + theta)
            for gp in (0.3, 0.8):
                for rho in (1.0, 2.0):
                    with self.subTest(theta=theta, gp=gp, rho=rho):
                        vals = [chart.evaluate(self._W, gp, rho, o)
                                for o in octants]
                        for o, v in zip(octants[1:], vals[1:]):
                            np.testing.assert_allclose(
                                v, vals[0], rtol=1e-12, atol=1e-12,
                                err_msg=f'octant {o:.4f} disagrees with the '
                                        'folded value')
                            self.n_checks += 1

    def test_fold_lands_on_folded_domain_value(self):
        """evaluate(-theta) equals the analytic cos(2 theta) at |theta|.

        Proves the fold maps to the RIGHT folded point (not just a consistent
        one): the stored residual is cos(2 theta) at 8 nodes, and cubic
        interpolation reproduces it to ~5e-4 at an off-grid theta.
        """
        chart = _build_fold_chart()
        for theta in (0.6, 1.0):
            with self.subTest(theta=theta):
                v = chart.evaluate(np.array([1.0]), 0.3, 1.0, -theta)
                expected = math.cos(2.0 * theta)
                self.assertLess(
                    abs(float(v[0].real) - expected), 2e-3,
                    f'folded evaluate at -{theta} gives {v[0].real:.4f}, '
                    f'expected cos(2 theta) = {expected:.4f}')
                self.n_checks += 1


class ThetaD2FoldSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the theta fold is load-bearing, not an accident of a flat chart.

    (1) The residual genuinely varies in theta (premise): evaluate at two
    distinct folded-domain angles differs.  (2) An independent NO-FOLD
    interpolator queried at a raw (non-folded) octant angle diverges from
    the folded answer -- the fold, not cubic-interpolation luck, produces
    the four-octant equality.
    """

    def test_residual_genuinely_varies_in_theta(self):
        """The chart is not theta-flat, so the fold test is non-vacuous."""
        chart = _build_fold_chart()
        w = np.array([1.0])
        v_low = chart.evaluate(w, 0.3, 1.0, 0.6)
        v_high = chart.evaluate(w, 0.3, 1.0, 1.2)
        self.assertGreater(
            abs(float(v_low[0].real - v_high[0].real)), 0.1,
            'residual is nearly flat in theta; the fold test would be vacuous')
        self.n_checks += 1

    def test_no_fold_interpolation_diverges(self):
        """A no-fold interpolator at a raw octant angle diverges from the fold."""
        chart = _build_fold_chart()
        theta = 0.6
        gp, rho, w = 0.3, 1.0, 1.0
        folded = chart.evaluate(np.array([w]), gp, rho, math.pi - theta)
        real_interp = RegularGridInterpolator(
            (chart.gamma_prime_grid, chart.rho_grid, chart.theta_grid,
             chart.log_w_grid),
            chart.real_coeffs, method='cubic', bounds_error=False,
            fill_value=None)
        points = np.array([[gp, rho, math.pi - theta, math.log(w)]])
        no_fold = float(real_interp(points)[0])
        self.assertGreater(
            abs(no_fold - folded[0].real), 1e-2,
            f'no-fold interpolation ({no_fold:.4f}) matched the folded value '
            f'({folded[0].real:.4f}); the fold is not doing anything')
        self.n_checks += 1


# ---------------------------------------------------------------------------
# Census mirror: serve_route_census.classify_draw routes the near-fold shell
# to the 12th SERVE_ROUTE ``low_w_diffractive_chart`` (served == counted).
# ---------------------------------------------------------------------------

#: Deterministic dimensionless-w band handed to the census mirror: the floor
#: (0.05) sits below the witness's ``farfield_w_floor`` (~1.24) and the
#: ceiling (1.0) is far below the Schwinger QD ceiling, so the above-ceiling
#: rung (intercept 3) is skipped and the low-w diffractive rung owns the band.
_CENSUS_W_GRID = np.array([0.05, 0.1, 0.2, 0.5, 1.0])

#: The near-fold-shell witness in the ``NEAR_FOLD_DECLINED_WITNESSES`` shape
#: of ``test_lensing_diffractive`` (``gamma=0.3, beta=-1.1`` at
#: ``Y_REF=(0.8, 0.4)``).  The census fixes ``kappa == beta == 0``, so the
#: lens-frame source is the beta-rotated ``Y_REF``; its DIRECTIONAL reduced
#: caustic ratio ``rho_dir = _caustic_rho(0.3, s, theta) = 1.247`` falls
#: inside the near-fold shell ``[RHO_LO, RHO_HI]`` (premise-asserted on the
#: directly-computed directional rho, NOT ``res.caustic_rho`` -- that field
#: is the SCALAR `ppgo_map.caustic_rho` gauge and must stay that way).
_NEAR_FOLD_WITNESS_GAMMA = 0.3
_NEAR_FOLD_WITNESS_BETA = -1.1
_WITNESS_Y_REF = (0.8, 0.4)
_witness_z = cmath.exp(-1j * _NEAR_FOLD_WITNESS_BETA) * complex(*_WITNESS_Y_REF)
NEAR_FOLD_WITNESS_Y1 = _witness_z.real
NEAR_FOLD_WITNESS_Y2 = _witness_z.imag


@functools.lru_cache(maxsize=1)
def _census_base_mods():
    """The real production-module bundle (loaded once, engine-free)."""
    return _census._load_production_modules()


def _census_mods(chart: LowWDiffractiveChart):
    """Synthetic `_ProductionModules` with ONLY the chart + w-band swapped.

    Pins the low-w diffractive chart and ``dimensionless_frequency`` to the
    deterministic ``_CENSUS_W_GRID``; every other field -- the real geometry
    class, ``farfield_w_floor``, ``reduced_shear``, ``caustic_rho``,
    ``w_low_fit`` -- stays the production object, so ``classify_draw`` runs
    the shipped waterfall unchanged.
    """
    base = _census_base_mods()
    return dataclasses.replace(
        base, low_w_chart=chart,
        dimensionless_frequency=lambda f_grid, m, z: np.array(
            _CENSUS_W_GRID, copy=True))


def _classify_near_fold_witness(chart: LowWDiffractiveChart):
    """Classify the near-fold-shell witness through the real waterfall."""
    return _census.classify_draw(
        _census_mods(chart), gamma=_NEAR_FOLD_WITNESS_GAMMA,
        m_lens_msun=1.0, y1=NEAR_FOLD_WITNESS_Y1, y2=NEAR_FOLD_WITNESS_Y2,
        f_grid=np.geomspace(20.0, 1024.0, _CENSUS_W_GRID.size),
        gamma_edges=_ppgo_map._gamma_band_edges())

def _witness_directional_rho() -> float:
    """Directional reduced caustic reach of the near-fold witness.

    The chart/census consult the DIRECTION-DEPENDENT `_caustic_rho` (the SAME
    discriminator the near-fold fence and serve use), NOT the scalar
    `ppgo_map.caustic_rho` gauge recorded in ``res.caustic_rho``.  Computed
    directly here so the shell-premise and covers-spy assertions are keyed on
    the directional value, never on a field that must stay scalar.
    """
    s = (NEAR_FOLD_WITNESS_Y1 * NEAR_FOLD_WITNESS_Y1
         + NEAR_FOLD_WITNESS_Y2 * NEAR_FOLD_WITNESS_Y2)
    theta = math.atan2(NEAR_FOLD_WITNESS_Y2, NEAR_FOLD_WITNESS_Y1)
    return _caustic_rho(abs(_NEAR_FOLD_WITNESS_GAMMA), s, theta)


class CensusMirrorTestCase(_BaseChartTestCase):
    """Census mirror: the near-fold shell counts as ``low_w_diffractive_chart``.

    The 12th ``SERVE_ROUTES`` label (positive-parity Rung P) is emitted by the
    REAL ``serve_route_census.classify_draw`` waterfall for the
    NEAR_FOLD_DECLINED_WITNESSES-shaped draw (``gamma=0.3, beta=-1.1`` at
    ``Y_REF``) when the chart's ``covers`` admits it, with EMPTY per-node
    kinds (a whole-band ANALYTIC serve -- zero engine demand) and therefore
    NOT in ``engine_residual``.  The route is governed by the SAME
    ``chart.covers`` predicate the production serve consults, so the census's
    served set equals the serve's counted set.
    """

    def test_serve_routes_header_lists_new_route(self):
        """SERVE_ROUTES has 12 entries, chart ordered before the fit split."""
        self.assertEqual(len(_census.SERVE_ROUTES), 12,
                         'SERVE_ROUTES must have exactly 12 draw-level routes')
        self.assertIn('low_w_diffractive_chart', _census.SERVE_ROUTES)
        self.assertLess(
            _census.SERVE_ROUTES.index('low_w_diffractive_chart'),
            _census.SERVE_ROUTES.index('diffractive_analytic'),
            'the chart route must precede the w_low_fit split in the '
            'decision-order header (it short-circuits the split)')
        self.n_checks += 1

    def test_near_fold_shell_witness_routes_to_chart(self):
        """The witness routes to 'low_w_diffractive_chart' with empty kinds.

        Premise (asserted on the DIRECTLY-computed directional rho, not the
        scalar ``res.caustic_rho`` gauge): the witness's directional reduced
        caustic ratio lies inside the near-fold shell, so the chart's shell
        clause (not the wall band) is what admits it -- the classification is
        the fence doing its job, not a degenerate refusal.
        """
        res = _classify_near_fold_witness(_build_coverage_chart())
        rho_dir = _witness_directional_rho()
        self.assertGreaterEqual(rho_dir, RHO_LO,
                                'premise lost: witness no longer in the shell')
        self.assertLessEqual(rho_dir, RHO_HI,
                             'premise lost: witness no longer in the shell')
        self.assertLess(_NEAR_FOLD_WITNESS_GAMMA, _WALL_GAMMA_PRIME,
                        'premise lost: witness now in the wall band')
        self.assertEqual(res.route, 'low_w_diffractive_chart')
        self.assertEqual(res.node_route_kinds, ())
        self.assertNotEqual(res.route, 'engine_residual')
        self.n_checks += 1

    def test_census_consults_chart_covers(self):
        """The census routes on ``chart.covers`` (reduced coords), exactly once.

        A spy on ``LowWDiffractiveChart.covers`` records the predicate the
        census consults: it must be called once with the reduced coordinates
        the production serve uses (``gamma_prime == gamma``, ``rho ==`` the
        DIRECTIONAL ``_caustic_rho``, the full served w band), never a
        re-derived shell/wall predicate nor the scalar ``res.caustic_rho``
        gauge.
        """
        chart = _build_coverage_chart()
        calls: list[tuple[float, float, np.ndarray | None]] = []
        real_covers = LowWDiffractiveChart.covers

        def spy(self, gamma_prime, rho, w=None):
            calls.append((gamma_prime, rho,
                          None if w is None else np.asarray(w).copy()))
            return real_covers(self, gamma_prime, rho, w)

        with mock.patch.object(LowWDiffractiveChart, 'covers', spy):
            res = _classify_near_fold_witness(chart)
        self.assertEqual(res.route, 'low_w_diffractive_chart')
        self.assertEqual(len(calls), 1,
                         'the census should consult covers exactly once')
        gp, rho, w = calls[0]
        self.assertAlmostEqual(gp, _NEAR_FOLD_WITNESS_GAMMA, delta=1e-12)
        self.assertAlmostEqual(rho, _witness_directional_rho(), delta=1e-12)
        np.testing.assert_array_equal(w, _CENSUS_W_GRID)
        self.n_checks += 1


class CensusMirrorSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the census route is governed by ``chart.covers``, not re-derived.

    The census must NOT re-derive the shell/wall predicate itself: flipping
    ``covers`` to False for the SAME witness (which genuinely lies inside the
    near-fold shell) must divert the route off ``low_w_diffractive_chart`` --
    here to ``engine_residual`` (the fit fence ``w_low_fit`` declines the
    shell, so the draw falls through to the exact-wave node pass).
    """

    def test_covers_false_diverts_off_chart_route(self):
        chart = _build_coverage_chart()
        with mock.patch.object(LowWDiffractiveChart, 'covers',
                               return_value=False):
            res = _classify_near_fold_witness(chart)
        self.assertNotEqual(
            res.route, 'low_w_diffractive_chart',
            'flipping covers to False left the route on the chart: the '
            'census is not actually consulting covers')
        self.assertEqual(
            res.route, 'engine_residual',
            'the declined near-fold witness should fall through to the '
            'engine node pass (w_low_fit declines the shell)')
        self.n_checks += 1


if __name__ == '__main__':
    main()
