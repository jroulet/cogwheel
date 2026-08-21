"""
Tests for `lensing.low_w_diffractive_chart` + its likelihood serve.

WHAT THIS SUITE ADJUDICATES
---------------------------
`LowWDiffractiveChart` holds a trained 4-D tensor-product interpolation of the
low-w diffractive residual

    r_new(w; gamma', rho, theta) = f_pure * sqrt(1 - gamma'^2) / F_ref(w)

where ``F_ref`` is the RHO-PARTITIONED uniform reference
`partitioned_reference` -- the Airy fold q=p Wronskian form
(``|F_ref|^2 ~ w^{1/3} Ai^2 + w^{-1/3} Ai'^2``, magnitude-renormalized to the
macro lead at low w), the restricted Pearcey cusp form (only on the ``b3 -> 0``
fold->cusp transition), or the macro lead carrier `born_lead_carrier`
(off-caustic) -- and
`LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve` re-modulates
it back to the full amplitude

    F_serve = mass_sheet_phase * F_ref(w) * sqrt_mu_full * r_new.

``F_ref`` replaces the point-mass prefactor ``C(w)`` ONLY; the macro
normalization ``sqrt(1 - gamma'^2)`` stays in the residual, so
``F_serve = mass_sheet_phase * f_pure / lam`` -- the exact engine.  The serve
is the Rung-P replacement for the exact Schwinger engine over the near-fold
shell (`w_low_fit` declines there -> None) and the wall band
(`gamma' > 0.5`).  Off-caustic cells split the band at the resolution boundary
``w * delta_tau = RHO_END``: resolved nodes use the two-image geometric sum,
unresolved nodes the macro-lead re-modulation.  Invariants:

  1. F_REF NON-VANISHING (`FrefNonVanishingTestCase`).  The q=p Wronskian
     combination ``w^{1/3} Ai(-xi)^2 + w^{-1/3} Ai'(-xi)^2`` is strictly
     positive at every node (the q=0 form would vanish at the Airy zeros
     ``xi = 2.338, 4.088``), and the built ``|F_ref|`` stays O(1)
     (``min|F_ref|/max|F_ref| >= 3e-1`` for the shell witness; the macro
     carrier's magnitude is EXACTLY w-independent).  Engine-free (geometry +
     ``scipy.special.airy``).

  2. RESIDUAL BOUNDEDNESS (`ResidualBoundednessTestCase`).  The residual
     ``r = f_pure * sqrt(1 - gamma'^2) / F_ref`` is a smooth O(1) complex
     function -- no magnitude collapse (``min|r|/max|r| >= 2e-1`` shell) and
     no Airy-zero crossing (unwrapped arg steps < pi/2).  The BROKEN
     representation (divide by ``prefactor_c``) collapses and jumps, pinned by
     a self-falsification class.

  3. SERVE-VS-ENGINE (`ServeEngineNodeExactTestCase`).  With an EXACT-residual
     chart the re-modulated F_serve must reproduce the exact engine to ~1e-14
     at grid NODES (the residual is node-exact by construction, so the only
     thing left is the re-modulation + normalization consistency).  This is
     the fast-tier form of the accuracy invariant -- STRONGER than the shipped
     chart's 1e-4 off-grid bar.  The shipped chart's off-grid accuracy is
     certified by ``scripts/train_low_w_diffractive_chart.py``'s margin report
     (a DRIVER post-build step, not a unit test).

  4. ONE-SIDED CONSERVATIVENESS (`ConservativenessTestCase`).  Cubic
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
recipe as ``test_lensing_diffractive._engine_reference_kappa``).  The chart
serve never calls ``f_schwinger`` for its band (the chart is the sole
serve-time source); the residual-boundedness spec evaluates ``f_schwinger``
directly because the geometric-optics 2-image sum is singular at the merging
fold pair (magnification -> inf) and so cannot stand in for ``f_pure`` in the
near-fold shell -- the exact engine at ``w <= 60`` (double-double path) is the
correct cheap oracle.  ``w`` stays <= 60 so the engine runs on its exact
double-double path (mpmath only above ~60).

TOLERANCES
----------
* `NODE_EXACT_TOL` = 1e-10: the node-exact re-modulation reconstructs
  F_engine to ~1e-15 (the residual/F_ref round-trip is node-exact to float64);
  1e-10 leaves ~1e5 margin.
* `SHELL_FREF_RATIO_TOL` = 3e-1 (shell): the measured |F_ref| min/max is
  0.515 (fold, under the macro-fold renormalization), far above the bar --
  the q=0 form would dive toward 0 at the Airy zeros.  The macro carrier's
  |F_ref| is EXACTLY ``sqrt_mu`` (w-independent), pinned separately.
* `SHELL_RESIDUAL_RATIO_TOL` = 2e-1 (shell): the measured fold residual
  min/max is 0.295, ~1.5x above the bar; the broken prefactor_c
  representation collapses to ~0.075 (3.9x below).
* `MACRO_RESIDUAL_RATIO_TOL` = 1e-1: the macro-carrier residual min/max over
  its unresolved domain is 0.758 (wall witness), far above the bar.
"""

from __future__ import annotations

import cmath
import dataclasses
import functools
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import TestCase, main, mock

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import airy

from cogwheel.lensing import likelihood as _likelihood_mod
from cogwheel.lensing import low_w_diffractive_chart as _lwd_module
from cogwheel.lensing import ppgo_map as _ppgo_map
from cogwheel.lensing import serve_route_census as _census
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.low_w_diffractive_chart import (
    RHO_HI, RHO_LO, _WALL_GAMMA_PRIME, _SCHEMA, _content_hash,
    LowWDiffractiveChart, partitioned_reference, reduced_source,
    _airy_fold_form, _pearcey_cusp_reference, _reduced_min_delay_separation)
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import operator as _operator
from cogwheel.lensing.chang_refsdal._airy_fold import (
    _merging_fold_pair, _soft_axis_cubic)
from cogwheel.lensing.chang_refsdal.operator import RHO_END, RHO_START
from cogwheel.lensing.chang_refsdal._diffractive import _caustic_rho
from cogwheel.lensing.chang_refsdal._gauge import smootherstep
from cogwheel.lensing.chang_refsdal._hyp1f1 import prefactor_c
from cogwheel.lensing.chang_refsdal._schwinger import f_schwinger


#: Reduced shears spanning the two chart-owned regions: the near-fold shell
#: (``gamma' = 0.3``, where ``w_low_fit`` declines) and the wall band
#: (``gamma' = 0.8``, ``gamma' > _WALL_GAMMA_PRIME = 0.5``).
NEAR_FOLD_GAMMA_PRIME = 0.3
WALL_GAMMA_PRIME = 0.8

#: Caustic-relative distances paired with the regions above.  BOTH fixtures
#: sit at rho = 1.0 (inside the shell fence ``[RHO_LO, RHO_HI] = [0.6, 1.4]``
#: and, for the wall fixture, on the caustic): the two regions are
#: distinguished by ``gamma'`` (0.3 shell vs 0.8 wall band), not by rho.
#: The fixtures sit ON the caustic (rho = 1.0) so they exercise the Airy fold
#: carrier (the caustic neighborhood ``[RHO_LO, RHO_HI]``).  The genuinely
#: EXTERIOR wall draw (rho ~ 2.0 > RHO_HI) is served by the MACRO carrier
#: (always buildable) and is pinned by `MacroCarrierReferenceTestCase` /
#: `PerCarrierServeAccuracyTestCase`; its coverage is also pinned by the
#: ``covers`` predicate test via `RHO_ABOVE_SHELL`.
NEAR_FOLD_RHO = 1.0
WALL_RHO = 1.0

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

#: Frequencies (grid NODES) exercising the serve-vs-engine / conservativeness
#: accuracy on the exact-residual chart.
SERVE_WS = (0.5, 2.0, 8.0)

#: Exact-residual chart grids (>= 4 nodes per axis for scipy cubic).  The
#: fixture coordinates (gamma' = 0.3/0.8, rho = 1.0, theta = 0.6,
#: w = 0.5/2/8) are INTERIOR nodes, so a float64 round-off in the serve's
#: coordinate reconstruction (~1e-16) cannot push them outside the grid and
#: trip `covers` (which uses inclusive <= on the grid edges).  The rho grid
#: is capped at 1.05 because beyond ~1.1 the theta-edge direction theta = 1.4
#: resolves to a cusp (b3 -> 0) which is currently DECLINED (the
#: restricted-Pearcey fallback is keyed on the wrong refusal -- see
#: `CuspFrefNonVanishingTestCase._witness`), and every cell of the exact
#: chart must carry a finite residual.
_GAMMA_GRID = np.array([0.2, 0.3, 0.8, 0.9])
_RHO_GRID = np.array([0.5, 0.8, 1.0, 1.05])
_THETA_GRID = np.array([0.2, 0.6, 1.0, 1.4])
_W_GRID = np.array([0.3, 0.5, 2.0, 8.0])

#: Serve-vs-engine accuracy bar at grid nodes (see module docstring).
NODE_EXACT_TOL = 1e-10

#: Scalar derate used to prove the serve applies it multiplicatively.
_CONSERVATIVE_DERATE = 0.5

#: Eigenframe source angle of the SHELL witness for the F_ref / residual
#: specs.  theta = 1.4 (a `_THETA_GRID` node) is where the near-fold
#: residual is cleanest: min|r|/max|r| = 0.295 with a 0.47-rad unwrapped arg
#: step, a wide margin over the 2e-1 ratio bar and the pi/2 arg-jump guard.
#: (At THETA_NODE = 0.6 the residual's arg winds ~3 rad -- an on-caustic
#: higher-order term F_ref does not cancel -- so the shell witness uses its
#: own angle rather than the serve fixture's THETA_NODE.)
FREF_SHELL_THETA = 1.4

#: Synthetic dimensionless-frequency grid spanning [0.02, 60] (log-uniform,
#: 24 nodes) for the F_ref-non-vanishing and residual-boundedness witnesses.
_FREF_W_GRID = np.geomspace(0.02, 60.0, 24)

#: min|F_ref|/max|F_ref| bar for the FOLD shell witness (the q=p Wronskian
#: form never vanishes; the q=0 form dives toward 0 at the Airy zeros
#: xi = 2.338, 4.088).  Measured 0.515 on `_FREF_W_GRID` under the macro-fold
#: low-w renormalization (was 0.772 before the renormalization; the fold
#: reference is now even more O(1)).
SHELL_FREF_RATIO_TOL = 3e-1

#: min|r|/max|r| bar for the FOLD residual (no magnitude collapse).  Measured
#: 0.295 on `_FREF_W_GRID`; the broken prefactor_c representation collapses
#: to ~0.075 (3.9x below the bar).
SHELL_RESIDUAL_RATIO_TOL = 2e-1

#: min|r|/max|r| bar for the MACRO-carrier residual over its unresolved
#: domain (``w < w_split``) -- the macro residual stays O(1) there (measured
#: 0.758 for the wall witness); it only oscillates/collapses above the
#: resolved/unresolved split, which is outside the guard's domain.
MACRO_RESIDUAL_RATIO_TOL = 1e-1

#: Max unwrapped |arg r| step between adjacent w-nodes (pi/2): a zero
#: crossing forces a >= pi jump.
RESIDUAL_ARG_STEP_TOL = 0.5 * math.pi

#: (gamma', rho, theta, f_ref_bar, residual_bar, label) witnesses for the
#: FOLD-carrier specs (F_ref-non-vanishing + residual-boundedness).  The fold
#: shell draw (gamma'=0.3, rho=1.0, theta=1.4) is inside the caustic
#: neighborhood (``RHO_LO <= rho <= RHO_HI``), so `partitioned_reference`
#: builds the Airy fold carrier.  The wall-band exterior draw (gamma'=0.8,
#: rho=2.0) is OFF-caustic (``rho > RHO_HI``) and is served by the macro
#: carrier, whose invariants are pinned separately by
#: `MacroCarrierReferenceTestCase`.
_FREF_WITNESSES = (
    (NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO, FREF_SHELL_THETA,
     SHELL_FREF_RATIO_TOL, SHELL_RESIDUAL_RATIO_TOL, 'near_fold_shell'),
)

#: Wall-band EXTERIOR witness (off-caustic, ``rho > RHO_HI``) -- now BUILDABLE
#: via the macro lead carrier (`born_lead_carrier`) instead of the (retired)
#: fold/cusp form.  Pinned by `MacroCarrierReferenceTestCase`.
MACRO_WITNESS = (WALL_GAMMA_PRIME, 2.0, THETA_NODE)

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


def _residual_at(w_grid: np.ndarray, gamma_prime: float, rho: float,
                 theta: float) -> np.ndarray | None:
    """Airy-anchored residual ``r_new = f_pure * sqrt(1 - gamma'**2) / F_ref``.

    Mirrors ``scripts/train_low_w_diffractive_chart._residual_at``: rebuilds
    the reduced eigenframe source from the chart coordinates via
    `reduced_source`, builds the rho-partitioned uniform reference
    ``F_ref`` ONCE on ``w_grid`` via `partitioned_reference`, then evaluates
    the exact engine per node and divides.  Returns ``None`` when ``F_ref``
    is unbuildable (no merging fold pair / degenerate fold frame) -- the same
    sentinel the trainer treats as a declined cell.
    """
    y_eig = reduced_source(gamma_prime, rho, theta)
    f_ref, _kind = partitioned_reference(w_grid, gamma_prime, rho, y_eig)
    if f_ref is None:
        return None
    residual = np.empty(w_grid.size, dtype=complex)
    for i_w, w in enumerate(w_grid):
        f_pure = f_schwinger(float(w), y_eig, gamma_prime)
        residual[i_w] = (f_pure * math.sqrt(1.0 - gamma_prime * gamma_prime)
                         / f_ref[i_w])
    return residual


def _serve_farfield(chart: LowWDiffractiveChart, lens: dict,
                    dense_w: np.ndarray) -> np.ndarray | None:
    """Return the re-modulated ``farfield`` (F_serve) from the REAL serve.

    Calls the shipped ``LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve``
    bound to a bare namespace, intercepting ``reconstruct_farfield`` (the
    shared, already-tested reconstruction tail) to capture its ``envelope``
    argument.  With ``geom.t_min = 0`` the frame-demodulation phase vanishes,
    so the captured envelope IS the re-modulated farfield -- the quantity
    under test.  Returns ``None`` when the serve declines (chart absent / out
    of coverage / reduced-shear refusal / unbuildable ``F_ref``).
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
def _build_exact_chart() -> LowWDiffractiveChart:
    """Exact-residual chart over `_GAMMA_GRID` x `_RHO_GRID` x `_THETA_GRID`
    x `_W_GRID` (256 engine calls, ~8 s), at unit de-rate.

    Cached so every test class shares one build.  The residual is the
    Airy-anchored exact engine value in the chart's reduced frame
    (``r_new = f_pure * sqrt(1 - gamma'^2) / F_ref``), so `_serve_farfield`
    on a grid NODE reconstructs `_engine_reference_kappa` to ~1e-15.  Every
    cell must be ``F_ref``-buildable (the rho grid is capped accordingly);
    an unbuildable cell raises rather than being silently zero-filled, so a
    future grid edit surfaces loudly.
    """
    real = np.zeros((4, 4, 4, 4), dtype=float)
    imag = np.zeros((4, 4, 4, 4), dtype=float)
    for i, gp in enumerate(_GAMMA_GRID):
        for j, rho in enumerate(_RHO_GRID):
            for k, theta in enumerate(_THETA_GRID):
                r = _residual_at(_W_GRID, float(gp), float(rho),
                                 float(theta))
                if r is None:
                    raise AssertionError(
                        f'F_ref unbuildable at chart node '
                        f'(gamma_prime={gp}, rho={rho}, theta={theta}); the '
                        'exact chart grid must be F_ref-buildable')
                real[i, j, k, :] = r.real
                imag[i, j, k, :] = r.imag
    return LowWDiffractiveChart(
        gamma_prime_grid=_GAMMA_GRID, rho_grid=_RHO_GRID,
        theta_grid=_THETA_GRID, w23_grid=_W_GRID ** (2.0 / 3.0),
        real_coeffs=real, imag_coeffs=imag, derate=1.0)


class _BaseChartTestCase(TestCase):
    """Anti-vacuity base: fail if a test makes no comparison."""

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('no comparisons were made; the test asserted nothing')


class ServeEngineNodeExactTestCase(_BaseChartTestCase):
    """Serve-vs-engine accuracy: the re-modulated F_serve == the engine.

    With the EXACT-residual chart, evaluating at grid NODES must reproduce
    `_engine_reference_kappa` to ~1e-15: the residual is node-exact by
    construction (the trainer stores ``r_new = f_pure * sqrt(1-gamma'^2) /
    F_ref`` and the serve re-modulates ``F_ref * sqrt_mu_full * r_new``, which
    cancels exactly), so any residual mismatch is a re-modulation or
    normalization defect (missed/doubled F_ref or mass-sheet phase, a dropped
    sqrt(1-gamma'^2), or an inconsistency between the training residual and
    the serve's reconstruction).  This is the fast-tier form of the accuracy
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


def _fold_xi(w_grid: np.ndarray, gamma_prime: float,
             source: np.ndarray) -> np.ndarray | None:
    """Airy fold control ``xi = (3 w delta_tau / 4)**(2/3)``.

    Reconstructs ``delta_tau = tau_minus - tau_plus`` from the merging fold
    pair -- the same `_merging_fold_pair` and delay difference
    `partitioned_reference` uses -- so the independent Airy evaluation below
    operates on the identical fold control.  ``None`` when the merging fold
    pair is absent.
    """
    matrix = geometry.macro_matrix(gamma_prime, 0.0, 0.0)
    images = geometry.find_images(source, matrix)
    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        return None
    tau_plus, tau_minus = pair
    delta_tau = tau_minus - tau_plus
    return (3.0 * np.asarray(w_grid, dtype=float) * delta_tau / 4.0
            ) ** (2.0 / 3.0)


def _wronskian_combination(w_grid: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """q=p Wronskian ``w^{1/3} Ai(-xi)^2 + w^{-1/3} Ai'(-xi)^2``.

    Strictly positive for every ``xi`` (Ai and Ai' never vanish together --
    the Airy Wronskian ``Ai Bi' - Ai' Bi = 1/pi``); ``|F_ref|^2`` is this
    combination times the ``w``-independent factor ``4 pi |p|^2``.
    """
    ai_value, aip_value, _, _ = airy(-xi)
    w = np.asarray(w_grid, dtype=float)
    return (w ** (1.0 / 3.0) * ai_value ** 2
            + w ** (-1.0 / 3.0) * aip_value ** 2)


def _q0_airy_form(w_grid: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """q=0 leading-order form ``w^{1/3} Ai(-xi)^2`` (the Ai channel alone).

    A q=0 (leading-order) F_ref would be proportional to this; it dives to 0
    at the Airy zeros ``xi = 2.338, 4.088``, unlike the q=p Wronskian.
    """
    ai_value, _, _, _ = airy(-xi)
    return np.asarray(w_grid, dtype=float) ** (1.0 / 3.0) * ai_value ** 2


def _residual_metrics(gamma_prime: float, rho: float, theta: float,
                      w_grid: np.ndarray):
    """Return ``(ratio, arg_step, prefactor_ratio, prefactor_arg_step)``.

    ``ratio`` / ``arg_step`` are the F_ref-anchored residual
    ``r = f_pure * sqrt(1 - gamma'^2) / F_ref`` metrics; the ``prefactor_*``
    pair is the SAME ``f_pure`` divided by the OLD ``prefactor_c``
    denominator (the broken representation), for the discriminator.  ``None``
    when ``F_ref`` is unbuildable.  ``f_pure`` is the exact engine
    (``f_schwinger`` at ``w <= 60``) -- the geometric-optics 2-image sum is
    singular at the merging fold pair, so it cannot stand in for ``f_pure``.
    """
    source = reduced_source(gamma_prime, rho, theta)
    f_ref, _kind = partitioned_reference(w_grid, gamma_prime, rho, source)
    if f_ref is None:
        return None
    sq = math.sqrt(1.0 - gamma_prime * gamma_prime)
    f_pure = np.array([f_schwinger(float(w), source, gamma_prime)
                       for w in w_grid])
    r_new = f_pure * sq / f_ref
    c = np.array([prefactor_c(float(w)) for w in w_grid])
    r_old = f_pure * sq / c

    def _metrics(residual: np.ndarray) -> tuple[float, float]:
        magnitude = np.abs(residual)
        arg = np.unwrap(np.angle(residual))
        return (float(magnitude.min() / magnitude.max()),
                float(np.max(np.abs(np.diff(arg)))))

    return _metrics(r_new) + _metrics(r_old)


class FrefNonVanishingTestCase(_BaseChartTestCase):
    """F_ref non-vanishing: the q=p Wronskian form never collapses (spec 1).

    `partitioned_reference` builds the q=p uniform Airy fold form
    ``F_ref = 2 sqrt(pi) p [w^{1/6} Ai(-xi) - i w^{-1/6} Ai'(-xi)] * carrier``
    whose magnitude ``|F_ref|^2 = 4 pi |p|^2 (w^{1/3} Ai^2 + w^{-1/3} Ai'^2)``
    is the Wronskian combination -- strictly positive at every node (Ai and
    Ai' never vanish together), unlike the q=0 leading-order form
    ``w^{1/3} Ai(-xi)^2`` which dives to 0 at the Airy zeros xi=2.338, 4.088.
    Engine-free (geometry + ``scipy.special.airy``).
    """

    def _witness_data(self):
        """Yield ``(label, f_ref, wronskian, bar, h, gp)`` per witness.

        ``h`` is the macro-fold low-w renormalization factor
        ``smootherstep(w * |delta_tau|, RHO_START, RHO_END)``: ``0`` on the
        unresolved nodes (where ``|F_ref| -> sqrt_mu``) and ``1`` on the
        resolved nodes (where the raw q=p Wronskian fold form survives).
        """
        for gp, rho, theta, bar, _rbar, label in _FREF_WITNESSES:
            source = reduced_source(gp, rho, theta)
            f_ref, _kind = partitioned_reference(_FREF_W_GRID, gp, rho,
                                                 source)
            if f_ref is None:
                self.fail(f'F_ref unbuildable at witness '
                          f'(gamma_prime={gp}, rho={rho}, theta={theta})')
            xi = _fold_xi(_FREF_W_GRID, gp, source)
            if xi is None:
                self.fail(f'merging fold pair absent at witness '
                          f'(gamma_prime={gp}, rho={rho}, theta={theta})')
            wronskian = _wronskian_combination(_FREF_W_GRID, xi)
            matrix = geometry.macro_matrix(gp, 0.0, 0.0)
            images = geometry.find_images(source, matrix)
            pair = _merging_fold_pair(images, source, matrix)
            tau_plus, tau_minus = pair
            delta_tau = tau_minus - tau_plus
            h = smootherstep(_FREF_W_GRID * abs(delta_tau), RHO_START,
                             RHO_END)
            yield label, f_ref, wronskian, bar, h, gp

    def test_wronskian_strictly_positive(self):
        """w^{1/3} Ai^2 + w^{-1/3} Ai'^2 > 0 at every node (q=p never vanishes)."""
        for label, _f_ref, wronskian, _bar, _h, _gp in self._witness_data():
            with self.subTest(label=label):
                self.assertTrue(
                    np.all(wronskian > 0.0),
                    f'q=p Wronskian combination is not strictly positive at '
                    f'every node for {label}')
                self.n_checks += 1

    def test_fref_magnitude_stays_o1(self):
        """min|F_ref|/max|F_ref| >= bar (no Airy-zero dive)."""
        for label, f_ref, _wronskian, bar, _h, _gp in self._witness_data():
            with self.subTest(label=label):
                magnitude = np.abs(f_ref)
                ratio = float(magnitude.min() / magnitude.max())
                self.assertGreaterEqual(
                    ratio, bar,
                    f'min|F_ref|/max|F_ref| = {ratio:.3f} < {bar} for '
                    f'{label}: F_ref collapses (a q=0 Airy-zero signature)')
                self.n_checks += 1

    def test_magnitude_tracks_wronskian_form(self):
        """|F_ref| tracks the Wronskian (resolved) and sqrt_mu (unresolved).

        Under the macro-fold renormalization ``F_ref *= h + (1-h)
        sqrt_mu/|F_ref|``:
        * resolved nodes (``h == 1``): ``|F_ref|^2 / (w^{1/3} Ai^2 +
          w^{-1/3} Ai'^2)`` equals ``4 pi |p|^2`` -- constant in w (the raw
          q=p Wronskian form, both Ai and Ai' channels).  A q=0 regression
          (dropping the Ai' channel) makes the ratio strongly w-dependent.
        * unresolved nodes (``h == 0``): ``|F_ref| == sqrt_mu`` exactly -- the
          macro-lead normalization that keeps the residual ``r -> sqrt(1-gp^2)``
          O(1) at the band bottom instead of ``w^{-1/6}``-blown.
        """
        for label, f_ref, wronskian, _bar, h, gp in self._witness_data():
            with self.subTest(label=label):
                resolved = h == 1.0
                self.assertTrue(np.any(resolved),
                                'no resolved (h==1) nodes on the grid')
                self.n_checks += 1
                ratio = np.abs(f_ref[resolved]) ** 2 / wronskian[resolved]
                spread = float(ratio.max() / ratio.min())
                self.assertLess(
                    spread - 1.0, 1e-6,
                    f'|F_ref|^2 / Wronskian varies by {spread - 1.0:.2e} on '
                    f'resolved nodes for {label}: F_ref is not the pure q=p '
                    'Wronskian form there')
                self.n_checks += 1
                unresolved = h == 0.0
                self.assertTrue(np.any(unresolved),
                                'no unresolved (h==0) nodes on the grid')
                self.n_checks += 1
                sqrt_mu = 1.0 / math.sqrt(1.0 - gp * gp)
                np.testing.assert_allclose(
                    np.abs(f_ref[unresolved]), sqrt_mu, rtol=1e-12, atol=1e-12)
                self.n_checks += 1


class FrefNonVanishingSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the non-vanishing pin is the q=p Wronskian's doing.

    The q=0 leading-order form ``w^{1/3} Ai(-xi)^2`` (the Ai channel alone)
    MUST dive toward 0 at the Airy zeros -- its min/max over `_FREF_W_GRID`
    is far below the q=p bar.  If the q=0 form also stayed O(1), the
    non-vanishing pin would be vacuous.
    """

    def test_q0_form_collapses(self):
        """The q=0 leading-order form collapses (min/max << the q=p bar)."""
        for gp, rho, theta, bar, _rbar, label in _FREF_WITNESSES:
            source = reduced_source(gp, rho, theta)
            xi = _fold_xi(_FREF_W_GRID, gp, source)
            if xi is None:
                self.fail(f'merging fold pair absent at witness '
                          f'(gamma_prime={gp}, rho={rho}, theta={theta})')
            q0 = _q0_airy_form(_FREF_W_GRID, xi)
            ratio = float(q0.min() / q0.max())
            with self.subTest(label=label):
                self.assertLess(
                    ratio, bar,
                    f'q=0 form min/max = {ratio:.3f} is NOT below the bar '
                    f'{bar} for {label}; the q=p non-vanishing pin would be '
                    'vacuous')
                self.n_checks += 1


class ResidualBoundednessTestCase(_BaseChartTestCase):
    """Residual boundedness / no Airy-zero crossing (spec 2).

    The residual ``r = f_pure * sqrt(1 - gamma'^2) / F_ref`` is a smooth O(1)
    complex function: no magnitude collapse (min|r|/max|r| >= bar) and no
    Airy-zero crossing (unwrapped arg steps < pi/2).  ``f_pure`` is the EXACT
    engine -- the geometric-optics 2-image sum is singular at the merging
    fold pair (magnification -> inf), so it cannot stand in for ``f_pure`` in
    the near-fold shell; the double-double engine at ``w <= 60`` is the
    correct cheap oracle (the same oracle spec 3 admits).
    """

    def test_residual_has_no_magnitude_collapse(self):
        """min|r|/max|r| >= bar over the w^(2/3) grid."""
        for gp, rho, theta, _fbar, bar, label in _FREF_WITNESSES:
            metrics = _residual_metrics(gp, rho, theta, _FREF_W_GRID)
            if metrics is None:
                self.fail(f'F_ref unbuildable at witness '
                          f'(gamma_prime={gp}, rho={rho}, theta={theta})')
            ratio, _arg_step, _pr, _pa = metrics
            with self.subTest(label=label):
                self.assertGreaterEqual(
                    ratio, bar,
                    f'min|r|/max|r| = {ratio:.3f} < {bar} for {label}: the '
                    'residual collapses (a zero/denominator mismatch)')
                self.n_checks += 1

    def test_residual_has_no_zero_crossing(self):
        """max unwrapped |arg r| step < pi/2 between adjacent w-nodes."""
        for gp, rho, theta, _fbar, _rbar, label in _FREF_WITNESSES:
            metrics = _residual_metrics(gp, rho, theta, _FREF_W_GRID)
            if metrics is None:
                self.fail(f'F_ref unbuildable at witness '
                          f'(gamma_prime={gp}, rho={rho}, theta={theta})')
            _ratio, arg_step, _pr, _pa = metrics
            with self.subTest(label=label):
                self.assertLess(
                    arg_step, RESIDUAL_ARG_STEP_TOL,
                    f'max unwrapped |arg r| step = {arg_step:.3f} rad >= '
                    f'{RESIDUAL_ARG_STEP_TOL:.3f} for {label}: an Airy-zero '
                    'crossing (>= pi jump)')
                self.n_checks += 1


class ResidualBoundednessSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the residual-boundedness pin is the F_ref denominator's doing.

    Dividing the SAME ``f_pure`` by the OLD ``prefactor_c`` denominator (the
    point-mass C(w), no Airy structure) leaves the full Airy fold structure
    in the residual -- it collapses ~13-68x and its arg jumps >= pi/2, the
    exact failure signature the F_ref division avoids.
    """

    def test_prefactor_denominator_collapses_and_jumps(self):
        """r_old = f_pure * sqrt(1 - gamma'^2) / C(w) collapses and jumps."""
        for gp, rho, theta, _fbar, bar, label in _FREF_WITNESSES:
            metrics = _residual_metrics(gp, rho, theta, _FREF_W_GRID)
            if metrics is None:
                self.fail(f'F_ref unbuildable at witness '
                          f'(gamma_prime={gp}, rho={rho}, theta={theta})')
            _ratio, _arg_step, pr, pa = metrics
            with self.subTest(label=label):
                self.assertLess(
                    pr, bar,
                    f'prefactor_c residual min/max = {pr:.3f} is NOT below '
                    f'bar {bar} for {label}: the F_ref discriminator would be '
                    'vacuous')
                self.assertGreaterEqual(
                    pa, RESIDUAL_ARG_STEP_TOL,
                    f'prefactor_c residual arg step = {pa:.3f} < '
                    f'{RESIDUAL_ARG_STEP_TOL} for {label}: no zero-crossing '
                    'jump in the broken representation')
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

    def test_doubled_fref_breaks_node_exactness(self):
        """Applying F_ref twice blows the node-exact agreement."""
        real_fref = _likelihood_mod.partitioned_reference
        rel = self._node_rel_error_under(
            'partitioned_reference',
            lambda w, gp, rho, src: (real_fref(w, gp, rho, src)[0] ** 2,
                                     'airy_fold'))
        self.assertGreater(
            rel, NODE_EXACT_TOL,
            f'doubled F_ref left rel err {rel:.3e} <= {NODE_EXACT_TOL}; '
            'the re-modulation suite would not catch a doubled F_ref')
        self.n_checks += 1

    def test_unit_sqrt_mu_full_breaks_node_exactness(self):
        """Replacing sqrt_mu_full with 1 (dropped macro normalization) breaks
        exactness."""
        real_born = _likelihood_mod._born_factors

        def _unit_sqrt_mu(y1, y2, gamma, beta, kappa):
            phi = real_born(y1, y2, gamma, beta, kappa)
            return (1.0,) + tuple(phi[1:])

        rel = self._node_rel_error_under('_born_factors', _unit_sqrt_mu)
        self.assertGreater(
            rel, NODE_EXACT_TOL,
            f'unit sqrt_mu_full left rel err {rel:.3e} <= {NODE_EXACT_TOL}; '
            'the re-modulation suite would not catch a dropped '
            'sqrt(1-gamma\'^2) / 1-lam normalization')
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
        'w23_grid': chart.w23_grid,
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
            chart.w23_grid, chart.real_coeffs, chart.imag_coeffs,
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
    w23_grid = np.array([0.3, 0.5, 2.0, 8.0]) ** (2.0 / 3.0)
    shape = (4, 4, 4, 4)
    rng = np.random.default_rng(0)
    # Non-trivial mask: the middle two rho columns are declined (near-fold
    # resonance-limited cells), so an all-False tamper CHANGES the stored
    # bytes -- the exact stale-mask corruption the hash must detect.
    declined_mask = np.zeros(shape[:3], dtype=bool)
    declined_mask[:, 1:3, :] = True
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, w23_grid=w23_grid,
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
    w23_grid = np.array([0.02, 0.1, 1.0, 8.0]) ** (2.0 / 3.0)
    shape = (4, 4, 4, 4)
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, w23_grid=w23_grid,
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
    w23_grid = np.geomspace(0.05, 8.0, n_w) ** (2.0 / 3.0)
    theta_2d = theta_grid.reshape(1, 1, -1, 1)
    real = np.broadcast_to(np.cos(2.0 * theta_2d),
                           (n_gp, n_rho, 8, n_w)).astype(float).copy()
    imag = np.zeros((n_gp, n_rho, 8, n_w), dtype=float)
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, w23_grid=w23_grid,
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
                      'w23_grid', 'real_coeffs', 'imag_coeffs',
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
        """A wall-band exterior draw (rho above the shell) is covered.

        Uses `RHO_ABOVE_SHELL` (a genuinely exterior rho) rather than
        `WALL_RHO` because the serve's re-modulation fixtures now sit
        on-caustic (rho = 1.0) where ``F_ref`` is buildable; the coverage
        predicate must still admit the exterior wall draw.
        """
        chart = _build_coverage_chart()
        self.assertTrue(chart.covers(WALL_GAMMA_PRIME, RHO_ABOVE_SHELL))
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

    def test_w_beyond_w23_ceiling_declines(self):
        """A band whose w extends beyond the trained w**(2/3) ceiling is declined."""
        chart = _build_coverage_chart()
        w_hi = float((chart.w23_grid[-1] + 0.1) ** 1.5)
        self.assertFalse(chart.covers(NEAR_FOLD_GAMMA_PRIME, NEAR_FOLD_RHO,
                                      np.array([w_hi])))
        self.n_checks += 1

    def test_w_within_w23_range_covers(self):
        """A band at the trained w**(2/3) ceiling (inclusive) is covered."""
        chart = _build_coverage_chart()
        w_hi = float(chart.w23_grid[-1] ** 1.5)
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
             chart.w23_grid),
            chart.real_coeffs, method='cubic', bounds_error=False,
            fill_value=None)
        points = np.array([[gp, rho, math.pi - theta, w ** (2.0 / 3.0)]])
        no_fold = float(real_interp(points)[0])
        self.assertGreater(
            abs(no_fold - folded[0].real), 1e-2,
            f'no-fold interpolation ({no_fold:.4f}) matched the folded value '
            f'({folded[0].real:.4f}); the fold is not doing anything')
        self.n_checks += 1


# ---------------------------------------------------------------------------
# Cusp-cell (Pearcey reference) fixtures -- the b3 -> 0 fold->cusp transition.
# ---------------------------------------------------------------------------

#: Reduced shear of the cusp-cell fixtures (wall band, gamma' > 0.5).
CUSP_GAMMA_PRIME = 0.8

#: Near-caustic rho (just outside the caustic) -- the cusp cell where the
#: uniform Pearcey reference is genuinely NON-VANISHING over the full chart
#: band.  At rho = 1.2, theta = 0.2 the soft-axis cubic coefficient is
#: b3 ~ 1.4e-15 (the fold degenerates), the Airy fold reference refuses, and
#: the Pearcey fallback builds a finite ``F_ref`` (measured
#: min|F_ref|/max|F_ref| ~ 0.25, ~4x spread).  Premise-asserted in the test
#: (b3 <= 1e-6, Airy form None), so a gate move fails loudly.
CUSP_RHO = 1.2

#: Eigenframe angle of the near-caustic cusp witness (inside the b3 -> 0
#: window theta in [0, 0.5]).
CUSP_THETA = 0.2

#: Exact-residual CUSP chart grids (>= 4 nodes per axis for scipy cubic;
#: the cusp cell (0.8, 1.2, 0.2) is an INTERIOR node).  The theta grid
#: avoids the declined band (theta ~ 0.25 at rho = 1.2) and the
#: far-exterior unbuildable corners, so every cell is F_ref-buildable
#: (fold OR Pearcey).
_CUSP_GP_GRID = np.array([0.6, 0.7, 0.8, 0.9])
_CUSP_RHO_GRID = np.array([0.8, 1.0, 1.2, 1.4])
_CUSP_THETA_GRID = np.array([0.15, 0.2, 0.3, 0.4])

#: theta sweep across the b3 -> 0 threshold at (gamma'=0.8, rho=1.2): the
#: cusp side (<= 0.2, Pearcey F_ref) through the fold side (>= 0.3, Airy
#: F_ref).  theta ~ 0.25 is the declined band (neither form serves) and is
#: skipped by the sweep.
_CUSP_THETA_SWEEP = (0.15, 0.2, 0.3, 0.35, 0.4, 0.5)

#: Max |F_ref| ratio between adjacent servable thetas across the handoff
#: (spec 3's "~3x" bar).  Measured 3.12x (at w ~ 0.65, where the cusp and
#: fold forms oscillate out of phase); 5.0 leaves headroom while still
#: catching an order-of-magnitude discontinuity (a q=0 collapse or a wrong
#: normalization would give >> 10x).
CONTINUITY_FREF_RATIO_TOL = 5.0

#: Non-vanishing upper bound on the Pearcey reference spread
#: (spec 2's ``max|F_ref|/min|F_ref| < 1e3`` sanity ceiling).  The measured
#: spread is ~4x, so 1e3 leaves ~250x headroom -- the load-bearing claim is
#: really ``min|F_ref| > 0`` (pinned separately), this ceiling only catches
#: a diverging reference.
CUSP_FREF_RATIO_TOL = 1e3


@functools.lru_cache(maxsize=1)
def _build_exact_cusp_chart() -> LowWDiffractiveChart:
    """Exact-residual chart whose cusp cell is an interior node (spec 1).

    Same construction as `_build_exact_chart` but on the CUSP grids, so the
    b3 -> 0 cusp cell (gamma'=0.8, rho=1.2, theta=0.2) is an interior grid
    node whose residual is Pearcey-anchored (``r = f_pure * sqrt(1-gp^2) /
    F_ref`` with the Pearcey ``F_ref``).  Every cell must be F_ref-buildable
    (fold OR Pearcey); an unbuildable cell raises rather than being silently
    zero-filled, so a future grid edit surfaces loudly.  Cached so the cusp
    tests share one build.
    """
    real = np.zeros((4, 4, 4, 4), dtype=float)
    imag = np.zeros((4, 4, 4, 4), dtype=float)
    for i, gp in enumerate(_CUSP_GP_GRID):
        for j, rho in enumerate(_CUSP_RHO_GRID):
            for k, theta in enumerate(_CUSP_THETA_GRID):
                r = _residual_at(_W_GRID, float(gp), float(rho),
                                 float(theta))
                if r is None:
                    raise AssertionError(
                        f'F_ref unbuildable at cusp-chart node '
                        f'(gamma_prime={gp}, rho={rho}, theta={theta}); the '
                        'cusp chart grid must be F_ref-buildable')
                real[i, j, k, :] = r.real
                imag[i, j, k, :] = r.imag
    return LowWDiffractiveChart(
        gamma_prime_grid=_CUSP_GP_GRID, rho_grid=_CUSP_RHO_GRID,
        theta_grid=_CUSP_THETA_GRID, w23_grid=_W_GRID ** (2.0 / 3.0),
        real_coeffs=real, imag_coeffs=imag, derate=1.0)


class CuspServeEngineNodeExactTestCase(_BaseChartTestCase):
    """Node-exact re-modulation for the Pearcey-anchored cusp cell (spec 1).

    The cusp-cell analogue of `ServeEngineNodeExactTestCase`: with an
    EXACT-residual chart whose b3 -> 0 cusp cell is an interior node (the
    trainer stores ``r = f_pure * sqrt(1-gp^2) / F_ref`` with the Pearcey
    ``F_ref``), the serve re-modulates ``F_ref * sqrt_mu_full * r`` back to
    the exact engine to ~1e-14 -- proving the Pearcey reference convention
    is consistent between train and serve (a missed/doubled F_ref, a dropped
    sqrt(1-gp^2), or a frame mismatch would break the cancellation).
    """

    def test_cusp_cell_serve_matches_engine_at_node(self):
        """|F_serve - F_engine| / |F_engine| <= NODE_EXACT_TOL at the cusp node."""
        chart = _build_exact_cusp_chart()
        for kappa, beta in ((k, b) for k in KAPPAS for b in BETAS):
            lens = _make_lens(CUSP_GAMMA_PRIME, CUSP_RHO, CUSP_THETA,
                              kappa, beta)
            dense_w = np.array(SERVE_WS)
            f_serve = _serve_farfield(chart, lens, dense_w)
            if f_serve is None:
                self.fail(f'serve declined the cusp-cell fixture '
                          f'(kappa={kappa}, beta={beta}); the exact cusp '
                          'chart must cover it')
            y = (lens['y1'], lens['y2'])
            for i, w in enumerate(SERVE_WS):
                with self.subTest(kappa=kappa, beta=beta, w=w):
                    f_engine = _engine_reference_kappa(
                        w, y, lens['gamma'], beta, kappa)
                    rel = abs(f_serve[i] - f_engine) / abs(f_engine)
                    self.assertLess(
                        rel, NODE_EXACT_TOL,
                        f'F_serve disagrees with the engine by {rel:.3e} at '
                        f'w={w:g}; the Pearcey re-modulation or residual '
                        'normalization is inconsistent')
                    self.n_checks += 1


class CuspFrefNonVanishingTestCase(_BaseChartTestCase):
    """Pearcey reference contract: finite, non-vanishing at a cusp cell.

    At a genuine b3 -> 0 cusp cell the fallback Pearcey reference is finite
    and non-vanishing over the full chart band (spec 2): the witness
    (gamma'=0.8, rho=1.2, theta=0.2) has b3 ~ 1.4e-15, the Airy fold form
    refuses, and the Pearcey fallback builds ``F_ref`` with ``min|F_ref| >
    0``.  Engine-free (geometry + the Pearcey form, no Schwinger call).
    """

    def _witness(self):
        """Return ``(source, f_ref, kind)`` with the cusp-cell premise asserted.

        The cusp witness (gamma'=0.8, rho=1.2, theta=0.2) has ``b3 ~ 1e-15``:
        the soft-axis cubic is degenerate (``b3 -> 0``, the fold->cusp
        transition), so the Airy fold form refuses and
        `partitioned_reference` must fall back to the RESTRICTED Pearcey cusp
        carrier (``kind == 'pearcey_cusp'``).

        NOTE (authoring-time state): this is RED until the cusp-transition
        detection is fixed.  `_airy_fold_form`
        (cogwheel/lensing/low_w_diffractive_chart.py) sets
        ``cusp_transition = True`` ONLY when `_soft_axis_cubic` returns
        ``None`` (``p <= 0`` / non-finite -- the image-at-point-mass case),
        but the genuine ``b3 -> 0`` fold->cusp transition is detected by
        `_fold_amplitudes` returning ``None`` (``abs(b3) <= _B3_MIN``).  The
        b3~1e-15 witness therefore gets ``(None, False)`` and is DECLINED
        (kind 'airy_fold' with f_ref None) instead of Pearcey-fallback, and
        the ``cusp_transition`` assertion below fails.  Flips green with zero
        further edits once ``cusp_transition`` is keyed on the
        `_fold_amplitudes` refusal.
        """
        source = reduced_source(CUSP_GAMMA_PRIME, CUSP_RHO, CUSP_THETA)
        nearest = geometry.nearest_caustic_point(
            CUSP_GAMMA_PRIME, 0.0, source, kappa=0.0)
        b3 = _soft_axis_cubic(nearest.image, nearest.soft_axis)
        if b3 is None:
            self.fail('premise lost: soft-axis cubic is degenerate at the '
                      'cusp witness')
        self.assertLessEqual(
            abs(b3), 1e-6,
            'premise lost: the witness is no longer a b3 -> 0 cusp cell '
            f'(b3 = {b3:.2e})')
        self.n_checks += 1
        airy, cusp_transition = _airy_fold_form(_FREF_W_GRID, CUSP_GAMMA_PRIME,
                                                source)
        self.assertIsNone(
            airy, 'premise lost: the Airy fold form no longer refuses at '
            'the cusp witness (b3 not ~ 0)')
        self.n_checks += 1
        self.assertTrue(
            cusp_transition,
            'the Airy refusal is not flagged as the b3 -> 0 fold->cusp '
            'transition; a genuine cusp cell is declined instead of routed '
            'to the restricted-Pearcey fallback')
        self.n_checks += 1
        pearcey = _pearcey_cusp_reference(_FREF_W_GRID, CUSP_GAMMA_PRIME,
                                          source)
        self.assertIsNotNone(
            pearcey, 'premise lost: the Pearcey fallback no longer builds '
            'the cusp reference')
        self.n_checks += 1
        f_ref, kind = partitioned_reference(_FREF_W_GRID, CUSP_GAMMA_PRIME,
                                            CUSP_RHO, source)
        return source, f_ref, kind

    def test_cusp_fref_is_finite_nonvanishing(self):
        """``partitioned_reference`` routes the cusp cell to the Pearcey carrier."""
        _source, f_ref, kind = self._witness()
        self.assertEqual(
            kind, 'pearcey_cusp',
            'the cusp cell is not routed to the restricted-Pearcey carrier')
        self.n_checks += 1
        self.assertIsNotNone(
            f_ref, 'partitioned_reference declined a non-vanishing cusp cell')
        self.n_checks += 1
        self.assertTrue(
            np.all(np.isfinite(f_ref)),
            'F_ref is not finite at the cusp witness')
        self.n_checks += 1
        magnitude = np.abs(f_ref)
        self.assertGreater(
            float(magnitude.min()), 0.0,
            'min|F_ref| == 0 at the cusp witness: a residual pole')
        self.n_checks += 1
        ratio = float(magnitude.max() / magnitude.min())
        self.assertLess(
            ratio, CUSP_FREF_RATIO_TOL,
            f'max|F_ref|/min|F_ref| = {ratio:.2e} >= {CUSP_FREF_RATIO_TOL}: '
            'the Pearcey reference diverges over the band')
        self.n_checks += 1


class FoldCuspContinuityTestCase(_BaseChartTestCase):
    """Fold/cusp continuity across the b3 -> 0 threshold.

    At fixed (gamma'=0.8, rho=1.2), theta sweeps from the cusp side
    (theta <= 0.2, Pearcey F_ref) to the fold side (theta >= 0.3, Airy
    F_ref).  ``|F_ref|`` at a shared w stays within ~3x across the handoff
    (the two normal forms are comparable in magnitude, not numerically
    equal), and the residual ``r = f_pure * sqrt(1-gp^2) / F_ref`` stays
    finite at every servable theta (no jump to a pole).

    NOTE (authoring-time state): ``test_handoff_visits_both_forms`` is RED
    until the cusp-transition detection is fixed (see
    `CuspFrefNonVanishingTestCase._witness`): the cusp side (theta <= 0.2)
    is currently DECLINED, so the sweep sees only 'airy_fold'.
    """

    def _servable_frefs(self):
        """Yield ``(theta, kind, f_ref)`` for the servable sweep points."""
        for theta in _CUSP_THETA_SWEEP:
            source = reduced_source(CUSP_GAMMA_PRIME, CUSP_RHO, theta)
            f_ref, kind = partitioned_reference(_FREF_W_GRID, CUSP_GAMMA_PRIME,
                                                CUSP_RHO, source)
            if f_ref is None:
                continue  # declined band (theta ~ 0.25): neither form serves
            yield theta, kind, f_ref

    def test_handoff_visits_both_forms(self):
        """The sweep visits both the cusp and the fold carrier (premise)."""
        forms = {form for _t, form, _f in self._servable_frefs()}
        self.assertEqual(
            forms, {'airy_fold', 'pearcey_cusp'},
            f'handoff sweep saw forms {forms}, expected both the fold and '
            'the cusp carrier; the b3 -> 0 threshold is not being crossed')
        self.n_checks += 1

    def test_fref_continuous_across_handoff(self):
        """max |F_ref| ratio between adjacent servable thetas < tol."""
        points = list(self._servable_frefs())
        for (t_lo, _f_lo, ref_lo), (t_hi, _f_hi, ref_hi) in zip(
                points[:-1], points[1:]):
            ratio = np.maximum(
                np.abs(ref_lo) / np.abs(ref_hi),
                np.abs(ref_hi) / np.abs(ref_lo))
            with self.subTest(theta_lo=t_lo, theta_hi=t_hi):
                self.assertLess(
                    float(ratio.max()), CONTINUITY_FREF_RATIO_TOL,
                    f'|F_ref| jumps {ratio.max():.2f}x between theta={t_lo} '
                    f'and theta={t_hi}: the fold/cusp handoff is not '
                    'continuous')
                self.n_checks += 1

    def test_residual_finite_across_handoff(self):
        """The residual r is finite (no pole) at every servable theta."""
        sq = math.sqrt(1.0 - CUSP_GAMMA_PRIME * CUSP_GAMMA_PRIME)
        for theta, form, f_ref in self._servable_frefs():
            source = reduced_source(CUSP_GAMMA_PRIME, CUSP_RHO, theta)
            f_pure = np.array([f_schwinger(float(w), source,
                                           CUSP_GAMMA_PRIME)
                               for w in _FREF_W_GRID])
            r = f_pure * sq / f_ref
            with self.subTest(theta=theta, form=form):
                self.assertTrue(
                    np.all(np.isfinite(r)),
                    f'residual not finite at theta={theta} ({form} form): '
                    'a jump to a pole at the handoff')
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


# ---------------------------------------------------------------------------
# Trainer-script bridge + per-carrier serve-accuracy fixtures.
# ---------------------------------------------------------------------------

_TRAINER_SCRIPT_REL = os.path.join('..', '..', 'scripts',
                                   'train_low_w_diffractive_chart.py')


@functools.lru_cache(maxsize=1)
def _load_trainer_script():
    """Lazily import the low-w chart trainer (single source of truth).

    ``scripts/train_low_w_diffractive_chart.py`` owns the absolute
    carrier-adequacy guard (`_carrier_adequate`, `_ABS_GUARD_CEILING`,
    `_ABS_GUARD_FLOOR`) and the resolved/unresolved split helpers.  Importing
    it here -- rather than re-deriving the predicate -- means the guard test
    probes EXACTLY what the trainer ships, so the test cannot drift from the
    training domain.  Cached (module import is cheap: only constants and
    function defs, no engine calls).
    """
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), _TRAINER_SCRIPT_REL))
    modname = 'train_low_w_diffractive_chart'
    spec = importlib.util.spec_from_file_location(modname, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


#: Off-caustic far-exterior macro cell (``rho > RHO_HI``): the wall-band
#: exterior witness.  ``delta_tau ~ 8.17`` so ``w_split = RHO_END/delta_tau
#: ~ 0.49`` -- ``w = 0.2`` is UNRESOLVED (macro chart), ``w = 2.0/8.0`` are
#: RESOLVED (geometric two-image sum).
_FAR_EXT_GAMMA_GRID = np.array([0.6, 0.8, 0.9, 0.95])
_FAR_EXT_RHO_GRID = np.array([1.5, 2.0, 2.5, 3.0])
_FAR_EXT_THETA_GRID = np.array([0.2, 0.6, 1.0, 1.4])
_FAR_EXT_W_GRID = np.array([0.1, 0.2, 2.0, 8.0])

#: Off-caustic deep-interior macro cell (``rho < RHO_LO``): ``delta_tau
#: ~ 0.2`` so ``w_split ~ 20`` -- every served ``w <= 8`` is UNRESOLVED
#: (macro chart).
_DEEP_INT_GAMMA_GRID = np.array([0.6, 0.8, 0.9, 0.95])
_DEEP_INT_RHO_GRID = np.array([0.2, 0.3, 0.4, 0.5])
_DEEP_INT_THETA_GRID = np.array([0.2, 0.6, 1.0, 1.4])
_DEEP_INT_W_GRID = np.array([0.1, 0.2, 2.0, 8.0])

#: The disjoint per-carrier cells (gamma', rho, theta).
FOLD_CELL = (0.3, 1.0, 1.4)       #: caustic nbhd -> airy_fold (reuses `_build_exact_chart`)
FAR_EXT_CELL = (0.8, 2.0, 0.6)    #: off-caustic far exterior -> macro / geometric
DEEP_INT_CELL = (0.8, 0.3, 0.6)   #: off-caustic deep interior -> macro

#: The resolved (geometric) far-exterior serve is node-exact at grid nodes:
#: the baked residual ``r = f_pure * sqrt(1 - gamma'**2) / F_ref`` absorbs the
#: two-image geometric sum's finite-w deviation, and the serve re-modulates
#: with the SAME partitioned reference the trainer baked, so the residual
#: re-multiplied by ``F_ref`` cancels exactly (measured ~1e-16 vs the engine,
#: pinned by `NODE_EXACT_TOL` below the serve's 1e-10 node bar).  The bare
#: two-image sum alone errs ~1e-1 at w=2.0 and ~4.6e-3 at w=8.0 (``w *
#: delta_tau ~ 16/65``) and would NOT meet the serve's 1e-4-style accuracy --
#: that is precisely the failure mode the residual anchoring eliminates, so
#: serving the bare geometric sum without the residual is the regression this
#: pin refuses.


def _build_exact_chart_on(gamma_grid: np.ndarray, rho_grid: np.ndarray,
                          theta_grid: np.ndarray, w_grid: np.ndarray,
                          label: str) -> LowWDiffractiveChart:
    """Node-exact residual chart on explicit grids (every cell buildable)."""
    real = np.zeros((len(gamma_grid), len(rho_grid), len(theta_grid),
                     len(w_grid)), dtype=float)
    imag = np.zeros_like(real)
    for i, gp in enumerate(gamma_grid):
        for j, rho in enumerate(rho_grid):
            for k, theta in enumerate(theta_grid):
                r = _residual_at(w_grid, float(gp), float(rho), float(theta))
                if r is None:
                    raise AssertionError(
                        f'F_ref unbuildable at {label} chart node '
                        f'(gamma_prime={gp}, rho={rho}, theta={theta}); the '
                        f'{label} grid must be F_ref-buildable')
                real[i, j, k, :] = r.real
                imag[i, j, k, :] = r.imag
    return LowWDiffractiveChart(
        gamma_prime_grid=gamma_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, w23_grid=w_grid ** (2.0 / 3.0),
        real_coeffs=real, imag_coeffs=imag, derate=1.0)


@functools.lru_cache(maxsize=1)
def _build_far_exterior_chart() -> LowWDiffractiveChart:
    """Node-exact macro-residual chart with (0.8, 2.0, 0.6) an interior node."""
    return _build_exact_chart_on(_FAR_EXT_GAMMA_GRID, _FAR_EXT_RHO_GRID,
                                 _FAR_EXT_THETA_GRID, _FAR_EXT_W_GRID,
                                 'far_exterior')


@functools.lru_cache(maxsize=1)
def _build_deep_interior_chart() -> LowWDiffractiveChart:
    """Node-exact macro-residual chart with (0.8, 0.3, 0.6) an interior node."""
    return _build_exact_chart_on(_DEEP_INT_GAMMA_GRID, _DEEP_INT_RHO_GRID,
                                 _DEEP_INT_THETA_GRID, _DEEP_INT_W_GRID,
                                 'deep_interior')


def _serve_cell(chart: LowWDiffractiveChart, gp: float, rho: float,
                theta: float, w: float) -> np.ndarray | None:
    """Serve a single reduced cell at frequency ``w`` (kappa = beta = 0)."""
    lens = _make_lens(gp, rho, theta, 0.0, 0.0)
    return _serve_farfield(chart, lens, np.asarray([w], dtype=float))


def _reduced_w_split(gp: float, rho: float, theta: float) -> float:
    """Resolved/unresolved split frequency ``RHO_END / delta_tau``.

    ``delta_tau`` is the smallest pairwise real-image delay gap measured via
    `_reduced_min_delay_separation` (``inf`` if fewer than two real images).
    """
    source = reduced_source(gp, rho, theta)
    delta_tau = _reduced_min_delay_separation(gp, source)
    if delta_tau <= 0.0:
        return math.inf
    return RHO_END / delta_tau


#: Macro-carrier reference grid, confined BELOW the wall witness's
#: resolved/unresolved split (``w_split = RHO_END / delta_tau ~ 0.49`` for
#: ``MACRO_WITNESS``).  Every node is unresolved, so `partitioned_reference`
#: returns the PURE macro carrier (``kind == 'macro'``, w-independent
#: ``|F_ref| == sqrt_mu``) -- the object `MacroCarrierReferenceTestCase`
#: pins.  A grid that spanned resolved nodes would route the witness to the
#: 'geometric' (split) carrier instead.
_MACRO_FREF_W_GRID = np.geomspace(0.02, 0.4, 8)


class MacroCarrierReferenceTestCase(_BaseChartTestCase):
    """Macro-lead carrier invariants for the off-caustic (wall-band) band.

    `partitioned_reference` serves the off-caustic bands (``rho < RHO_LO`` or
    ``rho > RHO_HI``) with the macro lead carrier `born_lead_carrier`
    (``sqrt(mu_macro) exp(1j w phi_geo)``) on their UNRESOLVED nodes.  For
    positive parity its magnitude is ``sqrt(mu_macro) = 1/sqrt(1 - gamma'^2)``,
    EXACTLY w-independent -- so ``F_ref`` is trivially non-vanishing, and the
    residual ``r = f_pure sqrt(1-gp^2) / F_ref`` is bounded O(1) over the
    UNRESOLVED domain ``w < w_split`` (the same domain `_carrier_adequate`
    guards).  The wall-band EXTERIOR witness (gamma'=0.8, rho=2.0, theta=0.6)
    -- previously DECLINED by the fold/cusp form -- is now buildable here.
    The reference is built on `_MACRO_FREF_W_GRID`, a grid confined below
    the witness's ``w_split ~ 0.49`` so every node is unresolved and the
    witness routes to the pure 'macro' carrier (its resolved nodes would
    route it to the split 'geometric' carrier).
    """

    def _ref(self):
        gp, rho, theta = MACRO_WITNESS
        source = reduced_source(gp, rho, theta)
        f_ref, kind = partitioned_reference(_MACRO_FREF_W_GRID, gp, rho,
                                            source)
        return gp, rho, theta, source, f_ref, kind

    def test_wall_exterior_routes_to_macro(self):
        """The wall-band exterior witness is macro-served, not declined."""
        gp, rho, _theta, _src, f_ref, kind = self._ref()
        self.assertGreater(rho, RHO_HI,
                           'premise lost: witness no longer off-caustic')
        self.n_checks += 1
        self.assertEqual(
            kind, 'macro',
            'the wall-band exterior witness is not routed to the macro carrier')
        self.n_checks += 1
        self.assertIsNotNone(
            f_ref,
            'the wall-band exterior witness is declined (should be macro-served)')
        self.n_checks += 1

    def test_macro_fref_magnitude_w_independent(self):
        """|F_ref| == sqrt_mu_macro, constant in w (positive parity)."""
        gp, _rho, _theta, _src, f_ref, _kind = self._ref()
        sqrt_mu = 1.0 / math.sqrt(1.0 - gp * gp)
        magnitude = np.abs(f_ref)
        np.testing.assert_allclose(magnitude, sqrt_mu, rtol=1e-12, atol=1e-12)
        self.n_checks += 1
        self.assertAlmostEqual(
            float(magnitude.min()), float(magnitude.max()), delta=1e-12,
            msg='|F_ref| is not w-independent (macro carrier magnitude drifts)')
        self.n_checks += 1

    def test_macro_residual_bounded_below_split(self):
        """min|r|/max|r| >= MACRO_RESIDUAL_RATIO_TOL over w < w_split."""
        gp, rho, theta, source, f_ref, _kind = self._ref()
        w_split = _reduced_w_split(gp, rho, theta)
        self.assertTrue(math.isfinite(w_split),
                        'premise lost: no resolved nodes for the witness')
        self.n_checks += 1
        domain = _MACRO_FREF_W_GRID < w_split
        self.assertTrue(np.all(domain),
                        'premise lost: a macro-grid node is resolved')
        self.n_checks += 1
        sq = math.sqrt(1.0 - gp * gp)
        f_pure = np.array([f_schwinger(float(w), source, gp)
                           for w in _MACRO_FREF_W_GRID])
        residual = f_pure * sq / f_ref
        mag = np.abs(residual[domain])
        ratio = float(mag.min() / mag.max())
        self.assertGreaterEqual(
            ratio, MACRO_RESIDUAL_RATIO_TOL,
            f'macro residual min/max = {ratio:.3f} < {MACRO_RESIDUAL_RATIO_TOL} '
            'over the unresolved domain: the macro carrier residual collapses')
        self.n_checks += 1


def _synthetic_residual(n: int, scale: float = 1.0) -> np.ndarray:
    """A bounded O(1) complex residual (``|r| = 0.55``), uniformly scaled."""
    return np.full(n, 0.55 + 0.0j) * scale


#: Synthetic dimensionless-frequency grid for the carrier-adequacy guard.
_GUARD_W_GRID = np.geomspace(0.02, 60.0, 16)


class CarrierAdequacyGuardTestCase(_BaseChartTestCase):
    """Absolute carrier-adequacy guard (trainer `_carrier_adequate`).

    A cell whose residual magnitude is 3-4 orders off normalization is
    DECLINED (folded into ``declined_mask``, NEVER de-rated): ``sup |r| <=
    _ABS_GUARD_CEILING (1e3)`` AND ``inf |r| >= _ABS_GUARD_FLOOR (1e-3)``
    over the carrier-specific domain -- the full ``w_grid`` for the
    caustic-neighborhood carriers (airy_fold / pearcey_cusp), only
    ``w < w_split`` for the macro carrier (resolved-node interference dips
    above the split are expected and must not false-trigger the floor).
    """

    def setUp(self):
        super().setUp()
        self._trainer = _load_trainer_script()

    def test_good_carrier_admits(self):
        """A residual with |r| in [1e-2, 1e2] over the domain admits."""
        guard = self._trainer._carrier_adequate
        res = _synthetic_residual(_GUARD_W_GRID.size)  # |r| = 0.55
        self.assertTrue(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1
        self.assertTrue(guard(res, 'macro', _GUARD_W_GRID, 2.0))
        self.n_checks += 1

    def test_overscaled_residual_declines_via_ceiling(self):
        """|r| ~ 1e4 -> declined by the ceiling (never a de-rate)."""
        guard = self._trainer._carrier_adequate
        res = _synthetic_residual(_GUARD_W_GRID.size, scale=1e4)  # |r| = 5500
        self.assertFalse(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1

    def test_underscaled_residual_declines_via_floor(self):
        """|r| ~ 1e-4 -> declined by the floor."""
        guard = self._trainer._carrier_adequate
        res = _synthetic_residual(_GUARD_W_GRID.size, scale=1e-4)  # |r| = 5.5e-5
        self.assertFalse(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1

    def test_macro_dip_above_split_only_admits(self):
        """A macro residual dipping below 1e-3 ONLY above w_split still admits.

        The macro guard excludes the resolved region (``w >= w_split``), so a
        dip confined there must not false-trigger the floor; the SAME dip
        inside the guarded domain must decline.
        """
        guard = self._trainer._carrier_adequate
        w_split = 2.0
        res_above = _synthetic_residual(_GUARD_W_GRID.size)
        res_above[_GUARD_W_GRID >= w_split] = 1e-6
        self.assertTrue(guard(res_above, 'macro', _GUARD_W_GRID, w_split))
        self.n_checks += 1
        res_below = _synthetic_residual(_GUARD_W_GRID.size)
        res_below[_GUARD_W_GRID < w_split] = 1e-6
        self.assertFalse(guard(res_below, 'macro', _GUARD_W_GRID, w_split))
        self.n_checks += 1


class CarrierAdequacyGuardSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the guard's CEILING/FLOOR constants are load-bearing (teeth).

    Each admit/decline decision flips when the bound constant moves 10x: a
    residual at |r| = 550 admits under CEILING=1e3 but declines under
    CEILING=1e2; a residual at |r| = 5.5e-3 admits under FLOOR=1e-3 but
    declines under FLOOR=1e-2.  Patching the module global (read at call time
    via ``function.__globals__``) is the standard teeth lever.
    """

    def setUp(self):
        super().setUp()
        self._trainer = _load_trainer_script()

    def test_ceiling_is_load_bearing(self):
        guard = self._trainer._carrier_adequate
        res = _synthetic_residual(_GUARD_W_GRID.size, scale=1e3)  # |r| = 550
        self.assertTrue(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1
        with mock.patch.object(self._trainer, '_ABS_GUARD_CEILING', 1e2):
            self.assertFalse(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1

    def test_floor_is_load_bearing(self):
        guard = self._trainer._carrier_adequate
        res = _synthetic_residual(_GUARD_W_GRID.size, scale=1e-2)  # |r| = 5.5e-3
        self.assertTrue(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1
        with mock.patch.object(self._trainer, '_ABS_GUARD_FLOOR', 1e-2):
            self.assertFalse(guard(res, 'airy_fold', _GUARD_W_GRID, math.inf))
        self.n_checks += 1


class PerCarrierServeAccuracyTestCase(_BaseChartTestCase):
    """Per-carrier served-accuracy pins (disjoint cells, engine oracle).

    Each cell is served by the REAL serve (`_low_w_diffractive_chart_serve`
    via `_serve_farfield`) and compared against the independent engine oracle
    `_engine_reference_kappa` at ``kappa = beta = 0`` (the reduced-frame
    ``f_schwinger``).  Chart-served nodes (fold / macro-unresolved /
    deep-interior) are node-exact (<= 1e-10 at grid nodes); the geometric
    (resolved far-exterior) nodes are node-exact TOO, because the baked
    residual ``r = f_pure * sqrt(1 - gamma'**2) / F_ref`` absorbs the
    two-image geometric sum's finite-w deviation and the serve re-modulates
    with the same partitioned reference the trainer baked (measured ~1e-16 at
    grid nodes; a BARE geometric sum would fail the 1e-10 bar, which is the
    regression the anchoring eliminates).  Off-grid interpolation accuracy is
    a DRIVER concern (the full-bake margin report), not a sparse-chart unit
    test.  Every pin premise-checks its served w is on the correct side of
    ``w_split`` by MEASURING ``delta_tau`` via
    `_reduced_min_delay_separation`.

    The fifth cell -- the b3 -> 0 cusp cell (Pearcey carrier) -- is covered by
    `CuspServeEngineNodeExactTestCase` / `CuspFrefNonVanishingTestCase` and is
    RED until the cusp-transition detection is fixed (see
    `CuspFrefNonVanishingTestCase._witness`).
    """

    def _assert_serve(self, chart, gp, rho, theta, w, tol):
        """Serve one w and assert |F_serve - F_engine| / |F_engine| <= tol."""
        f_serve = _serve_cell(chart, gp, rho, theta, w)
        if f_serve is None:
            self.fail(f'serve declined cell (gamma_prime={gp}, rho={rho}, '
                      f'theta={theta}, w={w}); the chart must cover it')
        lens = _make_lens(gp, rho, theta, 0.0, 0.0)
        y = (lens['y1'], lens['y2'])
        f_engine = _engine_reference_kappa(w, y, lens['gamma'], 0.0, 0.0)
        rel = abs(f_serve[0] - f_engine) / abs(f_engine)
        self.assertLess(
            rel, tol,
            f'|F_serve-F_engine|/|F_engine| = {rel:.3e} >= {tol} at w={w:g} '
            f'(gamma_prime={gp}, rho={rho}, theta={theta})')
        self.n_checks += 1

    def test_fold_cell_serve(self):
        """The caustic-neighborhood fold cell is node-exact (airy_fold)."""
        chart = _build_exact_chart()
        gp, rho, theta = FOLD_CELL
        self.assertGreaterEqual(rho, RHO_LO,
                                'premise lost: fold cell no longer in shell')
        self.assertLessEqual(rho, RHO_HI,
                             'premise lost: fold cell no longer in shell')
        self.n_checks += 1
        for w in SERVE_WS:  # grid nodes
            with self.subTest(w=w):
                self._assert_serve(chart, gp, rho, theta, w, NODE_EXACT_TOL)

    def test_geometric_far_exterior_serve(self):
        """Resolved far-exterior nodes are node-exact (anchored geometric sum).

        The chart's residual is the two-image geometric sum divided by the
        trainer's partitioned reference, so the re-modulated serve equals the
        engine to ~1e-16 at grid nodes (the residual does NOT serve the bare
        geometric sum).  Same ``NODE_EXACT_TOL`` bar as the fold/macro/
        deep-interior pins.
        """
        chart = _build_far_exterior_chart()
        gp, rho, theta = FAR_EXT_CELL
        w_split = _reduced_w_split(gp, rho, theta)
        for w in (2.0, 8.0):
            with self.subTest(w=w):
                self.assertGreaterEqual(
                    w, w_split,
                    'premise lost: served w is no longer resolved '
                    f'(w={w} < w_split={w_split})')
                self.n_checks += 1
                self._assert_serve(chart, gp, rho, theta, w, NODE_EXACT_TOL)

    def test_macro_far_exterior_serve(self):
        """Unresolved far-exterior node is the macro-lead re-modulation."""
        chart = _build_far_exterior_chart()
        gp, rho, theta = FAR_EXT_CELL
        w_split = _reduced_w_split(gp, rho, theta)
        self.assertLess(
            0.2, w_split,
            f'premise lost: w=0.2 is no longer unresolved (w_split={w_split})')
        self.n_checks += 1
        self._assert_serve(chart, gp, rho, theta, 0.2, NODE_EXACT_TOL)

    def test_deep_interior_macro_serve(self):
        """Deep-interior macro cell (all served w unresolved) is node-exact."""
        chart = _build_deep_interior_chart()
        gp, rho, theta = DEEP_INT_CELL
        w_split = _reduced_w_split(gp, rho, theta)
        self.assertGreater(
            w_split, 8.0,
            f'premise lost: served w now resolved (w_split={w_split})')
        self.n_checks += 1
        for w in (0.2, 2.0, 8.0):  # grid nodes
            with self.subTest(w=w):
                self._assert_serve(chart, gp, rho, theta, w, NODE_EXACT_TOL)


# ---------------------------------------------------------------------------
# Rho-partition continuity (spec: no step in F_serve at rho = RHO_HI) and the
# macro-fold low-w normalization (spec: fold residual -> sqrt(1-gp^2) at the
# band bottom, not 0).
# ---------------------------------------------------------------------------

#: Wall-band reduced shear of the rho-split continuity sweep.  gamma' = 0.8
#: (> _WALL_GAMMA_PRIME = 0.5) is REQUIRED: the coverage union
#: ``(RHO_LO <= rho <= RHO_HI) or (gamma' > _WALL_GAMMA_PRIME)`` admits the
#: off-caustic side of RHO_HI only through the wall-band clause.  At
#: gamma' < 0.5 a rho just above RHO_HI is neither shell nor wall, so the
#: serve DECLINES it and the sweep could not cross the boundary.
CONTINUITY_GAMMA_PRIME = 0.8

#: Eigenframe angle of the continuity sweep (an interior
#: `_CONTINUITY_THETA_GRID` node; theta = 1.4 resolves to a cusp at
#: rho ~ 1.3 and is declined there).
CONTINUITY_THETA = 0.8

#: rho nodes straddling the RHO_HI partition (INTERIOR `_CONTINUITY_RHO_GRID`
#: nodes, so the serve's ~1e-16 coordinate round-off cannot push them outside
#: the grid).  ``CONTINUITY_FOLD_RHO`` is the caustic (airy_fold) side,
#: ``CONTINUITY_MACRO_RHO`` the off-caustic (macro) side of RHO_HI = 1.4.
CONTINUITY_FOLD_RHO = 1.3
CONTINUITY_MACRO_RHO = 1.45

#: Continuity-sweep chart grids.  rho spans RHO_HI = 1.4 with two nodes on
#: the caustic (fold) side and two on the off-caustic (macro) side; every
#: (gamma', rho, theta) cell is F_ref-buildable (airy_fold for
#: rho <= RHO_HI, macro for rho > RHO_HI).  gamma' and theta are interior
#: nodes for the same round-off reason.
_CONTINUITY_GP_GRID = np.array([0.6, 0.8, 0.9, 0.95])
_CONTINUITY_RHO_GRID = np.array([1.1, 1.3, 1.45, 1.6])
_CONTINUITY_THETA_GRID = np.array([0.6, 0.8, 1.0, 1.2])

#: Low-w continuity witness frequency (a `_W_GRID` node).  w = 0.3 is
#: UNRESOLVED on both sides of RHO_HI (``w * delta_tau < RHO_END``; the
#: smallest macro-side ``w_split = RHO_END / delta_tau`` is ~0.455 at
#: rho = 1.6), so every swept rho is served by the chart re-modulation
#: (airy_fold F_ref on the caustic side, macro-lead F_ref off-caustic) --
#: the branch where the served amplitude is node-exact and the partition
#: handover must be continuous.
CONTINUITY_W = 0.3

#: Band-bottom residual-approach bar (macro-fold normalization spec):
#: ``| |r(w_bottom)| - sqrt(1-gp^2) | <= MACRO_FOLD_LOWW_APPROACH_TOL``.
#: Measured 1.5e-2 at the fold witness (gamma'=0.3, rho=1.0, theta=1.4,
#: w_bottom = 0.02); the raw (un-renormalized) fold form sits 4.1e-1 BELOW
#: sqrt(1-gp^2) because its residual dives toward 0 (the w^{-1/6} divergence
#: of |F_ref|) instead of asymptoting to sqrt(1-gp^2).  5e-2 leaves 3.3x
#: margin over the measured deviation while being 8x below the raw form's.
MACRO_FOLD_LOWW_APPROACH_TOL = 5e-2


@functools.lru_cache(maxsize=1)
def _build_continuity_chart() -> LowWDiffractiveChart:
    """Node-exact residual chart spanning the RHO_HI partition.

    Same construction as `_build_exact_chart` but on the continuity grids, so
    the straddling rho nodes (1.3 fold / 1.45 macro) are interior nodes whose
    residuals are anchored to the correct carrier (airy_fold vs macro).
    Cached so the continuity tests share one build.
    """
    return _build_exact_chart_on(_CONTINUITY_GP_GRID, _CONTINUITY_RHO_GRID,
                                 _CONTINUITY_THETA_GRID, _W_GRID,
                                 'rho_split_continuity')


def _fold_witness_h(gamma_prime: float, rho: float, theta: float,
                    w_grid: np.ndarray) -> np.ndarray:
    """Macro-fold handover ``h = smootherstep(w |delta_tau|, RHO_START, RHO_END)``.

    ``delta_tau`` comes from the merging fold pair (the same
    `partitioned_reference` uses); ``h == 0`` marks the unresolved nodes where
    the macro-fold renormalization is fully ON (``|F_ref| -> sqrt_mu``).
    Raises loudly if the merging fold pair is absent (premise), so a fixture
    move surfaces rather than silently mis-measuring the band bottom.
    """
    source = reduced_source(gamma_prime, rho, theta)
    matrix = geometry.macro_matrix(gamma_prime, 0.0, 0.0)
    images = geometry.find_images(source, matrix)
    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        raise AssertionError(
            f'merging fold pair absent at (gamma_prime={gamma_prime}, '
            f'rho={rho}, theta={theta})')
    tau_plus, tau_minus = pair
    delta_tau = tau_minus - tau_plus
    return smootherstep(np.asarray(w_grid) * abs(delta_tau),
                        RHO_START, RHO_END)


class RhoPartitionContinuityTestCase(_BaseChartTestCase):
    """Served amplitude is continuous across the rho = RHO_HI partition.

    At fixed (gamma' = 0.8, theta = 0.8, w = 0.3), rho sweeps from the
    caustic-neighborhood side (``CONTINUITY_FOLD_RHO = 1.3``, airy_fold
    carrier) to the off-caustic side (``CONTINUITY_MACRO_RHO = 1.45``, macro
    carrier).  The residual ``r`` is carrier-relative and legitimately
    changes form at the boundary, but the SERVED amplitude ``F_serve =
    mass_sheet_phase * F_ref * sqrt_mu_full * r`` must not step: it
    re-modulates back to ``mass_sheet_phase * f_pure / lam`` (the engine) on
    BOTH sides.  With an EXACT-residual chart every swept rho is a grid node,
    so ``F_serve`` reproduces the engine to ~1e-15 -- far stronger than the
    spec's 1e-4.  A carrier partition mis-wiring (the wrong F_ref on one
    side) breaks the re-modulation and shows as a discontinuity at RHO_HI
    (pinned by `RhoPartitionContinuitySelfFalsificationTestCase`).
    """

    def test_sweep_straddles_the_carrier_partition(self):
        """Premise: the two rho nodes sit on opposite carriers of RHO_HI."""
        self.assertLess(CONTINUITY_FOLD_RHO, RHO_HI,
                        'premise lost: fold rho no longer inside the shell')
        self.assertGreater(CONTINUITY_MACRO_RHO, RHO_HI,
                           'premise lost: macro rho no longer off-caustic')
        self.n_checks += 1
        # Assert the partition on the single UNRESOLVED frequency: the full
        # ``_W_GRID`` spans resolved nodes (w = 2.0 / 8.0 > w_split), which
        # would route the off-caustic side to the split 'geometric' carrier
        # instead of 'macro'.  At ``CONTINUITY_W = 0.3`` (below w_split on
        # both sides) the off-caustic node is the pure macro carrier.
        for rho in (CONTINUITY_FOLD_RHO, CONTINUITY_MACRO_RHO):
            source = reduced_source(CONTINUITY_GAMMA_PRIME, rho,
                                    CONTINUITY_THETA)
            _f_ref, kind = partitioned_reference(np.array([CONTINUITY_W]),
                                                 CONTINUITY_GAMMA_PRIME, rho,
                                                 source)
            expected = 'airy_fold' if rho <= RHO_HI else 'macro'
            with self.subTest(rho=rho):
                self.assertEqual(
                    kind, expected,
                    f'rho={rho} built carrier {kind!r}, expected {expected!r}; '
                    'the sweep no longer crosses the RHO_HI partition')
                self.n_checks += 1

    def test_continuity_w_is_unresolved_on_both_sides(self):
        """Premise: w=0.3 is below w_split on the macro side (no geometric)."""
        w_split = _reduced_w_split(CONTINUITY_GAMMA_PRIME, CONTINUITY_MACRO_RHO,
                                   CONTINUITY_THETA)
        self.assertLess(
            CONTINUITY_W, w_split,
            f'premise lost: w={CONTINUITY_W} is resolved on the macro side '
            f'(w_split={w_split}); the geometric branch would not be node-exact')
        self.n_checks += 1

    def test_served_amplitude_continuous_across_rho_split(self):
        """|F_serve - F_engine| / |F_engine| <= NODE_EXACT_TOL on both sides."""
        chart = _build_continuity_chart()
        for rho in (CONTINUITY_FOLD_RHO, CONTINUITY_MACRO_RHO):
            f_serve = _serve_cell(chart, CONTINUITY_GAMMA_PRIME, rho,
                                  CONTINUITY_THETA, CONTINUITY_W)
            if f_serve is None:
                self.fail(f'serve declined rho={rho} at the partition sweep; '
                          'the continuity chart must cover it')
            lens = _make_lens(CONTINUITY_GAMMA_PRIME, rho, CONTINUITY_THETA,
                              0.0, 0.0)
            f_engine = _engine_reference_kappa(
                CONTINUITY_W, (lens['y1'], lens['y2']), lens['gamma'], 0.0,
                0.0)
            rel = abs(f_serve[0] - f_engine) / abs(f_engine)
            with self.subTest(rho=rho):
                self.assertLess(
                    rel, NODE_EXACT_TOL,
                    f'F_serve disagrees with the engine by {rel:.3e} at '
                    f'rho={rho} (w={CONTINUITY_W}): the served amplitude '
                    'steps at the RHO_HI partition')
                self.n_checks += 1


class RhoPartitionContinuitySelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the continuity pin catches a carrier partition mis-wiring.

    Forcing the SERVE to re-modulate the caustic-side (fold) residual with
    the macro carrier (the wrong F_ref for that residual) breaks the
    node-exact re-modulation -- ``F_ref * r`` no longer equals ``f_pure *
    sqrt(1-gp^2)`` -- so ``F_serve`` steps away from the engine: the exact
    discontinuity at RHO_HI the continuity test guards.
    """

    def test_miswired_fold_side_breaks_continuity(self):
        """Forcing the macro carrier on the fold side breaks node-exactness."""
        chart = _build_continuity_chart()
        real_ref = _likelihood_mod.partitioned_reference

        def _macro_everywhere(w, gp, rho, source):
            return real_ref(w, gp, 2.0, source)  # force the macro carrier

        with mock.patch.object(_likelihood_mod, 'partitioned_reference',
                               _macro_everywhere):
            f_serve = _serve_cell(chart, CONTINUITY_GAMMA_PRIME,
                                  CONTINUITY_FOLD_RHO, CONTINUITY_THETA,
                                  CONTINUITY_W)
        if f_serve is None:
            self.fail('serve declined under the mis-wiring')
        lens = _make_lens(CONTINUITY_GAMMA_PRIME, CONTINUITY_FOLD_RHO,
                          CONTINUITY_THETA, 0.0, 0.0)
        f_engine = _engine_reference_kappa(
            CONTINUITY_W, (lens['y1'], lens['y2']), lens['gamma'], 0.0, 0.0)
        rel = abs(f_serve[0] - f_engine) / abs(f_engine)
        self.assertGreater(
            rel, NODE_EXACT_TOL,
            f'mis-wired fold side left rel err {rel:.3e} <= {NODE_EXACT_TOL}; '
            'the continuity pin would not catch a partition mis-wiring')
        self.n_checks += 1


class MacroFoldNormalizationTestCase(_BaseChartTestCase):
    """Macro-fold low-w normalization: the fold residual asymptotes to
    sqrt(1-gp^2) at the band bottom, not 0.

    The macro-fold renormalization in `_airy_fold_form`
    (``f_ref *= h + (1-h) sqrt_mu / |f_ref|`` with ``h = smootherstep(w
    |delta_tau|, RHO_START, RHO_END)``) keeps the fold-side residual
    ``r = f_pure sqrt(1-gp^2) / F_ref`` O(1) at low w: at ``h == 0``,
    ``|F_ref| -> sqrt_mu`` and ``|f_pure| -> sqrt_mu``, so ``|r| ->
    sqrt(1-gp^2)``.  WITHOUT it the raw fold form diverges like ``w^{-1/6}``
    and the residual dives toward 0.  The renormalization factor is real and
    positive, so the Airy phase is preserved (the arg-step guard in
    `ResidualBoundednessTestCase` still holds).  This pins the RESIDUAL-level
    asymptote; the F_ref-level statement (``|F_ref| == sqrt_mu`` at ``h ==
    0``) is pinned separately by
    `FrefNonVanishingTestCase.test_magnitude_tracks_wronskian_form`.
    """

    def test_fold_residual_approaches_sqrt_1_gp2_at_band_bottom(self):
        """| |r(w_bottom)| - sqrt(1-gp^2) | <= MACRO_FOLD_LOWW_APPROACH_TOL."""
        gp = NEAR_FOLD_GAMMA_PRIME
        rho = NEAR_FOLD_RHO
        theta = FREF_SHELL_THETA
        residual = _residual_at(_FREF_W_GRID, gp, rho, theta)
        if residual is None:
            self.fail('F_ref unbuildable at the fold witness')
        h = _fold_witness_h(gp, rho, theta, _FREF_W_GRID)
        unresolved = h == 0.0
        self.assertTrue(np.any(unresolved),
                        'premise lost: no unresolved (h==0) nodes on the grid')
        self.n_checks += 1
        bottom = int(np.flatnonzero(unresolved)[0])  # lowest-w unresolved node
        sqrt_1_gp2 = math.sqrt(1.0 - gp * gp)
        deviation = abs(abs(residual[bottom]) - sqrt_1_gp2)
        self.assertLess(
            deviation, MACRO_FOLD_LOWW_APPROACH_TOL,
            f'|r(w_bottom)| = {abs(residual[bottom]):.4f} deviates from '
            f'sqrt(1-gp^2) = {sqrt_1_gp2:.4f} by {deviation:.2e}; the fold '
            'residual does not asymptote to sqrt(1-gp^2) at the band bottom')
        self.n_checks += 1


class MacroFoldNormalizationSelfFalsificationTestCase(_BaseChartTestCase):
    """Prove the band-bottom asymptote is the renormalization's doing.

    Forcing ``h = 1`` everywhere (``smootherstep -> 1.0``) leaves the raw q=p
    fold form un-renormalized, whose magnitude diverges like ``w^{-1/6}``:
    the residual then dives toward 0 at the band bottom instead of
    asymptoting to sqrt(1-gp^2).  The same band-bottom probe must then
    deviate from sqrt(1-gp^2) by far more than the approach bar.
    """

    def test_unrenormalized_fold_residual_dives_below_bar(self):
        """The raw (h=1) fold residual at the band bottom is far from sqrt(1-gp^2)."""
        gp = NEAR_FOLD_GAMMA_PRIME
        rho = NEAR_FOLD_RHO
        theta = FREF_SHELL_THETA
        source = reduced_source(gp, rho, theta)
        with mock.patch.object(_lwd_module, 'smootherstep',
                               lambda x, lo, hi: 1.0):
            f_ref_raw, kind = partitioned_reference(_FREF_W_GRID, gp, rho,
                                                    source)
        self.assertEqual(kind, 'airy_fold')
        self.n_checks += 1
        sq = math.sqrt(1.0 - gp * gp)
        f_pure = np.array([f_schwinger(float(w), source, gp)
                           for w in _FREF_W_GRID])
        residual_raw = f_pure * sq / f_ref_raw
        deviation = abs(abs(residual_raw[0]) - sq)
        self.assertGreater(
            deviation, MACRO_FOLD_LOWW_APPROACH_TOL,
            f'un-renormalized band-bottom residual deviates from sqrt(1-gp^2) '
            f'by only {deviation:.2e} <= {MACRO_FOLD_LOWW_APPROACH_TOL}; the '
            'macro-fold normalization pin would be vacuous')
        self.n_checks += 1


if __name__ == '__main__':
    main()
