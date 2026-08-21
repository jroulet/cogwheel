"""
Tests for the low-w near-fold-shell residual chart (``low_w_shell_chart``).

This suite certifies the macro-lead demodulated-DIFFERENCE representation
that ``LowWShellChart`` interpolates and the likelihood's Rung-P shell serve
re-assembles.  Three invariants, one per ``TestCase`` family:

1. ``ShellResidualFiniteTestCase`` -- the difference residual
   ``R = f_pure - carrier`` has NO poles.  The quotient form this chart
   replaces divided by a carrier that beats through zero; the difference
   ``f_pure - carrier`` carries the SAME beat on both sides, so it is
   everywhere finite and O(1).  Pinned as: ``R`` finite at every witness
   node, ``max|R| / max|F_serve| <= 10``, and the re-modulated ``|F_serve|``
   never collapses toward zero or explodes (held inside a
   ``[0.5, 5] x sqrt(mu_macro)`` band).  The residual here is the ANALYTIC
   difference (oracle minus carrier), computed directly -- never via
   ``chart.evaluate`` (F002: a test must not call the chart under test to
   assert the chart's own correctness).

2. ``ShellNodeExactAccuracyTestCase`` -- a synthetic EXACT-residual chart
   (coefficients baked from ``f_schwinger - born_lead_carrier`` AT the grid
   nodes, so interpolation error is identically zero at nodes) round-trips
   the serve to the oracle.  Node-exact agreement is pinned to 1e-10 (the
   float64 round-trip is ~1e-15, leaving ~1e5 margin).  Off-grid agreement
   is NOT the spec's 1e-4: at a synthetic ~500-node scale the cubic theta
   interpolation overshoots to ~2e-2 (the 1e-4 bar is the TRAINED chart's
   acceptance -- ~16 theta / ~14 log-w nodes -- a driver full-bake concern,
   not a unit-testable invariant at this scale).  The off-grid test instead
   pins a MEASURED bar (0.1, ~5x above the measured worst) so a broken
   re-modulation is still caught off-grid.

3. ``ShellBornBoundaryContinuityTestCase`` -- the ``rho = 1.4`` handoff
   between the shell chart (``rho <= RHO_HI``) and the Born rung
   (``rho > _BORN_RHO_FLOOR``) has no gap and no step: the two boundaries
   are the SAME constant (1.4), the shell chart's served ``F`` at the
   boundary matches the oracle, and the Born carrier-only certificate
   REFUSES at ``rho = 1.4`` (the residual is not negligible near the
   caustic), so the Born side falls through to the exact engine rather than
   serving a finite-but-wrong carrier -- the honest form of the "no step"
   acceptance (Professor Q3b: no analytic arm serves the near-caustic shell
   at 1e-4 without the trained residual).

``ShellFalsificationTestCase`` closes the loop: it bakes deliberately-wrong
residuals (doubled carrier, unit ``sqrt_mu``, zero residual) and asserts the
node-exact invariant GOES RED on them, so a green suite is evidence and not
decoration.

4. ``ShellServeNodeExactTestCase`` (INS-2-001 port) -- the END-TO-END
   production serve.  The synthetic exact-residual chart is bound to a bare
   namespace and the SHIPPED
   ``LensedRelativeBinningLikelihood._low_w_shell_chart_serve`` is driven
   directly with the ``reconstruct_farfield`` tail intercepted (``t_min =
   0``, so the captured envelope IS the re-modulated farfield).  The
   re-modulated ``F`` must match the mass-sheet-map engine oracle
   ``_engine_reference_kappa`` to node-exact tolerance at every grid node --
   for BOTH ``kappa = 0`` and ``kappa != 0`` (at ``kappa = 0`` the
   mass-sheet phase and ``1/lam`` collapse to identity, so only a
   ``kappa != 0`` fixture can catch a serve that drops the gauge
   composition).  The band splits at ``w_shell = 1/delta_min``: the
   below-split nodes are the chart composition (the accuracy pin), and the
   above-split nodes are hosted through ``_engine_farfield_total`` (stubbed
   to the oracle, never engine-vs-engine) -- the split-mask wiring and the
   no-step continuity are pinned, not the engine.

5. ``ShellLoadContractTestCase`` (INS-2-001 port) -- the npz load contract.
   A schema-tagged, content-hashed artifact round-trips bit-identically; a
   missing/foreign schema, a missing ``content_hash``, and a tampered
   coefficient under a STALE hash all hard-refuse with a ``ValueError``
   naming the training script.  The positive control (a tampered value
   RE-hashed to consistency loads cleanly) pins the saver<->loader
   hash-field agreement; a tampered PROVENANCE (excluded from the hash)
   still loads -- the hash covers the physics arrays only.

Tolerances: node-exact 1e-10 (float64 round-trip ~1e-15); residual ratio
cap 10 (measured O(1) <= 0.9 over the witness cells; the quotient form blew
past this by ~1000x at carrier zeros); the ``F_serve`` magnitude band
``[0.5, 5] x sqrt(mu_macro)`` (measured values stay within ~[1.0, 2.0] of
the macro magnitude for the witness cells); off-grid 0.1 (measured ~2e-2 at
a 480-node fixture).
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

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import operator as _operator
from cogwheel.lensing.chang_refsdal._born import (
    born_carrier_omitted_term, born_lead_carrier)
from cogwheel.lensing.chang_refsdal._schwinger import f_schwinger
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.low_w_shell_chart import (
    RHO_HI, RHO_LO, _SCHEMA, _content_hash, LowWShellChart, reduced_source,
    _reduced_min_delay_separation)
from cogwheel.lensing import likelihood as _likelihood

#: Diagnostic output directory (plots are evidence, not assertions).
OUTPUT_DIR = Path(__file__).parent / 'output'

#: Node-exact agreement required of the re-modulated serve at grid nodes.
#: The float64 round-trip is bit-exact; 1e-10 leaves ~1e5 margin.
_NODE_EXACT_TOL = 1e-10

#: Off-grid interpolation bar (theta / log-w midpoints) at synthetic scale.
#: See the module docstring: the trained chart's 1e-4 acceptance is a driver
#: full-bake concern; a 480-node fixture measures ~2e-2, pinned 5x above.
_OFF_GRID_TOL = 1e-1

#: Hard cap on max|R| / max|F_serve|.  The difference residual is O(1)
#: (measured <= 0.9 over the witness cells); the quotient form it replaces
#: blew past this by ~1000x at carrier zeros.
_RESIDUAL_RATIO_CAP = 10.0

#: Re-modulated |F_serve| must stay within [floor, ceiling] x sqrt(mu_macro):
#: a collapse toward 0 or an explosion flags a wrong reconstruction.
_FSERVE_FLOOR_FRAC = 0.5
_FSERVE_CEILING_FRAC = 5.0

#: Relative-error bar for the falsification controls: a wrong residual must
#: push the node-exact error clearly above the 1e-10 pin (measured ~0.4-1.0).
_FALSIFY_REL_ERR = 1e-3

#: Witness cells (gamma_prime, rho, theta): the b3->0 cell (0.8, 1.1, 0.2)
#: plus a mid-shear shell cell and a low-shear interior-shell cell.
_WITNESS_CELLS = ((0.8, 1.1, 0.2), (0.5, 1.2, 0.5), (0.3, 0.9, 1.0))

#: Dimensionless-frequency grid for the no-poles sweep (w <= 1).
_W_GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])


def _sqrt_mu_macro(gamma_prime: float) -> float:
    """Return ``sqrt(|mu_macro|) = 1/sqrt(|1 - gamma'**2|)`` at kappa = 0."""
    return 1.0 / math.sqrt(abs(1.0 - gamma_prime ** 2))


def _lead_carrier(w: float, source: np.ndarray,
                  gamma_prime: float) -> complex:
    """Return the macro-lead Born carrier at the reduced source."""
    return born_lead_carrier(w, float(source[0]), float(source[1]),
                             gamma_prime)


def _analytic_residual(w: float, source: np.ndarray,
                       gamma_prime: float) -> complex:
    """Return ``R = f_pure - carrier`` -- the difference the chart stores."""
    return f_schwinger(w, source, gamma_prime) - _lead_carrier(
        w, source, gamma_prime)


def _reconstruct_f(w: np.ndarray, carrier: np.ndarray,
                   residual: np.ndarray, *, lam: float = 1.0,
                   kappa: float = 0.0, s: float = 0.0) -> np.ndarray:
    """Re-assemble the served ``F`` from the carrier and residual.

    The likelihood serve reconstructs
    ``F_abs = mass_sheet_phase * (carrier + R) / lam`` with
    ``mass_sheet_phase = exp(0.5j w (log(lam) - kappa s))``.  The chart is
    a ``kappa = 0`` reference surface (its axes carry gamma', rho, theta --
    no kappa / beta), so ``lam = 1`` and the mass-sheet phase is identically
    unity; the formula collapses to ``carrier + residual``.  The full form is
    kept so the kappa = 0 reduction is documented, not assumed.
    """
    phase = 0.5j * np.asarray(w, dtype=float) * (
        math.log(lam) - kappa * s)
    return np.exp(phase) * (carrier + residual) / lam


@functools.lru_cache(maxsize=1)
def _exact_residual_chart() -> LowWShellChart:
    """Build a small synthetic chart whose coeffs are the EXACT residual.

    Coefficients are ``f_schwinger - born_lead_carrier`` AT the grid nodes,
    so ``chart.evaluate`` reproduces the stored residual bit-exactly at every
    node (cubic interpolation passes through the data).  The grids are
    deliberately small (4 x 4 x 6 x 5 = 480 engine nodes) to keep the build
    a few seconds while still spanning a representative shell box and
    including the witness points ``gamma_prime = 0.8`` and ``rho = 1.4`` as
    interior/edge nodes.
    """
    gamma_prime_grid = np.array([0.3, 0.5, 0.8, 0.9])
    rho_grid = np.array([0.6, 0.9, 1.2, 1.4])
    theta_grid = np.linspace(0.0, math.pi / 2, 6)
    log_w_grid = np.log(np.array([0.05, 0.1, 0.15, 0.2, 0.3]))
    w_grid = np.exp(log_w_grid)
    shape = (len(gamma_prime_grid), len(rho_grid), len(theta_grid),
             len(w_grid))
    real = np.empty(shape)
    imag = np.empty(shape)
    for i, gamma_prime in enumerate(gamma_prime_grid):
        for j, rho in enumerate(rho_grid):
            for k, theta in enumerate(theta_grid):
                source = reduced_source(float(gamma_prime), float(rho),
                                        float(theta))
                for ell, w in enumerate(w_grid):
                    residual = _analytic_residual(float(w), source,
                                                  float(gamma_prime))
                    real[i, j, k, ell] = residual.real
                    imag[i, j, k, ell] = residual.imag
    return LowWShellChart(gamma_prime_grid=gamma_prime_grid,
                          rho_grid=rho_grid, theta_grid=theta_grid,
                          log_w_grid=log_w_grid, real_coeffs=real,
                          imag_coeffs=imag, provenance={})


def _save_no_poles_plot(cells: dict) -> None:
    """Write |R| and |F_serve| vs w for each witness cell (diagnostic).

    A quotient-style pole would show as a >1e3 spike at an interference
    zero; the difference residual stays smooth and O(1).  Plotting is a
    best-effort diagnostic -- an absent matplotlib must not fail the suite.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (gamma_prime, rho, theta), (_s, residual, _c, f_serve) in \
            cells.items():
        label = f'gp={gamma_prime},rho={rho},th={theta:.2f}'
        axes[0].semilogx(_W_GRID, np.abs(residual), '.-', label=label)
        axes[1].semilogx(_W_GRID, np.abs(f_serve), '.-', label=label)
    axes[0].set_title('|R| = |f_pure - carrier|')
    axes[0].set_xlabel('w'); axes[0].legend(fontsize=7)
    axes[1].set_title('|F_serve| = |carrier + R|')
    axes[1].set_xlabel('w'); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'low_w_shell_chart_no_poles.png', dpi=90)
    plt.close(fig)


class ShellResidualFiniteTestCase(TestCase):
    """The difference residual is finite, O(1), and re-modulates sanely."""

    @classmethod
    def setUpClass(cls):
        cls.cells = {}
        for gamma_prime, rho, theta in _WITNESS_CELLS:
            source = reduced_source(gamma_prime, rho, theta)
            residual = np.array([_analytic_residual(float(w), source,
                                                    gamma_prime)
                                 for w in _W_GRID], dtype=complex)
            carrier = np.array([_lead_carrier(float(w), source, gamma_prime)
                                for w in _W_GRID], dtype=complex)
            f_serve = _reconstruct_f(_W_GRID, carrier, residual)
            cls.cells[(gamma_prime, rho, theta)] = (source, residual,
                                                    carrier, f_serve)
        _save_no_poles_plot(cls.cells)

    def setUp(self):
        self.n_compared = 0

    def tearDown(self):
        if not self.n_compared:
            self.fail('no witness cell was compared; the test asserted '
                      'nothing')

    def test_residual_finite_everywhere(self):
        """R = f_pure - carrier is finite at every node (no poles)."""
        for (gamma_prime, rho, theta), (_s, residual, _c, _f) in \
                self.cells.items():
            self.n_compared += 1
            self.assertTrue(
                np.isfinite(residual).all(),
                f'non-finite residual at (gp={gamma_prime}, rho={rho}, '
                f'theta={theta}): {residual}')

    def test_residual_magnitude_bounded(self):
        """max|R| / max|F_serve| stays O(1) (<= 10), unlike the quotient."""
        for cell, (_s, residual, _c, f_serve) in self.cells.items():
            self.n_compared += 1
            ratio = np.abs(residual).max() / np.abs(f_serve).max()
            self.assertLessEqual(
                ratio, _RESIDUAL_RATIO_CAP,
                f'cell {cell}: residual/F ratio {ratio:.3f} exceeds '
                f'{_RESIDUAL_RATIO_CAP}')

    def test_fserve_magnitude_well_behaved(self):
        """|F_serve| stays within [0.5, 5] x sqrt(mu_macro) (no collapse)."""
        for (gamma_prime, rho, theta), (_s, _r, _c, f_serve) in \
                self.cells.items():
            self.n_compared += 1
            sqrt_mu = _sqrt_mu_macro(gamma_prime)
            self.assertGreaterEqual(
                np.abs(f_serve).min(), _FSERVE_FLOOR_FRAC * sqrt_mu,
                f'cell (gp={gamma_prime}, rho={rho}, theta={theta}): '
                f'|F_serve| collapsed to {np.abs(f_serve).min():.3f}')
            self.assertLessEqual(
                np.abs(f_serve).max(), _FSERVE_CEILING_FRAC * sqrt_mu,
                f'cell (gp={gamma_prime}, rho={rho}, theta={theta}): '
                f'|F_serve| exploded to {np.abs(f_serve).max():.3f}')


class ShellNodeExactAccuracyTestCase(TestCase):
    """The serve round-trips an exact-residual chart to the oracle.

    The chart under test is a SYNTHETIC chart whose coefficients are baked
    from the ORACLE at the grid nodes (see ``_exact_residual_chart``).  The
    serve is re-assembled as ``F = carrier + chart.evaluate(...)`` and
    compared against a FRESH ``f_schwinger`` evaluation -- the oracle is
    independent of the chart's stored coefficients.
    """

    @classmethod
    def setUpClass(cls):
        cls.chart = _exact_residual_chart()
        cls.gamma_prime = 0.8
        cls.rho = 1.2
        cls.theta = float(cls.chart.theta_grid[2])  # pi/5, an interior node
        cls.source = reduced_source(cls.gamma_prime, cls.rho, cls.theta)

    def _reconstruct_at(self, w, gamma_prime, rho, theta):
        """Serve one frequency via the chart and re-modulate to F."""
        w_arr = np.atleast_1d(np.asarray(w, dtype=float))
        source = reduced_source(gamma_prime, rho, theta)
        residual = self.chart.evaluate(w_arr, gamma_prime, rho, theta)
        carrier = np.array([_lead_carrier(float(wi), source, gamma_prime)
                            for wi in w_arr], dtype=complex)
        return _reconstruct_f(w_arr, carrier, residual)

    def test_node_exact_at_grid_nodes(self):
        """F_serve matches f_schwinger to 1e-10 at every grid node."""
        worst = 0.0
        for w in np.exp(self.chart.log_w_grid):
            f_serve = self._reconstruct_at(
                w, self.gamma_prime, self.rho, self.theta)[0]
            oracle = f_schwinger(float(w), self.source, self.gamma_prime)
            worst = max(worst, abs(f_serve - oracle) / abs(oracle))
        self.assertLessEqual(
            worst, _NODE_EXACT_TOL,
            f'node-exact agreement broken: worst relative error {worst:.3e} '
            f'> {_NODE_EXACT_TOL}')

    def test_off_grid_within_measured_bar(self):
        """F_serve at theta / log-w midpoints stays within 0.1 of the oracle.

        This is NOT the trained chart's 1e-4 acceptance -- see the module
        docstring.  It pins that the re-modulation is correct OFF-grid too
        (the error is interpolation overshoot, ~2e-2, not a broken formula).
        """
        worst = 0.0
        for k in range(len(self.chart.theta_grid) - 1):
            theta_mid = 0.5 * (self.chart.theta_grid[k]
                               + self.chart.theta_grid[k + 1])
            for iw in range(len(self.chart.log_w_grid) - 1):
                w_mid = math.exp(0.5 * (self.chart.log_w_grid[iw]
                                        + self.chart.log_w_grid[iw + 1]))
                f_serve = self._reconstruct_at(
                    w_mid, self.gamma_prime, self.rho, theta_mid)[0]
                source = reduced_source(self.gamma_prime, self.rho, theta_mid)
                oracle = f_schwinger(w_mid, source, self.gamma_prime)
                worst = max(worst, abs(f_serve - oracle) / abs(oracle))
        self.assertLessEqual(
            worst, _OFF_GRID_TOL,
            f'off-grid agreement broken: worst relative error {worst:.3e} '
            f'> {_OFF_GRID_TOL}')


class ShellBornBoundaryContinuityTestCase(TestCase):
    """The rho = 1.4 shell / Born handoff has no gap and no step.

    The shell chart owns ``rho in [RHO_LO, RHO_HI]``; the Born rung serves
    ``rho > _BORN_RHO_FLOOR``.  Both boundaries are 1.4 -- the SAME float --
    so there is no gap and no overlap.  The shell chart is accurate at its
    outer boundary node (rho = 1.4), and the Born carrier-only certificate
    refuses there (the residual is not negligible near the caustic), so the
    Born side falls through to the exact engine: neither side serves a
    finite-but-wrong value at the handoff.
    """

    @classmethod
    def setUpClass(cls):
        cls.chart = _exact_residual_chart()
        cls.gamma_prime = 0.8
        cls.rho_hi = RHO_HI
        cls.theta = float(cls.chart.theta_grid[2])

    def test_shell_outer_boundary_equals_born_floor(self):
        """RHO_HI and _BORN_RHO_FLOOR are the SAME constant (no gap/overlap)."""
        self.assertEqual(RHO_HI, _likelihood._BORN_RHO_FLOOR)

    def test_shell_chart_covers_up_to_boundary_only(self):
        """covers() admits rho <= 1.4 and declines rho > 1.4 at the box."""
        for rho in (RHO_HI - 0.01, RHO_HI):
            self.assertTrue(
                self.chart.covers(self.gamma_prime, rho, self.theta),
                f'rho={rho} should be inside [RHO_LO, RHO_HI]')
        self.assertFalse(
            self.chart.covers(self.gamma_prime, RHO_HI + 0.01, self.theta),
            f'rho={RHO_HI + 0.01} should be outside the shell box')

    def test_shell_serve_matches_oracle_at_boundary(self):
        """The shell chart's served F at rho = 1.4 matches the oracle.

        rho = 1.4 is a rho-grid node and the witness (gp, theta, w) are
        nodes, so the re-modulated F is node-exact against f_schwinger --
        the shell side of the handoff has no step.
        """
        worst = 0.0
        for w in np.exp(self.chart.log_w_grid):
            source = reduced_source(self.gamma_prime, self.rho_hi, self.theta)
            residual = self.chart.evaluate(np.array([w]), self.gamma_prime,
                                           self.rho_hi, self.theta)
            carrier = np.array([_lead_carrier(float(w), source,
                                              self.gamma_prime)])
            f_serve = _reconstruct_f(np.array([w]), carrier, residual)[0]
            oracle = f_schwinger(float(w), source, self.gamma_prime)
            worst = max(worst, abs(f_serve - oracle) / abs(oracle))
        self.assertLessEqual(
            worst, _NODE_EXACT_TOL,
            f'shell boundary serve broken: worst relative error {worst:.3e}')

    def test_born_carrier_only_refuses_at_boundary(self):
        """The Born carrier-only certificate refuses at rho = 1.4.

        The Born rung's beyond-box serve keeps ONLY the lead carrier (zero
        residual) and gates it with a carrier-relative truncation certificate
        at the band ceiling.  Near the caustic (rho = 1.4) the omitted term is
        O(1) -- safety * estimate far exceeds the certificate bar -- so the
        carrier-only serve refuses and the Born side falls through to the
        exact engine.  This is the honest "no step on the Born side": there
        is no rho where the Born rung serves a finite-but-wrong carrier.
        """
        source = reduced_source(self.gamma_prime, self.rho_hi, self.theta)
        est = born_carrier_omitted_term(0.1, float(source[0]),
                                        float(source[1]), self.gamma_prime)
        self.assertGreater(
            _likelihood._SADDLE_FARFIELD_SAFETY * est,
            _likelihood._SADDLE_FARFIELD_CERT_BAR,
            f'Born carrier-only certificate unexpectedly admits rho=1.4: '
            f'safety*est = {_likelihood._SADDLE_FARFIELD_SAFETY * est:.4g} '
            f'<= bar {_likelihood._SADDLE_FARFIELD_CERT_BAR}')


class ShellFalsificationTestCase(TestCase):
    """Prove this suite is able to FAIL on a wrong residual.

    A green node-exact suite is worth only as much as its ability to go red.
    These tests bake deliberately-wrong residuals (a doubled carrier, a unit
    ``sqrt_mu`` carrier, and an identically zero residual) and assert the
    node-exact invariant catches each -- the reconstructed ``F`` must drift
    far past the 1e-10 pin.  They also confirm the no-poles ratio cap has
    teeth against an oversized residual.
    """

    @classmethod
    def setUpClass(cls):
        cls.chart = _exact_residual_chart()
        cls.gamma_prime = 0.8
        cls.rho = 1.2
        cls.theta = float(cls.chart.theta_grid[2])
        cls.source = reduced_source(cls.gamma_prime, cls.rho, cls.theta)

    def _worst_node_error(self, mode: str) -> float:
        """Worst node relative error of ``F`` rebuilt from a wrong residual.

        ``mode`` selects the wrong residual: 'doubled_carrier' bakes
        ``R = f_pure - 2 * carrier``, 'unit_sqrt_mu' bakes
        ``R = f_pure - carrier / sqrt_mu`` (magnitude forced to unity), and
        'zero' bakes ``R = 0`` (the serve forgot the residual).  The serve
        re-modulates with the REAL carrier in every case, exactly as the
        production path would.
        """
        worst = 0.0
        sqrt_mu = _sqrt_mu_macro(self.gamma_prime)
        for w in np.exp(self.chart.log_w_grid):
            f_pure = f_schwinger(float(w), self.source, self.gamma_prime)
            carrier = _lead_carrier(float(w), self.source, self.gamma_prime)
            if mode == 'doubled_carrier':
                residual = f_pure - 2.0 * carrier
            elif mode == 'unit_sqrt_mu':
                residual = f_pure - carrier / sqrt_mu
            elif mode == 'zero':
                residual = 0.0
            else:
                raise ValueError(f'unknown falsification mode {mode!r}')
            f_serve = _reconstruct_f(np.array([w]), np.array([carrier]),
                                     np.array([residual]))[0]
            worst = max(worst, abs(f_serve - f_pure) / abs(f_pure))
        return worst

    def test_doubled_carrier_breaks_node_exactness(self):
        """A doubled carrier in the residual breaks node-exact agreement."""
        self.assertGreater(
            self._worst_node_error('doubled_carrier'), _FALSIFY_REL_ERR,
            'doubled carrier did not break node-exactness; the node-exact '
            'pin is vacuous')

    def test_unit_sqrt_mu_breaks_node_exactness(self):
        """A unit-sqrt_mu carrier in the residual breaks node-exactness."""
        self.assertGreater(
            self._worst_node_error('unit_sqrt_mu'), _FALSIFY_REL_ERR,
            'unit sqrt_mu did not break node-exactness; the node-exact pin '
            'is vacuous')

    def test_zero_residual_breaks_node_exactness(self):
        """A zero residual (forgotten) breaks node-exact agreement."""
        self.assertGreater(
            self._worst_node_error('zero'), _FALSIFY_REL_ERR,
            'zero residual did not break node-exactness; the node-exact pin '
            'is vacuous')

    def test_oversized_residual_trips_ratio_cap(self):
        """The no-poles ratio cap (10) trips on a residual 100x too large.

        The cap is ``max|R| / max|F_serve| <= 10``; scaling the genuine
        residual by 100 makes ``max|100 R| / max|F| ~ 100 x 0.2 >> 10``, so
        the bound must fire -- proving it is a real bound, not a vacuous one
        that would pass any residual.
        """
        residual = np.array([_analytic_residual(float(w), self.source,
                                                self.gamma_prime)
                             for w in _W_GRID], dtype=complex)
        carrier = np.array([_lead_carrier(float(w), self.source,
                                          self.gamma_prime)
                            for w in _W_GRID], dtype=complex)
        f_serve = _reconstruct_f(_W_GRID, carrier, residual)
        ratio = float(np.abs(100.0 * residual).max()
                      / np.abs(f_serve).max())
        self.assertGreater(
            ratio, _RESIDUAL_RATIO_CAP,
            f'a 100x residual did not trip the ratio cap: {ratio:.3f} <= '
            f'{_RESIDUAL_RATIO_CAP}')

def _rot_minus_beta(beta: float) -> np.ndarray:
    """Eigenframe rotation ``R(-beta)`` (2x2) for the mass-sheet oracle."""
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    return np.array([[cos_b, sin_b], [-sin_b, cos_b]])


def _engine_reference_kappa(w: float, y, gamma: float, beta: float,
                            kappa: float) -> complex:
    """Exact engine amplitude at ``kappa >= 0`` via the mass-sheet map.

    Independent oracle for the end-to-end serve test: reuses the shipped
    ``operator._mass_sheet_map`` reduction plus ``f_schwinger`` (mirroring
    ``test_lensing_diffractive._engine_reference_kappa``).  The serve never
    calls the engine for its chart band, so this is a genuine second
    derivation.  At ``kappa = 0`` it collapses to the eigenframe form
    ``f_schwinger(w, R(-beta) y, gamma)`` exactly.
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
    ``|y'| = rho * |y_c(theta)|`` along ``theta``, rotated back by ``beta``,
    rescaled by ``sqrt(1 - kappa)``; ``gamma = gamma' * (1 - kappa)``.  The
    serve reconstructs ``gamma', rho, theta`` from this lens, so a fixture
    here is a node-exact round trip through the serve.
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


def _serve_farfield(chart: LowWShellChart, lens: dict,
                    dense_w: np.ndarray) -> tuple | None:
    """Return ``(envelope, host_calls)`` from the PRODUCTION serve.

    Calls the shipped ``LensedRelativeBinningLikelihood._low_w_shell_chart_serve``
    bound to a bare namespace, intercepting ``reconstruct_farfield`` (the
    shared, already-tested reconstruction tail) to capture its ``envelope``
    argument.  With ``geom.t_min = 0`` the frame-demodulation phase vanishes
    (``_frame_phase(w, 0) == 0``), so the captured envelope IS the
    re-modulated farfield -- the quantity under test.  The above-``w_shell``
    host ``self._engine_farfield_total`` is stubbed to the oracle
    (``_engine_reference_kappa``), so the resolved sub-band is never compared
    engine-vs-engine (the F002 tautology trap); the stub records its call
    arguments so the test can pin the split-mask wiring.  Returns ``None``
    when the serve declines.
    """
    captured: dict[str, np.ndarray] = {}
    host_calls: list[np.ndarray] = []

    def _capture(w, envelope, delays, saddle_kernels, real_mask, definition,
                 t_min):
        captured['envelope'] = np.asarray(envelope).copy()
        return (np.zeros((w.shape[0], 4), dtype=complex),
                np.zeros(w.shape[0], dtype=complex))

    def _host(lens_, sub_w):
        host_calls.append(np.asarray(sub_w, dtype=float).copy())
        return np.array(
            [_engine_reference_kappa(float(wi), (lens_['y1'], lens_['y2']),
                                     lens_['gamma'], lens_['beta'],
                                     lens_['kappa'])
             for wi in sub_w], dtype=complex)

    instance = types.SimpleNamespace(low_w_shell_chart=chart)
    instance._reduce_dense_kernels = lambda kernels: (np.zeros(1), np.zeros(1))
    instance._image_delays = lambda lens, geom: None
    instance._engine_farfield_total = _host
    geom = types.SimpleNamespace(
        t_min=0.0, delays=np.zeros(4), saddle_kernels=np.zeros((1, 4)),
        real_mask=np.array([True, True, False, False]))
    with mock.patch.object(_likelihood, 'reconstruct_farfield', _capture):
        result = LensedRelativeBinningLikelihood._low_w_shell_chart_serve(
            instance, lens, dense_w, geom)
    if result is None:
        return None
    return captured['envelope'], host_calls


def _flip_first_real_coeff(coeffs: np.ndarray) -> np.ndarray:
    """Flip one ``real_coeff`` (element ``[0, 0, 0, 0]``) in a copy."""
    flipped = np.array(coeffs, dtype=float)
    flipped[0, 0, 0, 0] += 1.0
    return flipped


def _save_chart_artifact(path: Path, chart: LowWShellChart,
                         schema: str = _SCHEMA,
                         content_hash: str | None = None,
                         drop_keys: tuple[str, ...] = ()) -> None:
    """Write a chart npz in the training script's save format.

    ``content_hash`` defaults to the correct `_content_hash` over the chart's
    stored fields -- the 4 grid axes + both coefficient arrays (the exact
    float64 bytes, matching ``scripts/train_low_w_shell_chart.py``);
    ``drop_keys`` removes named keys before writing to exercise the
    missing-key refusal paths.
    """
    arrays = {
        'gamma_prime_grid': chart.gamma_prime_grid,
        'rho_grid': chart.rho_grid,
        'theta_grid': chart.theta_grid,
        'log_w_grid': chart.log_w_grid,
        'real_coeffs': chart.real_coeffs,
        'imag_coeffs': chart.imag_coeffs,
        'provenance': np.array(json.dumps(chart.provenance)),
        'schema': np.array(schema),
    }
    if content_hash is None:
        content_hash = _content_hash(
            chart.gamma_prime_grid, chart.rho_grid, chart.theta_grid,
            chart.log_w_grid, chart.real_coeffs, chart.imag_coeffs)
    arrays['content_hash'] = np.array(content_hash)
    for key in drop_keys:
        arrays.pop(key, None)
    np.savez(path, **arrays)


def _tamper_artifact(path: Path, key: str, mutation) -> np.ndarray:
    """Rewrite ``key`` in an already-saved artifact, keeping ``content_hash``.

    Loads the arrays, applies ``mutation`` to the named array, and re-saves
    the whole artifact with the ORIGINAL content hash -- the exact "stale
    hash" corruption the loader must refuse.  Returns the mutated array so
    the caller can premise-assert the tamper was non-trivial.
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    mutated = mutation(np.asarray(arrays[key]))
    arrays[key] = mutated
    np.savez(path, **arrays)
    return mutated


def _round_trip_chart() -> LowWShellChart:
    """Small synthetic chart (seeded RNG coefficients) for load tests.

    Engine-free by construction: the load contract tests the npz format and
    the content hash, not the physics, so seeded RNG coefficients stand in
    for a trained chart.
    """
    shape = (4, 4, 6, 5)
    rng = np.random.default_rng(0)
    return LowWShellChart(
        gamma_prime_grid=np.array([0.3, 0.5, 0.8, 0.9]),
        rho_grid=np.array([0.6, 0.9, 1.2, 1.4]),
        theta_grid=np.linspace(0.0, math.pi / 2, 6),
        log_w_grid=np.log(np.array([0.05, 0.1, 0.15, 0.2, 0.3])),
        real_coeffs=rng.standard_normal(shape),
        imag_coeffs=rng.standard_normal(shape),
        provenance={'scale': 'test'})

class ShellServeNodeExactTestCase(TestCase):
    """End-to-end PRODUCTION serve matches the mass-sheet-map oracle.

    INS-2-001 port of the deleted diffractive suite's
    ``ServeEngineNodeExactTestCase``: the synthetic exact-residual chart is
    bound to a bare namespace and the SHIPPED
    ``LensedRelativeBinningLikelihood._low_w_shell_chart_serve`` is driven
    directly, with the ``reconstruct_farfield`` tail intercepted (``t_min =
    0``, so the captured envelope IS the re-modulated farfield).  The
    re-modulated ``F`` must match ``_engine_reference_kappa`` to node-exact
    tolerance at every grid node for BOTH ``kappa = 0`` and ``kappa != 0``.
    The ``kappa != 0`` case is mandatory: at ``kappa = 0`` the mass-sheet
    phase and ``1/lam`` collapse to identity, so a kappa = 0-only test cannot
    detect a serve that drops the gauge composition.

    The band splits at ``w_shell = 1 / delta_min`` (measured ``~0.22`` for
    the witness cell, interior to the chart's log-w window): the below-split
    nodes are the CHART composition ``mass_sheet_phase * (carrier + R) / lam``
    -- the node-exact accuracy pin (engine-free for the chart band, per the
    brief's "no f_schwinger on the serve path"); the above-split nodes are
    hosted through ``_engine_farfield_total`` (stubbed to the oracle, so no
    engine-vs-engine tautology) and the test pins the split-mask WIRING plus
    the no-step continuity, not the engine.  The serve declines (returns
    ``None``) for rho outside ``[RHO_LO, RHO_HI]`` via its own inline gates.
    """

    @classmethod
    def setUpClass(cls):
        cls.chart = _exact_residual_chart()
        cls.gamma_prime = 0.8
        cls.rho = 1.2
        cls.theta = float(cls.chart.theta_grid[2])  # pi/5, an interior node
        cls.dense_w = np.exp(cls.chart.log_w_grid)

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if not self.n_checks:
            self.fail('no serve comparison was made; the test asserted '
                      'nothing')

    def _expected_split(self):
        """Re-derive the below/above split from the shared primitives.

        The serve computes ``w_shell = 1/delta_min`` (``_reduced_min_delay_
        separation`` on the reduced matrix) and splits via
        ``_band_split_mask`` intersected with the chart's log-w window.  This
        is a structural WIRING prediction -- it pins that the serve's host
        receives exactly the resolved nodes -- not the value oracle.
        """
        source = reduced_source(self.gamma_prime, self.rho, self.theta)
        delta_min = _reduced_min_delay_separation(self.gamma_prime, source)
        w_shell = 1.0 / delta_min if delta_min > 0.0 else math.inf
        _active, below = _likelihood._band_split_mask(self.dense_w, w_shell)
        in_log_w = ((np.log(self.dense_w) >= self.chart.log_w_grid[0])
                    & (np.log(self.dense_w) <= self.chart.log_w_grid[-1]))
        return below & in_log_w

    def test_remodulated_serve_matches_engine_at_nodes(self):
        """|F_serve - F_engine| / |F_engine| <= 1e-10 at every grid node.

        The below-split nodes exercise the full chart composition in the
        kappa != 0 gauge; the above-split nodes are host-routed (stubbed to
        the oracle) and pinned to be bit-identical to the host -- the two
        sides agree with the SAME oracle, so the split is step-free.
        """
        below = self._expected_split()
        self.assertTrue(below.any(),
                        'premise lost: no chart-served nodes below w_shell')
        self.assertTrue((~below).any(),
                        'premise lost: no hosted nodes above w_shell')
        for kappa, beta in ((0.0, 0.0), (0.2, 0.3)):
            with self.subTest(kappa=kappa, beta=beta):
                lens = _make_lens(self.gamma_prime, self.rho, self.theta,
                                  kappa, beta)
                served = _serve_farfield(self.chart, lens, self.dense_w)
                self.assertIsNotNone(
                    served, 'serve declined the exact-chart witness; the '
                    'chart must cover it')
                envelope, host_calls = served
                self.assertEqual(len(host_calls), 1,
                                 'the above-split host must be called once')
                np.testing.assert_array_equal(
                    host_calls[0], self.dense_w[~below],
                    err_msg='the above-split host must receive exactly the '
                    'resolved (above w_shell) nodes -- split wiring drifted')
                y = (lens['y1'], lens['y2'])
                for i, w in enumerate(self.dense_w):
                    oracle = _engine_reference_kappa(
                        float(w), y, lens['gamma'], beta, kappa)
                    rel = abs(envelope[i] - oracle) / abs(oracle)
                    if below[i]:
                        self.n_checks += 1
                        self.assertLess(
                            rel, _NODE_EXACT_TOL,
                            f'chart-composed F_serve disagrees with the '
                            f'engine by {rel:.3e} at w={w:g} (kappa='
                            f'{kappa}, beta={beta}); the re-modulation or '
                            'gauge composition is inconsistent')
                    else:
                        self.n_checks += 1
                        self.assertLess(
                            rel, _NODE_EXACT_TOL,
                            f'host-routed F_serve disagrees with the oracle '
                            f'by {rel:.3e} at w={w:g} (kappa={kappa}, '
                            f'beta={beta}); the split handoff is not '
                            'step-free')

    def test_serve_declines_outside_shell_band(self):
        """rho outside [RHO_LO, RHO_HI] returns None (inline gate, not covers)."""
        for rho in (RHO_LO - 0.1, RHO_HI + 0.1):
            with self.subTest(rho=rho):
                lens = _make_lens(self.gamma_prime, rho, self.theta,
                                  0.0, 0.0)
                self.assertIsNone(
                    _serve_farfield(self.chart, lens, self.dense_w),
                    f'serve must decline rho={rho} outside the shell band')
                self.n_checks += 1

class ShellLoadContractTestCase(TestCase):
    """Schema + content-hash npz round-trip and hard-refusal contract.

    INS-2-001 port of the deleted diffractive suite's ``LoadContractTestCase``
    for ``LowWShellChart``.  A schema-tagged, content-hashed artifact
    round-trips bit-identically through ``LowWShellChart.load``; a
    missing/foreign schema, a missing ``content_hash``, and a tampered grid /
    coefficient value under a STALE hash hard-refuse with a ``ValueError``
    naming ``scripts/train_low_w_shell_chart.py``.  The positive control (a
    tampered value RE-hashed to consistency loads cleanly) pins the
    saver<->loader hash-field agreement (the INS-2-002 bug class); a tampered
    PROVENANCE (excluded from the content hash) still loads -- the hash
    covers the physics arrays only, provenance is audit metadata.
    """

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if not self.n_checks:
            self.fail('no load-contract check was made; the test asserted '
                      'nothing')

    def test_round_trip_is_bit_identical(self):
        """Every stored field survives a save -> load round-trip exactly."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart)
            loaded = LowWShellChart.load(path)
        for field in ('gamma_prime_grid', 'rho_grid', 'theta_grid',
                      'log_w_grid', 'real_coeffs', 'imag_coeffs'):
            np.testing.assert_array_equal(
                getattr(loaded, field), getattr(chart, field),
                err_msg=f'{field} did not round-trip bit-identically')
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
                LowWShellChart.load(path)
        self.assertIn('train_low_w_shell_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_foreign_schema_hard_refuses(self):
        """A foreign schema tag raises ValueError naming the script."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart, schema='foreign_schema_v0')
            with self.assertRaises(ValueError) as ctx:
                LowWShellChart.load(path)
        self.assertIn('train_low_w_shell_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_missing_content_hash_hard_refuses(self):
        """An artifact without a ``content_hash`` key raises ValueError naming the script."""
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart, drop_keys=('content_hash',))
            with self.assertRaises(ValueError) as ctx:
                LowWShellChart.load(path)
        self.assertIn('train_low_w_shell_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_tampered_coefficient_hard_refuses(self):
        """A flipped ``real_coeff`` under a STALE hash raises ValueError.

        Single canonical pin for the hash family: ``_content_hash`` treats
        all six fields uniformly (the 4 grid axes + ``real_coeffs`` +
        ``imag_coeffs``), so one flipped real coefficient is the tamper.  The
        mutation keeps the ORIGINAL stored hash -- the exact "stale hash"
        corruption the loader must detect.
        """
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart)
            mutated = _tamper_artifact(path, 'real_coeffs',
                                       _flip_first_real_coeff)
            self.assertFalse(
                np.array_equal(mutated, chart.real_coeffs),
                'premise lost: the tamper was a no-op')
            with self.assertRaises(ValueError) as ctx:
                LowWShellChart.load(path)
        self.assertIn('train_low_w_shell_chart.py', str(ctx.exception))
        self.n_checks += 1

    def test_rehashed_tamper_loads_cleanly(self):
        """A tampered coefficient with a FRESH hash round-trips (positive control).

        The same tamper as ``test_tampered_coefficient_hard_refuses``, but
        RE-hashed so the artifact is self-consistent again.  That pair is
        what shows the refusal there was the content hash's doing -- not a
        format or shape guard -- and pins the saver<->loader hash-field
        agreement (both must hash the identical six fields in the identical
        order, the INS-2-002 bug class).
        """
        chart = _round_trip_chart()
        tampered = dataclasses.replace(
            chart, real_coeffs=_flip_first_real_coeff(chart.real_coeffs))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, tampered)
            loaded = LowWShellChart.load(path)
        np.testing.assert_array_equal(loaded.real_coeffs,
                                      tampered.real_coeffs)
        self.n_checks += 1

    def test_tampered_provenance_still_loads(self):
        """A tampered ``provenance`` loads cleanly (excluded from the hash).

        The content hash covers the physics arrays ONLY -- provenance is
        audit metadata, deliberately outside the hash.  Tampering it while
        keeping the stored hash (which never covered it) must NOT refuse:
        this pins the hash's field set, the negative of the coefficient
        tamper.
        """
        chart = _round_trip_chart()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.npz'
            _save_chart_artifact(path, chart)
            mutated = _tamper_artifact(
                path, 'provenance',
                lambda a: np.array(json.dumps({'tampered': True})))
            self.assertNotEqual(mutated, chart.provenance,
                                'premise lost: the provenance tamper was a '
                                'no-op')
            loaded = LowWShellChart.load(path)
        self.assertEqual(loaded.provenance, {'tampered': True})
        self.n_checks += 1


if __name__ == '__main__':
    main()
