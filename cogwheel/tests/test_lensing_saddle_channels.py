"""
Tests for the macro-saddle extension of the topology-stable
Chang-Refsdal channel layer (`lensing.chang_refsdal.channels`).

Build 7b lifted the saddle guards: `ChangRefsdalChannels.evaluate` now
serves macro-saddle hosts ``0 < 1 - kappa < |gamma|`` through the SAME
parity-blind SACR-C construction it uses at positive parity.  On a
saddle host the channel layer DELEGATES every frequency node to the
batched operator call (`operator.F_op_grid`), whose saddle arm owns the
per-node geometric-vs-Schwinger routing internally; positive parity is
byte-identical to before.  This suite pins the seven properties that
make the saddle path trustworthy on the resolved two-image saddle of
the design note (``gamma' = 1.3``, ``kappa = 0``, ``y = (0.4, 0.3)``)
and its four-image interior neighbours.

WHY THE ORACLES ARE INDEPENDENT (FINDINGS F002)
-----------------------------------------------
Two accuracy gates are judged against oracles built from OUTSIDE the
channel/operator/Schwinger stack:

* ``_oracle_channel_total`` (test 5) reconstructs the exact saddle
  amplification from a PURE-mpmath 1D Schwinger-parameter integral
  (`_oracle_saddle_amplitude`, the representation copied verbatim from
  `test_lensing_schwinger._oracle_1d`) times the ``exp(-i w t_min)``
  carrier evaluated from `geometry.delay` alone.  It imports ONLY
  `geometry` and `mpmath` -- never `channels`, `operator`, `_schwinger`
  or the likelihood.
* ``_geometric_image_sum`` (test 6) is the leading stationary-phase
  image sum assembled from `geometry.find_images` / `delay` /
  `magnification` / `morse_index` alone.

`OracleIndependenceGuardTestCase` enforces the independence by AST
inspection of each oracle's own source (the committed
`test_lensing_gauge` / `test_lensing_channels` / `test_lensing_schwinger`
import-walking idiom), and proves the guard itself can go red.

CARRIER CONVENTION (the ``t_min`` shift)
----------------------------------------
`ChangRefsdalPartition.exact_total` carries ``exp(-i w t_min)`` relative
delays (``t_min`` the minimum absolute Fermat delay).  Every oracle here
reconstructs the same convention: the amplitude oracle is multiplied by
``exp(-i w t_min)`` (test 5), and the geometric image sum runs over the
minimum-relative delays ``tau_a - t_min`` (test 6).  Where a phase
intercept is extracted (test 7) the carrier is first undone by
``exp(+i w t_min)``.

A NODE-COUNT NOTE (test 2, reported not fixed)
----------------------------------------------
Over a full 2-decade window ``[0.5, 50]`` the strong-shear LOO stop
(`_LOO_STOP_STRONG = 1e-3`, keyed on ``gamma' >= 0.5``) converges the
SACR-C envelope grid at ``N ~ 40-42`` nodes for the saddle configs here
(below the `_LOO_MAX_NODES = 48` safety cap, so the adaptive stop is
doing its job, not saturating).  The measured true reconstruction error
is ``~1-4e-4``, an order inside the ``1e-3`` gate, so the stop is
conservative; the brief's aspirational ``N <= 30`` is met only over
<=1-decade windows.  The load-bearing gate here is the reconstruction
accuracy (`_RECON_INTERP_GATE`), with the node count asserted to
converge below the production cap.

Every ``<Thing>TestCase`` derives from `SaddleChannelsTestCase`, whose
`tearDown` FAILS a test that ran zero comparisons.  The AST guard, the
under-seeded envelope grid, and the raw divergent kernel are the
falsification partners proving the gates can go red.
"""
from __future__ import annotations

import ast
import cmath
import inspect
import math
import pathlib
import textwrap
from unittest import TestCase, main

import numpy as np
from scipy.interpolate import CubicSpline

import mpmath

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

from cogwheel.lensing.chang_refsdal import channels, geometry, _gauge
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, _loo_stop_for_lens,
    _LOO_MAX_NODES, _LOO_SEED_NODES)

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


# =====================================================================
# Configuration and gates (measured constants documented inline).
# =====================================================================

#: The design-note resolved two-image saddle acceptance config.
GAMMA_ACC = 1.3
Y_ACC = (0.4, 0.3)

#: Two-decade dimensionless-frequency window, ~40 nodes (tests 1/3).
W_LO, W_HI = 0.5, 50.0

#: The fixed topology-stable label count, declared locally so a channel
#: regression cannot silently redefine it.
N_CHANNELS = 4

#: Test 1: flat reconstruction gate on the SACR-C switched-analytic +
#: envelope decomposition (measured worst ~1.8e-16, so >1e2 headroom).
RECON_GATE = 1e-13

#: Test 2: the null-safe envelope interpolation gate (report build3f
#: gate 2/3).  Measured saddle worst ~4.4e-4, so ~2x headroom.
RECON_INTERP_GATE = 1e-3

#: Test 3: switch-suppressed analytic trial ceiling on a caustic
#: crossing (measured worst ~1.32).  The un-switched analytic kernel it
#: caps reaches ~1e18 near the fold, so the switch is load-bearing.
SWITCH_CROSSING_CEILING = 2.0
#: Ceiling on the MEASURED generic-scan max |S_a H_a| (research 2.4-2.8;
#: this sampling measures ~1.52 -- recorded in the assertion message).
SWITCH_SCAN_CEILING = 4.0
#: Floor the raw (un-switched) analytic kernel must clear on a crossing,
#: proving the switch suppresses a genuinely divergent target.
RAW_KERNEL_FLOOR = 1e3

#: Test 4: absolute ceiling on the reconstructed-total step between
#: adjacent lobe-jump path points at fixed w (measured ~1.2e-7).
LOBE_JUMP_CEILING = 1e-6

#: Test 5: relative gate against the independent mpmath oracle
#: (measured worst ~1.3e-14).
ORACLE_RTOL = 1e-9

#: Test 6: leading stationary-phase agreement gate (measured 1.2-1.9e-2).
GEOMETRIC_GATE = 5e-2

#: Test 7: the literal macro-magnification limit and its gates.  |F|
#: extrapolated intercept rel 1e-6 (measured 1.4e-7); Morse-phase
#: intercept -> 0 within 5e-4 (measured 1.5e-7); the w=1e-4 magnitude
#: residual <= 1e-3 (measured 4.4e-5).
MACRO_LIMIT_INTERCEPT_RTOL = 1e-6
MORSE_PHASE_TOL = 5e-4
DEEP_MAGNITUDE_RTOL = 1e-3
DEEP_WS = (1e-4, 1e-3, 1e-2)

#: mpmath oracle quadrature calibration (the `test_lensing_schwinger`
#: research-note scaling, reproduced so this suite's oracle is
#: self-contained).
_ORACLE_DPS_BASE = 30
_ORACLE_WAVELENGTHS_PER_PANEL = 8.0
_ORACLE_MAXDEGREE = 5
_ORACLE_EXTRA_MARGIN = 40.0
_ORACLE_MIN_PANELS = 12

#: Production engine names the independent oracles must NEVER reference
#: (F002: an oracle that reaches into the code under test cannot fail).
_ENGINE_FORBIDDEN = frozenset({
    'channels', 'operator', '_schwinger', 'likelihood', '_gauge',
    'ChangRefsdalChannels', 'ChangRefsdalPartition',
    'reconstruct_from_envelope', 'reconstructed_total', 'envelope_total',
    'switched_analytic_channels', 'channels_from_envelope',
    'f_schwinger', 'F_op', 'F_op_grid', 'geometric_amplification',
    'select_branch', '_channel_switch', '_exact_total', 'image_kernel'})

#: The stricter set the PURE-mpmath amplitude oracle is held to: no
#: engine, and no numpy/numba either (a float64 shortcut would defeat
#: the arbitrary-precision independence).
_STRICT_FORBIDDEN = _ENGINE_FORBIDDEN | frozenset({'numpy', 'np', 'numba'})


# ---------------------------------------------------------------------
# The AST import-walking idiom (copied from test_lensing_channels /
# test_lensing_schwinger) and the independent oracle path.
# ---------------------------------------------------------------------

def _referenced_names(func):
    """Return every name a function's own source references.

    The committed `test_lensing_gauge` import walk extended with
    ``ast.Name`` ids and ``ast.Attribute`` attribute names, so a
    forbidden dependency entering as ``operator.F_op`` or a bare
    ``f_schwinger`` name is caught, not only as an import statement.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or '').split('.')[0])
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    names.discard('')
    return names


def _oracle_saddle_amplitude(w, y1, y2, gamma_prime):
    """Pure-mpmath saddle amplification ``F_{0, gamma'}(w, y)``.

    The 1D Schwinger-parameter representation copied verbatim from
    `test_lensing_schwinger._oracle_1d` with ``a = 1 - gamma'``,
    ``b = 1 + gamma'`` (research note Sec. 6.1), regularized by one
    integration by parts and quadratured in ``u = ln t`` at
    ``dps = 30 + ceil(w)``.  Shares MATHEMATICS with production but zero
    CODE: no double-double arithmetic, no numpy, just arbitrary
    precision -- so agreement with the channel layer is a real check.
    """
    a = 1.0 - gamma_prime
    b = 1.0 + gamma_prime
    dps = _ORACLE_DPS_BASE + int(math.ceil(w))
    with mpmath.workdps(dps):
        w_ = mpmath.mpf(w)
        s = mpmath.mpc(0, w_ / 2)
        branch_a = mpmath.mpc(0, w_ * mpmath.mpf(a) / 2)
        branch_b = mpmath.mpc(0, w_ * mpmath.mpf(b) / 2)
        amp1 = (w_ * mpmath.mpf(y1)) ** 2 / 4
        amp2 = (w_ * mpmath.mpf(y2)) ** 2 / 4

        def kernel(t):
            da = t - branch_a
            db = t - branch_b
            return (mpmath.exp(-amp1 / da - amp2 / db)
                    / (mpmath.sqrt(da) * mpmath.sqrt(db)))

        def kernel_derivative(t):
            da = t - branch_a
            db = t - branch_b
            return kernel(t) * (amp1 / da ** 2 + amp2 / db ** 2
                                - 1 / (2 * da) - 1 / (2 * db))

        t_cap = w_ * (abs(mpmath.mpf(a)) + abs(mpmath.mpf(b)) + 2) / 2
        u_mid = mpmath.log(t_cap)
        margin = mpmath.pi * w_ / 4 + _ORACLE_EXTRA_MARGIN
        wavelength = 4 * mpmath.pi / w_
        n_panels = max(
            _ORACLE_MIN_PANELS,
            int(mpmath.ceil(margin / (_ORACLE_WAVELENGTHS_PER_PANEL
                                      * wavelength))))
        part_a = mpmath.quad(
            lambda u: (mpmath.exp((s + 1) * u)
                       * kernel_derivative(mpmath.exp(u))),
            mpmath.linspace(u_mid - margin, u_mid, n_panels + 1),
            maxdegree=_ORACLE_MAXDEGREE)
        tail = mpmath.quad(
            lambda u: mpmath.exp(s * u) * kernel(mpmath.exp(u)),
            mpmath.linspace(u_mid, u_mid + margin, n_panels + 1),
            maxdegree=_ORACLE_MAXDEGREE)
        raw = t_cap ** s * kernel(t_cap) / s - part_a / s + tail

        prefactor = mpmath.mpc(0, -w_ / 2)
        source_phase = mpmath.exp(
            1j * w_ * (mpmath.mpf(y1) ** 2 + mpmath.mpf(y2) ** 2) / 2)
        result = prefactor * source_phase * raw / mpmath.gamma(s)
    return result


def _saddle_t_min(gamma, y):
    """Minimum absolute Fermat delay from `geometry` alone.

    The ``exp(-i w t_min)`` carrier reference `ChangRefsdalPartition`
    subtracts.  Built from `geometry.macro_matrix` / `find_images` /
    `delay` -- never from the channel layer's own bookkeeping.
    """
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    source = np.asarray(y, dtype=float)
    images = geometry.find_images(source, matrix)
    return min(float(geometry.delay(image, source, matrix))
               for image in images)


def _oracle_channel_total(w, gamma, y):
    """`ChangRefsdalPartition.exact_total` an independent oracle.

    The pure-mpmath saddle amplitude times the ``exp(-i w t_min)``
    carrier reconstructed from `geometry`.  ``kappa = 0`` and
    ``beta = 0`` here, so the reduced shear is ``gamma`` and the
    eigenframe source is ``y`` unrotated.  Imports ONLY `geometry` and
    `mpmath` (plus numpy plumbing for the source vector); references
    nothing in the channel/operator/Schwinger stack.
    """
    t_min = _saddle_t_min(gamma, y)
    amplitude = _oracle_saddle_amplitude(w, y[0], y[1], gamma)
    return amplitude * mpmath.exp(mpmath.mpc(0, -float(w) * t_min))


def _geometric_image_sum(w, gamma, y):
    """Leading stationary-phase image sum from `geometry` alone.

    ``sum_a sqrt|mu_a| exp(i w (tau_a - t_min) - i pi/2 n_a)`` in the
    minimum-relative-delay convention of `exact_total`, assembled from
    `geometry.find_images` / `delay` / `magnification` / `morse_index`
    -- never `operator.geometric_amplification`.
    """
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    source = np.asarray(y, dtype=float)
    images = geometry.find_images(source, matrix)
    delays = np.array([geometry.delay(image, source, matrix)
                       for image in images])
    t_min = float(delays.min())
    total = 0j
    for image, tau in zip(images, delays):
        mu = geometry.magnification(image, matrix)
        n_a = geometry.morse_index(image, matrix)
        total += (math.sqrt(abs(mu))
                  * cmath.exp(1j * float(w) * (float(tau) - t_min)
                              - 0.5j * math.pi * n_a))
    return total


#: The oracle-path functions whose engine-independence the AST guard
#: pins, with the forbidden set each is held to.
_STRICT_ORACLES = (_oracle_saddle_amplitude,)
_ENGINE_ORACLES = (_saddle_t_min, _oracle_channel_total,
                   _geometric_image_sum)


def _save_figure(fig, name):
    """Save a diagnostic figure, swallowing any backend error."""
    if not _HAVE_MPL:
        return
    try:
        _OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(_OUTPUT_DIR / name, dpi=90, bbox_inches='tight')
    except Exception:  # pragma: no cover - environment dependent
        pass
    finally:
        plt.close(fig)


class SaddleChannelsTestCase(TestCase):
    """Base class: seeded rng plus the anti-vacuity comparison tally.

    `tearDown` FAILS a test whose sweep ran zero comparisons, so a
    silently-skipped gate cannot masquerade as green.
    """

    _expect_checks = True

    def setUp(self):
        self.rng = np.random.default_rng(20260720)
        self.n_checks = 0

    def tearDown(self):
        if self._expect_checks and self.n_checks == 0:
            self.fail('vacuous test: no comparison ran, so nothing was '
                      'asserted')


class OracleIndependenceGuardTestCase(SaddleChannelsTestCase):
    """The independent oracles must not touch the code under test (F002).

    A green suite is worth only its ability to fail; an oracle judged
    against the very stack it certifies could not.  These checks parse
    each oracle's own source and assert the channel/operator/Schwinger
    stack never appears in it, then prove the same checker flags a
    function that DOES reach into it.
    """

    def test_pure_mpmath_amplitude_references_no_float64_or_engine(self):
        """The arbitrary-precision amplitude oracle references neither
        the engine nor numpy/numba (a float64 shortcut)."""
        for func in _STRICT_ORACLES:
            with self.subTest(oracle=func.__name__):
                overlap = _referenced_names(func) & _STRICT_FORBIDDEN
                self.n_checks += 1
                self.assertFalse(
                    overlap,
                    f'{func.__name__} references forbidden names '
                    f'{sorted(overlap)}; the mpmath oracle is not a '
                    'pure arbitrary-precision independent check (F002)')

    def test_carrier_and_geometric_oracles_reference_no_engine(self):
        """The geometry-carrier and image-sum oracles reference nothing
        in the channel/operator/Schwinger stack."""
        for func in _ENGINE_ORACLES:
            with self.subTest(oracle=func.__name__):
                overlap = _referenced_names(func) & _ENGINE_FORBIDDEN
                self.n_checks += 1
                self.assertFalse(
                    overlap,
                    f'{func.__name__} reaches into the engine via '
                    f'{sorted(overlap)}; it must be built from geometry '
                    '/ numpy / mpmath alone (F002)')

    def test_guard_itself_can_go_red(self):
        """A function that DOES reach the engine is flagged by the same
        checker, so the guard above is not vacuous."""
        def _tainted_oracle(w):
            return operator.F_op(w)  # noqa: F821 - forbidden on purpose

        overlap = _referenced_names(_tainted_oracle) & _ENGINE_FORBIDDEN
        self.n_checks += 1
        self.assertTrue(
            overlap,
            'the AST guard failed to flag a function that references '
            'the engine (operator.F_op); the import guard cannot go red')


class ChannelIdentityResidualTestCase(SaddleChannelsTestCase):
    """The saddle SACR-C decomposition reconstructs the exact total.

    On the resolved two-image saddle, both faces of the partition -- the
    switched-analytic + single-envelope form
    ``F = sum_a S_a H_a e^{i w tau_a} + e^{i w tau_c} E`` and the
    per-image ``F = sum_a e^{i w tau_a} K_a`` form -- must reproduce
    `exact_total` to roundoff across the two-decade window.
    """

    def test_switched_analytic_plus_envelope_reconstructs_the_total(self):
        """Envelope reconstruction and per-image reconstruction both
        equal `exact_total` to <= 1e-13, over ~40 nodes in [0.5, 50]."""
        w = np.geomspace(W_LO, W_HI, 40)
        partition = channels.ChangRefsdalChannels(w).evaluate(
            gamma=GAMMA_ACC, y=np.array(Y_ACC))

        envelope_recon = _gauge.envelope_total(
            w, partition.delays, partition.saddle_kernels,
            partition.switch, partition.critical_delay, partition.envelope)
        kernel_recon = _gauge.reconstructed_total(
            w, partition.delays, partition.kernels)

        env_residual = np.abs(envelope_recon - partition.exact_total)
        ker_residual = np.abs(kernel_recon - partition.exact_total)
        self.n_checks += int(env_residual.size + ker_residual.size)

        self.assertLessEqual(
            float(np.max(env_residual)), RECON_GATE,
            f'SACR-C envelope reconstruction residual '
            f'{float(np.max(env_residual)):.3e} exceeds {RECON_GATE:g}')
        self.assertLessEqual(
            float(np.max(ker_residual)), RECON_GATE,
            f'per-image reconstruction residual '
            f'{float(np.max(ker_residual)):.3e} exceeds {RECON_GATE:g}')

        if _HAVE_MPL:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.semilogy(w, np.maximum(env_residual, 1e-18), 'o-',
                        label='|envelope recon - exact|')
            ax.semilogy(w, np.maximum(ker_residual, 1e-18), 's-',
                        label='|kernel recon - exact|')
            ax.axhline(RECON_GATE, color='crimson', ls='--',
                       label=f'{RECON_GATE:g} gate')
            ax.set_xlabel('w (dimensionless frequency)')
            ax.set_ylabel('reconstruction residual')
            ax.set_title("saddle SACR-C reconstruction residual "
                         "(gamma'=1.3, y=(0.4,0.3))")
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)
            _save_figure(fig, 'saddle_channel_identity_residual.png')


class EnvelopeNodeCountTestCase(SaddleChannelsTestCase):
    """The SACR-C envelope interpolates the saddle total within 1e-3.

    The OWED replacement for the retired coarse-node interpolation gate
    (`test_lensing_fast_path`, report build3f gate 2/3), on the saddle
    domain: the leave-one-out-adaptive envelope node set
    (`_envelope_loo_nodes`) reconstructs ``F`` on a dense grid to
    ``max_f |F_interp - F_dense| / max_f |F_dense| < 1e-3``, and the
    node count CONVERGES below the `_LOO_MAX_NODES` safety cap (the
    adaptive stop is doing its job, not saturating).

    The node grid is obtained through production's own
    `LensedRelativeBinningLikelihood._envelope_loo_nodes`; the method
    touches no instance state, so it is exercised through an
    uninitialised instance rather than a full (event-data) likelihood
    fixture -- the FAST path.  See the module docstring for the measured
    ``N ~ 40-42`` over two decades (the brief's aspirational ``<= 30``
    holds only over <=1-decade windows -- reported, not fixed).

    The expensive engine evaluations (the LOO refinement, the dense
    reference partition, and the seed partition per config) are done
    ONCE in `setUpClass` and shared by the accuracy and the
    self-falsification tests, so no saddle Schwinger node is evaluated
    twice.
    """

    #: Reference grid the interpolation error is sampled on.  A modest
    #: 64-point grid resolves the smooth envelope-interpolation error
    #: (measured worst ~2-4e-4, unchanged from a 400-point grid) while
    #: keeping every saddle Schwinger node count small.
    _DENSE_W = np.geomspace(W_LO, W_HI, 64)
    _CONFIGS = (
        ('two-image g=1.3', 1.3, (0.4, 0.3)),
        ('two-image g=1.2', 1.2, (0.1, -0.2)),
        ('four-image g=1.3', 1.3, (-1.31, 0.15)),
    )
    #: config label -> dict(n_nodes, loo_error, seed_error, stop).
    _CACHE: dict = {}

    @classmethod
    def setUpClass(cls):
        """Evaluate the LOO grid, dense reference, and seed grid once
        per config, caching the two reconstruction errors and the node
        count the tests assert on."""
        like = LensedRelativeBinningLikelihood.__new__(
            LensedRelativeBinningLikelihood)
        log_dense = np.log(cls._DENSE_W)
        cls._CACHE = {}
        for label, gamma, y in cls._CONFIGS:
            lens = dict(gamma=gamma, beta=0.0, kappa=0.0,
                        y1=y[0], y2=y[1])
            _, coarse_w, env_nodes = like._envelope_loo_nodes(
                lens, cls._DENSE_W)
            dense = channels.ChangRefsdalChannels(
                cls._DENSE_W).evaluate(gamma=gamma, y=np.array(y))
            seed_w = np.geomspace(W_LO, W_HI, _LOO_SEED_NODES)
            seed = channels.ChangRefsdalChannels(seed_w).evaluate(
                gamma=gamma, y=np.array(y))

            def recon_error(node_w, node_env, dense=dense):
                lw = np.log(node_w)
                envelope = (CubicSpline(lw, node_env.real)(log_dense)
                            + 1j * CubicSpline(lw, node_env.imag)(log_dense))
                _, total = channels.reconstruct_from_envelope(
                    cls._DENSE_W, envelope, dense.delays,
                    dense.saddle_kernels, dense.switch,
                    dense.critical_delay)
                scale = max(float(np.max(np.abs(dense.exact_total))), 1e-12)
                return float(np.max(np.abs(total - dense.exact_total))
                             / scale)

            cls._CACHE[label] = dict(
                n_nodes=int(coarse_w.size),
                loo_error=recon_error(coarse_w, env_nodes),
                seed_error=recon_error(seed_w, seed.envelope),
                stop=_loo_stop_for_lens(lens))

    def test_loo_envelope_reconstructs_within_gate_and_converges(self):
        """Every saddle config: N converges below the cap and the
        interpolated envelope reconstructs F within 1e-3."""
        for label, _, _ in self._CONFIGS:
            with self.subTest(config=label):
                cached = self._CACHE[label]
                n_nodes = cached['n_nodes']
                error = cached['loo_error']
                self.n_checks += 1
                self.assertLess(
                    n_nodes, _LOO_MAX_NODES,
                    f'{label}: LOO envelope grid saturated the '
                    f'{_LOO_MAX_NODES}-node cap (N={n_nodes}) rather than '
                    'converging on the stop; the envelope is not '
                    'interpolable at this resolution')
                self.n_checks += 1
                self.assertLess(
                    error, RECON_INTERP_GATE,
                    f'{label}: envelope reconstruction error {error:.3e} '
                    f'exceeds {RECON_INTERP_GATE:g} at N={n_nodes} nodes '
                    f'(stop={cached["stop"]:g})')

    def test_under_seeded_envelope_grid_breaches_the_gate(self):
        """The refinement is load-bearing: the bare `_LOO_SEED_NODES`
        seed grid (no refinement) reconstructs F far outside the 1e-3
        gate, so the gate above is not vacuously satisfiable."""
        for label, _, _ in self._CONFIGS:
            with self.subTest(config=label):
                error = self._CACHE[label]['seed_error']
                self.n_checks += 1
                self.assertGreater(
                    error, RECON_INTERP_GATE,
                    f'{label}: the {_LOO_SEED_NODES}-node seed grid '
                    f'already reconstructed F within {RECON_INTERP_GATE:g} '
                    f'(error {error:.3e}); the LOO refinement gate cannot '
                    'go red')


class SwitchWeightBoundTestCase(SaddleChannelsTestCase):
    """The criticality-separation switch keeps the analytic trial bounded.

    On a caustic crossing a near-critical image carries a divergent
    stationary-phase kernel ``H_a`` (measured up to ~1e18 near a fold),
    but its criticality separation ``|tau_a - tau_c| -> 0`` drives its
    switch ``S_a -> 0``, so the switched analytic trial ``S_a H_a`` the
    SACR-C decomposition carries stays O(1).  Away from any caustic the
    switch is on and the trial is the bounded physical kernel
    ``sqrt|mu_a|``.
    """

    _W = np.geomspace(W_LO, W_HI, 15)

    def _crossing_configs(self):
        """On-caustic RIGHT-lobe (center 0) fold and cusp crossings at
        eta = +-0.002, from `geometry.critical_point` with branch args.

        Provenance is the `test_lensing_saddle_geometry` two-lobe
        utilities: the right lobe is the ``theta - beta`` near-0 wedge;
        a generic lobe angle gives a FOLD (soft-axis merge direction),
        the wedge edge gives a CUSP (hard-axis direction).
        """
        tmax = 0.5 * np.arcsin((1.0 - 0.0) / abs(GAMMA_ACC))
        fold = geometry.critical_point(GAMMA_ACC, 0.18, 0.0, 0.0,
                                       branch=1)
        cusp = geometry.critical_point(GAMMA_ACC, tmax * 0.999, 0.0, 0.0,
                                       branch=1)
        configs = []
        for sign in (+1.0, -1.0):
            configs.append(
                (f'fold soft {sign:+.0f}',
                 fold.source + sign * 0.002 * fold.soft_axis))
            configs.append(
                (f'cusp hard {sign:+.0f}',
                 cusp.source + sign * 0.002 * cusp.hard_axis))
        return configs

    def test_switch_caps_the_divergent_trial_on_crossings(self):
        """max_a |S_a H_a| <= 2 on every crossing, while the un-switched
        |H_a| it caps is measured > 1e3 (the switch is load-bearing)."""
        max_switched = 0.0
        max_raw = 0.0
        for label, y in self._crossing_configs():
            partition = channels.ChangRefsdalChannels(self._W).evaluate(
                gamma=GAMMA_ACC, y=y)
            switched = np.abs(partition.switch * partition.saddle_kernels)
            raw = np.abs(partition.saddle_kernels)
            max_switched = max(max_switched, float(np.max(switched)))
            max_raw = max(max_raw, float(np.max(raw)))
            self.n_checks += 1
            self.assertLessEqual(
                float(np.max(switched)), SWITCH_CROSSING_CEILING,
                f'{label}: switched analytic trial |S_a H_a| reached '
                f'{float(np.max(switched)):.3e}, above the '
                f'{SWITCH_CROSSING_CEILING:g} crossing ceiling; the '
                'criticality switch is not suppressing the near-critical '
                'kernel')
        self.assertGreater(
            max_raw, RAW_KERNEL_FLOOR,
            f'the un-switched |H_a| peaked at only {max_raw:.3e} on the '
            f'crossings, below {RAW_KERNEL_FLOOR:g}: the switch had no '
            'divergent target to suppress, so the crossing gate is '
            'vacuous')

    def test_generic_scan_trial_stays_bounded(self):
        """A small seeded generic saddle scan measures max_a |S_a H_a|
        and asserts it below 4 (research 2.4-2.8; this sampling ~1.5)."""
        values = []
        attempts = 0
        while len(values) < 12 and attempts < 200:
            attempts += 1
            y = self.rng.uniform(-2.0, 2.0, size=2)
            try:
                partition = channels.ChangRefsdalChannels(
                    self._W).evaluate(gamma=GAMMA_ACC, y=y)
            except Exception:
                continue
            if partition.caustic_distance < 0.1:
                continue  # keep the scan on resolved, generic sources
            values.append(
                float(np.max(np.abs(partition.switch
                                    * partition.saddle_kernels))))
        self.assertGreater(len(values), 0,
                           'the generic scan found no resolved saddle '
                           'source; the sweep measured nothing')
        measured = max(values)
        self.n_checks += len(values)
        self.assertLess(
            measured, SWITCH_SCAN_CEILING,
            f'generic-scan max |S_a H_a| = {measured:.4f} (over '
            f'{len(values)} configs) reached the {SWITCH_SCAN_CEILING:g} '
            'ceiling; the analytic trial is no longer bounded away from '
            'the caustic')


class LobeJumpKernelContinuityTestCase(SaddleChannelsTestCase):
    """The reconstructed total is continuous across a lobe handoff.

    A macro saddle has two deltoid caustic lobes.  As a source between
    them crosses the symmetry axis the nearest lobe flips, so the parked
    critical carrier ``tau_c`` and the label bookkeeping jump.  The
    physical total is untouched by that internal handoff: walking a fine
    continued path across the crossover, the reconstructed total steps
    by no more than ~1e-6 between adjacent points at fixed w (measured
    ~1.2e-7), and reproduces `exact_total` at every point.
    """

    def test_reconstructed_total_has_no_step_across_the_lobe_jump(self):
        """~15 points across the inter-lobe crossover: the nearest lobe
        flips, but the reconstructed total steps by < 1e-6."""
        w = np.array([10.0, 13.0])
        y1_grid = np.linspace(-2e-3, 2e-3, 15)
        path = [dict(gamma=GAMMA_ACC, y=np.array([y1, 0.3]))
                for y1 in y1_grid]
        partitions = channels.ChangRefsdalChannels(w).evaluate_path(path)

        thetas = np.array([p.critical_theta for p in partitions])
        self.n_checks += 1
        self.assertTrue(
            float(thetas.min()) < 1.0 and float(thetas.max()) > 2.0,
            'the path never flipped the nearest lobe (critical theta '
            f'range [{thetas.min():.3f}, {thetas.max():.3f}]); it is not '
            'a lobe-jump crossing')

        reconstructed = np.array([p.reconstructed for p in partitions])
        exact = np.array([p.exact_total for p in partitions])
        recon_error = float(np.max(np.abs(reconstructed - exact)))
        self.n_checks += 1
        self.assertLessEqual(
            recon_error, RECON_GATE,
            f'the reconstruction departed from exact_total by '
            f'{recon_error:.3e} on the path; the continuity test would '
            'be reading a broken reconstruction')

        steps = np.abs(np.diff(reconstructed, axis=0))
        largest = float(np.max(steps))
        self.n_checks += int(steps.size)
        self.assertLessEqual(
            largest, LOBE_JUMP_CEILING,
            f'the reconstructed total jumped by {largest:.3e} between '
            f'adjacent points across the lobe handoff, above the '
            f'{LOBE_JUMP_CEILING:g} continuity ceiling')

        if _HAVE_MPL:
            magnitude = np.abs(reconstructed[:, 0])
            phase = np.angle(reconstructed[:, 0])
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.0),
                                           sharex=True)
            ax1.plot(y1_grid, magnitude, 'o-')
            ax1.set_ylabel('|F|')
            ax1.set_title(f'lobe-jump continuity (w={w[0]:g}, y2=0.3)')
            ax1.grid(True, alpha=0.3)
            ax2.plot(y1_grid, phase, 's-')
            ax2.set_ylabel('arg F [rad]')
            ax2.set_xlabel('y1 (source, crossing the symmetry axis)')
            ax2.grid(True, alpha=0.3)
            _save_figure(fig, 'saddle_lobe_jump_continuity.png')


class IndependentOracleTestCase(SaddleChannelsTestCase):
    """The channel total matches the independent mpmath oracle.

    At four frequencies on the acceptance config the channel-layer
    `exact_total` is compared to `_oracle_channel_total` -- the
    pure-mpmath 1D Schwinger amplitude times the ``exp(-i w t_min)``
    carrier from `geometry`, sharing no code with the channel/operator/
    Schwinger stack (the AST guard pins that).  Agreement to 1e-9
    (measured ~1e-14) means the delegation to the operator saddle arm
    reproduces the exact wave-optics amplification.
    """

    _WS = (5.0, 15.0, 30.0, 50.0)

    def test_channel_total_matches_mpmath_oracle(self):
        """|F_channel - F_oracle| / |F_oracle| < 1e-9 at w in
        {5, 15, 30, 50}."""
        w = np.array(self._WS)
        partition = channels.ChangRefsdalChannels(w).evaluate(
            gamma=GAMMA_ACC, y=np.array(Y_ACC))
        for index, frequency in enumerate(self._WS):
            with self.subTest(w=frequency):
                oracle = _oracle_channel_total(frequency, GAMMA_ACC, Y_ACC)
                got = mpmath.mpc(partition.exact_total[index])
                rel = float(abs(got - oracle) / abs(oracle))
                self.n_checks += 1
                self.assertLessEqual(
                    rel, ORACLE_RTOL,
                    f'w={frequency}: channel total departs from the '
                    f'independent mpmath oracle by rel {rel:.3e} '
                    f'(> {ORACLE_RTOL:g})')


class GeometricCrossCheckTestCase(SaddleChannelsTestCase):
    """The channel total approaches the stationary-phase image sum.

    At high frequency the exact total approaches the leading geometric
    image sum ``sum_a sqrt|mu_a| e^{i w (tau_a - t_min) - i pi/2 n_a}``
    (`_geometric_image_sum`, from `geometry` alone).  The residual is
    the ``O(1/w)`` Fresnel correction the channel kernels carry and the
    leading sum omits, so agreement improves with w.
    """

    _WS = (40.0, 50.0)

    def test_channel_total_approaches_geometric_sum(self):
        """Agreement < 5e-2 at w in {40, 50}, improving with w."""
        w = np.array(self._WS)
        partition = channels.ChangRefsdalChannels(w).evaluate(
            gamma=GAMMA_ACC, y=np.array(Y_ACC))
        relatives = []
        for index, frequency in enumerate(self._WS):
            geometric = _geometric_image_sum(frequency, GAMMA_ACC, Y_ACC)
            rel = float(abs(complex(partition.exact_total[index])
                            - geometric) / abs(geometric))
            relatives.append(rel)
            self.n_checks += 1
            self.assertLessEqual(
                rel, GEOMETRIC_GATE,
                f'w={frequency}: channel total departs from the '
                f'independent geometric image sum by rel {rel:.3e} '
                f'(> {GEOMETRIC_GATE:g})')
        self.n_checks += 1
        self.assertLess(
            relatives[1], relatives[0],
            f'geometric agreement did not improve with w '
            f'({relatives}); the leading stationary-phase limit is not '
            'being approached')


class MacroLimitTestCase(SaddleChannelsTestCase):
    """The saddle deep-band macro-magnification and Morse-phase limits.

    As ``w -> 0`` the exact total approaches the LITERAL macro
    magnification ``1 / sqrt(gamma^2 - 1)`` (the saddle F009-S limit,
    ``|mu_macro| = 1/|det A|``) with a linear-in-w correction, and its
    Morse phase approaches ``-pi/2`` (a single macro saddle).  Both
    limits are hard-coded closed forms built from the raw shear -- never
    rebuilt from the engine (F002).
    """

    def test_magnitude_extrapolates_to_the_literal_macro_limit(self):
        """|F| approaches the literal 1/sqrt(1.3^2 - 1): the w=1e-4
        residual is <= 1e-3, the residual shrinks monotonically toward
        small w, and the extrapolated intercept matches the literal to
        rel 1e-6."""
        literal = 1.0 / math.sqrt(GAMMA_ACC ** 2 - 1.0)
        w = np.array(DEEP_WS)
        partition = channels.ChangRefsdalChannels(w).evaluate(
            gamma=GAMMA_ACC, y=np.array(Y_ACC))
        magnitudes = np.abs(partition.exact_total)
        residuals = np.abs(magnitudes - literal) / literal

        self.n_checks += 1
        self.assertLessEqual(
            float(residuals[0]), DEEP_MAGNITUDE_RTOL,
            f'|F| at w={DEEP_WS[0]} misses the literal macro limit '
            f'1/sqrt(gamma^2-1)={literal:.6f}: rel {float(residuals[0]):.3e}')
        self.n_checks += 1
        self.assertLess(
            float(residuals[0]), float(residuals[1]),
            'the |F| residual does not shrink from w=1e-3 to w=1e-4; the '
            'flat macro limit is not being approached')
        self.n_checks += 1
        self.assertLess(
            float(residuals[1]), float(residuals[2]),
            'the |F| residual does not shrink from w=1e-2 to w=1e-3; the '
            'flat macro limit is not being approached')

        # Extrapolated intercept: |F| = m0 + a1 w ln(w/2) + a2 w (the
        # F009-S deep-band drift model), solved on the three decades.
        design = np.array([[1.0, wi * math.log(wi / 2.0), wi]
                           for wi in DEEP_WS])
        intercept = float(np.linalg.solve(design, magnitudes)[0])
        intercept_rel = abs(intercept - literal) / literal
        self.n_checks += 1
        self.assertLess(
            intercept_rel, MACRO_LIMIT_INTERCEPT_RTOL,
            f'the extrapolated |F| intercept {intercept:.8f} misses the '
            f'literal {literal:.8f} by rel {intercept_rel:.3e} '
            f'(> {MACRO_LIMIT_INTERCEPT_RTOL:g})')

    def test_morse_phase_intercept_is_minus_half_pi(self):
        """arg(F e^{+i pi/2}) with the t_min carrier undone extrapolates
        to 0 within 5e-4 -- the saddle Morse ``e^{-i pi/2}`` phase."""
        w = np.array(DEEP_WS)
        partition = channels.ChangRefsdalChannels(w).evaluate(
            gamma=GAMMA_ACC, y=np.array(Y_ACC))
        t_min = partition.t_min
        phases = []
        design = []
        for index, frequency in enumerate(DEEP_WS):
            # Undo the exp(-i w t_min) carrier, then test the Morse phase.
            amplitude = (partition.exact_total[index]
                         * np.exp(1j * frequency * t_min))
            phases.append(cmath.phase(amplitude
                                      * cmath.exp(0.5j * math.pi)))
            design.append([1.0, frequency * math.log(frequency / 2.0),
                           frequency])
        intercept = float(np.linalg.solve(np.array(design),
                                          np.array(phases))[0])
        self.n_checks += 1
        self.assertLess(
            abs(intercept), MORSE_PHASE_TOL,
            f'the Morse-phase intercept arg(F e^(i pi/2)) -> {intercept:.3e} '
            f'is not 0 within {MORSE_PHASE_TOL:g}; the saddle '
            'e^(-i pi/2) deep-band phase law is violated')


if __name__ == '__main__':
    main()
