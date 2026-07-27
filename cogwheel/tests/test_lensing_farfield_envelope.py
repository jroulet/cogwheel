"""Tests for the far-field envelope redefinition in ``channels``.

Build 8g-b redefined the exterior (far-field) surrogate label.  The OLD
label was ``ChangRefsdalPartition.envelope`` -- the caustic-region SACR-C
transition envelope, demodulated at the critical carrier ``tau_c`` whose
lobe is chosen by ``geometry.nearest_caustic_point``.  On the astroid
lobe-equidistance lines (the "diagonals") that carrier flips between two
nearly-equidistant lobes; a well-resolved exterior image then reads as
near-critical, its switch turns off, its full oscillation is left
un-subtracted, and the stored envelope jumps by ~1500x mid-tile.  No
spline can fit that discontinuity, so the surrogate chart it trains is
garbage on the good side of the flip.

The NEW label is ``channels.farfield_envelope_from_partition``:

    E_ff(w) = F(w) - sum_{a real} H_a(w) * exp(1j*w*tau_a),

the post-geometric-optics remainder built by forcing the SACR-C switch
to ``1`` on every real channel and parking the carrier at ``tau_c = 0``
-- so no caustic lobe is consulted at all.  This suite pins the three
properties the Architect specified.

WHY THESE TOLERANCES
--------------------
* Q6a continuity (`EnvelopeContinuityAcrossDiagonalTestCase`).  The OLD
  label's jump across the flip line is asserted only ``>= 100x`` even
  though it is measured at ~1500x: the gate is a REACHABLE-RED threshold
  a full order of magnitude below the data, not a knife-edge on the flip
  location.  The NEW label is gated at an ABSOLUTE ``5e-3`` on
  ``max|E_ff|`` (measured ~1.9e-4) and an adjacent-sample continuity of
  ``1e-3`` (measured ~5e-6); both are far-field magnitudes, ``|E_ff|`` is
  the ``~1e-4`` post-optics remainder, so an absolute bound is the right
  currency and relative tolerance would be meaningless at the interference
  nulls the sweep passes through.

* Q6c invariance (`LobeAssignmentInvarianceTestCase`).  E_ff must be
  independent of the lobe assignment to ``1e-12 * max|F|`` -- it is
  ``0`` by construction, because `farfield_envelope_from_partition` reads
  only ``{w, real_mask, exact_total, delays, saddle_kernels}`` and never
  the lobe-dependent ``{critical_delay, switch, envelope}``.  The
  companion teeth assert the OLD label genuinely MOVES (~0.2) under the
  same flip, so the invariance is a removed degree of freedom rather than
  a vacuous truth.

* Q2/Q3 reconstruction (`ReconstructionExactnessTestCase`).  Adding the
  real saddle carriers back to ``E_ff`` must return ``exact_total`` to
  ``1e-12 * max|F|`` across the full ``w``-band up to 60.  ``E_ff`` is
  ``exact_total`` minus those same carriers, so this is a CANCELLATION-
  SAFETY claim: the range-reduced ``_unit_carrier`` phases keep the
  subtract-then-add at machine precision (measured ``0``), and the
  ``~1e-4`` far-field remainder is not swamped.

INDEPENDENT ORACLE
------------------
The reconstruction oracle is ``ChangRefsdalPartition.exact_total`` -- the
engine's operator/Schwinger amplification total, which shares no code
with the SACR-C envelope projection under test.  The Q6a OLD-vs-NEW
comparison is intrinsic (both labels are computed side by side from the
same evaluated partition; no runtime flag selects one), and the Q6c
oracle is the invariance property itself.

SURROGATE-SIDE SPECS (WP1 + WP2)
--------------------------------
Three further classes drive real trained tiles through the production
surrogate (`FarFieldChart`, `LensAmplificationSurrogate`):

* `StraddlingTileTrainabilityTestCase` (Spec 1).  A far-field tile whose
  box straddles the astroid diagonal is fit under BOTH labels on an
  identical spline grid.  The NEW label's held-out F-normalized eps clears
  the production gate ``FARFIELD_EPS_GATE = 1e-3`` (measured ~1.6e-4); the
  OLD label -- the lobe-flip discontinuity inside the box -- is gated as a
  reachable-red foil above ``OLD_STRADDLING_EPS_MIN = 1.0`` (measured
  ~7.6e2, so the foil sits far below the data).  The eps is F-normalized
  by ``max|exact_total|`` because ``max|E_ff| ~ 1e-4`` is too small a
  denominator (this is exactly `surrogate_training._heldout_eps`'s rule).

* `ServingMirrorAcrossDiagonalTestCase` (Spec 2 / Q6b).  The served
  envelope is reconstructed to ``F`` through the likelihood far-field path
  (``switch = real_mask``, ``critical_delay = 0``) and compared to a fresh
  exact-engine ``F``.  The gate ``SERVE_MIRROR_TOL = 3e-3`` is Professor
  Q6b's ~3x headroom over the ``1e-3`` training gate (measured ~1.6e-4);
  the mirror error equals ``max|dE_ff| / max|F| = eps_ff`` because the
  real carriers cancel exactly on both sides.

* `DefinitionTagLoaderRefusalTestCase` (Spec 3 / F010).  The loader must
  hard-refuse a far-field artifact whose ``envelope_definition`` is absent
  or unknown (both the multi-chart and legacy single-box paths), with a
  ValueError naming the chart and instructing a rebuild, while a
  known-tagged chart loads and serves and a tube-only artifact is
  unaffected.  This is a boolean contract, not a tolerance.

The independent oracle throughout is the engine's ``exact_total`` (Spec 1
reference envelope and Spec 2 mirror target), which shares no code with the
spline emulator under test.  `NewGateSelfFalsificationTestCase` proves the
trainability, mirror and loader gates each go red under a corruption.

`FarfieldEnvelopeTestCase.tearDown` FAILS if a test's sweep ran zero
comparisons, and `SelfFalsificationTestCase` proves each gate can go red.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import inspect
import json
import pathlib
import subprocess
import sys
import tempfile
import types
from contextlib import ExitStack
from unittest import TestCase, main, mock

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

from cogwheel.lensing.chang_refsdal import channels, geometry, _gauge
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, ChangRefsdalPartition,
    farfield_envelope_from_partition, reconstruct_from_envelope)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal.operator import CancellationError
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)
from cogwheel.lensing.surrogate import (
    LensAmplificationSurrogate, FarFieldChart, TubeChart,
    _FARFIELD_ENVELOPE_DEFINITION, _KNOWN_FARFIELD_DEFINITIONS,
    _FARFIELD_AXIS_SCHEMA)
from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing import surrogate_training

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'

#: Measured diagonal configuration (Build 8g-b): the shear and first
#: source coordinate whose ``y2`` sweep crosses a lobe-equidistance line.
GAMMA_DIAGONAL = 0.0387
Y1_DIAGONAL = 1.3

#: Second-coordinate sweep, 33 nodes with step 0.0125 so it lands ON
#: ``y2 = 1.250`` and ``y2 = 1.275`` (the two theta_c flip lines) WITH
#: samples on both sides of each -- it straddles both flips.
Y2_SWEEP = np.linspace(1.10, 1.50, 33)

#: Frequencies at which the OLD/NEW labels are compared across the sweep.
CONTINUITY_W = np.array([5.0, 20.0, 60.0])

#: The OLD label's magnitude jump across the flip line, as a ratio of the
#: larger to the smaller adjacent sample.  Measured ~1492x; gated an order
#: of magnitude lower so the positive control is not perched on the flip.
OLD_JUMP_MIN_RATIO = 100.0

#: Absolute ceiling on ``max|E_ff|`` across the whole sweep (measured
#: ~1.9e-4); the far-field remainder is ``~1e-4``.
NEW_ENVELOPE_MAX = 5.0e-3

#: Absolute ceiling on the largest adjacent-sample change of ``|E_ff|``
#: over the sweep (measured ~5e-6): the fix is smooth, a spline can fit it.
NEW_CONTINUITY_MAX = 1.0e-3

#: Full ``w``-band for the reconstruction / cancellation-safety test,
#: including ``w`` up to 60 (the exact-Schwinger ceiling).
RECON_BAND = np.linspace(1.0, 60.0, 160)

#: Machine-precision reconstruction / invariance gate, F-normalized by
#: ``max|F| = max|exact_total|`` (measured ``0`` for both).
MACHINE_REL_TOL = 1.0e-12

#: Minimum OLD-label movement under a lobe flip that certifies the flip is
#: real (the teeth for the invariance test); measured ~0.2.
OLD_LOBE_SENSITIVITY_MIN = 1.0e-2

#: An exterior on-diagonal point for the invariance and reconstruction
#: fixtures (two real images, well outside the caustic).
DIAGONAL_EXTERIOR = (1.3, 1.3)

# --------------------------------------------------------------------------
# Build 8g-b / WP1+WP2 surrogate-side specs (trainability, serving mirror,
# definition-tag loader refusal).  The three classes below train real
# far-field tiles from the engine and drive them through the production
# surrogate machinery (`FarFieldChart`, `LensAmplificationSurrogate`).
# --------------------------------------------------------------------------

#: Engine refusals to skip while sampling a training grid or held-out set.
_ENGINE_REFUSALS = (LensDomainError, CancellationError,
                    SchwingerCertificationError)

#: Shear band and half-width of the synthetic far-field training tiles.
TILE_GAMMA_BAND = (0.02, 0.06)
TILE_HALF = 0.03

#: Far-field ``w``-band and grid resolution for a trained tile.  Fourteen
#: log-``w`` nodes span ``[5, 60]`` (the exact-Schwinger ceiling); a
#: ``4 x 4 x 7`` source/shear grid resolves the smooth remainder.
TILE_W_RANGE = (5.0, 60.0)
TILE_N_GAMMA, TILE_N_Y1, TILE_N_Y2, TILE_N_W = 4, 4, 7, 14

#: A tile whose ``y2`` box ``[1.23, 1.29]`` STRADDLES the astroid diagonal
#: flip line ``y2 = 1.25`` (the case the OLD label cannot fit).
STRADDLING_TILE_CENTER = (1.30, 1.26)

#: A tile whose ``y2`` box ``[1.42, 1.48]`` sits away from any flip line
#: (the on-axis control for the trainability histogram).
ON_AXIS_TILE_CENTER = (1.30, 1.45)

#: The astroid diagonal flip line the straddling tile crosses; held-out
#: points on both sides of it certify the "across the diagonal" claim.
DIAGONAL_FLIP_Y2 = 1.25

#: Production far-field gate (``surrogate_training.farfield_eps_max``): the
#: held-out F-normalized envelope error a chart must clear to be served.
#: The NEW label clears it on BOTH tiles (measured ~1.6e-4 straddling,
#: ~1.4e-4 on-axis).
FARFIELD_EPS_GATE = 1.0e-3

#: Reachable-red floor the OLD label's held-out eps must EXCEED on the
#: straddling tile (measured ~7.6e2 -- the lobe-flip discontinuity makes
#: the tile unfittable).  Gated far below the data so the foil is not
#: perched on a knife edge.
OLD_STRADDLING_EPS_MIN = 1.0

#: Minimum ratio by which the NEW label beats the OLD on the straddling
#: tile (measured ~4.7e6); ties the two labels in one comparison.
NEW_OVER_OLD_RATIO_MIN = 1.0e3

#: Q6b serving-mirror tolerance: ``max_w|F_serve - F_engine| / max_w|F|``
#: reconstructed through the likelihood far-field path.  ~3x headroom over
#: the ``1e-3`` gate (measured ~1.6e-4).
SERVE_MIRROR_TOL = 3.0e-3

#: Interior ``w``-grid used for held-out evaluation; nudged just inside the
#: band so ``log(exp(log_w_grid))`` round-off cannot push an endpoint
#: outside the chart's certified ``ln w`` box (the serve() band guard is
#: strict, `_log_w_band_inside`).
_W_EVAL = np.geomspace(TILE_W_RANGE[0] * 1.003, TILE_W_RANGE[1] * 0.997, 40)


def _held_out_samples(center: tuple[float, float], count: int,
                      seed: int) -> list[tuple[float, float, float]]:
    """Random ``(gamma, y1, y2)`` inside the inner 80% of a tile box."""
    rng = np.random.default_rng(seed)
    y1_lo, y1_hi = center[0] - TILE_HALF * 0.8, center[0] + TILE_HALF * 0.8
    y2_lo, y2_hi = center[1] - TILE_HALF * 0.8, center[1] + TILE_HALF * 0.8
    return [(float(rng.uniform(*TILE_GAMMA_BAND)),
             float(rng.uniform(y1_lo, y1_hi)),
             float(rng.uniform(y2_lo, y2_hi)))
            for _ in range(count)]


def _box_to_caustic_fixed(y1_range: tuple[float, float],
                          y2_range: tuple[float, float], n1: int, n2: int,
                          gamma_range: tuple[float, float] = TILE_GAMMA_BAND
                          ) -> tuple[np.ndarray, np.ndarray]:
    """``(rho_grid, theta_c_grid)`` for a rectangular eigenframe box.

    Maps every corner of the ``gamma_range x y1_range x y2_range`` box
    through `_to_caustic_fixed` (EACH corner's own ``gamma``, i.e. its own
    ``_caustic_reach``) and returns the enclosing ``n1``/``n2``-node axes.

    A per-corner hull -- rather than a single fixed reference reach -- is
    safe HERE specifically: every tile in this file sits deep in the far
    field (``rho`` from ~5 to ~45 measured across the full
    ``TILE_GAMMA_BAND = (0.02, 0.06)``, since ``|y| ~ 0.2-1.7`` while
    ``reach`` is only ``~0.04-0.12`` even at the band edges), so the union
    over gamma never approaches the ``rho ~ 1`` caustic boundary the way
    it does for the wider, closer-to-unity-rho boxes in
    `test_lensing_surrogate.py` (whose `_train` docstring records that
    exact failure mode) -- a per-corner hull is the more literal
    "same physical box" conversion and is used here because it is safe.
    A held-out query anywhere in the box, evaluated at ITS OWN gamma via
    `LensAmplificationSurrogate.serve`, therefore always falls inside the
    trained ``rho_grid``/``theta_c_grid`` bounds.
    """
    rhos, theta_cs = [], []
    for gamma in np.linspace(*gamma_range, 5):
        for y1 in y1_range:
            for y2 in y2_range:
                rho, theta_c = surrogate_module._to_caustic_fixed(
                    float(gamma), y1, y2)
                rhos.append(rho)
                theta_cs.append(theta_c)
    return (np.linspace(min(rhos), max(rhos), n1),
            np.linspace(min(theta_cs), max(theta_cs), n2))


@functools.lru_cache(maxsize=None)
def _train_tile(center: tuple[float, float], label: str) -> FarFieldChart:
    """Fit a `FarFieldChart` to a fixed engine grid under ``label``.

    ``label='new'`` fits the redefined far-field remainder
    ``farfield_envelope_from_partition`` (what the production trainer uses);
    ``label='old'`` fits the legacy ``partition.envelope`` on the SAME axes,
    so the two are compared under an identical spline fit -- the only
    difference is the label being interpolated.  Points the engine refuses
    (or that return a non-finite envelope) are recorded as refused.

    The chart's spatial axes are the caustic-fixed ``(rho, theta_c)``
    (Build 8h-b3): the physical box is UNCHANGED (``center +/- TILE_HALF``
    in eigenframe ``(y1, y2)``), only the coordinate the label is fitted
    over changes -- see `_box_to_caustic_fixed`.
    """
    gamma_grid = np.linspace(*TILE_GAMMA_BAND, TILE_N_GAMMA)
    rho_grid, theta_c_grid = _box_to_caustic_fixed(
        (center[0] - TILE_HALF, center[0] + TILE_HALF),
        (center[1] - TILE_HALF, center[1] + TILE_HALF),
        TILE_N_Y1, TILE_N_Y2)
    log_w_grid = np.linspace(np.log(TILE_W_RANGE[0]), np.log(TILE_W_RANGE[1]),
                             TILE_N_W)
    w_grid = np.exp(log_w_grid)
    shape = (TILE_N_W, TILE_N_GAMMA, TILE_N_Y1, TILE_N_Y2)
    envelope_real = np.zeros(shape)
    envelope_imag = np.zeros(shape)
    refused: list[tuple[float, float, float]] = []
    for ig, gamma in enumerate(gamma_grid):
        for i1, rho in enumerate(rho_grid):
            for i2, theta_c in enumerate(theta_c_grid):
                y1, y2 = surrogate_module._from_caustic_fixed(
                    float(gamma), float(rho), float(theta_c))
                engine = ChangRefsdalChannels(w_grid)
                engine.reset()
                try:
                    partition = engine.evaluate(
                        gamma=float(gamma), y=(float(y1), float(y2)),
                        beta=0.0, kappa=0.0)
                except _ENGINE_REFUSALS:
                    refused.append((float(gamma), float(rho), float(theta_c)))
                    continue
                envelope = (farfield_envelope_from_partition(partition)
                            if label == 'new'
                            else np.asarray(partition.envelope))
                if not np.all(np.isfinite(envelope)):
                    refused.append((float(gamma), float(rho), float(theta_c)))
                    continue
                envelope_real[:, ig, i1, i2] = envelope.real
                envelope_imag[:, ig, i1, i2] = envelope.imag
    refused_points = (np.array(refused) if refused
                      else np.empty((0, 3), dtype=float))
    return FarFieldChart.from_values(
        gamma_grid=gamma_grid, rho_grid=rho_grid, theta_c_grid=theta_c_grid,
        log_w_grid=log_w_grid, envelope_real=envelope_real,
        envelope_imag=envelope_imag, image_count=2, parity=1,
        refused_points=refused_points)


@functools.lru_cache(maxsize=None)
def _held_out_eps_list(center: tuple[float, float], label: str
                       ) -> tuple[float, ...]:
    """Per-sample held-out F-normalized envelope error for a trained tile.

    Serves each held-out geometry point through a one-chart surrogate and
    compares to a fresh engine reference under the chart's own label -- the
    NEW label F-normalized by ``max|exact_total|`` (``max|E_ff| ~ 1e-4`` is
    too small a denominator), the OLD label by ``max|E|``, exactly as
    `surrogate_training._heldout_eps` does.
    """
    chart = _train_tile(center, label)
    surrogate = LensAmplificationSurrogate([chart], {})
    errors: list[float] = []
    for gamma, y1, y2 in _held_out_samples(center, 50, seed=1):
        engine = ChangRefsdalChannels(_W_EVAL)
        engine.reset()
        try:
            partition = engine.evaluate(
                gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        except _ENGINE_REFUSALS:
            continue
        if label == 'new':
            reference = farfield_envelope_from_partition(partition)
            denom = float(np.max(np.abs(partition.exact_total))) or 1.0
        else:
            reference = np.asarray(partition.envelope)
            denom = float(np.max(np.abs(reference))) or 1.0
        if not np.all(np.isfinite(reference)):
            continue
        emulated, served, _definition = surrogate.serve(
            _W_EVAL, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=partition.caustic_distance, theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        if not served:
            continue
        errors.append(float(np.max(np.abs(emulated - reference)) / denom))
    return tuple(errors)


@functools.lru_cache(maxsize=None)
def _partition(w_key: tuple[float, ...], gamma: float, y1: float,
               y2: float) -> ChangRefsdalPartition:
    """Evaluate one fresh, reset partition (cached by its arguments).

    Uses `ChangRefsdalChannels.reset` so the labeling is the deterministic
    far-proposal labeling -- the total is label-invariant, so no
    continuation state leaks between the independent fixtures here.
    """
    engine = ChangRefsdalChannels(np.asarray(w_key, dtype=float))
    engine.reset()
    return engine.evaluate(gamma=gamma, y=(y1, y2))


def _far_field_switch(partition: ChangRefsdalPartition) -> np.ndarray:
    """Dense ``(n_w, 4)`` switch: ``1`` on real channels, ``0`` elsewhere.

    The exact switch `farfield_envelope_from_partition` uses internally
    and the one the likelihood's dense reconstruction passes to
    `reconstruct_from_envelope` (``real_mask`` broadcast to the band).
    """
    switch = np.zeros((partition.w.shape[0], channels._N_CHANNELS),
                      dtype=float)
    switch[:, np.asarray(partition.real_mask, dtype=bool)] = 1.0
    return switch


def _two_nearest_lobes(gamma: float, source: np.ndarray, beta: float,
                       kappa: float) -> list[float]:
    """Polar angles of the two nearest caustic lobes, by a coarse scan.

    Independent of `channels`: locates local minima of the source-to-
    caustic distance ``|caustic(theta) - source|`` over a uniform angular
    grid, using only `geometry.critical_point`.  Returns the angles of the
    two nearest local minima, nearest first.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
    distance = np.array([
        np.hypot(*(np.asarray(geometry.critical_point(
            gamma, theta, beta, kappa)[1]) - source))
        for theta in thetas])
    minima = [
        (distance[i], thetas[i])
        for i in range(len(thetas))
        if distance[i] < distance[i - 1]
        and distance[i] < distance[(i + 1) % len(thetas)]]
    minima.sort()
    return [theta for _, theta in minima[:2]]


def _lobe_flipped_partition(partition: ChangRefsdalPartition
                            ) -> ChangRefsdalPartition:
    """Same real geometry, critical carrier forced to the OTHER lobe.

    Recomputes the lobe-dependent ``critical_delay``, ``switch`` and
    ``envelope`` at the second-nearest caustic lobe (a genuinely different
    ``tau_c``), leaving ``exact_total``, ``delays``, ``saddle_kernels`` and
    ``real_mask`` -- everything `farfield_envelope_from_partition` reads --
    untouched.  The OLD ``envelope`` field therefore genuinely moves while
    E_ff must not.
    """
    source = np.asarray(partition.source, dtype=float)
    angles = _two_nearest_lobes(partition.gamma, source, partition.beta,
                                partition.kappa)
    candidates = []
    for theta in angles:
        image = np.asarray(geometry.critical_point(
            partition.gamma, theta, partition.beta, partition.kappa)[0])
        tau_c = geometry.delay(image, source, partition.matrix) \
            - partition.t_min
        candidates.append(tau_c)
    # The OTHER lobe = the candidate whose carrier delay differs most from
    # the one the partition actually chose.
    other = max(candidates,
                key=lambda tau_c: abs(tau_c - partition.critical_delay))
    switch = channels._channel_switch(
        partition.w, partition.delays, partition.real_mask, other)
    weights = channels._envelope_weights(switch)
    _, envelope = _gauge.switched_analytic_channels(
        partition.w, partition.exact_total, partition.delays,
        partition.saddle_kernels, switch, other, weights)
    return dataclasses.replace(
        partition, critical_delay=float(other), switch=switch,
        envelope=envelope)


class FarfieldEnvelopeTestCase(TestCase):
    """Base carrying the F-normalized assertion and anti-vacuity guard."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        # A sweep that skipped every comparison asserts nothing; fail loudly
        # rather than read green.
        self.assertGreater(
            self.comparisons, 0,
            'no comparisons were made -- the test asserted nothing')

    def assert_within(self, value: float, tol: float, message: str) -> None:
        """Assert ``value <= tol`` and count the comparison."""
        self.comparisons += 1
        self.assertLessEqual(value, tol, message)


class EnvelopeContinuityAcrossDiagonalTestCase(FarfieldEnvelopeTestCase):
    """Q6a: the OLD label jumps across the flip; the NEW label is smooth.

    Sweeps ``y2`` across the two lobe-equidistance lines at ``y1 = 1.3``,
    ``gamma = 0.0387`` and, at each node, computes BOTH labels from the
    SAME evaluated partition (no runtime flag): the OLD
    ``partition.envelope`` and the NEW
    `farfield_envelope_from_partition`.  The OLD magnitude jumps by
    ``>= 100x`` across the flip line, while the NEW label stays bounded
    (``<= 5e-3``) and adjacent-continuous (``<= 1e-3``).
    """

    @classmethod
    def setUpClass(cls) -> None:
        w_key = tuple(float(v) for v in CONTINUITY_W)
        old = []
        new = []
        for y2 in Y2_SWEEP:
            partition = _partition(w_key, GAMMA_DIAGONAL, Y1_DIAGONAL,
                                   float(y2))
            old.append(np.abs(partition.envelope))
            new.append(np.abs(farfield_envelope_from_partition(partition)))
        cls.old = np.array(old)          # (n_y2, n_w)
        cls.new = np.array(new)          # (n_y2, n_w)
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(CONTINUITY_W), figsize=(13, 4),
                                 sharex=True)
        for col, (ax, w) in enumerate(zip(axes, CONTINUITY_W)):
            ax.semilogy(Y2_SWEEP, cls.old[:, col], 'r.-',
                        label='|E_old| (partition.envelope)')
            ax.semilogy(Y2_SWEEP, cls.new[:, col], 'b.-',
                        label='|E_ff| (far-field)')
            for flip in (1.250, 1.275):
                ax.axvline(flip, color='0.6', ls=':', lw=0.8)
            ax.set_title(f'w = {w:g}')
            ax.set_xlabel('y2')
        axes[0].set_ylabel('envelope magnitude')
        axes[0].legend(fontsize=7)
        fig.suptitle('Q6a: old label spikes at the flip, far-field is flat')
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'farfield_envelope_continuity_across_diagonal.png',
                    dpi=110)
        plt.close(fig)

    def test_old_label_jumps_across_the_flip_line(self):
        """The OLD envelope's largest adjacent ratio exceeds 100x."""
        for col, w in enumerate(CONTINUITY_W):
            column = self.old[:, col]
            adjacent = np.maximum(column[:-1], column[1:])
            floor = np.minimum(column[:-1], column[1:])
            # Guard the ratio against exact-zero minima (the flip drives the
            # OLD label to ~1e-16); a floor of 1e-12 keeps the ratio finite
            # and still lets a genuine >=100x jump register.
            ratio = adjacent / np.maximum(floor, 1e-12)
            self.comparisons += 1
            self.assertGreaterEqual(
                float(ratio.max()), OLD_JUMP_MIN_RATIO,
                f'old label did not jump at w={w:g}: '
                f'max adjacent ratio {ratio.max():.1f}')

    def test_far_field_label_stays_below_the_absolute_ceiling(self):
        """``max|E_ff|`` over the whole sweep is at far-field magnitude."""
        self.assert_within(
            float(self.new.max()), NEW_ENVELOPE_MAX,
            f'far-field label exceeded {NEW_ENVELOPE_MAX:g}: '
            f'max|E_ff| = {self.new.max():.3e}')

    def test_far_field_label_is_adjacent_continuous(self):
        """Adjacent ``|E_ff|`` samples never jump by more than 1e-3."""
        adjacent = np.max(np.abs(np.diff(self.new, axis=0)))
        self.assert_within(
            float(adjacent), NEW_CONTINUITY_MAX,
            f'far-field label was discontinuous: max adjacent change '
            f'{adjacent:.3e}')


class LobeAssignmentInvarianceTestCase(FarfieldEnvelopeTestCase):
    """Q6c: E_ff is invariant under a lobe flip; the OLD label is not.

    At the on-diagonal exterior worst case, builds the partition and a
    lobe-flipped counterpart whose critical carrier is forced to the
    other (second-nearest) caustic lobe.  ``E_ff`` must agree to
    ``1e-12 * max|F|`` -- the removed degree of freedom -- while the OLD
    ``envelope`` field genuinely moves, proving the invariance is not
    vacuous.
    """

    @classmethod
    def setUpClass(cls) -> None:
        w_key = tuple(float(v) for v in CONTINUITY_W)
        cls.partition = _partition(
            w_key, GAMMA_DIAGONAL, *DIAGONAL_EXTERIOR)
        cls.flipped = _lobe_flipped_partition(cls.partition)
        cls.f_scale = float(np.max(np.abs(cls.partition.exact_total)))
        cls.e_reference = farfield_envelope_from_partition(cls.partition)
        cls.e_flipped = farfield_envelope_from_partition(cls.flipped)

    def test_far_field_label_is_invariant_under_lobe_flip(self):
        """E_ff agrees between the two lobe assignments to machine zero."""
        deviation = float(np.max(np.abs(self.e_reference - self.e_flipped)))
        self.assert_within(
            deviation / self.f_scale, MACHINE_REL_TOL,
            f'far-field label depended on the lobe: relative deviation '
            f'{deviation / self.f_scale:.3e}')

    def test_the_flip_genuinely_moves_the_old_label(self):
        """The OLD envelope moves under the same flip (the teeth)."""
        movement = float(np.max(np.abs(
            self.partition.envelope - self.flipped.envelope)))
        self.comparisons += 1
        self.assertGreaterEqual(
            movement, OLD_LOBE_SENSITIVITY_MIN,
            f'the lobe flip did not move the OLD label ({movement:.3e}); '
            f'the invariance test would be vacuous')

    def test_the_two_lobes_are_genuinely_distinct(self):
        """The forced carrier delay differs from the chosen one."""
        gap = abs(self.flipped.critical_delay
                  - self.partition.critical_delay)
        self.comparisons += 1
        self.assertGreater(
            gap, 1e-3,
            f'the "other" lobe carried the same delay ({gap:.3e}); the '
            f'flip did nothing')


class ReconstructionExactnessTestCase(FarfieldEnvelopeTestCase):
    """Q2/Q3: adding the real carriers back to E_ff returns exact_total.

    Forms ``E_ff`` on the full ``w``-band (up to 60) and reconstructs
    ``F`` two ways -- through the public `reconstruct_from_envelope` and
    through `_gauge.envelope_total`, both with ``switch = real_mask`` and
    ``critical_delay = 0`` -- comparing to the untouched engine oracle
    ``partition.exact_total``.  The range-reduced carriers keep the
    subtract-then-add at the ``1e-12`` floor.
    """

    @classmethod
    def setUpClass(cls) -> None:
        w_key = tuple(float(v) for v in RECON_BAND)
        cls.partition = _partition(
            w_key, GAMMA_DIAGONAL, *DIAGONAL_EXTERIOR)
        cls.envelope = farfield_envelope_from_partition(cls.partition)
        cls.switch = _far_field_switch(cls.partition)
        cls.f_scale = float(np.max(np.abs(cls.partition.exact_total)))
        cls._plot()

    @classmethod
    def _reconstruct_public(cls) -> np.ndarray:
        _kernels, total = reconstruct_from_envelope(
            cls.partition.w, cls.envelope, cls.partition.delays,
            cls.partition.saddle_kernels, cls.switch, 0.0)
        return total

    @classmethod
    def _reconstruct_gauge(cls) -> np.ndarray:
        return _gauge.envelope_total(
            cls.partition.w, cls.partition.delays,
            cls.partition.saddle_kernels, cls.switch, 0.0, cls.envelope)

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        error = np.abs(cls._reconstruct_public()
                       - cls.partition.exact_total)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(cls.partition.w, np.maximum(error, 1e-18), 'b.-',
                    label='|F_recon - F_exact|')
        ax.axhline(MACHINE_REL_TOL * cls.f_scale, color='r', ls='--',
                   label='1e-12 * max|F|')
        ax.set_xlabel('w')
        ax.set_ylabel('reconstruction error')
        ax.set_title('Q2/Q3: reconstruction sits at the machine floor')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'farfield_envelope_reconstruction_error.png', dpi=110)
        plt.close(fig)

    def test_public_reconstruct_from_envelope_is_exact(self):
        """`reconstruct_from_envelope` reproduces ``exact_total``."""
        error = float(np.max(np.abs(
            self._reconstruct_public() - self.partition.exact_total)))
        self.assert_within(
            error / self.f_scale, MACHINE_REL_TOL,
            f'reconstruct_from_envelope departed from exact_total by '
            f'{error / self.f_scale:.3e} (relative)')

    def test_gauge_envelope_total_is_exact(self):
        """`_gauge.envelope_total` reproduces ``exact_total``."""
        error = float(np.max(np.abs(
            self._reconstruct_gauge() - self.partition.exact_total)))
        self.assert_within(
            error / self.f_scale, MACHINE_REL_TOL,
            f'envelope_total departed from exact_total by '
            f'{error / self.f_scale:.3e} (relative)')

    def test_the_band_reaches_the_exact_schwinger_ceiling(self):
        """The reconstruction band includes ``w`` up to 60."""
        self.comparisons += 1
        self.assertGreaterEqual(
            float(self.partition.w.max()), 60.0,
            'the reconstruction band did not reach w = 60')


class SelfFalsificationTestCase(FarfieldEnvelopeTestCase):
    """Prove each green gate can actually go red under a corruption.

    A numerical suite whose gates cannot fail is decoration.  These tests
    feed the OLD (buggy) label or a corrupted envelope into the SAME
    thresholds the passing tests use, and assert the thresholds catch it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.w_key = tuple(float(v) for v in CONTINUITY_W)
        cls.partition = _partition(
            cls.w_key, GAMMA_DIAGONAL, *DIAGONAL_EXTERIOR)
        cls.flipped = _lobe_flipped_partition(cls.partition)
        cls.f_scale = float(np.max(np.abs(cls.partition.exact_total)))

    def test_old_label_violates_the_continuity_gate(self):
        """The OLD label jumps across the sweep, tripping ``NEW`` gates."""
        old = np.array([
            np.abs(_partition(self.w_key, GAMMA_DIAGONAL, Y1_DIAGONAL,
                              float(y2)).envelope)
            for y2 in Y2_SWEEP])
        self.comparisons += 1
        # The OLD label would fail BOTH far-field gates it is the foil for.
        self.assertGreater(
            float(np.max(np.abs(np.diff(old, axis=0)))), NEW_CONTINUITY_MAX,
            'the OLD label passed the continuity gate -- the gate has no '
            'teeth')
        self.assertGreater(
            float(old.max()), NEW_ENVELOPE_MAX,
            'the OLD label passed the magnitude ceiling -- no teeth')

    def test_old_label_violates_the_invariance_gate(self):
        """Using the OLD envelope as the label breaks lobe invariance."""
        deviation = float(np.max(np.abs(
            self.partition.envelope - self.flipped.envelope)))
        self.comparisons += 1
        self.assertGreater(
            deviation / self.f_scale, MACHINE_REL_TOL,
            'the OLD lobe-dependent envelope satisfied the invariance gate '
            '-- the gate would not catch the bug it exists for')

    def test_corrupted_envelope_breaks_reconstruction(self):
        """A perturbed ``E_ff`` no longer reconstructs ``exact_total``."""
        band_key = tuple(float(v) for v in RECON_BAND)
        partition = _partition(band_key, GAMMA_DIAGONAL, *DIAGONAL_EXTERIOR)
        envelope = farfield_envelope_from_partition(partition)
        switch = _far_field_switch(partition)
        f_scale = float(np.max(np.abs(partition.exact_total)))
        total = _gauge.envelope_total(
            partition.w, partition.delays, partition.saddle_kernels,
            switch, 0.0, envelope * (1.0 + 1e-6))
        error = float(np.max(np.abs(total - partition.exact_total)))
        self.comparisons += 1
        self.assertGreater(
            error / f_scale, MACHINE_REL_TOL,
            'a 1e-6 perturbation of E_ff still reconstructed exact_total '
            'within 1e-12 -- the reconstruction gate has no teeth')


class StraddlingTileTrainabilityTestCase(FarfieldEnvelopeTestCase):
    """Spec 1: a diagonal-straddling tile trains below the gate under NEW.

    Trains the SAME straddling far-field tile under both labels (a fixed
    engine grid, identical spline fit) and measures the held-out
    F-normalized eps.  The NEW label clears the production gate
    ``1e-3``; the OLD label -- whose lobe-flip discontinuity sits inside
    the box -- is orders of magnitude worse.  An on-axis tile confirms the
    NEW label is uniformly good, not tuned to one box.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.straddling_new = np.array(
            _held_out_eps_list(STRADDLING_TILE_CENTER, 'new'))
        cls.straddling_old = np.array(
            _held_out_eps_list(STRADDLING_TILE_CENTER, 'old'))
        cls.on_axis_new = np.array(
            _held_out_eps_list(ON_AXIS_TILE_CENTER, 'new'))
        cls.on_axis_old = np.array(
            _held_out_eps_list(ON_AXIS_TILE_CENTER, 'old'))
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.logspace(-5, 3, 33)
        ax.hist(cls.straddling_new, bins=bins, alpha=0.6,
                label='NEW, straddling', color='C0')
        ax.hist(cls.on_axis_new, bins=bins, alpha=0.6,
                label='NEW, on-axis', color='C2')
        ax.hist(cls.straddling_old, bins=bins, alpha=0.6,
                label='OLD, straddling', color='C3')
        ax.hist(cls.on_axis_old, bins=bins, alpha=0.6,
                label='OLD, on-axis', color='C1')
        ax.axvline(FARFIELD_EPS_GATE, color='k', ls='--',
                   label=f'gate {FARFIELD_EPS_GATE:g}')
        ax.set_xscale('log')
        ax.set_xlabel('held-out F-normalized eps')
        ax.set_ylabel('count')
        ax.set_title('Spec 1: NEW label trains below the gate; OLD does not')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'farfield_trainability_eps_histogram.png',
                    dpi=110)
        plt.close(fig)

    def test_straddling_tile_trains_below_the_gate_under_new_label(self):
        """NEW straddling held-out eps clears the ``1e-3`` production gate."""
        self.assert_within(
            float(self.straddling_new.max()), FARFIELD_EPS_GATE,
            f'NEW straddling tile failed the gate: max eps '
            f'{self.straddling_new.max():.3e}')

    def test_straddling_tile_fails_the_gate_under_old_label(self):
        """OLD straddling held-out eps blows past the gate (the foil)."""
        old_max = float(self.straddling_old.max())
        self.comparisons += 1
        self.assertGreater(
            old_max, OLD_STRADDLING_EPS_MIN,
            f'OLD straddling tile unexpectedly fit ({old_max:.3e}); the '
            f'trainability contrast would be vacuous')

    def test_new_label_beats_old_by_orders_of_magnitude(self):
        """The NEW-vs-OLD held-out eps ratio exceeds ``1e3`` straddling."""
        ratio = (float(self.straddling_old.max())
                 / float(self.straddling_new.max()))
        self.comparisons += 1
        self.assertGreater(
            ratio, NEW_OVER_OLD_RATIO_MIN,
            f'NEW beat OLD by only {ratio:.2e}x on the straddling tile')

    def test_new_label_also_trains_below_the_gate_on_axis(self):
        """NEW is uniformly good: an on-axis tile also clears the gate."""
        self.assert_within(
            float(self.on_axis_new.max()), FARFIELD_EPS_GATE,
            f'NEW on-axis tile failed the gate: max eps '
            f'{self.on_axis_new.max():.3e}')


class ServingMirrorAcrossDiagonalTestCase(FarfieldEnvelopeTestCase):
    """Spec 2 (Q6b): served envelope reconstructs ``F`` across the diagonal.

    Serves the trained straddling chart at held-out points on BOTH sides
    of the astroid flip line, reconstructs ``F`` through the likelihood
    far-field path (`reconstruct_from_envelope` with ``switch = real_mask``
    and ``critical_delay = 0``), and compares to a FRESH exact-engine
    ``F = partition.exact_total``.  The relative mismatch stays within the
    Q6b tolerance ``3e-3`` (~3x headroom over the training gate).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _train_tile(STRADDLING_TILE_CENTER, 'new')
        cls.surrogate = LensAmplificationSurrogate([cls.chart], {})
        cls.errors: list[float] = []
        cls.served_y2: list[float] = []
        cls.overlay = None
        for gamma, y1, y2 in _held_out_samples(
                STRADDLING_TILE_CENTER, 24, seed=7):
            served = cls._mirror_error(gamma, y1, y2)
            if served is None:
                continue
            error, f_serve, f_engine = served
            cls.errors.append(error)
            cls.served_y2.append(y2)
            if cls.overlay is None:
                cls.overlay = (f_serve, f_engine)
        cls._plot()

    @classmethod
    def _mirror_error(cls, gamma: float, y1: float, y2: float):
        """Serve, reconstruct F, and compare to the engine; ``None`` if
        the chart declined to serve the point."""
        geom = ChangRefsdalChannels(_W_EVAL).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        envelope, served, _definition = cls.surrogate.serve(
            _W_EVAL, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=geom.caustic_distance, theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()))
        if not served:
            return None
        far_switch = np.zeros((_W_EVAL.size, geom.real_mask.size))
        far_switch[:, np.asarray(geom.real_mask, dtype=bool)] = 1.0
        _kernels, f_serve = reconstruct_from_envelope(
            _W_EVAL, envelope, geom.delays, geom.saddle_kernels,
            far_switch, 0.0)
        engine = ChangRefsdalChannels(_W_EVAL)
        engine.reset()
        f_engine = engine.evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0).exact_total
        error = float(np.max(np.abs(f_serve - f_engine))
                      / np.max(np.abs(f_engine)))
        return error, f_serve, f_engine

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL or cls.overlay is None:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        f_serve, f_engine = cls.overlay
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(_W_EVAL, f_engine.real, 'k-', label='Re F_engine')
        ax.plot(_W_EVAL, f_serve.real, 'C0--', label='Re F_serve')
        ax.plot(_W_EVAL, f_engine.imag, 'k:', label='Im F_engine')
        ax.plot(_W_EVAL, f_serve.imag, 'C3-.', label='Im F_serve')
        ax.set_xlabel('w')
        ax.set_ylabel('F')
        ax.set_title('Spec 2: served F matches engine at a straddling point')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'farfield_serving_mirror_overlay.png',
                    dpi=110)
        plt.close(fig)

    def test_reconstructed_F_matches_engine_across_the_diagonal(self):
        """Every served held-out point reconstructs ``F`` within Q6b."""
        for error in self.errors:
            self.assert_within(
                error, SERVE_MIRROR_TOL,
                f'serving-mirror error {error:.3e} exceeded '
                f'{SERVE_MIRROR_TOL:g}')

    def test_held_out_points_straddle_the_flip_line(self):
        """The served set includes points on both sides of ``y2 = 1.25``."""
        served = np.array(self.served_y2)
        self.comparisons += 1
        self.assertTrue(
            np.any(served < DIAGONAL_FLIP_Y2)
            and np.any(served > DIAGONAL_FLIP_Y2),
            f'served points did not straddle y2 = {DIAGONAL_FLIP_Y2}: '
            f'range [{served.min():.3f}, {served.max():.3f}]')


def _synthetic_tube_chart() -> TubeChart:
    """A small, cheap `TubeChart` (no engine) for the tube-only artifact."""
    gamma_grid = np.linspace(0.1, 0.3, 4)
    u_grid = np.linspace(0.2, 0.5, 4)
    theta_grid = np.linspace(0.1, 0.5, 4)
    log_w_grid = np.linspace(np.log(5.0), np.log(60.0), 5)
    shape = (log_w_grid.size, gamma_grid.size, u_grid.size, theta_grid.size)
    values = np.ones(shape)
    return TubeChart.from_values(
        gamma_grid=gamma_grid, u_grid=u_grid, theta_grid=theta_grid,
        log_w_grid=log_w_grid, envelope_real=values,
        envelope_imag=0.1 * values, image_count=2, parity=1,
        eta_floor=0.04, eta_max=0.25)


def _legacy_single_box_arrays(chart: FarFieldChart, tag: str | None
                              ) -> dict:
    """Flat arrays mimicking an 8a legacy single-box artifact (no charts).

    Emits exactly the keys `_load_legacy_single_box` reads and, crucially,
    NO ``n_charts`` key -- so `LensAmplificationSurrogate.load` routes to
    the legacy path.  ``tag=None`` omits ``envelope_definition`` (a genuine
    pre-tag artifact); a string writes it.

    ALWAYS writes a valid ``axis_schema`` (Build 8h-b3): `chart` (from
    `_train_tile`) already carries the caustic-fixed ``(rho, theta_c)``
    axes the current loader unconditionally reads (``data['rho_grid']``,
    ``data['knot_rho']``, etc. -- it no longer reads ``y1_grid``/``y2_grid``
    at all), so this fixture is a genuine legacy artifact ONLY along the
    ``envelope_definition`` axis this test class targets, not the axis
    schema (a separate, later hard-refuse `_validate_farfield_axis_schema`
    would otherwise trip regardless of ``tag``, making every case here
    refuse for the wrong reason).
    """
    knot_log_w, knot_gamma, knot_rho, knot_theta_c = chart.knots
    arrays = {
        'gamma_grid': chart.gamma_grid, 'rho_grid': chart.rho_grid,
        'theta_c_grid': chart.theta_c_grid, 'log_w_grid': chart.log_w_grid,
        'real_coeffs': chart.real_coeffs, 'imag_coeffs': chart.imag_coeffs,
        'knot_log_w': knot_log_w, 'knot_gamma': knot_gamma,
        'knot_rho': knot_rho, 'knot_theta_c': knot_theta_c,
        'refused_points': chart.refused_points,
        'axis_schema': np.array(_FARFIELD_AXIS_SCHEMA),
        'provenance': np.array(json.dumps({}))}
    if tag is not None:
        arrays['envelope_definition'] = np.array(tag)
    return arrays


class DefinitionTagLoaderRefusalTestCase(FarfieldEnvelopeTestCase):
    """Spec 3 (F010): the loader hard-refuses an absent/unknown tag.

    A far-field chart trained before the Build 8g-b redefinition encodes
    the OLD lobe-flipping label and would reconstruct a finite-but-WRONG
    ``F`` under the new serving mirror.  The loader must refuse such an
    artifact (absent OR unknown ``envelope_definition``) with a clear
    ValueError naming the chart and instructing a rebuild -- through BOTH
    the multi-chart and the legacy single-box load paths -- while a chart
    carrying the known tag loads and serves, and a tube-only artifact
    (which never carries the tag) is unaffected.
    """

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.chart = _train_tile(STRADDLING_TILE_CENTER, 'new')

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def _save_multichart(self, name: str) -> pathlib.Path:
        """Save a one-far-field-chart surrogate; return its ``.npz`` path."""
        surrogate = LensAmplificationSurrogate([self.chart], {})
        path = self.tmp / name
        surrogate.save(path)
        return path.with_suffix('.npz')

    @staticmethod
    def _rewrite_meta(path: pathlib.Path, out: pathlib.Path,
                      transform) -> None:
        """Re-save an artifact with ``chart0_meta`` JSON transformed."""
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        meta = json.loads(str(arrays['chart0_meta']))
        arrays['chart0_meta'] = np.array(json.dumps(transform(meta)))
        np.savez(out, **arrays)

    def _save_legacy(self, name: str, tag: str | None) -> pathlib.Path:
        path = self.tmp / name
        np.savez(path, **_legacy_single_box_arrays(self.chart, tag))
        return path.with_suffix('.npz')

    def test_multichart_missing_tag_refuses_with_rebuild_message(self):
        """A chart whose meta lacks the tag raises, naming the chart."""
        base = self._save_multichart('missing')
        corrupt = self.tmp / 'missing_corrupt.npz'
        self._rewrite_meta(
            base, corrupt,
            lambda meta: {k: v for k, v in meta.items()
                          if k != 'envelope_definition'})
        self.comparisons += 1
        with self.assertRaises(ValueError) as ctx:
            LensAmplificationSurrogate.load(corrupt)
        message = str(ctx.exception)
        self.assertIn('chart 0', message)
        self.assertIn('rebuild', message)

    def test_multichart_unknown_tag_refuses_naming_the_tag(self):
        """A chart with an unknown tag raises, naming the offending tag."""
        base = self._save_multichart('unknown')
        corrupt = self.tmp / 'unknown_corrupt.npz'
        self._rewrite_meta(
            base, corrupt,
            lambda meta: {**meta, 'envelope_definition': 'legacy_v1_label'})
        self.comparisons += 1
        with self.assertRaises(ValueError) as ctx:
            LensAmplificationSurrogate.load(corrupt)
        message = str(ctx.exception)
        self.assertIn('legacy_v1_label', message)
        self.assertIn('rebuild', message)

    def test_known_tag_loads_and_serves(self):
        """A chart carrying the known tag loads and serves normally."""
        path = self._save_multichart('known')
        surrogate = LensAmplificationSurrogate.load(path)
        self.assertEqual(len(surrogate.charts), 1)
        gamma, y1, y2 = _held_out_samples(STRADDLING_TILE_CENTER, 1, seed=3)[0]
        geom = ChangRefsdalChannels(_W_EVAL).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        _envelope, served, definition = surrogate.serve(
            _W_EVAL, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=geom.caustic_distance, theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()))
        self.comparisons += 1
        self.assertTrue(served, 'the known-tag chart declined to serve')
        self.assertEqual(definition, _FARFIELD_ENVELOPE_DEFINITION)

    def test_legacy_single_box_missing_tag_refuses(self):
        """The legacy loader refuses an artifact with no tag."""
        path = self._save_legacy('legacy_notag', tag=None)
        self.comparisons += 1
        with self.assertRaises(ValueError) as ctx:
            LensAmplificationSurrogate.load(path)
        message = str(ctx.exception)
        self.assertIn('legacy single-box', message)
        self.assertIn('rebuild', message)

    def test_legacy_single_box_known_tag_loads(self):
        """The legacy loader accepts an artifact carrying the known tag."""
        path = self._save_legacy(
            'legacy_ok', tag=_FARFIELD_ENVELOPE_DEFINITION)
        surrogate = LensAmplificationSurrogate.load(path)
        self.comparisons += 1
        self.assertEqual(len(surrogate.charts), 1)
        self.assertIsInstance(surrogate.charts[0], FarFieldChart)

    def test_tube_only_artifact_is_unaffected(self):
        """A tube-only artifact (never tagged) loads without refusal."""
        surrogate = LensAmplificationSurrogate([_synthetic_tube_chart()], {})
        path = self.tmp / 'tube_only'
        surrogate.save(path)
        loaded = LensAmplificationSurrogate.load(path.with_suffix('.npz'))
        self.comparisons += 1
        self.assertEqual(len(loaded.charts), 1)
        self.assertIsInstance(loaded.charts[0], TubeChart)


class NewGateSelfFalsificationTestCase(FarfieldEnvelopeTestCase):
    """Prove the three surrogate-side gates can actually go red.

    A gate that cannot fail is decoration.  These feed the OLD label into
    the trainability gate, an additively-corrupted envelope into the
    serving-mirror gate, and a de-tagged artifact into the loader -- and
    assert each is caught.
    """

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def test_trainability_gate_rejects_the_old_label(self):
        """The OLD straddling tile blows past the ``1e-3`` gate."""
        old = np.array(_held_out_eps_list(STRADDLING_TILE_CENTER, 'old'))
        self.comparisons += 1
        self.assertGreater(
            float(old.max()), FARFIELD_EPS_GATE,
            'the OLD label cleared the trainability gate -- the gate has '
            'no teeth')

    def test_mirror_gate_rejects_a_corrupted_served_envelope(self):
        """An additive envelope corruption breaks the serving mirror.

        The served ``E_ff`` is the ``~1e-4`` far-field remainder, so a
        corruption must be ADDITIVE (of order the gate itself) to move
        ``F = E_ff + carriers`` by more than ``3e-3`` -- a multiplicative
        perturbation of a ``1e-4`` quantity would be invisible against
        ``|F| ~ 1``.
        """
        chart = _train_tile(STRADDLING_TILE_CENTER, 'new')
        surrogate = LensAmplificationSurrogate([chart], {})
        gamma, y1, y2 = _held_out_samples(STRADDLING_TILE_CENTER, 1, seed=5)[0]
        geom = ChangRefsdalChannels(_W_EVAL).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        envelope, served, _definition = surrogate.serve(
            _W_EVAL, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=geom.caustic_distance, theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()))
        self.assertTrue(served, 'fixture point did not serve')
        far_switch = np.zeros((_W_EVAL.size, geom.real_mask.size))
        far_switch[:, np.asarray(geom.real_mask, dtype=bool)] = 1.0
        _kernels, f_corrupt = reconstruct_from_envelope(
            _W_EVAL, envelope + 0.01, geom.delays, geom.saddle_kernels,
            far_switch, 0.0)
        engine = ChangRefsdalChannels(_W_EVAL)
        engine.reset()
        f_engine = engine.evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0).exact_total
        error = float(np.max(np.abs(f_corrupt - f_engine))
                      / np.max(np.abs(f_engine)))
        self.comparisons += 1
        self.assertGreater(
            error, SERVE_MIRROR_TOL,
            f'a 0.01 additive envelope corruption still reconstructed F '
            f'within {SERVE_MIRROR_TOL:g} (error {error:.3e}) -- no teeth')

    def test_detagging_a_serving_artifact_flips_it_to_refused(self):
        """Only the tag stands between a serving chart and a refusal.

        Same spline coefficients, two saved variants: one carrying the
        known tag (loads and serves) and one with the tag stripped
        (refuses).  This proves the loader refusal is consequential -- it
        rejects an otherwise-valid artifact solely on the missing tag.
        """
        chart = _train_tile(STRADDLING_TILE_CENTER, 'new')
        base = self.tmp / 'ff'
        LensAmplificationSurrogate([chart], {}).save(base)
        base = base.with_suffix('.npz')
        with np.load(base, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        stripped = dict(arrays)
        meta = json.loads(str(arrays['chart0_meta']))
        stripped['chart0_meta'] = np.array(json.dumps(
            {k: v for k, v in meta.items() if k != 'envelope_definition'}))
        detagged = self.tmp / 'ff_detagged.npz'
        np.savez(detagged, **stripped)
        # Tagged variant loads and serves.
        tagged = LensAmplificationSurrogate.load(base)
        self.comparisons += 1
        self.assertEqual(len(tagged.charts), 1)
        # De-tagged variant -- identical coefficients -- is refused.
        self.comparisons += 1
        with self.assertRaises(ValueError):
            LensAmplificationSurrogate.load(detagged)


# ==========================================================================
# Build 8g-b extension: node-convergence probe (Q7 / acceptance d), tube
# byte-identity (acceptance e), and far-field gate-currency mutation check.
# These three shards are additive to the far-field-envelope suite above.
# ==========================================================================

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: An exterior far-field tile centre well outside the caustic (the caustic
#: for ``gamma ~ 0.04`` is a sub-``0.1`` astroid near the origin), so every
#: grid point resolves two real images and the smooth remainder ``E_ff``.
EXTERIOR_TILE_CENTER = (1.5, 1.5)
EXTERIOR_TILE_HALF = 0.2

#: A deliberately oversized exterior tile pushed toward the caustic (still
#: two-image everywhere); the smooth NEW label is fit by the authorized grid
#: even here, which is why the Q7 recommendation is uniform across the
#: exterior rather than tuned to one box.
OVERSIZED_TILE_CENTER = (0.5, 0.5)
OVERSIZED_TILE_HALF = 0.3

#: Fixed shear-node and ``w``-node counts for the convergence tiles.  Both
#: are held constant while the ``y`` node count is swept, so the curve
#: isolates the source-plane resolution (``surrogate_training`` shares the
#: 4-node shear axis and the 14-node log-``w`` axis).
EXTERIOR_N_GAMMA = 4
EXTERIOR_N_W = 14

#: The ``y``-node counts swept for the convergence curve: a coarser variant,
#: the Professor-authorized ``5 x 5`` start, and a finer variant.  All three
#: are ``>= 4`` (the `_validate_axis` minimum).
CONVERGENCE_NODE_COUNTS = (4, 5, 7)
AUTHORIZED_N_Y = 5

#: The finer grid must not make the held-out eps materially WORSE than the
#: authorized grid: ``eps(fine) <= CONVERGENCE_PLATEAU_RATIO * eps(auth)``.
#: Measured ratio ~1.00 -- the authorized grid is already on the converged
#: plateau (the residual is the ``w``-spline / far-field-magnitude floor,
#: not ``y`` resolution), so adding nodes cannot rescue a chart by chance.
CONVERGENCE_PLATEAU_RATIO = 2.0

#: HEAD's far-field eps bar before the Build 8g-b redefinition.  The campaign
#: TIGHTENED the bar to `FARFIELD_EPS_GATE` (``1e-3``); it was never widened
#: to admit the OLD label.  The convergence probe asserts the served bar is
#: at most this (a strict tightening, the "never widen admittance" direction).
HEAD_FARFIELD_EPS_MAX = 3.0e-3

#: The tube eps bar (`surrogate_training.TrainingConfig.tube_eps_max`), which
#: Build 8g-b left UNCHANGED at ``5e-2`` (only the far-field bar/currency
#: moved).  Asserted equal on HEAD and branch.
TUBE_EPS_MAX = 5.0e-2

#: Additive coefficient perturbation for the gate-currency mutation, as a
#: fraction of ``max|F|``.  A ``5e-3 * max|F|`` bump lifts the F-normalized
#: held-out eps to ~5.3e-3 (measured), ~5x over the ``1e-3`` gate -- RED.
MUTATION_COEFF_FRACTION = 5.0e-3

#: Floor the SAME healthy chart's held-out eps must EXCEED when (wrongly)
#: normalized by ``max|E_ff| ~ 1e-4`` instead of ``max|F|``: measured ~1.5,
#: i.e. the tiny-denominator currency thrashes a perfectly good chart to
#: O(1).  This is why the production gate F-normalizes.
EFFNORM_THRASH_MIN = 1.0e-1


@functools.lru_cache(maxsize=None)
def _head_module(rel_path: str, mod_name: str):
    """Import a module from its HEAD revision, side by side with the branch.

    ``git show HEAD:<rel_path>`` is written to a real temporary ``.py`` file
    and imported under ``mod_name`` (registered in ``sys.modules`` first so
    any dataclass field resolution succeeds).  Used to prove the tube path
    is byte-identical to HEAD after the additive far-field changes.
    """
    source = subprocess.check_output(
        ['git', 'show', f'HEAD:{rel_path}'], cwd=_REPO_ROOT).decode()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    tmp.write(source)
    tmp.close()
    spec = importlib.util.spec_from_file_location(mod_name, tmp.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _exterior_samples(center: tuple[float, float], half: float, count: int,
                      seed: int) -> list[tuple[float, float, float]]:
    """Random ``(gamma, y1, y2)`` inside the inner 80% of an exterior tile."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(*TILE_GAMMA_BAND)),
             float(rng.uniform(center[0] - 0.8 * half, center[0] + 0.8 * half)),
             float(rng.uniform(center[1] - 0.8 * half, center[1] + 0.8 * half)))
            for _ in range(count)]


@functools.lru_cache(maxsize=None)
def _train_exterior_chart(center: tuple[float, float], half: float,
                          n_y: int, n_w: int) -> FarFieldChart:
    """Fit a `FarFieldChart` to the NEW far-field label on an exterior tile.

    A parameterized companion to `_train_tile`: the source box is
    ``center +/- half`` with ``n_y x n_y`` nodes and ``n_w`` log-``w`` nodes,
    the shear axis is the fixed 4-node `TILE_GAMMA_BAND`.  Always fits the
    production label `farfield_envelope_from_partition`.  Spatial axes are
    caustic-fixed ``(rho, theta_c)`` (Build 8h-b3) via `_box_to_caustic_fixed`
    -- the SAME fixed-reach convention `_train_tile` uses.
    """
    # Unlike `_train_tile` (a narrow, off-origin box where a per-gamma hull
    # is needed so randomly-gamma'd held-out queries stay contained), this
    # tile can be centred near the origin with a wide half (`OVERSIZED_TILE_*`),
    # spanning a wide angular sweep whose per-gamma hull would badly dilate
    # the trained rho/theta_c box (measured: dilated hull held-out eps
    # ~3.8e-2 for the oversized tile vs ~2.6e-3 at a single band-midpoint
    # reach).  Use the FIXED band-midpoint reach here for a materially
    # tighter box; `_chart_eps`/`_exterior_eps` already skip any held-out
    # sample that ends up not served, so the narrower containment this
    # trades away costs no assertion.
    gamma_grid = np.linspace(*TILE_GAMMA_BAND, EXTERIOR_N_GAMMA)
    gamma_mid = 0.5 * sum(TILE_GAMMA_BAND)
    rho_grid, theta_c_grid = _box_to_caustic_fixed(
        (center[0] - half, center[0] + half),
        (center[1] - half, center[1] + half), n_y, n_y,
        gamma_range=(gamma_mid, gamma_mid))
    log_w_grid = np.linspace(np.log(TILE_W_RANGE[0]), np.log(TILE_W_RANGE[1]),
                             n_w)
    w_grid = np.exp(log_w_grid)
    shape = (n_w, EXTERIOR_N_GAMMA, n_y, n_y)
    envelope_real = np.zeros(shape)
    envelope_imag = np.zeros(shape)
    refused: list[tuple[float, float, float]] = []
    for ig, gamma in enumerate(gamma_grid):
        for i1, rho in enumerate(rho_grid):
            for i2, theta_c in enumerate(theta_c_grid):
                y1, y2 = surrogate_module._from_caustic_fixed(
                    float(gamma), float(rho), float(theta_c))
                engine = ChangRefsdalChannels(w_grid)
                engine.reset()
                try:
                    partition = engine.evaluate(
                        gamma=float(gamma), y=(float(y1), float(y2)),
                        beta=0.0, kappa=0.0)
                except _ENGINE_REFUSALS:
                    refused.append((float(gamma), float(rho), float(theta_c)))
                    continue
                envelope = farfield_envelope_from_partition(partition)
                if not np.all(np.isfinite(envelope)):
                    refused.append((float(gamma), float(rho), float(theta_c)))
                    continue
                envelope_real[:, ig, i1, i2] = envelope.real
                envelope_imag[:, ig, i1, i2] = envelope.imag
    refused_points = (np.array(refused) if refused
                      else np.empty((0, 3), dtype=float))
    return FarFieldChart.from_values(
        gamma_grid=gamma_grid, rho_grid=rho_grid, theta_c_grid=theta_c_grid,
        log_w_grid=log_w_grid, envelope_real=envelope_real,
        envelope_imag=envelope_imag, image_count=2, parity=1,
        refused_points=refused_points)


def _chart_eps(chart: FarFieldChart, center: tuple[float, float], half: float,
               *, normalization: str, seed: int = 1, count: int = 30
               ) -> float:
    """Max held-out far-field eps of ``chart`` over an exterior tile.

    Serves each held-out ``(gamma, y1, y2)`` through a one-chart surrogate
    and compares to the fresh engine label `farfield_envelope_from_partition`
    on the interior ``_W_EVAL`` grid (dodging the log-``w`` endpoint round-off
    that silently un-serves a training-grid endpoint).  ``normalization='F'``
    divides by ``max|exact_total|`` -- the production far-field currency
    (`surrogate_training._heldout_eps`); ``normalization='Eff'`` divides by
    ``max|E_ff|`` -- the tiny-denominator currency the gate deliberately does
    NOT use.
    """
    surrogate = LensAmplificationSurrogate([chart], {})
    errors: list[float] = []
    for gamma, y1, y2 in _exterior_samples(center, half, count, seed):
        engine = ChangRefsdalChannels(_W_EVAL)
        engine.reset()
        try:
            partition = engine.evaluate(
                gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        except _ENGINE_REFUSALS:
            continue
        reference = farfield_envelope_from_partition(partition)
        if not np.all(np.isfinite(reference)):
            continue
        if normalization == 'F':
            denom = float(np.max(np.abs(partition.exact_total))) or 1.0
        else:
            denom = float(np.max(np.abs(reference))) or 1.0
        emulated, served, _definition = surrogate.serve(
            _W_EVAL, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=partition.caustic_distance, theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        if not served:
            continue
        errors.append(float(np.max(np.abs(emulated - reference)) / denom))
    return max(errors) if errors else float('nan')


@functools.lru_cache(maxsize=None)
def _exterior_eps(center: tuple[float, float], half: float, n_y: int,
                  n_w: int, normalization: str) -> float:
    """Cached max held-out eps for a freshly trained exterior tile."""
    chart = _train_exterior_chart(center, half, n_y, n_w)
    return _chart_eps(chart, center, half, normalization=normalization)


@functools.lru_cache(maxsize=None)
def _center_f_scale(center: tuple[float, float]) -> float:
    """``max|exact_total|`` at a tile centre (for sizing a coefficient bump)."""
    engine = ChangRefsdalChannels(_W_EVAL)
    engine.reset()
    partition = engine.evaluate(
        gamma=float(np.mean(TILE_GAMMA_BAND)), y=center, beta=0.0, kappa=0.0)
    return float(np.max(np.abs(partition.exact_total)))


def _head_git_default(rel_path: str, field: str) -> float:
    """Read a ``field: float = <value>`` dataclass default from HEAD source."""
    source = subprocess.check_output(
        ['git', 'show', f'HEAD:{rel_path}'], cwd=_REPO_ROOT).decode()
    import re
    match = re.search(rf'{field}\s*:\s*float\s*=\s*([0-9eE.+-]+)', source)
    if match is None:
        raise AssertionError(f'{field} default not found in HEAD {rel_path}')
    return float(match.group(1))


def _tube_probe_configs() -> list[dict]:
    """A fixed probe set of `TubeChart.from_values` inputs (no engine).

    Deterministic random value tensors on three distinct near-caustic tube
    boxes; the values stand in for engine labels for a pure byte-identity
    comparison of the tube construction/serialization/serving machinery.
    """
    configs = []
    for seed, (g_lo, g_hi, eta_floor, eta_max) in enumerate(
            [(0.10, 0.30, 0.04, 0.25),
             (0.40, 0.70, 0.02, 0.16),
             (1.20, 1.60, 0.05, 0.30)]):
        rng = np.random.default_rng(seed)
        gamma_grid = np.linspace(g_lo, g_hi, 4)
        u_grid = np.linspace(np.sqrt(eta_floor), np.sqrt(eta_max), 4)
        theta_grid = np.linspace(0.1, 0.6, 4)
        log_w_grid = np.linspace(np.log(5.0), np.log(60.0), 5)
        shape = (log_w_grid.size, 4, 4, 4)
        configs.append(dict(
            gamma_grid=gamma_grid, u_grid=u_grid, theta_grid=theta_grid,
            log_w_grid=log_w_grid,
            envelope_real=rng.standard_normal(shape),
            envelope_imag=rng.standard_normal(shape),
            image_count=2, parity=(1 if g_lo < 1.0 else -1),
            eta_floor=eta_floor, eta_max=eta_max))
    return configs


def _tube_probe_queries() -> list[dict]:
    """Serving queries landing inside the first `_tube_probe_configs` box."""
    return [dict(gamma=0.20, y1=0.0, y2=0.0, beta=0.0, eta=0.09, theta=0.30,
                 image_count=2),
            dict(gamma=0.15, y1=0.0, y2=0.0, beta=0.0, eta=0.16, theta=0.45,
                 image_count=2),
            dict(gamma=0.28, y1=0.0, y2=0.0, beta=0.0, eta=0.05, theta=0.20,
                 image_count=2)]


class FarFieldNodeConvergenceTestCase(FarfieldEnvelopeTestCase):
    """Spec A (Professor Q7, acceptance d): node-convergence of ``eps_ff``.

    Trains the NEW smooth far-field label on a fixed exterior tile at three
    source-node counts -- a coarser variant, the Professor-authorized
    ``5 x 5`` start, and a finer variant -- holding the 4-node shear axis
    and 14-node log-``w`` axis fixed.  The F-normalized held-out eps is
    certified against the production gate ``FARFIELD_EPS_GATE = 1e-3`` at
    every node count, and the finer grid must not be materially WORSE than
    the authorized one (the residual is the far-field-magnitude / ``w``-
    spline floor, not ``y`` resolution -- the authorized grid is already on
    the converged plateau).

    "Promote, never widen": the SAME production gate is applied at every
    node count; the probe would respond to a coarse failure by adding
    ``y``-nodes, never by relaxing the bar.  The self-falsification test
    below feeds an additively-corrupted authorized chart through the same
    currency and confirms it goes RED, so a false pass is impossible.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.eps_by_n = {
            n_y: _exterior_eps(EXTERIOR_TILE_CENTER, EXTERIOR_TILE_HALF,
                               n_y, EXTERIOR_N_W, 'F')
            for n_y in CONVERGENCE_NODE_COUNTS}
        cls.oversized_eps = _exterior_eps(
            OVERSIZED_TILE_CENTER, OVERSIZED_TILE_HALF, AUTHORIZED_N_Y,
            EXTERIOR_N_W, 'F')
        cls.head_bar = _head_git_default(
            'cogwheel/lensing/surrogate_training.py', 'farfield_eps_max')
        # Build-report record: (tile size, node count, eps_ff) per the Q7 ask.
        print('\n[Q7 far-field node-convergence] tile '
              f'{EXTERIOR_TILE_CENTER} +/- {EXTERIOR_TILE_HALF}, '
              f'{EXTERIOR_N_GAMMA} shear x {EXTERIOR_N_W} log-w nodes:')
        for n_y in CONVERGENCE_NODE_COUNTS:
            print(f'    {n_y}x{n_y} y-nodes -> eps_ff = '
                  f'{cls.eps_by_n[n_y]:.3e}  (gate {FARFIELD_EPS_GATE:g})')
        print(f'    oversized {OVERSIZED_TILE_CENTER} +/- '
              f'{OVERSIZED_TILE_HALF} at {AUTHORIZED_N_Y}x{AUTHORIZED_N_Y} '
              f'-> eps_ff = {cls.oversized_eps:.3e}')
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        counts = list(CONVERGENCE_NODE_COUNTS)
        eps = [cls.eps_by_n[n] for n in counts]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(counts, eps, 'bo-', label='eps_ff (F-normalized)')
        ax.axhline(FARFIELD_EPS_GATE, color='r', ls='--',
                   label=f'gate {FARFIELD_EPS_GATE:g}')
        ax.axvline(AUTHORIZED_N_Y, color='0.6', ls=':',
                   label=f'authorized {AUTHORIZED_N_Y}x{AUTHORIZED_N_Y}')
        ax.set_xlabel('y-nodes per side')
        ax.set_ylabel('held-out eps_ff')
        ax.set_title('Q7: far-field eps_ff vs y-node count (exterior tile)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'farfield_node_convergence_eps.png', dpi=110)
        plt.close(fig)

    def test_authorized_grid_clears_the_gate(self):
        """The Professor-authorized ``5 x 5`` grid passes ``1e-3``."""
        eps = self.eps_by_n[AUTHORIZED_N_Y]
        self.assertFalse(np.isnan(eps),
                         'authorized grid served no held-out point')
        self.assert_within(
            eps, FARFIELD_EPS_GATE,
            f'authorized {AUTHORIZED_N_Y}x{AUTHORIZED_N_Y} eps_ff {eps:.3e} '
            f'exceeded the gate {FARFIELD_EPS_GATE:g}')

    def test_every_swept_node_count_clears_the_same_gate(self):
        """Coarser and finer grids clear the SAME (never-widened) gate."""
        for n_y in CONVERGENCE_NODE_COUNTS:
            with self.subTest(n_y=n_y):
                eps = self.eps_by_n[n_y]
                self.assert_within(
                    eps, FARFIELD_EPS_GATE,
                    f'{n_y}x{n_y} eps_ff {eps:.3e} exceeded the gate')

    def test_finer_grid_stays_on_the_converged_plateau(self):
        """Adding ``y``-nodes does not make eps materially worse."""
        auth = self.eps_by_n[AUTHORIZED_N_Y]
        fine = self.eps_by_n[max(CONVERGENCE_NODE_COUNTS)]
        self.assert_within(
            fine, CONVERGENCE_PLATEAU_RATIO * auth,
            f'finer grid eps {fine:.3e} exceeded {CONVERGENCE_PLATEAU_RATIO}x '
            f'the authorized {auth:.3e}: not on the plateau')

    def test_oversized_exterior_tile_also_clears_gate(self):
        """The authorized grid fits an oversized exterior tile too (Q7 is
        a uniform exterior recommendation, not box-tuned)."""
        self.assert_within(
            self.oversized_eps, FARFIELD_EPS_GATE,
            f'oversized-tile eps_ff {self.oversized_eps:.3e} exceeded the gate')

    def test_served_bar_is_a_tightening_not_a_widening(self):
        """The branch bar equals ``1e-3`` and is at most HEAD's ``3e-3``."""
        branch_bar = surrogate_training.TrainingConfig().farfield_eps_max
        self.comparisons += 1
        self.assertEqual(branch_bar, FARFIELD_EPS_GATE)
        self.assert_within(
            branch_bar, self.head_bar,
            f'branch bar {branch_bar:g} is looser than HEAD '
            f'{self.head_bar:g} -- admittance was WIDENED, not tightened')

    def test_a_corrupted_chart_would_fail_the_convergence_gate(self):
        """Self-falsification: a coefficient-corrupted authorized chart, run
        through the SAME currency, exceeds the gate (a false pass is a
        correctness bug the probe must catch)."""
        healthy = _train_exterior_chart(
            EXTERIOR_TILE_CENTER, EXTERIOR_TILE_HALF, AUTHORIZED_N_Y,
            EXTERIOR_N_W)
        bump = MUTATION_COEFF_FRACTION * _center_f_scale(EXTERIOR_TILE_CENTER)
        corrupted = dataclasses.replace(
            healthy, real_coeffs=healthy.real_coeffs + bump)
        eps_bad = _chart_eps(corrupted, EXTERIOR_TILE_CENTER,
                             EXTERIOR_TILE_HALF, normalization='F')
        self.comparisons += 1
        self.assertGreater(
            eps_bad, FARFIELD_EPS_GATE,
            f'a corrupted chart passed the gate (eps {eps_bad:.3e}) -- the '
            f'convergence currency has no teeth')


class TubeByteIdentityTestCase(FarfieldEnvelopeTestCase):
    """Spec B (acceptance e): the tube path is byte-identical to HEAD.

    The Build 8g-b far-field redefinition is ADDITIVE -- it must not touch
    the tube (near-caustic) chart's construction, serving, serialization,
    or eps gate.  This shard loads HEAD's ``surrogate.py`` side by side with
    the branch, builds `TubeChart` charts from a fixed probe set of value
    tensors on both, and asserts:

    * training labels (coefficient tensors, knots, axes) agree to the byte;
    * served envelopes on a fixed query set agree to the byte;
    * the tube npz round-trips unchanged;
    * the tube eps bar (``tube_eps_max``) and the tube currency
      (``max|partition.envelope|`` normalization in `_heldout_eps`) are
      unchanged from HEAD.

    ``max|diff| == 0.0`` is the whole claim -- an independent HEAD build of
    the identical machinery, not a tolerance.
    """

    HEAD_MODULE = 'cogwheel/lensing/surrogate.py'
    HEAD_TRAINING = 'cogwheel/lensing/surrogate_training.py'

    @classmethod
    def setUpClass(cls) -> None:
        cls.head = _head_module(cls.HEAD_MODULE, 'cogwheel_head_surrogate_tube')
        cls.configs = _tube_probe_configs()
        cls.queries = _tube_probe_queries()

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    @staticmethod
    def _max_abs_diff(branch_array: np.ndarray, head_array: np.ndarray
                      ) -> float:
        """``max|branch - head|`` after asserting identical shapes."""
        branch_array = np.asarray(branch_array)
        head_array = np.asarray(head_array)
        if branch_array.shape != head_array.shape:
            return float('inf')
        if branch_array.size == 0:
            return 0.0
        return float(np.max(np.abs(branch_array - head_array)))

    def test_tube_training_labels_are_byte_identical(self):
        """Coefficient tensors, knots and axes match HEAD to the byte."""
        for index, config in enumerate(self.configs):
            with self.subTest(tube=index):
                branch = TubeChart.from_values(**config)
                head = self.head.TubeChart.from_values(**config)
                for name, got, ref in (
                        ('real_coeffs', branch.real_coeffs, head.real_coeffs),
                        ('imag_coeffs', branch.imag_coeffs, head.imag_coeffs),
                        ('gamma_grid', branch.gamma_grid, head.gamma_grid),
                        ('u_grid', branch.u_grid, head.u_grid),
                        ('theta_grid', branch.theta_grid, head.theta_grid),
                        ('log_w_grid', branch.log_w_grid, head.log_w_grid)):
                    self.assert_within(
                        self._max_abs_diff(got, ref), 0.0,
                        f'tube {index} {name} diverged from HEAD')
                for axis, (got, ref) in enumerate(
                        zip(branch.knots, head.knots)):
                    self.assert_within(
                        self._max_abs_diff(got, ref), 0.0,
                        f'tube {index} knot axis {axis} diverged from HEAD')

    def test_tube_served_envelopes_are_byte_identical(self):
        """Serving the probe queries yields byte-identical envelopes."""
        config = self.configs[0]
        branch_sur = LensAmplificationSurrogate(
            [TubeChart.from_values(**config)], {})
        head_sur = self.head.LensAmplificationSurrogate(
            [self.head.TubeChart.from_values(**config)], {})
        for index, query in enumerate(self.queries):
            with self.subTest(query=index):
                branch_res = branch_sur.serve(_W_EVAL, **query)
                head_res = head_sur.serve(_W_EVAL, **query)
                env_b, served_b = branch_res[0], branch_res[1]
                env_h, served_h = head_res[0], head_res[1]
                self.comparisons += 1
                self.assertTrue(
                    served_b and served_h,
                    f'query {index} was not served on both '
                    f'(branch={served_b}, head={served_h})')
                self.assert_within(
                    self._max_abs_diff(env_b, env_h), 0.0,
                    f'served tube envelope {index} diverged from HEAD')

    def test_tube_npz_round_trips_unchanged(self):
        """Saving and reloading a tube-only artifact preserves the chart."""
        config = self.configs[0]
        original = TubeChart.from_values(**config)
        surrogate = LensAmplificationSurrogate([original], {})
        path = self.tmp / 'tube_only'
        surrogate.save(path)
        loaded = LensAmplificationSurrogate.load(path.with_suffix('.npz'))
        self.comparisons += 1
        self.assertIsInstance(loaded.charts[0], TubeChart)
        restored = loaded.charts[0]
        self.assert_within(
            self._max_abs_diff(restored.real_coeffs, original.real_coeffs),
            0.0, 'tube real_coeffs changed across an npz round-trip')
        self.assert_within(
            self._max_abs_diff(restored.imag_coeffs, original.imag_coeffs),
            0.0, 'tube imag_coeffs changed across an npz round-trip')

    def test_tube_eps_bar_is_unchanged(self):
        """``tube_eps_max`` equals ``5e-2`` on both HEAD and branch."""
        branch_bar = surrogate_training.TrainingConfig().tube_eps_max
        head_bar = _head_git_default(self.HEAD_TRAINING, 'tube_eps_max')
        self.comparisons += 1
        self.assertEqual(branch_bar, TUBE_EPS_MAX)
        self.comparisons += 1
        self.assertEqual(head_bar, TUBE_EPS_MAX)

    def test_tube_eps_currency_is_unchanged(self):
        """The tube branch of ``_heldout_eps`` still normalizes by
        ``max|partition.envelope|`` on both HEAD and branch."""
        branch_src = pathlib.Path(
            surrogate_training.__file__).read_text()
        head_src = subprocess.check_output(
            ['git', 'show', f'HEAD:{self.HEAD_TRAINING}'],
            cwd=_REPO_ROOT).decode()
        needle = 'env_true = np.asarray(partition.envelope)'
        for label, src in (('branch', branch_src), ('head', head_src)):
            self.comparisons += 1
            self.assertIn(
                needle, src,
                f'{label} _heldout_eps no longer uses the tube currency')


class FarFieldGateCurrencyMutationTestCase(FarfieldEnvelopeTestCase):
    """Spec C: the F-normalized eps enforces the right quantity.

    Trains a healthy new-definition far-field chart that passes the gate,
    then perturbs its load-bearing real spline coefficients by an additive
    ``MUTATION_COEFF_FRACTION * max|F|``.  Because the cubic B-spline basis
    is a partition of unity, a constant coefficient bump shifts the
    reconstructed envelope by ~that constant, so the F-normalized held-out
    eps jumps to ~``MUTATION_COEFF_FRACTION`` (~5x the ``1e-3`` gate) and
    the chart goes RED.

    The companion test shows the SAME healthy chart is thrashed to O(1)
    when (wrongly) normalized by the tiny ``max|E_ff| ~ 1e-4`` denominator:
    that is exactly why the production currency divides by ``max|F|`` and
    not by ``max|E_ff|`` -- the F-normalization enforces the physically
    meaningful error without a knife-edge denominator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.healthy = _train_exterior_chart(
            EXTERIOR_TILE_CENTER, EXTERIOR_TILE_HALF, AUTHORIZED_N_Y,
            EXTERIOR_N_W)
        cls.f_scale = _center_f_scale(EXTERIOR_TILE_CENTER)
        cls.bump = MUTATION_COEFF_FRACTION * cls.f_scale
        cls.perturbed = dataclasses.replace(
            cls.healthy, real_coeffs=cls.healthy.real_coeffs + cls.bump)
        cls.eps_healthy_f = _chart_eps(
            cls.healthy, EXTERIOR_TILE_CENTER, EXTERIOR_TILE_HALF,
            normalization='F')
        cls.eps_bad_f = _chart_eps(
            cls.perturbed, EXTERIOR_TILE_CENTER, EXTERIOR_TILE_HALF,
            normalization='F')
        cls.eps_healthy_eff = _chart_eps(
            cls.healthy, EXTERIOR_TILE_CENTER, EXTERIOR_TILE_HALF,
            normalization='Eff')
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['healthy\n(F-norm)', 'perturbed\n(F-norm)',
                  'healthy\n(Eff-norm)']
        values = [cls.eps_healthy_f, cls.eps_bad_f, cls.eps_healthy_eff]
        ax.bar(labels, values, color=['C2', 'C3', 'C1'])
        ax.axhline(FARFIELD_EPS_GATE, color='k', ls='--',
                   label=f'gate {FARFIELD_EPS_GATE:g}')
        ax.set_yscale('log')
        ax.set_ylabel('held-out eps_ff')
        ax.set_title('Spec C: F-normalized gate currency mutation check')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'farfield_gate_currency_mutation.png',
                    dpi=110)
        plt.close(fig)

    def test_healthy_chart_passes_under_F_normalization(self):
        """The unperturbed chart clears the gate under ``max|F|`` currency."""
        self.assertFalse(np.isnan(self.eps_healthy_f),
                         'healthy chart served no held-out point')
        self.assert_within(
            self.eps_healthy_f, FARFIELD_EPS_GATE,
            f'healthy chart failed the gate (eps {self.eps_healthy_f:.3e})')

    def test_perturbed_chart_goes_red_under_F_normalization(self):
        """The coefficient-corrupted chart exceeds the gate."""
        self.comparisons += 1
        self.assertGreater(
            self.eps_bad_f, FARFIELD_EPS_GATE,
            f'perturbed chart did not go red: eps {self.eps_bad_f:.3e} '
            f'<= gate {FARFIELD_EPS_GATE:g} -- the F-normalized currency is '
            f'not enforcing the injected error')

    def test_perturbation_is_the_load_bearing_change(self):
        """The mutation, not sampling noise, drives the failure: the bad
        eps is at least an order of magnitude above the healthy eps."""
        self.comparisons += 1
        self.assertGreater(
            self.eps_bad_f, 10.0 * self.eps_healthy_f,
            f'perturbed eps {self.eps_bad_f:.3e} is not clearly above the '
            f'healthy eps {self.eps_healthy_f:.3e}')

    def test_tiny_denominator_currency_would_thrash_a_good_chart(self):
        """Normalizing by ``max|E_ff| ~ 1e-4`` (the currency the gate does
        NOT use) drives the healthy chart to O(1) -- the reason F-norm is
        the right currency and does not thrash on a tiny denominator."""
        self.comparisons += 1
        self.assertGreater(
            self.eps_healthy_eff, EFFNORM_THRASH_MIN,
            f'Eff-normalized eps {self.eps_healthy_eff:.3e} did not thrash -- '
            f'the tiny-denominator hazard the F-norm avoids is not exhibited')


# ==========================================================================
# Build 8h-a / WP4: edge-annulus far-field tile subdivision
# (`surrogate_training._subdivide_farfield_tile`).
#
# A far-field tile whose held-out eps fails the registration bar is halved
# ONCE (single level, no recursion) into up to four children at
# ``(cx +/- h/2, cy +/- h/2)`` each with half ``h/2``.  Each child is
# re-admitted through the PARENT's OWN region predicate (carried verbatim,
# never re-derived from geometry): an exterior parent uses the min-corner
# exclusion test, an interior parent the max-corner containment test.  A
# disk-excluded child is DROPPED silently (recorded only in the subdivision
# summary, packed in neither the charts nor the failing-chart reports).  An
# admitted child retrains on the parent's inherited (already ppGO-trimmed)
# ``w_range`` and re-gates against the SAME ``farfield_eps_max`` bar: a
# passing child is packed, a still-failing child is recorded with its
# ``gate_reason`` but NOT packed.
#
# These classes drive the REAL ``_subdivide_farfield_tile`` on synthetic
# gated tiles with the four engine-backed helpers
# (`_build_farfield_chart`, `_farfield_heldout_samples`, `_heldout_eps`,
# `_load_or_build`) replaced by physics-free stand-ins, so the WP4 dispatch
# logic is exercised deterministically -- the engine's fit quality is a
# separate physics claim owned by the trainability classes above.  The gate
# arithmetic (`_gate_chart` / `_chart_gated`) and the admission geometry are
# the REAL production code, never stubbed.  `SubdivisionSelfFalsificationTest
# Case` proves the packing gate, the admission predicate and the exact-centre
# assertion each go red under a corruption.
# ==========================================================================

#: The production far-field registration bar the children are re-gated
#: against (`TrainingConfig.farfield_eps_max`, currently ``1e-3``).
_SUB_FARFIELD_BAR = surrogate_training.TrainingConfig().farfield_eps_max

#: Synthetic held-out eps a stand-in child reports: one an order of
#: magnitude BELOW the bar (packs) and one well ABOVE it (recorded, gated).
_SUB_PASS_EPS = 1.0e-4
_SUB_FAIL_EPS = 5.0e-3

#: Exterior parent straddling the exclusion disk: with half ``0.4`` and
#: ``exclusion_radius = 1.0`` the two inner children (min corner ``0.8``)
#: are disk-excluded and the two outer children (min corner ``1.2``) admit.
_EXT_PARENT_CENTER = (1.2, 0.0)
_EXT_PARENT_HALF = 0.4
_EXT_EXCLUSION_RADIUS = 1.0

#: Interior parent straddling the admit disk: with half ``0.2`` and
#: ``interior_admit_radius = 0.5`` only the origin-ward child (far corner
#: ``0.495``) admits; the other three (far corner ``>= 0.652``) are excluded.
_INT_PARENT_CENTER = (0.35, 0.35)
_INT_PARENT_HALF = 0.2
_INT_ADMIT_RADIUS = 0.5

#: A parent whose exterior/interior admission sets DIFFER, used to prove the
#: child admission is governed by the parent's region key, not re-derived
#: geometry.  Exterior (``exclusion_radius = 0.5``) excludes the two inner
#: children; interior (``interior_admit_radius = 1.5``) admits all four.
_FLIP_PARENT_CENTER = (0.6, 0.0)
_FLIP_PARENT_HALF = 0.4
_FLIP_EXCLUSION_RADIUS = 0.5
_FLIP_ADMIT_RADIUS = 1.5

#: The parent's already-ppGO-trimmed w-band, inherited verbatim by children.
_SUB_W_RANGE = (5.0, 60.0)


def _fake_heldout_samples(gamma_band, box_center, half, config, rng):
    """Patched stand-in: the sample list is irrelevant to the WP4 dispatch."""
    return []


def _make_fake_build(build_calls: list) -> object:
    """A `_build_farfield_chart` stand-in recording every child box it builds.

    Returns a lightweight ``SimpleNamespace`` chart (never touches the engine)
    tagged with the exact ``box_center`` / ``half`` / ``w_range`` it received,
    so the test can assert the children were built at ``(cx +/- h/2, cy +/-
    h/2)`` with half ``h/2`` on the inherited w-band.
    """
    def _fake_build_farfield_chart(*, gamma_band, parity, box_center, half,
                                   w_range, config):
        build_calls.append({'center': tuple(box_center), 'half': float(half),
                            'w_range': tuple(w_range)})
        chart = types.SimpleNamespace(
            image_count=2, parity=parity,
            refused_points=np.empty((0, 3), dtype=float),
            center=tuple(box_center), half=float(half),
            w_range=tuple(w_range))
        return chart, config.n_gamma * config.n_rho * config.n_theta_c, 0
    return _fake_build_farfield_chart


def _make_fake_eps(eps_for) -> object:
    """A `_heldout_eps` stand-in returning ``eps_for(child_center)``.

    ``eps_for`` maps a child centre to the synthetic held-out eps; return
    ``float('nan')`` to simulate a cancellation fail (zero served points).
    """
    def _fake_heldout_eps(chart, samples, provenance):
        return float(eps_for(chart.center))
    return _fake_heldout_eps


def _fake_load_or_build(path, build_fn, provenance):
    """A `_load_or_build` stand-in: run the build closure, skip the npz I/O.

    Mirrors the fresh-build branch of `surrogate_training._load_or_build`
    (report assembly, ``heldout_eps`` surfaced) so the REAL `_gate_chart`
    sees the same report shape, without a filesystem round-trip.
    """
    chart, calls, refused, report_extra = build_fn()
    report = {'engine_calls': int(calls), 'refused_points': int(refused),
              'build_seconds': 0.0, **report_extra}
    return chart, report, False


def _expected_children(center: tuple[float, float], half: float
                       ) -> list[dict]:
    """The four ``(cx +/- h/2, cy +/- h/2)`` child boxes in production order.

    Reproduces `_subdivide_farfield_tile`'s row-major loop (``sx`` outer in
    ``(-1, 1)``, ``sy`` inner) with the SAME float arithmetic, so the centres
    are bit-for-bit comparable to the ones the code builds.
    """
    cx, cy = center
    child_half = 0.5 * float(half)
    out: list[dict] = []
    ci = 0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            out.append({'ci': ci, 'half': child_half,
                        'center': (float(cx) + sx * child_half,
                                   float(cy) + sy * child_half)})
            ci += 1
    return out


def _run_subdivision(*, center: tuple[float, float], half: float,
                     region: str, eps_for,
                     exclusion_radius: float = 1.0e9,
                     interior_admit_radius: float = 1.0e9,
                     w_range: tuple[float, float] = _SUB_W_RANGE,
                     parent_tag: str = 'chart_pos_s0_ff_2_3',
                     si: int = 0, m_lo: float = 20.0, m_hi: float = 40.0
                     ) -> types.SimpleNamespace:
    """Drive the real `_subdivide_farfield_tile` on one synthetic gated tile.

    The four engine-backed helpers are patched on the ``surrogate_training``
    module (the globals the subdivision resolves at call time); spies on
    `_apply_ppgo_trim` and `_stratum_w_range` witness that NO per-child
    re-trim fires.  Returns a namespace bundling the subdivision summary, the
    (mutated in place) ``charts`` / ``chart_reports`` accumulators, the list
    of built child boxes, and the re-trim spy call counts.
    """
    tile = {'center': center, 'half': half, 'w_range': w_range,
            'region': region, 'si': si, 'm_lo': m_lo, 'm_hi': m_hi}
    charts: list = []
    chart_reports: list[dict] = []
    build_calls: list[dict] = []
    config = surrogate_training.TrainingConfig()
    ppgo_spy = mock.Mock(name='_apply_ppgo_trim')
    wrange_spy = mock.Mock(name='_stratum_w_range')
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        stack.enter_context(mock.patch.object(
            surrogate_training, '_build_farfield_chart',
            _make_fake_build(build_calls)))
        stack.enter_context(mock.patch.object(
            surrogate_training, '_farfield_heldout_samples',
            _fake_heldout_samples))
        stack.enter_context(mock.patch.object(
            surrogate_training, '_heldout_eps', _make_fake_eps(eps_for)))
        stack.enter_context(mock.patch.object(
            surrogate_training, '_load_or_build', _fake_load_or_build))
        stack.enter_context(mock.patch.object(
            surrogate_training, '_apply_ppgo_trim', ppgo_spy))
        stack.enter_context(mock.patch.object(
            surrogate_training, '_stratum_w_range', wrange_spy))
        summary = surrogate_training._subdivide_farfield_tile(
            tile=tile, parent_tag=parent_tag, band=(0.02, 0.06), parity=1,
            config=config, rng=np.random.default_rng(0),
            outdir=pathlib.Path(tmp), exclusion_radius=exclusion_radius,
            interior_admit_radius=interior_admit_radius,
            charts=charts, chart_reports=chart_reports)
    return types.SimpleNamespace(
        summary=summary, charts=charts, chart_reports=chart_reports,
        build_calls=build_calls, ppgo_calls=ppgo_spy.call_count,
        wrange_calls=wrange_spy.call_count, parent_tag=parent_tag,
        tile=tile, config=config)


def _plot_subdivision_tree(result: types.SimpleNamespace, region: str,
                           filename: str) -> None:
    """Diagnostic: parent box -> child boxes coloured by disposition."""
    if not _HAVE_MPL:
        return
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {'packed': 'C2', 'recorded_gated': 'C3',
              'disk_excluded': '0.7'}
    cx, cy = result.tile['center']
    half = result.tile['half']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half,
                               fill=False, edgecolor='k', lw=1.5,
                               label='parent tile'))
    for child in result.summary['children']:
        ccx, ccy = child['center']
        ch = child['half']
        res = child['result']
        ax.add_patch(plt.Rectangle(
            (ccx - ch, ccy - ch), 2 * ch, 2 * ch, alpha=0.5,
            facecolor=colors.get(res, 'C0'), edgecolor='k', lw=0.6))
        eps = child.get('eps')
        tag = (f"c{child['ci']}\n{res}"
               + ('' if eps is None else f"\neps={eps:.1e}"))
        ax.annotate(tag, (ccx, ccy), ha='center', va='center', fontsize=7)
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.margins(0.3)
    ax.set_title(f'WP4 edge-annulus subdivision ({region} parent)\n'
                 f'bar={_SUB_FARFIELD_BAR:g}')
    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / filename, dpi=110)
    plt.close(fig)


class ExteriorEdgeAnnulusSubdivisionTestCase(FarfieldEnvelopeTestCase):
    """WP4: an exterior gated tile halves, drops disk-excluded children, and
    packs/records the admitted ones by the re-gate outcome.

    Fixture (`_EXT_PARENT_CENTER`, half ``0.4``, ``exclusion_radius = 1.0``):
    the two inner children (``ci0``, ``ci1`` at ``y1 = 1.0``) are disk-
    excluded; the two outer children (``ci2``, ``ci3`` at ``y1 = 1.4``) admit.
    ``ci2`` (``y2 < 0``) is handed a passing eps and ``ci3`` (``y2 > 0``) a
    failing eps, so ONE admitted child packs and ONE is recorded gated -- both
    dispositions in a single run.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=_EXT_EXCLUSION_RADIUS,
            eps_for=lambda c: (_SUB_FAIL_EPS if c[1] > 0.0 else _SUB_PASS_EPS))
        cls.expected = _expected_children(_EXT_PARENT_CENTER, _EXT_PARENT_HALF)
        cls.tag = cls.result.parent_tag
        _plot_subdivision_tree(cls.result, 'exterior',
                               'farfield_subdivision_tree_exterior.png')

    def _child_summary(self, ci: int) -> dict:
        return next(c for c in self.result.summary['children']
                    if c['ci'] == ci)

    def _report_named(self, name: str) -> dict | None:
        return next((r for r in self.result.chart_reports
                     if r.get('name') == name), None)

    def test_children_halve_into_exact_quarter_boxes(self):
        """Every child sits at ``(cx +/- h/2, cy +/- h/2)`` with half ``h/2``.

        Guard (b): an off-by-half (using ``h`` instead of ``h/2``) still gates
        green by luck, so the centres and half-width are pinned EXACTLY.
        Admitted children are checked against the un-rounded box the builder
        received; all four against the 6-dp summary record.
        """
        built = {call['center']: call for call in self.result.build_calls}
        for child in self.expected:
            summ = self._child_summary(child['ci'])
            self.comparisons += 1
            self.assertEqual(
                summ['center'],
                [round(child['center'][0], 6), round(child['center'][1], 6)],
                f'child {child["ci"]} summary centre is not (cx+/-h/2, cy+/-h/2)')
            self.assertEqual(round(0.5 * _EXT_PARENT_HALF, 6), summ['half'],
                             f'child {child["ci"]} half-width is not h/2')
        for call in self.result.build_calls:
            self.comparisons += 1
            self.assertIn(
                call['center'], {ch['center'] for ch in self.expected},
                'a child was built at a centre other than (cx+/-h/2, cy+/-h/2)')
            self.assertEqual(0.5 * _EXT_PARENT_HALF, call['half'],
                             'a child was built with a half-width != h/2')

    def test_disk_excluded_children_absent_from_packed_and_recorded(self):
        """Guard (a): the two inner children are dropped -- present in NEITHER
        the packed charts NOR the recorded-failure reports (only the summary).
        """
        for ci in (0, 1):
            summ = self._child_summary(ci)
            self.comparisons += 1
            self.assertEqual('disk_excluded', summ['admission'])
            self.assertEqual('disk_excluded', summ['result'])
            self.assertIsNone(self._report_named(f'{self.tag}_c{ci}'),
                              f'disk-excluded child c{ci} leaked into reports')
            self.assertNotIn(
                (float(summ['center'][0]), float(summ['center'][1])),
                {ch.center for ch in self.result.charts},
                f'disk-excluded child c{ci} leaked into packed charts')

    def test_passing_child_is_packed(self):
        """The admitted child under the bar is packed into ``charts`` and
        recorded WITHOUT a gated marker."""
        self.comparisons += 1
        self.assertEqual(1, len(self.result.charts),
                         'exactly one child should have packed')
        packed = self.result.charts[0]
        self.assertEqual(self.expected[2]['center'], packed.center,
                         'the packed child is not ci2')
        report = self._report_named(f'{self.tag}_c2')
        self.assertIsNotNone(report)
        self.assertNotIn('gated', report)
        self.assertEqual(self.tag, report['subdivided_from'])
        self.assertEqual(self._child_summary(2)['result'], 'packed')

    def test_failing_child_is_recorded_gated_not_packed(self):
        """Guard: the admitted child above the bar is recorded with its
        gate-reason but NOT packed into ``charts``."""
        report = self._report_named(f'{self.tag}_c3')
        self.comparisons += 1
        self.assertIsNotNone(report, 'the failing child was not recorded')
        self.assertTrue(report.get('gated'))
        self.assertEqual('eps_above_bar', report['gate_reason'])
        self.assertEqual(self.tag, report['subdivided_from'])
        self.assertNotIn(self.expected[3]['center'],
                         {ch.center for ch in self.result.charts},
                         'the failing child was packed despite gating')
        self.assertEqual(self._child_summary(3)['result'], 'recorded_gated')

    def test_accuracy_fail_reason_is_eps_above_bar(self):
        """Guard (d): an accuracy fail carries ``'eps_above_bar'`` so the
        serving ladder routes it correctly.

        (The Architect brief calls this reason ``heldout_eps``; the production
        `_chart_gated` returns ``'eps_above_bar'`` for an eps-over-bar fail,
        which is the string asserted here -- verified against source.)
        """
        self.comparisons += 1
        self.assertEqual('eps_above_bar',
                         self._child_summary(3)['gate_reason'])
        self.assertEqual(_SUB_FARFIELD_BAR, self._child_summary(3)['bar'])

    def test_children_inherit_parent_w_range_verbatim(self):
        """The children retrain on the parent's already-trimmed ``w_range``;
        no per-child ``_stratum_w_range`` / ``_apply_ppgo_trim`` recompute."""
        self.comparisons += 1
        self.assertEqual(0, self.result.ppgo_calls,
                         '_apply_ppgo_trim was called during subdivision')
        self.assertEqual(0, self.result.wrange_calls,
                         '_stratum_w_range was called during subdivision')
        for call in self.result.build_calls:
            self.comparisons += 1
            self.assertEqual(_SUB_W_RANGE, call['w_range'],
                             'a child did not inherit the parent w_range')

    def test_child_tags_are_row_major_and_carry_provenance(self):
        """Admitted children are tagged ``{parent}_c{ci}`` in row-major order
        with a ``subdivided_from`` back-reference."""
        names = [r['name'] for r in self.result.chart_reports]
        self.comparisons += 1
        self.assertEqual([f'{self.tag}_c2', f'{self.tag}_c3'], names)
        for report in self.result.chart_reports:
            self.assertEqual(self.tag, report['subdivided_from'])

    def test_no_second_halving_depth_is_one(self):
        """Guard (c): a still-failing child spawns NO grandchildren -- exactly
        one build per admitted child and no ``{parent}_c#_c#`` report."""
        self.comparisons += 1
        self.assertEqual(2, len(self.result.build_calls),
                         'more builds than admitted children -> recursion')
        for report in self.result.chart_reports:
            self.assertNotRegex(
                report['name'], r'_c\d+_c\d+',
                'a grandchild tag appeared -> a second halving fired')

    def test_summary_records_all_four_children_with_census_fields(self):
        """The subdivision summary carries every child (packed, recorded, and
        disk-excluded) with the census fields the ladder attributes."""
        children = self.result.summary['children']
        self.comparisons += 1
        self.assertEqual(4, len(children))
        self.assertEqual('exterior', self.result.summary['region'])
        self.assertEqual([0, 1, 2, 3], [c['ci'] for c in children])
        for ci in (2, 3):
            summ = self._child_summary(ci)
            for key in ('eps', 'bar', 'gate_reason', 'result'):
                self.assertIn(key, summ,
                              f'child c{ci} summary missing {key!r}')


class InteriorEdgeAnnulusSubdivisionTestCase(FarfieldEnvelopeTestCase):
    """WP4: an INTERIOR gated tile re-admits children by the parent's
    max-corner containment predicate, packs the admitted one, and drops the
    disk-excluded ones.

    Fixture (`_INT_PARENT_CENTER`, half ``0.2``, ``interior_admit_radius =
    0.5``): only the origin-ward child ``ci0`` (far corner ``0.495``) lies
    wholly inside the admit disk; ``ci1``/``ci2``/``ci3`` (far corner ``>=
    0.652``) straddle it and are dropped.  The admitted child is handed a
    passing eps.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _run_subdivision(
            center=_INT_PARENT_CENTER, half=_INT_PARENT_HALF,
            region='interior', interior_admit_radius=_INT_ADMIT_RADIUS,
            eps_for=lambda c: _SUB_PASS_EPS)
        cls.tag = cls.result.parent_tag
        _plot_subdivision_tree(cls.result, 'interior',
                               'farfield_subdivision_tree_interior.png')

    def _child_summary(self, ci: int) -> dict:
        return next(c for c in self.result.summary['children']
                    if c['ci'] == ci)

    def test_interior_admission_uses_max_corner_predicate(self):
        """Only the origin-ward child is admitted; the three edge children are
        disk-excluded by the ``hypot(|cx|+h, |cy|+h) <= admit_radius`` test."""
        self.comparisons += 1
        self.assertEqual('admitted', self._child_summary(0)['admission'])
        for ci in (1, 2, 3):
            self.assertEqual('disk_excluded',
                             self._child_summary(ci)['admission'],
                             f'child c{ci} should straddle the admit disk')

    def test_admitted_interior_child_packed_with_interior_region(self):
        """The admitted child packs and its report carries the parent's
        ``interior`` region key (never re-derived to exterior)."""
        self.comparisons += 1
        self.assertEqual(1, len(self.result.charts))
        report = next(r for r in self.result.chart_reports
                      if r['name'] == f'{self.tag}_c0')
        self.assertEqual('interior', report['region'])
        self.assertNotIn('gated', report)
        self.assertEqual('interior', self.result.summary['region'])

    def test_disk_excluded_interior_children_dropped(self):
        """The three straddling children appear in neither packed nor
        recorded lists."""
        names = {r['name'] for r in self.result.chart_reports}
        self.comparisons += 1
        for ci in (1, 2, 3):
            self.assertNotIn(f'{self.tag}_c{ci}', names)
        self.assertEqual([f'{self.tag}_c0'],
                         [r['name'] for r in self.result.chart_reports])

    def test_interior_child_inherits_parent_w_range(self):
        """The admitted interior child retrains on the inherited w_range with
        no per-child re-trim."""
        self.comparisons += 1
        self.assertEqual(0, self.result.ppgo_calls)
        self.assertEqual(0, self.result.wrange_calls)
        self.assertEqual(1, len(self.result.build_calls))
        self.assertEqual(_SUB_W_RANGE, self.result.build_calls[0]['w_range'])


class CancellationChildSubdivisionTestCase(FarfieldEnvelopeTestCase):
    """WP4 guard (d): a child that re-nans (a genuine cancellation in the same
    parity/gamma cell) is recorded with ``'nan_eps'`` -- not special-cased,
    not packed.  Halving cannot fix a cancellation, so both admitted children
    of the exterior fixture are handed a NaN eps.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=_EXT_EXCLUSION_RADIUS,
            eps_for=lambda c: float('nan'))
        cls.tag = cls.result.parent_tag

    def test_cancellation_children_carry_nan_eps_reason(self):
        """Both admitted children are gated with reason ``'nan_eps'``."""
        for ci in (2, 3):
            report = next(r for r in self.result.chart_reports
                          if r['name'] == f'{self.tag}_c{ci}')
            self.comparisons += 1
            self.assertTrue(report.get('gated'))
            self.assertEqual('nan_eps', report['gate_reason'])

    def test_nan_eps_children_are_not_packed(self):
        """A cancellation fail leaves ``charts`` empty; the windows fall to the
        serving ladder (never numerical quadrature)."""
        self.comparisons += 1
        self.assertEqual(0, len(self.result.charts),
                         'a NaN-eps child was packed despite gating')
        gated = [r for r in self.result.chart_reports if r.get('gated')]
        self.assertEqual(2, len(gated))
        for report in gated:
            self.assertTrue(np.isnan(report['heldout_eps']),
                            'a cancellation child recorded a finite eps')


class RegionKeyConsistencySubdivisionTestCase(FarfieldEnvelopeTestCase):
    """WP4 guard (e): the child admission is governed by the PARENT's region
    key, never re-derived from the child's geometry.

    The SAME parent box (`_FLIP_PARENT_CENTER`, half ``0.4``) is subdivided
    twice.  As an EXTERIOR parent (``exclusion_radius = 0.5``) the two inner
    children are disk-excluded; as an INTERIOR parent (``interior_admit_radius
    = 1.5``) all four admit.  A single fixed predicate could not produce both
    admission sets, so the differing outcome proves the region key drives the
    decision.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.exterior = _run_subdivision(
            center=_FLIP_PARENT_CENTER, half=_FLIP_PARENT_HALF,
            region='exterior', exclusion_radius=_FLIP_EXCLUSION_RADIUS,
            eps_for=lambda c: _SUB_PASS_EPS)
        cls.interior = _run_subdivision(
            center=_FLIP_PARENT_CENTER, half=_FLIP_PARENT_HALF,
            region='interior', interior_admit_radius=_FLIP_ADMIT_RADIUS,
            eps_for=lambda c: _SUB_PASS_EPS)

    @staticmethod
    def _admissions(result) -> list[str]:
        return [c['admission'] for c in
                sorted(result.summary['children'], key=lambda c: c['ci'])]

    def test_region_governs_admission_not_geometry(self):
        """Identical geometry -> different admission sets under the two region
        keys (exterior excludes two children, interior admits all four)."""
        self.comparisons += 1
        self.assertEqual(
            ['disk_excluded', 'disk_excluded', 'admitted', 'admitted'],
            self._admissions(self.exterior))
        self.assertEqual(['admitted'] * 4, self._admissions(self.interior))
        self.assertNotEqual(self._admissions(self.exterior),
                            self._admissions(self.interior),
                            'the region key did not change the admission set')

    def test_summaries_carry_the_parent_region(self):
        """Each summary reports the parent's own region string."""
        self.comparisons += 1
        self.assertEqual('exterior', self.exterior.summary['region'])
        self.assertEqual('interior', self.interior.summary['region'])

    def test_child_reports_never_flip_region(self):
        """Every admitted child inherits the parent's region key verbatim."""
        self.comparisons += 1
        for report in self.exterior.chart_reports:
            self.assertEqual('exterior', report['region'])
        for report in self.interior.chart_reports:
            self.assertEqual('interior', report['region'])


class SubdivisionSelfFalsificationTestCase(FarfieldEnvelopeTestCase):
    """Prove the WP4 gates can each go red: the packing gate, the admission
    predicate, and the exact-centre assertion.
    """

    def test_lowering_eps_flips_recorded_to_packed(self):
        """The packed-vs-recorded split is driven by the eps gate, not luck:
        the SAME failing child (ci3), given an eps below the bar, now packs.
        """
        baseline = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=_EXT_EXCLUSION_RADIUS,
            eps_for=lambda c: (_SUB_FAIL_EPS if c[1] > 0.0 else _SUB_PASS_EPS))
        healed = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=_EXT_EXCLUSION_RADIUS,
            eps_for=lambda c: _SUB_PASS_EPS)
        self.comparisons += 1
        self.assertEqual(1, len(baseline.charts),
                         'baseline should pack only the passing child')
        self.assertEqual(2, len(healed.charts),
                         'lowering the failing eps below the bar did not pack '
                         'the previously-recorded child -- the gate has no '
                         'teeth')

    def test_shrinking_exclusion_admits_a_previously_excluded_child(self):
        """The admission predicate has teeth: a smaller exclusion radius admits
        an inner child that was disk-excluded before."""
        excluded = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=_EXT_EXCLUSION_RADIUS,
            eps_for=lambda c: _SUB_PASS_EPS)
        admitted = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=0.5,
            eps_for=lambda c: _SUB_PASS_EPS)
        n_excluded = sum(c['admission'] == 'disk_excluded'
                         for c in excluded.summary['children'])
        n_admitted = sum(c['admission'] == 'disk_excluded'
                         for c in admitted.summary['children'])
        self.comparisons += 1
        self.assertEqual(2, n_excluded)
        self.assertLess(n_admitted, n_excluded,
                        'shrinking the exclusion radius did not admit any '
                        'previously-excluded child -- no teeth')

    def test_off_by_half_center_would_be_caught(self):
        """The exact-centre assertion is load-bearing: the code centres children
        at ``cx +/- h/2`` (half ``h/2``), NOT the off-by-half ``cx +/- h``."""
        result = _run_subdivision(
            center=_EXT_PARENT_CENTER, half=_EXT_PARENT_HALF,
            region='exterior', exclusion_radius=_EXT_EXCLUSION_RADIUS,
            eps_for=lambda c: _SUB_PASS_EPS)
        built = {call['center'] for call in result.build_calls}
        cx, cy = _EXT_PARENT_CENTER
        off_by_half = {(cx + sx * _EXT_PARENT_HALF, cy + sy * _EXT_PARENT_HALF)
                       for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)}
        self.comparisons += 1
        self.assertTrue(built, 'no child was built')
        self.assertFalse(built & off_by_half,
                         'a child was built at cx+/-h (off-by-half) -- the '
                         'exact-centre guard would not catch the bug')


if __name__ == '__main__':
    main()
