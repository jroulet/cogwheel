"""Invariants of the beat-free tube demodulation reference ``F_ref``.

This suite pins the three properties the Architect specified for the
beat-free residual tube surrogate, whose single authoritative reference is
:func:`cogwheel.lensing.surrogate._tube_f_ref`.  The build stores the smooth
residual ``r = E / F_ref`` and the serve recovers ``E = r * F_ref`` at the
query source; ``F_ref`` is the uniform-Airy fold form built from the SAME
merging fold pair the SACR-C envelope beats over, re-referenced to the
critical carrier ``tau_c = virtual_delay - t_min`` and evaluated with the
Airy-derivative amplitude ``q = p`` (NOT the shipped ``q = 0``).

1.  **NON-VANISHING REFERENCE (safety property).**  ``|F_ref|`` never dips to
    zero -- not even at the Airy zeros ``xi = 2.338, 4.088, 5.521`` where the
    ``q = 0`` form vanishes -- so the serve-side division ``E / F_ref`` never
    hits a carrier zero.  With ``q = p`` the Airy Wronskian makes ``|F_ref|^2
    proportional to w^{1/3} Ai^2 + w^{-1/3} Ai'^2``, strictly positive.  A
    regression to the shipped ``q = 0`` would trip the strict floor.

2.  **ROUND-TRIP ENVELOPE IDENTITY (wiring property).**  Re-modulating the
    stored residual by ``F_ref`` recomputed at the node source returns the
    stored engine envelope exactly, and the SACR-C reconstruction of that
    served envelope reproduces the exact total to the machine floor.  A
    carrier/frame mismatch (``F_ref`` referenced to ``tau_bar`` instead of
    ``tau_c``) would leave a ``w``-dependent fringe.

3.  **DRY: BUILDER CALLS ``_merging_fold_pair``.**  ``F_ref`` reads its delay
    separation ``Delta_tau`` from the shared
    :func:`~cogwheel.lensing.chang_refsdal._airy_fold._merging_fold_pair`
    rather than re-deriving it; the reference is invoked once per build and
    perturbing its return measurably moves ``F_ref``.

4.  **D2 INVARIANCE OF ``F_ref`` (serve-at-raw-eigenframe correctness).**  The
    macro amplification is even in each eigenframe source component, so
    ``F_ref`` is invariant under the three non-trivial D2 reflections
    ``(-y1, y2)``, ``(y1, -y2)``, ``(-y1, -y2)``.  The serve decision to
    re-modulate at the RAW eigenframe source (no theta-fold) rests on this;
    a refactor breaking D2 would corrupt the served envelope silently.  The
    reflected reference matches the raw one to ``1e-11`` relative (delays are
    computed, not analytic), and the engine envelope matches to ``1e-12`` so
    the residual ``r = E / F_ref`` inherits the same D2.

5.  **LARGE-xi CALIBRATION LIMIT (sigma/xi factor-of-two lock).**  On a
    well-resolved fold pair (``w Delta_tau >> 1``) the uniform-Airy
    ``F_ref`` approaches its stationary-phase asymptote.  Confronted against
    an INDEPENDENT closed-form asymptote ``S_asymp`` (built from the Airy
    large-argument expansion, NOT from production), the ratio
    ``F_ref / S_asymp`` tends to a single ``w``-independent complex constant;
    a broken ``sigma`` or a ``xi`` factor-of-two flips the ``zeta`` phase and
    destroys that constancy.

6.  **TUBE-ARC BUILDABILITY (four-band fast smoke).**  Each of the four
    Professor bands -- astroid small-gamma (incl. ``gamma ~ 0.045``), astroid
    large-gamma, saddle ``gamma = 1.2``, and ``gamma = 0.4`` -- must build a
    MINIMAL tube chart on ``arcs[0]`` and serve a handful of in-band held-out
    queries without error, covering at least one node and returning a finite
    served envelope.  This guards against a band whose arcs drift below four
    images or refuse everywhere; it is a fast SMOKE invariant, NOT the eps
    accuracy sweep.

Push-back / substitution (spec 5, LARGE-xi)
-------------------------------------------
The Architect's literal ``F_ref_geo = exp(-i w tau_c) [sqrt|mu_+| exp(i w
tau_+) + sqrt|mu_-| exp(i (w tau_- - pi/2))]`` is the ``q = 0`` geometric
two-image sum; the shipped ``F_ref`` uses ``q = p`` (Airy-derivative
amplitude), whose large-xi asymptote carries the DISTINCT amplitude pair
``C1 = (3 Delta_tau / 4)**(-1/6)`` and ``C2 = (3 Delta_tau / 4)**(1/6)``
(measured ~2.7x apart), plus a global normalization/phase the residual ``r``
absorbs.  A direct ``F_ref == F_ref_geo`` pin therefore FAILS by construction
on correct code (it certifies a convention production does not use).  The
invariant that genuinely locks the ``sigma``/``xi`` factor-of-two -- and is
what the description is really after -- is that ``F_ref / S_asymp`` is a
single ``w``-independent constant across ``w Delta_tau in [40, 200]``, with
``S_asymp`` the ``q = p`` stationary-phase asymptote derived here from the
Airy large-argument expansion (``Ai(-xi) ~ pi**-1/2 xi**-1/4 cos(zeta -
pi/4)``, ``Ai'(-xi) ~ pi**-1/2 xi**1/4 sin(zeta - pi/4)``,
``zeta = (2/3) xi**3/2 = w Delta_tau / 2``).  The factor-of-two foil (halving
``zeta``) breaks the constancy by ``O(10)``, giving the test its teeth.

Independent oracles
-------------------
The NON-VANISHING floor is confronted against an INDEPENDENT
`scipy.special.airy` evaluation of ``Ai`` / ``Ai'`` at the exact Airy zeros
(never a re-transcription of the production bracket).  The ROUND-TRIP oracle
is the shipping engine partition (`ChangRefsdalChannels.evaluate`) and its
``exact_total`` -- the served envelope is rebuilt through the shipping
`reconstruct_from_envelope`.  The DRY oracle is the shipping
`_merging_fold_pair` itself, spied via ``mock`` on the binding
``surrogate._tube_f_ref`` actually calls.

Patch-target note (load-bearing)
--------------------------------
`surrogate.py` binds ``_merging_fold_pair`` into its own namespace at import
(``from ..._airy_fold import (..., _merging_fold_pair, ...)``), so the call
inside ``_tube_f_ref`` resolves to ``surrogate._merging_fold_pair`` -- the
DRY spy patches THAT binding.  Patching ``_airy_fold._merging_fold_pair``
would be a silent no-op for ``_tube_f_ref`` and give a false-green DRY test.

Tolerances
----------
``_ROUNDTRIP_RTOL = 1e-12`` (spec 2): the SACR-C reconstruction algebra is
exact for any envelope, so the only error is float64 round-off; ``1e-12`` is a
self-falsifying margin (a wrong-frame carrier is a whole-number relative
error).  ``_NONVANISH_FLOOR_FRAC = 1e-8`` (spec 1): a deliberately loose
"never near zero" guard -- the ``q = p`` reference clears it by many orders
while the ``q = 0`` regression lands exactly on zero.
"""
from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import os
import unittest
from unittest import mock

import numpy as np
from scipy.special import airy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing import surrogate as sg
from cogwheel.lensing.surrogate import _tube_f_ref, LensAmplificationSurrogate
from cogwheel.lensing.chang_refsdal._airy_fold import (
    airy_fold_value, _merging_fold_pair)
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, reconstruct_from_envelope, _frame_delays)
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.surrogate_training import (
    TrainingConfig, band_caustic_structure, _tube_training_arcs,
    _min_curvature_radius, _tube_source, _build_tube_chart,
    _tube_heldout_samples, _ENGINE_REFUSALS)

#: Gamma band + representative gamma whose astroid tube arc supplies the
#: derived real 4-image interior node (mirrors the sibling nyquist suite).
_BAND: tuple[float, float] = (0.37, 0.43)
#: Representative convergence ratio inside ``_BAND``.
_GAMMA: float = 0.4
#: Caustic-sampling resolution for structure detection / curvature radius.
_N_SAMPLES: int = 200

#: First three zeros of ``Ai`` (magnitudes), where ``Ai(-xi) = 0`` -- the
#: exact loci at which the shipped ``q = 0`` reference vanishes.
_AIRY_ZEROS: tuple[float, ...] = (2.33810741045976703849,
                                  4.08794944413097061664,
                                  5.52055982809555105913)

#: Strict non-vanishing floor fraction (spec 1): assert
#: ``abs(F_ref) >= _NONVANISH_FLOOR_FRAC * (2 sqrt(pi) p w**(1/6))``.
_NONVANISH_FLOOR_FRAC: float = 1e-8

#: Relative tolerance for the round-trip / reconstruction identities (spec 2).
_ROUNDTRIP_RTOL: float = 1e-12

#: Relative move a perturbed ``_merging_fold_pair`` must induce in ``F_ref``.
_DRY_PERTURB_MIN: float = 1e-6

#: Output directory for diagnostic plots.
_OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), 'output')

#: Minimum magnitude of BOTH eigenframe source components for the off-axis
#: D2 / large-xi node -- an on-axis node makes some D2 reflections coincide
#: with the raw source, hollowing out the invariant.
_OFFAXIS_MIN_COMP: float = 0.10

#: Relative tolerance for D2 invariance of ``F_ref`` (spec 4).  Delays are
#: computed (not analytic), so the reflection agreement floors near ``1e-11``.
_D2_FREF_RTOL: float = 1e-11

#: Relative tolerance for the paired engine-envelope D2 identity (spec 4).
_D2_ENV_RTOL: float = 1e-12

#: A generic (non-D2) source rotation, in radians, used as the D2 self-
#: falsification foil -- small enough to keep the node in the 4-image
#: interior yet large enough to move ``F_ref`` by ``O(0.1)``.
_D2_ROTATION_ANGLE: float = 0.05

#: Minimum relative move the generic rotation must induce (self-falsification).
_D2_ROTATION_MIN_MOVE: float = 1e-2

#: ``w Delta_tau`` window over which ``F_ref / S_asymp`` must be constant
#: (spec 5).  Lower bound >= 40 keeps the pair well-resolved; the reference is
#: engine-free so ``w`` may run far past the DD engine's fast band.
_LARGEXI_WDT_RANGE: tuple[float, float] = (40.0, 200.0)

#: Number of ``w`` nodes across ``_LARGEXI_WDT_RANGE``.
_LARGEXI_N_W: int = 24

#: Tolerance on the coefficient of variation of ``F_ref / S_asymp`` (spec 5).
_LARGEXI_RATIO_TOL: float = 1.5e-2

#: Four Professor bands for the buildability smoke (spec 6):
#: ``(band, parity)`` -- astroid small-gamma, astroid large-gamma, saddle,
#: and the representative ``gamma = 0.4`` astroid band.
_BUILDABILITY_BANDS: tuple[tuple[tuple[float, float], int], ...] = (
    ((0.03, 0.06), 1),
    ((0.70, 0.80), 1),
    ((1.10, 1.30), -1),
    ((0.37, 0.43), 1),
)


def _find_offaxis_interior_node(
        min_comp: float = _OFFAXIS_MIN_COMP,
        ) -> tuple[np.ndarray, float, np.ndarray]:
    """Derive a real 4-image interior node OFF both eigenframe axes.

    Like :func:`_find_interior_node` but rejects near-axis nodes (where a D2
    reflection would collide with the raw source), returning the largest-
    ``Delta_tau`` 4-image node whose both source components exceed
    ``min_comp``.  Needed for the D2 and large-xi specs, which are hollow on
    an on-axis source.
    """
    matrix = geometry.macro_matrix(_GAMMA)
    config = TrainingConfig()
    structure = band_caustic_structure(_BAND, 1, n_samples=_N_SAMPLES)
    arc = _tube_training_arcs(structure, 1)[0]
    r_min = _min_curvature_radius(_BAND, arc, _N_SAMPLES)
    eta_max = config.f_max * r_min

    thetas = np.linspace(arc.theta_lo, arc.theta_hi, _N_SAMPLES)
    best_source: np.ndarray | None = None
    best_gap = -math.inf
    for theta in thetas:
        source = _tube_source(_GAMMA, float(theta), eta_max, arc.branch,
                              arc.inward_sign)
        if min(abs(float(source[0])), abs(float(source[1]))) < min_comp:
            continue
        try:
            images = geometry.find_images(source, matrix)
        except geometry.LensDomainError:
            continue
        if len(images) != 4:
            continue
        pair = _merging_fold_pair(images, source, matrix)
        if pair is None:
            continue
        gap = pair[1] - pair[0]
        if gap > best_gap:
            best_gap = gap
            best_source = source
    if best_source is None:
        raise AssertionError(
            'fixture premise lost: no off-axis (both |y| > '
            f'{min_comp}) 4-image interior node found on the gamma=0.4 arc.')
    return best_source, eta_max, matrix


def _geometric_fold_inputs(
        gamma: float, source: np.ndarray,
        ) -> tuple[float, float, float]:
    """Extract ``(Delta_tau, tau_bar, critical_delay)`` from live geometry.

    Replicates ONLY the geometric-delay extraction of `_tube_f_ref` (image
    finding, the merging fold pair, and the ``t_min``-relative carriers) --
    NOT the uniform-Airy amplitude form under test.  These are the delay
    INPUTS the asymptote oracle shares with production; the property being
    tested (the ``sigma``/``xi`` factor-of-two convention) lives entirely in
    how the reference turns them into an oscillation, which the oracle
    reconstructs from the Airy large-argument expansion independently.
    """
    matrix = geometry.macro_matrix(float(gamma), 0.0, 0.0)
    images, _absolute, t_min = _frame_delays(source, matrix)
    caustic = geometry.nearest_caustic_point(float(gamma), 0.0, source,
                                             kappa=0.0)
    virtual_delay = geometry.delay(caustic.image, source, matrix)
    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        raise AssertionError('fixture premise lost: no merging fold pair.')
    tau_plus, tau_minus = pair
    delta_tau = tau_minus - tau_plus
    tau_bar = 0.5 * (tau_plus + tau_minus) - t_min
    critical_delay = virtual_delay - t_min
    return float(delta_tau), float(tau_bar), float(critical_delay)


def _stationary_phase_asymptote(
        w_grid: np.ndarray, delta_tau: float, tau_bar: float,
        critical_delay: float, zeta_factor: float = 1.0) -> np.ndarray:
    """Independent ``q = p`` stationary-phase asymptote ``S_asymp(w)``.

    Built from the Airy large-argument expansion (NOT from production):

        Ai(-xi)  ~ pi^{-1/2} xi^{-1/4} cos(zeta - pi/4)
        Ai'(-xi) ~ pi^{-1/2} xi^{ 1/4} sin(zeta - pi/4)
        zeta = (2/3) xi^{3/2} = w Delta_tau / 2 ,  xi = (3 w Delta_tau/4)^{2/3}

    so with ``q = p`` the ``w`` powers cancel and

        S_asymp = [C1 cos(zeta - pi/4) - i C2 sin(zeta - pi/4)]
                  * exp(i w (tau_bar - critical_delay)) ,

    with ``C1 = (3 Delta_tau/4)^{-1/6}``, ``C2 = (3 Delta_tau/4)^{1/6}``.  The
    production ``F_ref`` differs from this only by the ``w``-independent
    complex constant ``2 p exp(i sigma)`` (absorbed by the residual).
    ``zeta_factor`` scales ``zeta`` for the factor-of-two self-falsification
    foil.
    """
    w_grid = np.asarray(w_grid, dtype=float)
    xi = (3.0 * w_grid * delta_tau / 4.0) ** (2.0 / 3.0)
    zeta = zeta_factor * (2.0 / 3.0) * xi ** 1.5
    c1 = (3.0 * delta_tau / 4.0) ** (-1.0 / 6.0)
    c2 = (3.0 * delta_tau / 4.0) ** (1.0 / 6.0)
    bracket = (c1 * np.cos(zeta - math.pi / 4.0)
               - 1j * c2 * np.sin(zeta - math.pi / 4.0))
    return bracket * np.exp(1j * w_grid * (tau_bar - critical_delay))


def _find_interior_node() -> tuple[np.ndarray, float, np.ndarray]:
    """Derive a real 4-image interior tube source with a defined ``F_ref``.

    Scans the production gamma=0.4 astroid tube arc for the interior theta at
    which the merging fold pair resolves most robustly (largest ``Delta_tau``)
    and returns ``(source, eta_max, matrix)``.  Deriving the node from the
    live boundary -- rather than pinning a literal source -- keeps the fixture
    valid if the caustic geometry shifts.
    """
    matrix = geometry.macro_matrix(_GAMMA)
    config = TrainingConfig()
    structure = band_caustic_structure(_BAND, 1, n_samples=_N_SAMPLES)
    arc = _tube_training_arcs(structure, 1)[0]
    r_min = _min_curvature_radius(_BAND, arc, _N_SAMPLES)
    eta_max = config.f_max * r_min

    thetas = np.linspace(arc.theta_lo, arc.theta_hi, _N_SAMPLES)
    best_source: np.ndarray | None = None
    best_gap = -math.inf
    for theta in thetas:
        source = _tube_source(_GAMMA, float(theta), eta_max, arc.branch,
                              arc.inward_sign)
        try:
            images = geometry.find_images(source, matrix)
        except geometry.LensDomainError:
            continue
        if len(images) != 4:
            continue
        pair = _merging_fold_pair(images, source, matrix)
        if pair is None:
            continue
        gap = pair[1] - pair[0]
        if gap > best_gap:
            best_gap = gap
            best_source = source
    if best_source is None:
        raise AssertionError(
            'fixture premise lost: no robust 4-image interior tube node found '
            'on the gamma=0.4 astroid arc.')
    return best_source, eta_max, matrix


class _TubeBeatFreeTestCase(unittest.TestCase):
    """Shared real 4-image interior node + anti-vacuity bookkeeping.

    ``setUpClass`` derives the node once and asserts the domain premise (the
    real `_tube_f_ref` returns a finite non-``None`` reference on it).  An
    anti-vacuity ``tearDown`` fails any test that made zero comparisons, so a
    silently-skipping body can never read green.
    """

    #: Incremented by each test; ``tearDown`` fails the test if it stays zero.
    comparisons: int

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        cls.source, cls.eta_max, cls.matrix = _find_interior_node()
        #: Tube-band engine grid.  Capped at ``w = 60`` so
        #: `ChangRefsdalChannels.evaluate` stays on the exact DD path (``w >
        #: 60`` routes to the slow mpmath path, ``w > 150`` hard-refuses);
        #: ``F_ref`` itself is cheap at any ``w``, but the round-trip oracle
        #: is the engine, so the shared grid honours the engine's fast band.
        cls.w_grid = np.geomspace(2.0, 60.0, 24)
        fref = _tube_f_ref(cls.w_grid, _GAMMA, cls.source)
        if fref is None or not np.all(np.isfinite(fref)):
            raise AssertionError(
                'fixture premise lost: _tube_f_ref refuses the derived '
                'interior node -- the beat-free reference is undefined there.')
        cls.fref = fref

        # Shipping engine partition at the same node: the round-trip oracle.
        partition = ChangRefsdalChannels(cls.w_grid).evaluate(
            gamma=_GAMMA, y=(float(cls.source[0]), float(cls.source[1])),
            beta=0.0, kappa=0.0)
        cls.partition = partition
        cls.envelope = np.asarray(partition.envelope)
        cls.f_scale = float(np.max(np.abs(partition.exact_total)))
        if len(partition.images) != 4:
            raise AssertionError(
                'fixture premise lost: the engine reports a non-interior '
                f'({len(partition.images)}-image) node.')

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no comparisons executed -- test body was vacuous.')


class NonVanishingReferenceTestCase(_TubeBeatFreeTestCase):
    """Spec 1: ``|F_ref|`` clears a strict positive floor everywhere.

    The synthetic ``(w, xi)`` grid deliberately lands ``xi`` ON the Airy zeros
    where ``Ai(-xi) = 0``; with ``q = p`` the reference survives on its
    ``Ai'`` term, so ``|F_ref| >= _NONVANISH_FLOOR_FRAC * (2 sqrt(pi) p
    w**(1/6))`` at every node.  The real `_tube_f_ref` on the derived node is
    checked against the same floor built from its own inner amplitude.
    """

    #: Synthetic amplitudes: |F_ref| scales linearly with ``p`` and is
    #: independent of ``tau_bar``/``sigma`` (pure carrier phase), so a clean
    #: unit amplitude with zero phases isolates the magnitude invariant.
    _P: float = 1.0
    _TAU_BAR: float = 0.0
    _SIGMA: float = 0.0

    def _synthetic_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Frequency and xi nodes; xi includes the exact Airy zeros."""
        w_nodes = np.geomspace(2.0, 400.0, 12)
        # Airy zeros plus a few generic interior/near-zero controls.
        xi_nodes = np.array(
            sorted(set(_AIRY_ZEROS)
                   | {0.0, 0.5, 1.5, 3.0, 4.5, 6.0}), dtype=float)
        return w_nodes, xi_nodes

    def test_synthetic_reference_clears_floor_at_airy_zeros(self) -> None:
        w_nodes, xi_nodes = self._synthetic_grid()
        for w, xi in itertools.product(w_nodes, xi_nodes):
            with self.subTest(w=w, xi=xi):
                value = airy_fold_value(
                    float(w), self._TAU_BAR, float(xi),
                    self._P, self._P, self._SIGMA)
                floor = (_NONVANISH_FLOOR_FRAC
                         * 2.0 * math.sqrt(math.pi) * self._P
                         * float(w) ** (1.0 / 6.0))
                self.comparisons += 1
                self.assertGreaterEqual(
                    abs(value), floor,
                    f'|F_ref| dipped below the non-vanishing floor at '
                    f'w={w:.3g}, xi={xi:.4f} -- a q->0 carrier-zero '
                    f'regression.')

    def test_independent_airy_wronskian_lower_bound(self) -> None:
        # Independent oracle: at an Airy zero Ai(-xi)=0 so |F_ref| collapses to
        # its Ai' term, 2 sqrt(pi) p w^{-1/6} |Ai'(-xi)|.  Confront the
        # production bracket against a fresh scipy.special.airy evaluation.
        w_nodes, _ = self._synthetic_grid()
        for w, xi in itertools.product(w_nodes, _AIRY_ZEROS):
            with self.subTest(w=w, xi=xi):
                _ai, aip, _bi, _bip = airy(-float(xi))
                expected = (2.0 * math.sqrt(math.pi) * self._P
                            * float(w) ** (-1.0 / 6.0) * abs(aip))
                value = airy_fold_value(
                    float(w), self._TAU_BAR, float(xi),
                    self._P, self._P, self._SIGMA)
                self.comparisons += 1
                # |Ai(-xi)| is ~1e-6 at these tabulated zeros, so allow a
                # loose relative agreement -- the point is |F_ref| ~ the Ai'
                # term, decisively non-zero, not a tight match.
                self.assertAlmostEqual(
                    abs(value) / expected, 1.0, delta=1e-3,
                    msg=f'|F_ref| at Airy zero xi={xi} departs from the '
                        f'independent Wronskian floor.')

    def test_real_reference_is_strictly_positive(self) -> None:
        # The shipping _tube_f_ref on the real node clears a strict positive
        # floor across the whole tube-band grid (no carrier zero in serving).
        floor = np.percentile(np.abs(self.fref), 1) * 1e-3
        self.assertGreater(floor, 0.0)
        for value in self.fref:
            self.comparisons += 1
            self.assertGreater(
                abs(value), floor,
                'shipping F_ref dipped toward zero on the real node grid.')

    def test_plot_reference_magnitude(self) -> None:
        # Diagnostic: |F_ref| vs xi at fixed w, contrasting q=p (survives) with
        # the q=0 regression (dips to ~0 at each Airy zero).
        xi = np.linspace(0.0, 6.5, 400)
        w = 100.0
        mag_qp = np.array([
            abs(airy_fold_value(w, 0.0, float(x), 1.0, 1.0, 0.0)) for x in xi])
        mag_q0 = np.array([
            abs(airy_fold_value(w, 0.0, float(x), 1.0, 0.0, 0.0)) for x in xi])
        self.comparisons += 1
        self.assertTrue(np.all(mag_qp > 0.0))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xi, mag_qp, 'b-', label='q = p (beat-free reference)')
        ax.plot(xi, mag_q0, 'r--', label='q = 0 (shipped regression)')
        for zero in _AIRY_ZEROS:
            ax.axvline(zero, color='k', ls=':', lw=0.6)
        ax.set_xlabel('xi')
        ax.set_ylabel('|F_ref|  (w = 100)')
        ax.set_title('Non-vanishing reference: q=p never touches zero')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'tube_beat_free_reference_magnitude.png'), dpi=110)
        plt.close(fig)


class RoundTripEnvelopeIdentityTestCase(_TubeBeatFreeTestCase):
    """Spec 2: served envelope round-trips and reconstructs to exact_total.

    The build stores ``r = E / F_ref``; the serve recovers
    ``E_served = r * F_ref`` with ``F_ref`` recomputed at the node source.
    Here the build and serve source coincide, so the recovery is exact by
    construction -- the LOAD-BEARING content is that the SACR-C reconstruction
    of ``E_served`` reproduces the engine ``exact_total`` to the machine
    floor.  A carrier/frame mismatch would break this.
    """

    def test_residual_remodulation_recovers_stored_envelope(self) -> None:
        # E_stored / F_ref, then re-modulated by F_ref recomputed at the node
        # source, returns E_stored to machine precision.
        r_stored = self.envelope / self.fref
        fref_serve = _tube_f_ref(self.w_grid, _GAMMA, self.source)
        self.assertIsNotNone(fref_serve)
        e_served = r_stored * np.asarray(fref_serve)
        self.comparisons += e_served.size
        np.testing.assert_allclose(
            e_served, self.envelope, rtol=_ROUNDTRIP_RTOL, atol=0.0,
            err_msg='r * F_ref(node) departed from the stored envelope -- a '
                    'residual/reference wiring or frame mismatch.')

    def test_reconstruction_of_served_envelope_is_exact(self) -> None:
        # SACR-C reconstruction of the SERVED envelope reproduces exact_total.
        r_stored = self.envelope / self.fref
        e_served = r_stored * self.fref
        _kernels, total = reconstruct_from_envelope(
            self.partition.w, e_served, self.partition.delays,
            self.partition.saddle_kernels, self.partition.switch,
            self.partition.critical_delay)
        error = float(np.max(np.abs(total - self.partition.exact_total)))
        self.comparisons += total.size
        self.assertLessEqual(
            error / self.f_scale, _ROUNDTRIP_RTOL,
            f'reconstruct_from_envelope(E_served) departed from exact_total by '
            f'{error / self.f_scale:.3e} (relative) -- the SACR-C algebra is '
            f'exact for any envelope, so this is a wiring bug.')

    def test_plot_frame_fringe_diagnostic(self) -> None:
        # Diagnostic: |E_served - E_stored| for the correct (tau_c) frame vs a
        # wrong (tau_bar) frame.  The wrong frame leaves a w-carrier fringe
        # exp(i w (tau_c - tau_bar)); the correct frame sits at the floor.
        critical = float(self.partition.critical_delay)
        r_stored = self.envelope / self.fref
        e_correct = r_stored * self.fref
        # Wrong frame: undo the exp(-i w tau_c) re-referencing so F_ref sits in
        # the tau_bar frame -- the residual then carries the leftover carrier.
        fref_wrong = self.fref * np.exp(1j * self.w_grid * critical)
        e_wrong = r_stored * fref_wrong
        self.comparisons += 1
        self.assertTrue(np.all(np.isfinite(e_correct)))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(
            self.w_grid,
            np.maximum(np.abs(e_correct - self.envelope), 1e-18),
            'b.-', label='|E_served - E_stored|  (tau_c frame)')
        ax.semilogy(
            self.w_grid,
            np.maximum(np.abs(e_wrong - self.envelope), 1e-18),
            'r.-', label='wrong tau_bar frame (carrier fringe)')
        ax.set_xlabel('w')
        ax.set_ylabel('envelope round-trip error')
        ax.set_title('Round-trip: correct frame at the floor, wrong frame rings')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'tube_beat_free_frame_fringe.png'), dpi=110)
        plt.close(fig)


class DryMergingFoldPairTestCase(_TubeBeatFreeTestCase):
    """Spec 3: ``_tube_f_ref`` reads ``Delta_tau`` from ``_merging_fold_pair``.

    The reference must not re-derive the merging pair.  A ``mock`` spy on the
    binding ``_tube_f_ref`` actually calls (``surrogate._merging_fold_pair``,
    imported into that namespace) proves the call happens; perturbing the
    pair's return measurably moves ``F_ref``, proving the value flows through.
    """

    def test_reference_invokes_merging_fold_pair(self) -> None:
        with mock.patch.object(
                sg, '_merging_fold_pair',
                wraps=_merging_fold_pair) as spy:
            out = _tube_f_ref(self.w_grid, _GAMMA, self.source)
        self.comparisons += 1
        self.assertIsNotNone(
            out, 'F_ref refused under the spy -- fixture premise lost.')
        self.assertGreaterEqual(
            spy.call_count, 1,
            '_tube_f_ref did not query _merging_fold_pair -- it may be '
            're-deriving the merging-pair delay separation.')

    def test_perturbing_fold_pair_moves_reference(self) -> None:
        baseline = _tube_f_ref(self.w_grid, _GAMMA, self.source)
        self.assertIsNotNone(baseline)
        baseline = np.asarray(baseline)

        def _widened(images, source, matrix):
            pair = _merging_fold_pair(images, source, matrix)
            if pair is None:
                return None
            tau_plus, tau_minus = pair
            mid = 0.5 * (tau_plus + tau_minus)
            half = 0.5 * (tau_minus - tau_plus) * 1.05  # widen Delta_tau 5%
            return (mid - half, mid + half)

        with mock.patch.object(sg, '_merging_fold_pair', side_effect=_widened):
            moved = _tube_f_ref(self.w_grid, _GAMMA, self.source)
        self.assertIsNotNone(moved)
        moved = np.asarray(moved)
        rel = (float(np.max(np.abs(moved - baseline)))
               / float(np.max(np.abs(baseline))))
        self.comparisons += 1
        self.assertGreater(
            rel, _DRY_PERTURB_MIN,
            f'perturbing _merging_fold_pair moved F_ref by only {rel:.3e} '
            f'(relative) -- the reference may ignore the fold-pair return and '
            f're-derive Delta_tau.')


class BeatFreeSelfFalsificationTestCase(_TubeBeatFreeTestCase):
    """Proves the three invariants can go red -- the suite has teeth.

    Twin 1 (spec 1): the shipped ``q = 0`` form DOES vanish at an Airy zero,
    tripping the non-vanishing floor -- the floor is not vacuous.
    Twin 2 (spec 2): reconstructing with a corrupted ``critical_delay``
    departs from ``exact_total`` -- the reconstruction pin is not vacuous.
    """

    def test_q0_regression_trips_the_floor(self) -> None:
        # The shipped q=0 reference vanishes at an Airy zero: |F_ref| falls
        # BELOW the floor that the q=p form clears, proving spec 1 has teeth.
        w = 100.0
        xi = _AIRY_ZEROS[0]
        value_q0 = airy_fold_value(w, 0.0, float(xi), 1.0, 0.0, 0.0)
        floor = (_NONVANISH_FLOOR_FRAC * 2.0 * math.sqrt(math.pi) * 1.0
                 * w ** (1.0 / 6.0))
        self.comparisons += 1
        self.assertLess(
            abs(value_q0), floor,
            'the q=0 regression did NOT trip the floor -- spec 1 would be '
            'vacuous.')

    def test_corrupted_critical_delay_breaks_reconstruction(self) -> None:
        # Reconstruct the (correct) served envelope with a bogus critical delay
        # and confirm it departs from exact_total by a whole-number relative
        # error, proving spec 2's reconstruction pin has teeth.
        e_served = (self.envelope / self.fref) * self.fref
        _kernels, bad_total = reconstruct_from_envelope(
            self.partition.w, e_served, self.partition.delays,
            self.partition.saddle_kernels, self.partition.switch,
            float(self.partition.critical_delay) + 0.5)
        error = float(np.max(np.abs(bad_total - self.partition.exact_total)))
        self.comparisons += 1
        self.assertGreater(
            error / self.f_scale, 1e-3,
            'a corrupted critical_delay did NOT move the reconstruction -- '
            'spec 2 would be vacuous.')


class _TubeOffAxisTestCase(unittest.TestCase):
    """Shared OFF-AXIS real 4-image interior node + anti-vacuity bookkeeping.

    The D2 and large-xi specs are hollow on a near-axis source (a reflection
    collides with the raw source; the pair barely resolves), so this base
    derives a node with both eigenframe components above ``_OFFAXIS_MIN_COMP``
    and asserts the domain premise.
    """

    comparisons: int

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        cls.source, cls.eta_max, cls.matrix = _find_offaxis_interior_node()
        cls.w_grid = np.geomspace(2.0, 60.0, 24)
        fref = _tube_f_ref(cls.w_grid, _GAMMA, cls.source)
        if fref is None or not np.all(np.isfinite(fref)):
            raise AssertionError(
                'fixture premise lost: _tube_f_ref refuses the off-axis '
                'interior node -- the beat-free reference is undefined there.')
        cls.fref = np.asarray(fref)

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no comparisons executed -- test body was vacuous.')


class D2InvarianceTestCase(_TubeOffAxisTestCase):
    """Spec 4: ``F_ref`` (and the engine envelope) are D2-invariant.

    ``F_ref`` at the three non-trivial D2 reflections of the off-axis source
    matches ``F_ref`` at the raw source to ``_D2_FREF_RTOL``; the paired
    engine envelope matches to ``_D2_ENV_RTOL`` so the residual ``r`` inherits
    the same symmetry.  A generic (non-D2) rotation moves ``F_ref`` by
    ``O(1)``, proving the invariance is a real property of the reflection
    group, not a numerical coincidence.
    """

    def _reflections(self) -> tuple[tuple[float, float], ...]:
        y1, y2 = float(self.source[0]), float(self.source[1])
        return ((-y1, y2), (y1, -y2), (-y1, -y2))

    def test_reference_is_d2_invariant(self) -> None:
        raw_scale = float(np.max(np.abs(self.fref)))
        for refl in self._reflections():
            with self.subTest(reflection=refl):
                fref_refl = _tube_f_ref(
                    self.w_grid, _GAMMA, np.array(refl, dtype=float))
                self.assertIsNotNone(
                    fref_refl,
                    f'F_ref refused at D2 reflection {refl} -- the reflected '
                    f'source should be an equivalent interior node.')
                fref_refl = np.asarray(fref_refl)
                gap = float(np.max(np.abs(fref_refl - self.fref)))
                self.comparisons += fref_refl.size
                self.assertLessEqual(
                    gap / raw_scale, _D2_FREF_RTOL,
                    f'F_ref at D2 reflection {refl} departed from the raw '
                    f'reference by {gap / raw_scale:.3e} (relative) -- a '
                    f'tau_bar/tau_c sign bug breaks D2 in the carrier phase.')

    def test_engine_envelope_is_d2_invariant(self) -> None:
        # Paired envelope pin: the engine amplification is even in each source
        # component, so the SACR-C envelope is D2-invariant to ~1e-12.
        y1, y2 = float(self.source[0]), float(self.source[1])
        raw = ChangRefsdalChannels(self.w_grid).evaluate(
            gamma=_GAMMA, y=(y1, y2), beta=0.0, kappa=0.0)
        env_raw = np.asarray(raw.envelope)
        scale = float(np.max(np.abs(env_raw)))
        for refl in self._reflections():
            with self.subTest(reflection=refl):
                part = ChangRefsdalChannels(self.w_grid).evaluate(
                    gamma=_GAMMA, y=refl, beta=0.0, kappa=0.0)
                env_refl = np.asarray(part.envelope)
                gap = float(np.max(np.abs(env_refl - env_raw)))
                self.comparisons += env_refl.size
                self.assertLessEqual(
                    gap / scale, _D2_ENV_RTOL,
                    f'engine envelope at D2 reflection {refl} departed by '
                    f'{gap / scale:.3e} (relative) -- amplification is not '
                    f'even in the source components as D2 requires.')

    def test_generic_rotation_moves_reference(self) -> None:
        # Self-falsification: a non-D2 rotation is NOT a symmetry, so it must
        # move F_ref by O(1) -- proving the D2 tolerance is not so loose that
        # any nearby source would pass.
        y1, y2 = float(self.source[0]), float(self.source[1])
        cos_a, sin_a = math.cos(_D2_ROTATION_ANGLE), math.sin(_D2_ROTATION_ANGLE)
        rotated = np.array(
            [cos_a * y1 - sin_a * y2, sin_a * y1 + cos_a * y2], dtype=float)
        fref_rot = _tube_f_ref(self.w_grid, _GAMMA, rotated)
        self.assertIsNotNone(fref_rot)
        fref_rot = np.asarray(fref_rot)
        rel = (float(np.max(np.abs(fref_rot - self.fref)))
               / float(np.max(np.abs(self.fref))))
        self.comparisons += 1
        self.assertGreater(
            rel, _D2_ROTATION_MIN_MOVE,
            f'a generic {_D2_ROTATION_ANGLE} rad rotation moved F_ref by only '
            f'{rel:.3e} -- the D2 invariance tolerance may be vacuously loose.')


class LargeXiCalibrationTestCase(_TubeOffAxisTestCase):
    """Spec 5: ``F_ref / S_asymp`` is a single ``w``-independent constant.

    On a well-resolved fold pair (``w Delta_tau in _LARGEXI_WDT_RANGE``) the
    production reference approaches its stationary-phase asymptote up to the
    global constant ``2 p exp(i sigma)`` the residual absorbs, so the ratio
    ``F_ref / S_asymp`` has a coefficient of variation below
    ``_LARGEXI_RATIO_TOL``.  The two self-falsification foils halve and
    double the asymptote's ``zeta`` -- a ``xi`` factor-of-two convention
    error -- and shatter the constancy, proving the pin locks the sigma/xi
    phase convention.

    Substitution note: the Architect's literal ``F_ref == F_ref_geo``
    (``q = 0`` geometric two-image sum) FAILS on correct code because the
    shipped reference uses ``q = p`` amplitudes ``C1 != C2``; see the module
    docstring's "Push-back / substitution" section.  This constant-ratio
    invariant is the genuine convention lock the description is after.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.delta_tau, cls.tau_bar, cls.critical_delay = _geometric_fold_inputs(
            _GAMMA, cls.source)
        wdt = np.linspace(_LARGEXI_WDT_RANGE[0], _LARGEXI_WDT_RANGE[1],
                          _LARGEXI_N_W)
        cls.large_w = wdt / cls.delta_tau
        fref = _tube_f_ref(cls.large_w, _GAMMA, cls.source)
        if fref is None or not np.all(np.isfinite(fref)):
            raise AssertionError(
                'fixture premise lost: _tube_f_ref refuses the large-w grid.')
        cls.large_fref = np.asarray(fref)

    def _cv(self, zeta_factor: float) -> float:
        s_asymp = _stationary_phase_asymptote(
            self.large_w, self.delta_tau, self.tau_bar, self.critical_delay,
            zeta_factor=zeta_factor)
        ratio = self.large_fref / s_asymp
        k_const = np.mean(ratio)
        return float(np.max(np.abs(ratio - k_const)) / abs(k_const))

    def test_reference_matches_independent_asymptote(self) -> None:
        cv = self._cv(1.0)
        self.comparisons += self.large_fref.size
        self.assertLessEqual(
            cv, _LARGEXI_RATIO_TOL,
            f'F_ref / S_asymp varied by {cv:.3e} across w Delta_tau in '
            f'{_LARGEXI_WDT_RANGE} -- the reference departs from its q=p '
            f'stationary-phase asymptote, indicating a sigma/xi phase bug.')

    def test_halved_zeta_foil_breaks_constancy(self) -> None:
        # Self-falsification: a xi factor-of-two error halves zeta; the ratio
        # then beats (CV >> tolerance).
        cv_bad = self._cv(0.5)
        self.comparisons += 1
        self.assertGreater(
            cv_bad, 10.0 * _LARGEXI_RATIO_TOL,
            'halving zeta did NOT break the constant ratio -- the sigma/xi '
            'convention lock would be vacuous.')

    def test_doubled_zeta_foil_breaks_constancy(self) -> None:
        # Self-falsification: the opposite factor-of-two error doubles zeta.
        cv_bad = self._cv(2.0)
        self.comparisons += 1
        self.assertGreater(
            cv_bad, 10.0 * _LARGEXI_RATIO_TOL,
            'doubling zeta did NOT break the constant ratio -- the sigma/xi '
            'convention lock would be vacuous.')

    def test_plot_ratio_constancy(self) -> None:
        # Diagnostic: |F_ref / S_asymp - K| for the correct asymptote (flat at
        # the floor) vs the halved-zeta foil (a growing beat).
        s_ok = _stationary_phase_asymptote(
            self.large_w, self.delta_tau, self.tau_bar, self.critical_delay)
        s_bad = _stationary_phase_asymptote(
            self.large_w, self.delta_tau, self.tau_bar, self.critical_delay,
            zeta_factor=0.5)
        ratio_ok = self.large_fref / s_ok
        ratio_bad = self.large_fref / s_bad
        wdt = self.large_w * self.delta_tau
        self.comparisons += 1
        self.assertTrue(np.all(np.isfinite(ratio_ok)))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(
            wdt, np.abs(ratio_ok - np.mean(ratio_ok)) / abs(np.mean(ratio_ok)),
            'b.-', label='correct zeta = w dtau / 2')
        ax.semilogy(
            wdt, np.abs(ratio_bad - np.mean(ratio_bad)) / abs(np.mean(ratio_bad)),
            'r.-', label='halved zeta (factor-of-two foil)')
        ax.axhline(_LARGEXI_RATIO_TOL, color='k', ls=':', lw=0.8)
        ax.set_xlabel('w * Delta_tau')
        ax.set_ylabel('|F_ref / S_asymp - K| / |K|')
        ax.set_title('Large-xi: ratio is constant only for the correct zeta')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'tube_beat_free_largexi_ratio.png'), dpi=110)
        plt.close(fig)


#: Minimal grid override for the buildability smoke (spec 6): the four gamma
#: axes must keep >= 4 nodes (`TubeChart.from_values` refuses fewer), so only
#: the ``w`` density and held-out count are trimmed to hold the four-band
#: build+serve under the fast-tier ceiling (measured ~9 s per band).
_SMOKE_CONFIG: TrainingConfig = dataclasses.replace(
    TrainingConfig(), w_nodes_per_decade=3, n_heldout=4)

#: Frequency band for the buildability smoke -- capped at ``w = 30`` so the
#: engine references stay on the fast DD path.
_SMOKE_W_RANGE: tuple[float, float] = (1.0, 30.0)


def _serve_probe(chart, samples: list[tuple[float, float, float]],
                 ) -> tuple[int, int, bool]:
    """Serve held-out queries through the one-chart surrogate guard stack.

    Mirrors the serve leg of `surrogate_training._heldout_eps` EXACTLY (same
    engine reference, same ``(eta, theta, image_count)`` gauge fed to
    `LensAmplificationSurrogate.serve`) but returns the raw coverage counts
    the buildability smoke needs -- ``(n_with_reference, n_served,
    all_served_finite)`` -- rather than an eps.  A ``_heldout_eps`` value
    alone is ambiguous here: a pure coverage-miss band returns a FINITE
    ``1.0`` with zero serves, so ``isfinite(eps)`` cannot witness "at least
    one node serves".  Counting serves directly does.
    """
    surrogate = LensAmplificationSurrogate([chart], {'schema': 'spec6-smoke'})
    w_grid = np.exp(chart.log_w_grid)
    n_ref = n_served = 0
    all_finite = True
    for gamma, y1, y2 in samples:
        channels = ChangRefsdalChannels(w_grid)
        try:
            partition = channels.evaluate(
                gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        except _ENGINE_REFUSALS:
            continue
        env_true = np.asarray(partition.envelope)
        if not np.all(np.isfinite(env_true)):
            continue
        n_ref += 1
        emulated, served, _definition = surrogate.serve(
            w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=partition.caustic_distance, theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        if served:
            n_served += 1
            if not np.all(np.isfinite(emulated)):
                all_finite = False
    return n_ref, n_served, all_finite


class TubeArcBuildabilityTestCase(unittest.TestCase):
    """Spec 6: each of the four Professor bands builds+serves a tube.

    For every ``(band, parity)`` in ``_BUILDABILITY_BANDS`` -- astroid
    small-gamma (incl. ``gamma ~ 0.045``), astroid large-gamma, saddle
    ``gamma = 1.2``, and the representative ``gamma = 0.4`` astroid -- a
    MINIMAL tube chart is built on ``arcs[0]`` under the beat-free residual
    representation and a handful of in-band held-out ``(gamma, y1, y2)``
    queries are served.  The invariant: the build raises no error, covers at
    least one build node (``refused < calls``), and serves at least one
    held-out query with a FINITE envelope.  This is a fast SMOKE gate against
    a band whose arcs drift below four images or whose beat-free reference
    refuses everywhere -- NOT the eps accuracy sweep.

    Parsimony note: `test_lensing_caustic_cusps.UniversalFMaxTestCase` builds
    its own positive/saddle bands, but it pins the ``f_max`` eps-BAR currency,
    not per-band buildability under the ``TUBE_BEAT_FREE_AIRY`` residual tag;
    this class is the distinct build+serve-non-empty invariant for the four
    Professor bands the description names, and shares no assertion with it.

    Anti-vacuity: ``comparisons`` counts each band actually built and probed,
    and ``tearDown`` fails the test if none were.
    """

    comparisons: int

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no bands were built -- the buildability smoke was vacuous.')

    def test_all_four_bands_build_and_serve(self) -> None:
        config = _SMOKE_CONFIG
        for band, parity in _BUILDABILITY_BANDS:
            with self.subTest(band=band, parity=parity):
                structure = band_caustic_structure(
                    band, parity, n_samples=config.n_caustic_samples)
                arcs = _tube_training_arcs(structure, parity)
                self.assertGreater(
                    len(arcs), 0,
                    f'band {band} parity {parity}: no tube arcs detected.')
                arc = arcs[0]
                r_min = _min_curvature_radius(band, arc,
                                              config.n_caustic_samples)
                eta_max = config.f_max * r_min
                eta_floor = config.f_floor * r_min
                gamma_grid = np.linspace(band[0], band[1], config.n_gamma)

                # Build must not raise (the core buildability claim).
                chart, calls, refused = _build_tube_chart(
                    gamma_grid=gamma_grid, arc=arc, parity=parity,
                    w_range=_SMOKE_W_RANGE, config=config,
                    eta_max=eta_max, eta_floor=eta_floor)
                self.comparisons += 1

                # Coverage > 0: at least one build node stored a residual
                # (F_ref buildable on the 4-image interior somewhere on the
                # arc), not an all-refused chart of zeros.
                self.assertLess(
                    refused, calls,
                    f'band {band} parity {parity}: the tube refused ALL '
                    f'{calls} grid nodes -- arcs drift below four images or '
                    f'the beat-free reference refuses everywhere.')

                # Serve a handful of in-band held-out queries: at least one
                # must serve with a finite envelope.
                rng = np.random.default_rng(2026)
                samples = _tube_heldout_samples(
                    band, arc, config, rng,
                    eta_max=eta_max, eta_floor=eta_floor)
                n_ref, n_served, all_finite = _serve_probe(chart, samples)
                self.assertGreater(
                    n_ref, 0,
                    f'band {band} parity {parity}: no held-out query had a '
                    f'valid engine reference -- fixture premise lost.')
                self.assertGreater(
                    n_served, 0,
                    f'band {band} parity {parity}: the tube served NONE of '
                    f'{n_ref} referenced held-out queries (coverage = 0).')
                self.assertTrue(
                    all_finite,
                    f'band {band} parity {parity}: a served held-out query '
                    f'returned a non-finite envelope.')


#: F083 build-node count per theta axis (spec-pinned).  The pre-fix beating
#: representation needed ~48 nodes; the beat-free residual must clear the eps
#: bar at TEN, leaving the 10-vs-48 margin visible -- do NOT raise this.
_F083_N_THETA: int = 10

#: F083 build/serve frequency band.  The Architect names ``w in [40, 80]``,
#: but `ChangRefsdalChannels.evaluate` routes ``w > 60`` to the slow mpmath
#: path (~15 s/eval, a ~40 min build) and hard-refuses ``w > 150``; the tube
#: envelope is smooth and the residual is band-independent above the first
#: Airy fringe, so the sweep is capped at ``w = 60`` to stay on the exact DD
#: path.  (Deviation documented in the change report.)
_F083_W_RANGE: tuple[float, float] = (40.0, 60.0)

#: Held-out OFF-node query count (node-midpoints, strict interior interp).
_F083_N_HELDOUT: int = 8

#: The acceptance bar: eps <= 0.0237 at ``n_theta = 10``.  A genuine anti-
#: regression gate -- the q=0 beating rep fails this at ten nodes.
_F083_EPS_BAR: float = 0.0237

#: Held-out queries sit at ``eta = 0.5 * eta_max`` (mid-shell interior).
_F083_ETA_FRAC: float = 0.5

#: Delta_tau knee threshold and inward stand-offs used to derive the robust
#: servable sub-arc from the binding corner's live merging-pair profile (the
#: full cusp-to-cusp arc has a non-monotone Delta_tau and must not be used).
_F083_DTAU_FRAC: float = 0.6
_F083_LO_STANDOFF: float = 0.20
_F083_HI_STANDOFF: float = 0.05


@dataclasses.dataclass(frozen=True)
class _F083Fixture:
    """The shared trimmed-sub-arc ``n_theta = 10`` beat-free tube chart."""

    chart: object
    arc: object
    eta_max: float
    eta_floor: float
    gamma_grid: np.ndarray
    config: TrainingConfig
    calls: int
    refused: int


def _f083_delta_tau(arc, gamma: float, theta: float,
                    eta: float) -> float | None:
    """Merging-pair delay gap ``Delta_tau`` at a tube node.

    Returns ``None`` wherever the source is non-finite, drops below four
    images, or the merging fold pair refuses -- the exact conditions under
    which `_tube_f_ref` (hence the beat-free residual) is undefined.  Used to
    locate the sub-arc where the fold resolves robustly along the whole run.
    """
    source = _tube_source(gamma, float(theta), eta, arc.branch,
                          arc.inward_sign)
    if not np.all(np.isfinite(source)):
        return None
    try:
        matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
        images, _absolute, _t_min = _frame_delays(source, matrix)
    except geometry.LensDomainError:
        return None
    if len(images) != 4:
        return None
    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        return None
    return float(pair[1] - pair[0])


@functools.lru_cache(maxsize=1)
def _f083_shared_tube() -> _F083Fixture:
    """Build the trimmed-sub-arc ``n_theta = 10`` beat-free tube chart ONCE.

    The build is ``n_gamma(4) x n_u(4) x n_theta(10) = 160`` sequential DD
    engine evaluations (~1 min, irreducible: the cubic spline floor forces
    ``>= 4`` nodes per axis and ``n_theta = 10`` is spec-pinned).  Caching it
    lets both `TubeF083AccuracySweepTestCase` and `RawSourceReModulationTest\
Case` share a SINGLE build, so the file pays it once.

    The servable sub-arc is derived from the LIVE merging-pair boundary, NOT
    pinned: scan the binding corner ``(gamma_hi, eta_max)`` -- which resolves
    narrowest and whose ``Delta_tau`` turns over first -- for the ``Delta_tau``
    peak, take the low knee as the first theta on the rise clearing
    ``_F083_DTAU_FRAC`` of the peak, then stand both bounds inward off the
    steep-rise / turnover ends into the smooth core.  A sub-arc robust at the
    binding corner is robust across the whole gamma axis.
    """
    config = dataclasses.replace(
        TrainingConfig(), n_gamma=4, n_u=4, n_theta=_F083_N_THETA,
        w_nodes_per_decade=4)
    structure = band_caustic_structure(_BAND, 1, n_samples=_N_SAMPLES)
    arc = _tube_training_arcs(structure, 1)[0]
    r_min = _min_curvature_radius(_BAND, arc, _N_SAMPLES)
    eta_max = config.f_max * r_min
    eta_floor = config.f_floor * r_min
    gamma_grid = np.linspace(_BAND[0], _BAND[1], config.n_gamma)

    scan = np.linspace(arc.theta_lo, arc.theta_hi, 80)
    dtau = np.array([_f083_delta_tau(arc, _BAND[1], float(t), eta_max)
                     or np.nan for t in scan])
    finite = np.isfinite(dtau)
    if not np.any(finite):
        raise AssertionError(
            'fixture premise lost: no resolvable merging-pair Delta_tau along '
            'the gamma=0.4 astroid tube arc -- the trim cannot be derived.')
    peak_idx = int(np.nanargmax(dtau))
    peak_val = float(dtau[peak_idx])
    lo_candidates = np.where(finite & (dtau >= _F083_DTAU_FRAC * peak_val))[0]
    lo_knee = float(scan[int(lo_candidates[0])])
    hi_peak = float(scan[peak_idx])
    span = hi_peak - lo_knee
    theta_lo = lo_knee + _F083_LO_STANDOFF * span
    theta_hi = hi_peak - _F083_HI_STANDOFF * span
    arc2 = dataclasses.replace(arc, theta_lo=theta_lo, theta_hi=theta_hi)

    chart, calls, refused = _build_tube_chart(
        gamma_grid=gamma_grid, arc=arc2, parity=1, w_range=_F083_W_RANGE,
        config=config, eta_max=eta_max, eta_floor=eta_floor)
    return _F083Fixture(
        chart=chart, arc=arc2, eta_max=eta_max, eta_floor=eta_floor,
        gamma_grid=gamma_grid, config=config, calls=calls, refused=refused)


class TubeF083AccuracySweepTestCase(unittest.TestCase):
    """F083: held-out eps of the beat-free tube at ``n_theta = 10``.

    The missing acceptance item -- the eps sweep `TubeArcBuildabilityTestCase`
    explicitly is NOT.  On the shared trimmed sub-arc, hold out
    ``_F083_N_HELDOUT`` OFF-node queries whose theta lie STRICTLY BETWEEN the
    ten build nodes (node-midpoints -> interior interpolation) at
    ``eta = 0.5 eta_max``, serve each through the real surrogate guard stack
    over ``w in _F083_W_RANGE``, and take the F_ref-NORMALIZED max relative
    error against the Schwinger-engine oracle.

    F_ref-normalization (NOT ``|exact_total|``) is load-bearing: the old q=0
    carrier's ``|exact_total|`` vanishes at the Airy zeros where ``F_ref`` does
    not, so an ``|exact_total|`` denominator spikes to false failures there.

    Cost: 160-node build (cached, shared) + 8 held-out x 3 w on ONE trimmed
    sub-arc, single band -- seconds beyond the shared build, well under the
    60 s per-test ceiling.  The measured ``(n_theta, eps)`` pair is emitted
    (stdout + a saved diagnostic plot) so the completion record can quote it.
    """

    comparisons: int

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        cls.fx = _f083_shared_tube()
        cls.surrogate = LensAmplificationSurrogate(
            [cls.fx.chart], {'schema': 'f083-sweep'})

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no held-out comparisons executed -- the eps sweep was vacuous.')

    def test_trimmed_run_refused_no_build_nodes(self) -> None:
        # Premise guard (spec): a non-zero build-refusal count means the
        # Delta_tau core drifted and the trim is stale -- fail LOUDLY rather
        # than certify eps on a partially-covered chart.
        self.comparisons += 1
        self.assertEqual(
            self.fx.refused, 0,
            f'trimmed sub-arc refused {self.fx.refused}/{self.fx.calls} build '
            f'nodes -- the servable core drifted; re-derive the trim.')

    def test_heldout_eps_within_bar_at_ten_nodes(self) -> None:
        fx = self.fx
        theta_nodes = np.sort(fx.chart.theta_grid)
        mids = 0.5 * (theta_nodes[:-1] + theta_nodes[1:])
        idx = np.unique(
            np.linspace(0, len(mids) - 1, _F083_N_HELDOUT).round().astype(int))
        held_thetas = mids[idx]
        eta_q = _F083_ETA_FRAC * fx.eta_max
        w_serve = np.linspace(_F083_W_RANGE[0], _F083_W_RANGE[1], 3)

        eps = 0.0
        refused_serve = 0
        per_point: list[tuple[float, float]] = []
        worst: tuple[float, np.ndarray, np.ndarray] | None = None
        for theta in held_thetas:
            source = _tube_source(_GAMMA, float(theta), eta_q, fx.arc.branch,
                                  fx.arc.inward_sign)
            y1, y2 = float(source[0]), float(source[1])
            channels = ChangRefsdalChannels(w_serve)
            try:
                partition = channels.evaluate(
                    gamma=_GAMMA, y=(y1, y2), beta=0.0, kappa=0.0)
            except _ENGINE_REFUSALS:
                refused_serve += 1
                continue
            env_true = np.asarray(partition.envelope)
            emulated, served, _definition = self.surrogate.serve(
                w_serve, gamma=_GAMMA, y1=y1, y2=y2, beta=0.0,
                eta=partition.caustic_distance, theta=partition.critical_theta,
                image_count=int(partition.real_mask.sum()))
            if not served:
                refused_serve += 1
                continue
            fref = np.abs(np.asarray(_tube_f_ref(w_serve, _GAMMA, source)))
            rel = float(np.max(np.abs(emulated - env_true) / fref))
            per_point.append((float(theta), rel))
            if worst is None or rel > worst[0]:
                worst = (rel, np.abs(emulated), np.abs(env_true))
            eps = max(eps, rel)
            self.comparisons += 1

        # Premise guard: every interior held-out query must serve on the
        # trimmed run, else the fixture no longer covers its own midpoints.
        self.assertEqual(
            refused_serve, 0,
            f'{refused_serve}/{_F083_N_HELDOUT} held-out midpoints failed to '
            f'serve -- fixture premise lost (trim drifted off the tube).')

        # Emit the measured (n_theta, eps) pair for the completion record.
        print(f'\n[F083] n_theta={_F083_N_THETA} eps={eps:.4e} '
              f'bar={_F083_EPS_BAR} w_range={_F083_W_RANGE} '
              f'per_point='
              f'{[(round(t, 4), float(f"{e:.3e}")) for t, e in per_point]}')
        self._plot(per_point, eps, w_serve, worst)

        self.assertLessEqual(
            eps, _F083_EPS_BAR,
            f'beat-free tube eps={eps:.4e} exceeds the {_F083_EPS_BAR} bar at '
            f'n_theta={_F083_N_THETA} -- the residual representation regressed '
            f'(a beating carrier would need ~48 nodes to clear this).')

    def _plot(self, per_point: list[tuple[float, float]], eps: float,
              w_serve: np.ndarray,
              worst: tuple[float, np.ndarray, np.ndarray] | None) -> None:
        # Diagnostic: per-midpoint eps bars (collapse to the smooth-variation
        # scale) plus the served-vs-oracle |E| overlay at the worst held-out
        # theta (a residual cos(w Delta_tau) beat would show here if the
        # representation regressed).
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
        thetas = [t for t, _ in per_point]
        rels = [r for _, r in per_point]
        ax0.semilogy(thetas, rels, 'bo-')
        ax0.axhline(_F083_EPS_BAR, color='r', ls='--', lw=0.9,
                    label=f'bar = {_F083_EPS_BAR}')
        ax0.set_xlabel('held-out theta (node-midpoint)')
        ax0.set_ylabel('F_ref-normalized max rel. error')
        ax0.set_title(f'F083 eps sweep (n_theta={_F083_N_THETA}, '
                      f'eps={eps:.3e})')
        ax0.legend(fontsize=8)
        if worst is not None:
            _rel, mag_served, mag_true = worst
            ax1.plot(w_serve, mag_true, 'k.-', label='engine oracle |E|')
            ax1.plot(w_serve, mag_served, 'b.--', label='served |E|')
            ax1.set_xlabel('w')
            ax1.set_ylabel('|E|')
            ax1.set_title('worst held-out theta: served vs oracle |E|')
            ax1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'tube_beat_free_f083_eps_sweep.png'), dpi=110)
        plt.close(fig)


class RawSourceReModulationTestCase(unittest.TestCase):
    """Cross-suite pin: ``F_ref`` is recomputed at the RAW D2 eigenframe.

    Narrow invariant, DISTINCT from `D2InvarianceTestCase`'s output-equality:
    the serve path re-modulates the stored residual with
    ``_tube_f_ref(w, gamma, [y1_eig, y2_eig])`` at the RAW eigenframe query
    source -- because ``F_ref`` is exactly D2-invariant -- while the theta-fold
    (via ``theta_to_s``) is applied ONLY to the residual's ``(u, s)``
    interpolation coordinate.  A regression that folded the source into the
    fundamental octant BEFORE ``F_ref`` would round-trip at fundamental-octant
    nodes yet silently corrupt every off-fundamental serve.

    Fixture: a servable query REFLECTED into a non-fundamental D2 octant (at
    least one negative eigenframe component), derived on the shared tube chart.
    The spy patches the ``surrogate._tube_f_ref`` binding the serve path calls
    (module-global, the same one `_tube_serves` probes) and captures the
    ``source`` argument.
    """

    comparisons: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.fx = _f083_shared_tube()
        cls.surrogate = LensAmplificationSurrogate(
            [cls.fx.chart], {'schema': 'f083-spy'})
        cls._derive_reflected_query()

    @classmethod
    def _derive_reflected_query(cls) -> None:
        """Find a non-fundamental D2-octant query that serves on the chart.

        Sign flips are exact D2 elements, so any octant image of a servable
        fundamental query shares its ``eta`` / image count / ``|F_ref|``.
        Scan the node-midpoints and the three non-identity sign reflections
        for the first image that (a) has ``>= 1`` negative eigenframe
        component and (b) actually serves with a finite envelope.
        """
        fx = cls.fx
        theta_nodes = np.sort(fx.chart.theta_grid)
        mids = 0.5 * (theta_nodes[:-1] + theta_nodes[1:])
        eta_q = _F083_ETA_FRAC * fx.eta_max
        w_serve = np.linspace(_F083_W_RANGE[0], _F083_W_RANGE[1], 3)
        for theta in mids:
            source = _tube_source(_GAMMA, float(theta), eta_q, fx.arc.branch,
                                  fx.arc.inward_sign)
            y1, y2 = float(source[0]), float(source[1])
            for sign_x, sign_y in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
                ry1, ry2 = sign_x * y1, sign_y * y2
                if min(ry1, ry2) >= 0.0:
                    continue  # must be genuinely non-fundamental
                channels = ChangRefsdalChannels(w_serve)
                try:
                    partition = channels.evaluate(
                        gamma=_GAMMA, y=(ry1, ry2), beta=0.0, kappa=0.0)
                except _ENGINE_REFUSALS:
                    continue
                emulated, served, _definition = cls.surrogate.serve(
                    w_serve, gamma=_GAMMA, y1=ry1, y2=ry2, beta=0.0,
                    eta=partition.caustic_distance,
                    theta=partition.critical_theta,
                    image_count=int(partition.real_mask.sum()))
                if served and np.all(np.isfinite(emulated)):
                    cls.w_serve = w_serve
                    cls.query = (ry1, ry2)
                    cls.partition = partition
                    return
        raise AssertionError(
            'fixture premise lost: no reflected non-fundamental D2 query '
            'served on the shared tube chart -- cannot test raw-source '
            're-modulation.')

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no serve executed -- the raw-source spy was vacuous.')

    def _serve_query(self, **patch):
        ry1, ry2 = self.query
        part = self.partition
        return self.surrogate.serve(
            self.w_serve, gamma=_GAMMA, y1=ry1, y2=ry2, beta=0.0,
            eta=part.caustic_distance, theta=part.critical_theta,
            image_count=int(part.real_mask.sum()), **patch)

    def test_fref_called_with_raw_eigenframe_source(self) -> None:
        ry1, ry2 = self.query
        raw = np.array([ry1, ry2], dtype=float)
        folded = np.abs(raw)  # the fundamental-octant image a regression uses
        real_fref = sg._tube_f_ref
        captured: list[np.ndarray] = []

        def spy(w_grid, gamma, source, *args, **kwargs):
            arr = np.asarray(source, dtype=float)
            if arr.shape == (2,):
                captured.append(arr.copy())
            return real_fref(w_grid, gamma, source, *args, **kwargs)

        with mock.patch.object(sg, '_tube_f_ref', new=spy):
            _emulated, served, _definition = self._serve_query()
        self.assertTrue(served, 'reflected query stopped serving under spy.')
        self.assertGreater(
            len(captured), 0,
            '_tube_f_ref was never called on the tube serve path.')

        # Every 2-vector _tube_f_ref saw is the RAW reflected query, NOT its
        # fundamental-octant reflection.
        self.assertGreater(
            np.max(np.abs(raw - folded)), 1e-6,
            'reflected query is not genuinely non-fundamental (setup error).')
        for source in captured:
            self.comparisons += 1
            np.testing.assert_allclose(
                source, raw, rtol=0, atol=1e-12,
                err_msg='_tube_f_ref received a folded/reflected source '
                        'instead of the raw eigenframe query -- the theta-fold '
                        'leaked into F_ref.')
            self.assertGreater(
                np.max(np.abs(source - folded)), 1e-6,
                '_tube_f_ref received the fundamental-octant image, so an '
                'off-fundamental serve would be silently wrong.')

    def test_fref_return_value_is_consumed(self) -> None:
        # Consumption pin: scaling F_ref by 2 must scale the served envelope by
        # 2 (E = r * F_ref).  A discarded F_ref would leave E unchanged.
        real_fref = sg._tube_f_ref

        def double(w_grid, gamma, source, *args, **kwargs):
            value = real_fref(w_grid, gamma, source, *args, **kwargs)
            return None if value is None else 2.0 * np.asarray(value)

        base, served_base, _d0 = self._serve_query()
        self.assertTrue(served_base, 'reflected query did not serve (base).')
        with mock.patch.object(sg, '_tube_f_ref', new=double):
            scaled, served_scaled, _d1 = self._serve_query()
        self.assertTrue(served_scaled, 'reflected query did not serve (2x).')

        mask = np.abs(base) > 1e-30
        self.assertTrue(
            np.any(mask), 'served envelope is identically zero (no teeth).')
        ratio = scaled[mask] / base[mask]
        self.comparisons += 1
        np.testing.assert_allclose(
            ratio, 2.0, rtol=1e-9,
            err_msg='doubling F_ref did NOT double the served envelope -- the '
                    'reference is discarded, not consumed as E = r * F_ref.')


if __name__ == '__main__':
    unittest.main()
