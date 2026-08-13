"""
Gauge-identity, frame-phase round-trip, and near/far handover tests for the
tier-1 far-from-caustic macro-saddle analytic serve rung -- all WITHOUT any
training or engine call on the numerically heavy paths.

The functions under test are the authoritative saddle GAUGE HOME
(`cogwheel.lensing.chang_refsdal._gauge`):

* ``_saddle_switch_delay(tau_min, w_min) == tau_min - RHO_END / w_min`` -- the
  delay at which the switch window closes, a fixed property of the BAND FLOOR
  (``w_min`` is the passed ``min(dense_w)``, NEVER a live serve-time ``w``).
* ``_saddle_phase_delay(tau_min) == tau_min`` -- the phase origin the stored
  envelope's carrier is demodulated against.

Housing both saddle tiers on the SAME two functions is what keeps the train
gauge and the serve gauge from skewing; these tests pin that identity, prove
the functions are pure, exercise the ``_frame_phase`` demod/re-mod telescoping
that the deferred tier-2 stored object relies on, and characterise the near/far
handover.

Tolerance rationale
-------------------
* ``IDENTITY_TOL = 1e-13`` -- the gauge closed forms are exact float
  arithmetic; ``1e-13`` is a generous slack above round-off on a single
  subtraction/division (measured residual is bit-exact / 0.0).
* ``ROUND_TRIP_TOL = 1e-13`` -- the ``exp(+1j w t_min)`` / ``exp(-1j w t_min)``
  demod/re-mod pair telescopes to machine precision when both sides reduce
  ``w t_min`` modulo ``2*pi`` (`_frame_phase`).  Measured recovery error for a
  smooth analytic envelope over the band is ``~2e-16`` (``w t_min`` up to
  ~300 rad), comfortably under ``1e-13``.
* ``MAG_CONTINUITY_TOL_TOP = 1e-6`` -- at the RESOLVED top of the band the
  tier-1 zero-envelope magnitude ``|F|`` matches the exact ``|F|`` to
  ``~1.4e-7`` on the at-the-floor fixture (``rho = 2.02``).
* ``MAG_CONTINUITY_TOL_BAND = 2e-4`` -- the whole-band envelope of the same
  magnitude residual (measured max ``~5.0e-5``).

  Both bounds were ``1e-3`` / ``5e-3`` until 2026-08-13, sized against a
  ``rho = 1.5`` fixture that the two-term serve gate REFUSES.  They were
  certifying the rung on a domain it never serves, at a residual 57x worse
  than the worst case it does, and read as the rung's accuracy bar to anyone
  who did not go check the fixture's rho.  The rule that fell out: when a
  gate term is added, every fixture that predates it must be re-checked for
  whether it is still INSIDE the served domain -- a passing test on a refused
  source certifies nothing.

MEASURED SPEC DISCREPANCY (flagged, do not "fix" by loosening physics).
----------------------------------------------------------------------
The SHARD C spec asked for ``|F_tier1 - F_nearcaustic| / |F| <= 1e-3`` *at the
resolvability boundary* ``w_lo * min_delta_tau = RHO_END`` (=4).  That is
physically unreachable: ``RHO_END = 4`` is the *resolvability* threshold (the
real image pair is individually resolved), NOT the *envelope-negligible*
threshold.  The dropped far-field envelope decays like ``|E_ff|/|F| ~
(w*min_delta_tau)^-1.5`` -- measured ``O(0.3)`` at ``w*min_delta_tau ~ 23`` and
only ``~4e-3`` at ``w*min_delta_tau ~ 113``.  So the COMPLEX difference is
``O(1)`` at the boundary (see `SaddleGaugeSelfFalsificationTestCase`).  What IS
true, and what these tests pin, is Professor Q5e's point that "only a phase
differs, which |F| cannot see": the MAGNITUDE ``|F_tier1|`` converges to
``|F_exact|`` (``~1e-3`` in the resolved regime), and -- exactly -- the tier-1
zero-envelope ``F`` is gauge-INDEPENDENT (bit-identical under the switch vs
phase gauge), so a mis-keyed regional gauge switch CANNOT introduce a jump in
the tier-1 branch.  That gauge-independence is the definitive form of the
spec's diagnostic.

Cost.  Every test is closed-form or a single ``ChangRefsdalChannels`` build on
a 40-node band; no training, no engine sweep.  The two ``evaluate`` builds
(exact total, one per fixture class) dominate at well under a second each.
Whole file runs in a few seconds on the fast tier.
"""
from __future__ import annotations

import importlib
import itertools
import math
import pathlib
import unittest

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels,
    FARFIELD_DIFFRACTIVE,
    FARFIELD_KERNEL_SUM,
    reconstruct_farfield,
    farfield_envelope_from_partition,
    _frame_phase,
)
from cogwheel.lensing.chang_refsdal.operator import RHO_END
from cogwheel.lensing.ppgo_map import caustic_rho
from cogwheel.lensing.chang_refsdal._gauge import (
    _saddle_switch_delay,
    _saddle_phase_delay,
    _RHO_END,
)
from cogwheel.lensing.likelihood import (
    _SADDLE_FARFIELD_RHO_FLOOR,
    _saddle_farfield_analytic_serves,
)


#: Band-floor dimensionless frequency of the representative chart band.
BAND_W_MIN = 12.0
#: Band-ceiling dimensionless frequency of the representative chart band.
BAND_W_MAX = 58.0
#: Number of log-spaced nodes across the band.
N_BAND = 40
#: Log-spaced band ``w`` grid shared by the round-trip / handover fixtures.
BAND_W = np.geomspace(BAND_W_MIN, BAND_W_MAX, N_BAND)

#: Independent hard-coded oracle for the resolvability scale (mirrors
#: ``operator.RHO_END``); used so the identity oracle never imports the
#: constant it is meant to bless.
RHO_END_ORACLE = 4.0

#: Absolute tolerance for the gauge closed-form identities (exact arithmetic).
IDENTITY_TOL = 1e-13
#: Absolute tolerance for the frame-phase demod/re-mod round-trip recovery.
ROUND_TRIP_TOL = 1e-13
#: Band-top magnitude-continuity tolerance (resolved regime; far-field bar).
#: Measured 1.4e-7 on the at-the-floor fixture; ~7x headroom.
MAG_CONTINUITY_TOL_TOP = 1e-6
#: Whole-band magnitude-continuity envelope tolerance.  Measured max 5.0e-5 on
#: the at-the-floor fixture; ~4x headroom.  Both bounds were 1e-3 / 5e-3 until
#: 2026-08-13, sized for the retired rho = 1.5 fixture the gate refuses -- a
#: bar 100x looser than the served domain supports, which would have read as
#: the rung's accuracy to anyone who did not check the fixture's rho.
MAG_CONTINUITY_TOL_BAND = 2e-4

#: Representative WELL-SEPARATED far-from-caustic macro saddle (gamma > 1):
#: 2 real images, ``min_delta_tau ~ 6.98``, so ``w_lo*min_delta_tau ~ 84`` at
#: the band floor -- tier-1 serves the whole band.  Magnitude residual is
#: ~2.9e-3 at the floor and ~3e-5 at the top.
SADDLE_GAMMA = 1.3
#: Caustic-relative radius of the handover fixture, placed JUST INSIDE the
#: production serve floor (`_SADDLE_FARFIELD_RHO_FLOOR` = 2.0).  This is the
#: rho at which the near->far handover actually occurs, so it is where the
#: continuity bound belongs and is the WORST admitted case (accuracy improves
#: monotonically outward: measured band-wide |F| residual 5.0e-5 at rho 2.00,
#: 4.6e-6 at 2.50, 1.5e-7 at 3.50).  The small 0.02 margin above the floor
#: keeps admission off an exact floating-point equality without materially
#: changing the residual (5.02e-5 at 2.00 vs 5.05e-5 here).
#:
#: This fixture was rho = 1.5 until 2026-08-13, which the two-term gate
#: REFUSES -- the continuity bounds were being certified on a domain the rung
#: never serves, at a residual 57x worse (2.9e-3) than the worst case it does.
SADDLE_RHO = 2.02
SADDLE_ANGLE = 0.0

#: Output directory for diagnostic plots.
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'


def _polar_source(rho: float, angle: float, gamma: float,
                  kappa: float = 0.0) -> np.ndarray:
    """Source position at ``rho`` times the caustic reach along ``angle``.

    Parameters
    ----------
    rho : float
        Radial multiple of the caustic reach (``> 1`` is exterior).
    angle : float
        Source-plane polar angle [radians].
    gamma : float
        External shear magnitude (``> 1`` for a macro saddle).
    kappa : float, optional
        External convergence.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` source position ``(y1, y2)``.
    """
    reach = geometry.r_caustic(gamma, angle, kappa=kappa)
    return rho * reach * np.array([math.cos(angle), math.sin(angle)])


def _saddle_delays(gamma: float, source: np.ndarray) -> np.ndarray:
    """Fermat delays of the images of ``source`` for a ``gamma`` macro matrix.

    Independent of the channels partition -- built straight from
    ``geometry.find_images`` / ``geometry.delay`` so it can serve as an
    oracle for the served ``min_delta_tau``.
    """
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    images = geometry.find_images(source, matrix)
    return np.array([geometry.delay(image, source, matrix)
                     for image in images], dtype=float)


def _min_delta_tau(delays: np.ndarray) -> float:
    """Narrowest positive pairwise delay gap of ``delays``."""
    ordered = np.sort(np.asarray(delays, dtype=float))
    gaps = np.diff(ordered)
    positive = gaps[gaps > 0.0]
    return float(positive.min())


def _save_diagnostic_plot(fig, name: str) -> None:
    """Save a matplotlib figure to the output directory."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / name, dpi=100, bbox_inches='tight')


class _SaddleGaugeTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity guard shared by every suite class."""

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        """Fail if zero comparisons ran (anti-vacuity)."""
        if self.n_checks == 0:
            self.fail(
                f'{self._testMethodName}: zero comparisons ran -- the test '
                f'is vacuous (all configs skipped or no assertion fired).')


# ======================================================================
# Part 1 -- GAUGE IDENTITY / PURITY
# ======================================================================
class SaddleGaugeIdentityTestCase(_SaddleGaugeTestCase):
    """The two gauge functions are the exact closed forms and are pure.

    ``w_min`` is a passed band-floor, never a live ``w`` -- so the gauge is a
    fixed property of the band, which is exactly what lets the deferred tier-2
    stored object share one authoritative gauge with the live rung.
    """

    #: (tau_min, w_min) probe grid spanning realistic saddle delays / floors.
    _TAU_MIN_GRID = (0.0, 1.0, 4.361, 5.2272, 8.1967, 20.0)
    _W_MIN_GRID = (12.0, 20.0, 37.5, 58.0)

    def test_switch_delay_matches_closed_form(self):
        """``_saddle_switch_delay`` == ``tau_min - RHO_END/w_min`` to 1e-13."""
        for tau_min, w_min in itertools.product(
                self._TAU_MIN_GRID, self._W_MIN_GRID):
            with self.subTest(tau_min=tau_min, w_min=w_min):
                # Independent oracle: hard-coded RHO_END_ORACLE, NOT the
                # module constant the function uses.
                oracle = tau_min - RHO_END_ORACLE / w_min
                got = _saddle_switch_delay(tau_min, w_min)
                self.n_checks += 1
                self.assertAlmostEqual(
                    got, oracle, delta=IDENTITY_TOL,
                    msg=f'switch delay {got!r} != oracle {oracle!r}')

    def test_phase_delay_is_bit_identity(self):
        """``_saddle_phase_delay(tau_min)`` returns ``tau_min`` bit-exactly."""
        for tau_min in self._TAU_MIN_GRID:
            with self.subTest(tau_min=tau_min):
                got = _saddle_phase_delay(tau_min)
                self.n_checks += 1
                # Bit-exact: it is the identity, not merely close.
                self.assertEqual(got, tau_min)

    def test_rho_end_mirror_is_consistent(self):
        """The mirrored ``_RHO_END`` equals ``operator.RHO_END`` == 4.0.

        Load-bearing: ``_saddle_switch_delay`` subtracts ``_RHO_END/w_min``;
        if the mirror drifted from the authoritative ``operator.RHO_END`` the
        switch gauge and the resolvability gate would key off different
        scales.
        """
        self.n_checks += 1
        self.assertEqual(_RHO_END, RHO_END)
        self.n_checks += 1
        self.assertEqual(_RHO_END, RHO_END_ORACLE)

    def test_switch_minus_phase_is_rho_end_over_w_min(self):
        """Gauge SPLIT is exactly ``-RHO_END/w_min`` (the window width)."""
        for tau_min, w_min in itertools.product(
                self._TAU_MIN_GRID, self._W_MIN_GRID):
            with self.subTest(tau_min=tau_min, w_min=w_min):
                split = (_saddle_switch_delay(tau_min, w_min)
                         - _saddle_phase_delay(tau_min))
                self.n_checks += 1
                self.assertAlmostEqual(
                    split, -RHO_END_ORACLE / w_min, delta=IDENTITY_TOL)

    def test_gauge_functions_are_pure_on_repeat(self):
        """Repeated calls return bit-identical values (no hidden state)."""
        for tau_min, w_min in itertools.product(
                self._TAU_MIN_GRID, self._W_MIN_GRID):
            with self.subTest(tau_min=tau_min, w_min=w_min):
                first_s = _saddle_switch_delay(tau_min, w_min)
                second_s = _saddle_switch_delay(tau_min, w_min)
                first_p = _saddle_phase_delay(tau_min)
                second_p = _saddle_phase_delay(tau_min)
                self.n_checks += 1
                self.assertEqual(first_s, second_s)
                self.n_checks += 1
                self.assertEqual(first_p, second_p)

    def test_gauge_home_is_single_authoritative_object(self):
        """An independent import site resolves the SAME function objects.

        Proves there is one authoritative gauge home -- no divergent copy a
        train path and a serve path could drift between.
        """
        fresh = importlib.import_module(
            'cogwheel.lensing.chang_refsdal._gauge')
        self.n_checks += 1
        self.assertIs(fresh._saddle_switch_delay, _saddle_switch_delay)
        self.n_checks += 1
        self.assertIs(fresh._saddle_phase_delay, _saddle_phase_delay)
        # ... and they agree numerically from the independent alias.
        for tau_min, w_min in itertools.product(
                self._TAU_MIN_GRID, self._W_MIN_GRID):
            with self.subTest(tau_min=tau_min, w_min=w_min):
                self.n_checks += 1
                self.assertEqual(
                    fresh._saddle_switch_delay(tau_min, w_min),
                    _saddle_switch_delay(tau_min, w_min))


# ======================================================================
# Part 2 -- SYNTHETIC DEMOD / RE-MOD ROUND-TRIP (de-risks deferred tier-2)
# ======================================================================
class SaddleFramePhaseRoundTripTestCase(_SaddleGaugeTestCase):
    """The ``_frame_phase`` demod/re-mod pair telescopes to machine precision.

    A smooth analytic envelope ``E(w)`` (a low-order polynomial in ``log w``,
    NO engine, NO training) is demodulated by ``exp(+1j w tau_phase)`` with
    ``tau_phase = _saddle_phase_delay(tau_min)`` -- mimicking
    `farfield_envelope_from_partition`'s stored label -- then rebuilt through
    `reconstruct_farfield`'s ``exp(-1j w t_min)`` frame round-trip under the
    ``FARFIELD_DIFFRACTIVE`` tag (all-zero switch -> the reconstruction returns
    the re-modulated envelope unchanged, isolating the frame telescoping).  The
    original field is recovered to ``ROUND_TRIP_TOL`` (measured ~2e-16), the
    guarantee the deferred tier-2 stored object relies on.
    """

    @classmethod
    def setUpClass(cls):
        cls.w = BAND_W
        source = _polar_source(SADDLE_RHO, SADDLE_ANGLE, SADDLE_GAMMA)
        cls.geom = ChangRefsdalChannels(cls.w).geometry_partition(
            gamma=SADDLE_GAMMA, y=(float(source[0]), float(source[1])),
            beta=0.0, kappa=0.0)
        cls.tau_min = float(cls.geom.t_min)
        cls.tau_phase = _saddle_phase_delay(cls.tau_min)
        # Smooth analytic envelope: low-order complex polynomial in log w.
        logw = np.log(cls.w)
        centred = (logw - logw.mean()) / np.ptp(logw)
        cls.envelope = ((0.7 - 0.9 * centred + 0.4 * centred**2)
                        + 1j * (0.3 + 0.5 * centred - 0.2 * centred**2))

    def _round_trip(self, t_min: float) -> np.ndarray:
        """Demodulate the envelope, then rebuild through the frame round-trip.

        Builds the STORED label exactly as `farfield_envelope_from_partition`
        does (``E * exp(+1j * _frame_phase(w, t_min))``) and feeds it to
        `reconstruct_farfield` under ``FARFIELD_DIFFRACTIVE`` (all-zero switch),
        so the returned total is the re-modulated envelope alone.
        """
        stored = self.envelope * np.exp(1j * _frame_phase(self.w, t_min))
        _kernels, total = reconstruct_farfield(
            self.w, stored, self.geom.delays, self.geom.saddle_kernels,
            self.geom.real_mask, FARFIELD_DIFFRACTIVE, t_min)
        return total

    def test_demod_remod_recovers_envelope(self):
        """Round-trip recovers ``E(w)`` to ``ROUND_TRIP_TOL``."""
        total = self._round_trip(self.tau_phase)
        err = float(np.max(np.abs(total - self.envelope)))
        self.n_checks += 1
        self.assertLess(
            err, ROUND_TRIP_TOL,
            msg=f'round-trip recovery error {err:.3e} >= '
                f'{ROUND_TRIP_TOL:.0e} (tau_phase={self.tau_phase:.4f}, '
                f'max w*tau={self.w.max() * self.tau_phase:.1f} rad)')

    def test_round_trip_is_deterministic(self):
        """Two identical round-trips are bit-identical (pure re-mod)."""
        first = self._round_trip(self.tau_phase)
        second = self._round_trip(self.tau_phase)
        self.n_checks += 1
        self.assertEqual(
            float(np.max(np.abs(first - second))), 0.0,
            msg='round-trip is not deterministic')

    def test_phase_delay_gauge_matches_partition_frame(self):
        """The phase gauge equals the partition frame origin it demodulates.

        ``_saddle_phase_delay(geom.t_min)`` is exactly ``geom.t_min`` (the
        min-relative frame origin the stored far-field label lives in), so the
        gauge and the producer's frame cannot differ.
        """
        self.n_checks += 1
        self.assertEqual(self.tau_phase, self.tau_min)

    def test_round_trip_recovery_plot(self):
        """Diagnostic: |E| and recovered |total| across the band."""
        total = self._round_trip(self.tau_phase)
        self.n_checks += 1
        self.assertEqual(total.shape, self.envelope.shape)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            ax0.plot(self.w, np.abs(self.envelope), 'k-', label='|E| input')
            ax0.plot(self.w, np.abs(total), 'r--', label='|total| recovered')
            ax0.set_ylabel('magnitude')
            ax0.legend()
            ax0.set_title('Part 2: frame-phase demod/re-mod round-trip')
            ax1.semilogy(self.w, np.abs(total - self.envelope) + 1e-18, 'b-')
            ax1.axhline(ROUND_TRIP_TOL, color='grey', ls=':',
                        label=f'tol {ROUND_TRIP_TOL:.0e}')
            ax1.set_xlabel('w')
            ax1.set_ylabel('|recovery error|')
            ax1.legend()
            _save_diagnostic_plot(
                fig, 'test_saddle_gauge_round_trip_recovery.png')
            plt.close(fig)
        except ImportError:
            pass


# ======================================================================
# Part 3 -- NEAR / FAR HANDOVER CONTINUITY
# ======================================================================
class SaddleHandoverContinuityTestCase(_SaddleGaugeTestCase):
    """Tier-1 (far) and the envelope-carrying (near) reconstruction agree.

    For the WELL-SEPARATED fixture (whole band resolved, ``w_lo*mdt ~ 84``):

    * the near-caustic envelope-carrying reconstruction -- the full producer
      label from `farfield_envelope_from_partition` rebuilt through
      `reconstruct_farfield` -- reproduces ``partition.exact_total``
      BIT-EXACTLY (the ``_frame_phase`` telescoping the mid-band label proves);
    * the tier-1 zero-envelope reconstruction is gauge-INDEPENDENT bit-exactly
      (switch gauge vs phase gauge give identical ``F``), so a mis-keyed
      regional gauge switch CANNOT jump the served field -- the definitive
      form of the spec's diagnostic;
    * the tier-1 magnitude tracks the exact magnitude to the far-field bar
      (``~3e-5`` at the resolved band top, ``~2.9e-3`` band-wide), so handing
      the serve from the near (exact) branch to the far (tier-1) branch moves
      ``|F_serve|`` by no more than the surrogate accuracy at every node.

    Note (measured, see the module docstring and
    `SaddleGaugeSelfFalsificationTestCase`): the ``w_lo*mdt = RHO_END = 4``
    resolvability boundary is NOT where the dropped envelope becomes negligible
    -- near a fold the tier-1 field is O(1) wrong even at ``w_lo*mdt ~ 25``.
    These agreement bounds are measured AT THE SERVE FLOOR (``rho = 2.02``,
    just inside `_SADDLE_FARFIELD_RHO_FLOOR`), which is where the near->far
    handover actually happens and therefore where the continuity bound is
    load-bearing.  It is also the worst admitted case: accuracy improves
    monotonically outward in ``rho``.  Resolvability alone does NOT buy this
    -- near a fold the tier-1 field is O(1) wrong even at ``w_lo*mdt ~ 25`` --
    which is exactly why the gate carries a rho term as well.
    """

    @classmethod
    def setUpClass(cls):
        cls.w = BAND_W
        source = _polar_source(SADDLE_RHO, SADDLE_ANGLE, SADDLE_GAMMA)
        y = (float(source[0]), float(source[1]))
        #: Production rho of this fixture, computed the way the live rung
        #: computes it, so the gate tests below probe the REAL served point
        #: rather than an invented admitted value.
        cls.rho = caustic_rho(
            SADDLE_GAMMA, float(np.hypot(y[0], y[1])), kappa=0.0)
        cls.part = ChangRefsdalChannels(cls.w).evaluate(
            gamma=SADDLE_GAMMA, y=y, beta=0.0, kappa=0.0)
        cls.geom = ChangRefsdalChannels(cls.w).geometry_partition(
            gamma=SADDLE_GAMMA, y=y, beta=0.0, kappa=0.0)
        cls.tau_min = float(cls.geom.t_min)
        cls.exact = cls.part.exact_total
        # Tier-1 zero-envelope reconstruction (the far serve rung).
        env0 = np.zeros(cls.w.shape, dtype=complex)
        _k, cls.f_tier1 = reconstruct_farfield(
            cls.w, env0, cls.geom.delays, cls.geom.saddle_kernels,
            cls.geom.real_mask, FARFIELD_KERNEL_SUM, cls.tau_min)
        # Resolved real-image delay gap (the eta driving the serve gate).
        real = np.asarray(cls.geom.real_mask, dtype=bool)
        cls.real_delays = np.asarray(cls.geom.delays)[real]
        cls.min_delta_tau = _min_delta_tau(cls.real_delays)

    def test_producer_round_trip_reproduces_exact(self):
        """Producer label -> `reconstruct_farfield` == ``exact_total`` exactly.

        This is the near-caustic (envelope-carrying) branch of the handover:
        `farfield_envelope_from_partition` demodulates the exact total to the
        stored label, and `reconstruct_farfield` re-modulates it back.  The
        ``_frame_phase`` demod/re-mod pair telescopes to machine precision, so
        the recovered ``F`` equals ``partition.exact_total`` to
        ``ROUND_TRIP_TOL`` (measured 0.0).
        """
        label = farfield_envelope_from_partition(
            self.part, FARFIELD_KERNEL_SUM)
        _k, f_near = reconstruct_farfield(
            self.w, label, self.geom.delays, self.geom.saddle_kernels,
            self.geom.real_mask, FARFIELD_KERNEL_SUM, self.tau_min)
        rel = float(np.max(np.abs(f_near - self.exact))
                    / np.max(np.abs(self.exact)))
        self.n_checks += 1
        self.assertLess(rel, ROUND_TRIP_TOL,
                        msg=f'producer round-trip rel err {rel:.3e}')

    def test_tier1_reconstruction_is_gauge_independent(self):
        """Tier-1 ``F`` is bit-identical under the switch vs phase gauge.

        The zero-envelope reconstruction's re-modulation multiplies
        ``0 * exp(-1j w t_min) = 0``, so the reconstructed total is the pure
        switched-kernel sum -- independent of the frame origin.  Feeding the
        SWITCH-gauge delay ``_saddle_switch_delay(tau_min, w_min)`` instead of
        the PHASE-gauge ``tau_min`` therefore gives a BIT-IDENTICAL field.  A
        mis-keyed regional gauge switch cannot introduce a jump here.
        """
        switch_delay = _saddle_switch_delay(self.tau_min, BAND_W_MIN)
        env0 = np.zeros(self.w.shape, dtype=complex)
        _k, f_switch_gauge = reconstruct_farfield(
            self.w, env0, self.geom.delays, self.geom.saddle_kernels,
            self.geom.real_mask, FARFIELD_KERNEL_SUM, switch_delay)
        self.n_checks += 1
        self.assertEqual(
            float(np.max(np.abs(self.f_tier1 - f_switch_gauge))), 0.0,
            msg='tier-1 reconstruction is not gauge-independent')

    def test_tier1_magnitude_matches_exact_at_band_top(self):
        """``|F_tier1|`` == ``|F_exact|`` at the resolved band top to 1e-3."""
        rel_top = float(abs(abs(self.f_tier1[-1]) - abs(self.exact[-1]))
                        / abs(self.exact[-1]))
        self.n_checks += 1
        self.assertLess(rel_top, MAG_CONTINUITY_TOL_TOP,
                        msg=f'band-top magnitude residual {rel_top:.3e}')

    def test_tier1_magnitude_matches_exact_across_band(self):
        """``|F_tier1|`` tracks ``|F_exact|`` band-wide within 5e-3.

        The per-node magnitude residual bounds the jump a near->far serve
        handover makes in ``|F_serve|`` at ANY node, so the served magnitude
        is continuous across the handover to the far-field bar.
        """
        mag_res = (np.abs(np.abs(self.f_tier1) - np.abs(self.exact))
                   / np.abs(self.exact))
        worst = float(mag_res.max())
        self.n_checks += 1
        self.assertLess(worst, MAG_CONTINUITY_TOL_BAND,
                        msg=f'band-wide magnitude residual {worst:.3e}')

    def test_gauge_rho_end_matches_the_operator_constant(self):
        """`_gauge._RHO_END` equals `operator.RHO_END`.

        `_gauge` is a shared leaf and `test_lensing_gauge` pins that it
        imports nothing but numpy -- naming `operator` as forbidden -- so it
        CANNOT bind the constant from `operator` the way `_born` does.  Its
        literal is a deliberate duplicate, and this is the pin that stops the
        duplicate drifting: the alternative (importing operator) breaks the
        leaf invariant, which is a worse failure than a guarded copy.
        """
        self.n_checks += 1
        self.assertEqual(
            _RHO_END, RHO_END,
            'the _gauge copy of the SACR-C resolution scale has drifted '
            'from operator.RHO_END; update the literal (do NOT import '
            'operator -- see GaugeIndependenceTestCase).')

    def test_accuracy_improves_outward_so_the_floor_is_worst_case(self):
        """Residual falls monotonically with ``rho`` across the served domain.

        This is the assumption that makes an AT-THE-FLOOR fixture a valid
        certification of the whole served domain: if the tier-1 residual did
        not improve outward, bounding it at ``rho = 2.02`` would say nothing
        about ``rho = 3``.  Pinned here because the claim is load-bearing for
        the suite's design and would otherwise live only in a docstring.
        """
        rhos = (SADDLE_RHO, 2.5, 3.0, 3.5)
        w = np.geomspace(BAND_W_MIN, BAND_W_MAX, 12)
        residuals = []
        for rho in rhos:
            src = _polar_source(rho, SADDLE_ANGLE, SADDLE_GAMMA)
            y = (float(src[0]), float(src[1]))
            geom = ChangRefsdalChannels(w).geometry_partition(
                gamma=SADDLE_GAMMA, y=y, beta=0.0, kappa=0.0)
            exact = ChangRefsdalChannels(w).evaluate(
                gamma=SADDLE_GAMMA, y=y, beta=0.0, kappa=0.0).exact_total
            _k, served = reconstruct_farfield(
                w, np.zeros(w.shape, dtype=complex), geom.delays,
                geom.saddle_kernels, geom.real_mask, FARFIELD_KERNEL_SUM,
                float(geom.t_min))
            residuals.append(float(
                (np.abs(np.abs(served) - np.abs(exact))
                 / np.abs(exact)).max()))
        for (r_in, res_in), (r_out, res_out) in zip(
                zip(rhos, residuals), zip(rhos[1:], residuals[1:])):
            self.n_checks += 1
            self.assertLess(
                res_out, res_in,
                f'residual did not improve outward: rho {r_in} -> {r_out} '
                f'gave {res_in:.3e} -> {res_out:.3e}.  The at-the-floor '
                f'fixture no longer bounds the served domain.')
        self.n_checks += 1
        self.assertEqual(
            residuals.index(max(residuals)), 0,
            f'the floor is not the worst admitted case: {residuals}')

    def test_serve_gate_has_single_monotone_boundary_crossing(self):
        """The eta serve gate flips once, at ``w_lo = RHO_END / mdt``.

        Sweeping the band floor ``w_lo`` (the resolvability parameter
        ``eta = w_lo * min_delta_tau``) across the boundary, the gate
        `_saddle_farfield_analytic_serves` is monotone False -> True with a
        single crossing at ``w_lo = RHO_END / min_delta_tau``.  A mis-keyed
        gauge switch region would show as a second crossing here.

        The gate has TWO terms; this fixture's own production ``rho`` clears
        the floor, so the caustic-proximity term is constantly True and this
        sweep probes the resolvability term alone -- at a point the rung
        genuinely serves.
        """
        crossing = RHO_END / self.min_delta_tau
        w_lo_grid = np.linspace(0.2 * crossing, 3.0 * crossing, 60)
        served = np.array([
            _saddle_farfield_analytic_serves(
                self.real_delays, float(w_lo), self.rho)
            for w_lo in w_lo_grid], dtype=bool)
        # Exactly one False->True transition, none True->False.
        transitions = np.diff(served.astype(int))
        self.n_checks += 1
        self.assertEqual(int((transitions == 1).sum()), 1,
                         msg='not exactly one False->True crossing')
        self.n_checks += 1
        self.assertEqual(int((transitions == -1).sum()), 0,
                         msg='gate is non-monotone (a True->False flip)')
        # Boundary bracketed: refuse just below, serve just above.
        self.n_checks += 1
        self.assertFalse(_saddle_farfield_analytic_serves(
            self.real_delays, crossing * (1.0 - 1e-6), self.rho))
        self.n_checks += 1
        self.assertTrue(_saddle_farfield_analytic_serves(
            self.real_delays, crossing * (1.0 + 1e-6), self.rho))

    def test_handover_continuity_plot(self):
        """Diagnostic: |F_tier1| vs |F_exact| across the band (no jump)."""
        self.n_checks += 1
        self.assertEqual(self.f_tier1.shape, self.exact.shape)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            ax0.plot(self.w, np.abs(self.exact), 'k-',
                     label='|F| exact (near serve)')
            ax0.plot(self.w, np.abs(self.f_tier1), 'r--',
                     label='|F| tier-1 (far serve)')
            ax0.set_ylabel('|F|')
            ax0.legend()
            ax0.set_title('Part 3: near/far handover -- |F_serve| continuity')
            mag_res = (np.abs(np.abs(self.f_tier1) - np.abs(self.exact))
                       / np.abs(self.exact))
            ax1.semilogy(self.w, mag_res + 1e-18, 'b-')
            ax1.axhline(MAG_CONTINUITY_TOL_BAND, color='grey', ls=':',
                        label=f'band tol {MAG_CONTINUITY_TOL_BAND:.0e}')
            ax1.set_xlabel('w')
            ax1.set_ylabel('||F_tier1|-|F_exact|| / |F_exact|')
            ax1.legend()
            _save_diagnostic_plot(
                fig, 'test_saddle_gauge_handover_continuity.png')
            plt.close(fig)
        except ImportError:
            pass


# ======================================================================
# Part 4 -- SELF-FALSIFICATION (the suite can go red)
# ======================================================================
class SaddleGaugeSelfFalsificationTestCase(_SaddleGaugeTestCase):
    """Each guard the suite relies on is shown to have teeth.

    Also encodes the FLAGGED spec discrepancy as a positive fact: near a fold
    the dropped far-field envelope is O(1), so the tier-1 serve gate is
    load-bearing (``RHO_END`` is a resolvability scale, not an
    envelope-negligible scale).
    """

    @classmethod
    def setUpClass(cls):
        cls.w = BAND_W
        # Well-separated fixture for the gauge / round-trip teeth.
        far = _polar_source(SADDLE_RHO, SADDLE_ANGLE, SADDLE_GAMMA)
        cls.geom = ChangRefsdalChannels(cls.w).geometry_partition(
            gamma=SADDLE_GAMMA, y=(float(far[0]), float(far[1])),
            beta=0.0, kappa=0.0)
        cls.tau_min = float(cls.geom.t_min)
        logw = np.log(cls.w)
        centred = (logw - logw.mean()) / np.ptp(logw)
        cls.envelope = ((0.7 - 0.9 * centred + 0.4 * centred**2)
                        + 1j * (0.3 + 0.5 * centred - 0.2 * centred**2))

    def test_wrong_switch_formula_is_detected(self):
        """A sign-flipped switch formula differs from the closed form.

        Proves `SaddleGaugeIdentityTestCase` has teeth: the WRONG
        ``tau_min + RHO_END/w_min`` deviates by ``2*RHO_END/w_min`` (measured
        ~0.67 at ``w_min = 12``), far above ``IDENTITY_TOL``.
        """
        for w_min in (12.0, 58.0):
            with self.subTest(w_min=w_min):
                correct = _saddle_switch_delay(self.tau_min, w_min)
                wrong = self.tau_min + RHO_END / w_min
                self.n_checks += 1
                self.assertGreater(abs(correct - wrong), 1e3 * IDENTITY_TOL)

    def test_wrong_frame_origin_breaks_round_trip(self):
        """Re-modulating against a WRONG ``t_min`` wrecks the round-trip.

        Storing against the correct phase gauge but reconstructing against a
        shifted frame origin leaves an O(1) recovery error (measured ~1.87),
        so `SaddleFramePhaseRoundTripTestCase` cannot pass by accident.
        """
        stored = self.envelope * np.exp(1j * _frame_phase(self.w, self.tau_min))
        _k, total = reconstruct_farfield(
            self.w, stored, self.geom.delays, self.geom.saddle_kernels,
            self.geom.real_mask, FARFIELD_DIFFRACTIVE, self.tau_min + 0.5)
        err = float(np.max(np.abs(total - self.envelope)))
        self.n_checks += 1
        self.assertGreater(err, 1e-2,
                           msg=f'wrong-frame round-trip error {err:.3e} '
                               f'unexpectedly small')

    def test_nonzero_envelope_makes_reconstruction_gauge_dependent(self):
        """With a NONZERO envelope the reconstruction IS gauge-dependent.

        Proves `test_tier1_reconstruction_is_gauge_independent` has teeth: the
        gauge-independence of tier-1 relies specifically on the ZERO envelope.
        Reconstructing a nonzero envelope against two different frame origins
        gives materially different ``F`` (measured ~2.08).
        """
        _k1, f1 = reconstruct_farfield(
            self.w, self.envelope, self.geom.delays, self.geom.saddle_kernels,
            self.geom.real_mask, FARFIELD_KERNEL_SUM, self.tau_min)
        _k2, f2 = reconstruct_farfield(
            self.w, self.envelope, self.geom.delays, self.geom.saddle_kernels,
            self.geom.real_mask, FARFIELD_KERNEL_SUM, self.tau_min + 1.0)
        self.n_checks += 1
        self.assertGreater(float(np.max(np.abs(f1 - f2))), 1e-2,
                           msg='nonzero envelope reconstruction was gauge '
                               'independent -- the zero-envelope invariant '
                               'would be vacuous')

    def test_dropping_envelope_is_order_one_near_a_fold(self):
        """Near the caustic the dropped envelope is O(1) -- gate is needed.

        Encodes the flagged spec discrepancy: at ``rho = 1.10`` (still
        formally resolved, ``w_lo*mdt ~ 31``) the tier-1 zero-envelope field
        differs from the exact total by O(1) somewhere in the band (measured
        max ~8), so ``RHO_END = 4`` is a RESOLVABILITY scale, not an
        envelope-negligible one, and the tier-1 rung MUST remain gated to the
        far, well-separated regime.
        """
        near = _polar_source(1.10, SADDLE_ANGLE, SADDLE_GAMMA)
        y = (float(near[0]), float(near[1]))
        part = ChangRefsdalChannels(self.w).evaluate(
            gamma=SADDLE_GAMMA, y=y, beta=0.0, kappa=0.0)
        geom = ChangRefsdalChannels(self.w).geometry_partition(
            gamma=SADDLE_GAMMA, y=y, beta=0.0, kappa=0.0)
        env0 = np.zeros(self.w.shape, dtype=complex)
        _k, f_tier1 = reconstruct_farfield(
            self.w, env0, geom.delays, geom.saddle_kernels, geom.real_mask,
            FARFIELD_KERNEL_SUM, float(geom.t_min))
        cplx_res = (np.abs(f_tier1 - part.exact_total)
                    / np.abs(part.exact_total))
        self.n_checks += 1
        self.assertGreater(float(cplx_res.max()), 0.1,
                           msg='dropped envelope was negligible near a fold -- '
                               'the resolvability caveat would be false')


if __name__ == '__main__':
    unittest.main()
