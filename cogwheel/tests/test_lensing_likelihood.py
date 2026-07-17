"""
Tests for `lensing.likelihood` -- the multi-component relative-binning
likelihood `LensedRelativeBinningLikelihood` for Chang--Refsdal
microlensed CBC signals.

WHAT THIS SUITE PINS (the crown gate, cogwheel value #1: correctness)
---------------------------------------------------------------------
The relative-binning ``lnlike`` is a fast approximation of a Gaussian
log-likelihood; it is worth nothing unless it reproduces the EXACT
answer.  These are the independent oracles used here, each chosen so the
check is not circular:

* ``lnlike`` (relative binning) vs ``lnlike_bruteforce`` (the same
  candidate strain assembled on the FULL FFT grid through the same
  `LensedWaveformGenerator`).  This isolates the *binning* approximation
  -- both paths use the identical amplification and waveform, so any
  disagreement is the moment/Taylor bin model, nothing else.  Swept over
  2-image, 4-image, near-fold, near-cusp, kappa!=0 and beta!=0 lens
  configurations, a candidate whose waveform is offset from the fiducial
  (exercising the residual-delay/linear-free machinery), and higher-mode
  content (IMRPhenomXPHM: |m| in {1, 2, 3, 4}).

* The ``F -> 1`` limit against cogwheel's EXISTING unlensed machinery --
  `CBCLikelihood.lnlike_fft` (exact, no binning).  It is not defined in
  the module under test, so this catches an overall NORMALIZATION error
  (a missing ``4 df``, a stray factor of two) that the same-generator
  brute-force comparison -- which shares the module's normalization --
  structurally cannot.  Two gates: a LOOSE factor gate on the seeded
  NOISY fixture (catches a gross O(SNR^2) error), and a TIGHT physically
  meaningful FLOOR on a ZERO-NOISE anchor (data == pure fiducial signal,
  so the ``<noise|delta-h>`` tail vanishes and only the ~1e-4 lensing
  residual plus a small template-construction ``delta-h`` remain).

  The ``F -> 1`` limit requires a TRIVIAL MACRO SECTOR.  operator.py's
  certified ``w -> 0`` limit is the macro constant ``1/sqrt(1 - gamma**2)``
  (at ``kappa = 0``) -- NOT unity.  A tiny-mass candidate at
  ``gamma = 0.20`` therefore tends to 1.0206207..., and the 0.1214 nat
  offset it showed against an unlensed reference was correct physics
  measured against the wrong oracle, not a normalization bug.  The
  zero-noise floor pair consequently uses ``gamma = kappa = 0``, where the
  unlensed limit genuinely holds.  Deleting the inconvenient
  configuration would be a dodge, so `MacroSectorContrastTestCase` keeps
  it: two tiny-mass candidates differing ONLY in shear, with the sheared
  one's offset PREDICTED in closed form from
  ``lnL(c*h0) - lnL(h0) = -0.5*(c - 1)**2*(h0|h0)`` at ``d == h0``.  That
  anchor goes red if the macro magnification is ever normalized out of
  the engine, or a small-``w`` short-circuit forcing ``F = 1`` returns.

* A PRODUCT-OF-SUMMARIES structural regression: at a near-fold
  configuration two images have near-degenerate delays and the cross
  term dominates ``(h_L|h_L)``.  Summarizing ``F`` and ``h`` separately
  and multiplying the summaries drops that cross term's in-bin phase
  structure; pinning ``lnlike`` to the exact ``lnlike_bruteforce`` there
  is the guard.

* A NEAR-CUSP regression PIN with a falsifying canary.  At the near-cusp
  source a real image's true cluster mate is a virtual label parked at
  the critical point; the production ``_channel_switch`` measures delay
  separation over the WHOLE cluster (paper Eq. delay-separation), keeping
  a still-merged channel in the bounded artificial gauge.  The pre-WP1
  rule measured it over REAL channels only and handed the channel to the
  divergent stationary-phase kernel, so the per-bin kernels ``K_a`` blew
  up and ``_norm_term`` squared them into a spurious ``(h|h)`` (FINDINGS
  F008, superseding the F006 edge-secant attribution).  The pin holds
  ``lnlike`` to ``lnlike_bruteforce`` at the corrected value AND
  demonstrates, on the SAME event/generator/bins, that monkeypatching the
  switch back to the real-only rule blows ``max|K_a|`` to ``>= 1e3 * |F|``
  while the production switch stays O(|F|) -- so the WP1 switch fix is
  load-bearing and the pin is non-vacuous.

* The ``LensedBinningError`` guard is FALSIFIED, not just described: it
  must fire both at construction (bins too coarse for ``delta_t_max``)
  and at evaluation (a candidate whose image delays exceed
  ``delta_t_max``).

* Timing assertions grounded in the additive ``M^2 + n_img^2`` design:
  (a) the relative-binning ``lnlike`` beats the full-grid matched filter
  ``lnlike_bruteforce`` by a conservative margin (the RB speed-up), and
  (b) the pure ``_data_term`` + ``_norm_term`` contraction is subdominant
  to the amplification-engine call (`_amplification_coefficients`, the
  unavoidable 1F1 special-function evaluation at
  ``n_bins * kernel_subsamples`` points).  We do NOT assert the
  contraction beats the coarse waveform call: that coarse
  ``get_strain_at_detectors(fbin, ...)`` is a per-eval CO-COST shared by
  RB, not RB's competitor; RB's competitor is the full-grid brute force.

* Macro-saddle (non-positive-parity) candidates make BOTH ``lnlike`` and
  ``lnlike_bruteforce`` raise `geometry.LensDomainError`, propagated
  unswallowed -- the likelihood-path analogue of the generator-boundary
  rejection (engine-refusal SYMMETRY: never one path returning a value
  while the other refuses).

DETERMINISM
-----------
Every stochastic input is seeded.  ``data.EventData.gaussian_noise``
draws its noise from its OWN ``np.random.default_rng(seed)``, so a bare
``np.random.seed`` is inert for the noise realization; the fixtures pass
an explicit integer ``seed=SEED``.  `DeterminismTestCase` proves the
seeded ``strain`` is bit-identical across repeated constructions and that
``lnlike`` / ``lnlike_bruteforce`` are exact pure functions of their
inputs (``assertEqual``, not almost-equal).

ANTI-VACUITY AND SELF-FALSIFICATION
-----------------------------------
`LensedLikelihoodTestCase.tearDown` fails a test that made zero
comparisons.  `SelfFalsificationTestCase` proves the central
brute-force-agreement gate can go red.

TOLERANCES (conservative; see per-constant notes)
-------------------------------------------------
The relative-binning-vs-brute-force tolerances are set to catch
STRUCTURAL errors (a wrong contraction is off by O(SNR^2), tens to
hundreds in lnL) with margin above the true binning floor, because the
floor itself is bin-density dependent.  The zero-noise F->1 floor is the
one physically tight gate (1e-2); it is deterministic and, unlike a
noisy floor, is not perched at a failure boundary that a noise draw could
tip over.
"""
from __future__ import annotations

import pathlib
import time
import warnings
from unittest import TestCase, main, mock

import numpy as np
from matplotlib import pyplot as plt

from cogwheel import data, waveform
from cogwheel.likelihood.relative_binning import RelativeBinningLikelihood
from cogwheel.lensing.chang_refsdal import _gauge, channels, geometry, operator
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, LensedBinningError,
    _data_term, _norm_term)

#: Higher-mode approximant so the mode-pair (``M^2``) contraction is
#: genuinely exercised (|m| in {1, 2, 3, 4}), not the trivial 22-only
#: case.  Built into ``waveform.APPROXIMANTS`` (precessing + HM).
APPROXIMANT = 'IMRPhenomXPHM'

#: Fixed seed for every stochastic input.  ``EventData.gaussian_noise``
#: uses ``np.random.default_rng(seed)`` internally, so this -- not a bare
#: ``np.random.seed`` -- is what makes the noise realization reproducible.
SEED = 20260717

#: Bin width [Hz] of the uniform relative-binning grid.  Chosen so the
#: lens-aware criterion ``pi * Delta_f_bin * delta_t_max`` clears the
#: default 0.5 rad tolerance at ``DELTA_T_MAX`` (pi*4*0.02 = 0.25) while
#: staying fine enough that the linear-in-bin component-ratio model is
#: accurate.
DF_BIN = 4.0

#: Largest relative image delay [s] the main fixture's bins support.
DELTA_T_MAX = 0.02

#: Lens mass [Msun] / redshift for the main (well-conditioned) fixture.
#: With ``xi = 8*pi*G*M_L*(1+z_L)/c**3`` these give in-band ``w = xi*f``
#: of order a few -- deep in the wave branch, far from any cancellation
#: refusal -- and image delays of a few ms, comfortably below
#: ``DELTA_T_MAX``.
M_LENS_MSUN = 90.0
Z_LENS = 0.4

#: Absolute / relative tolerance on ``lnlike`` vs ``lnlike_bruteforce``.
#: Conservative (see module docstring): a broken contraction fails by
#: tens-hundreds; the true binning floor is well below these.
RB_ATOL = 1.5
RB_RTOL = 1e-2

#: Tolerance on the LOOSE ``F -> 1`` normalization factor gate (noisy
#: fixture).  Catches an overall factor error (O(SNR^2)); lenient to
#: absorb the band-edge bookkeeping difference between the exact and
#: brute-force paths.  NOT the sub-nat floor (the zero-noise gate is).
NORM_TOL = 0.1

#: Tolerance on the TIGHT zero-noise ``F -> 1`` floor.  On data equal to
#: the pure fiducial signal the noise tail vanishes, leaving the ~1e-4
#: lensing residual plus a small deterministic template-construction
#: ``delta-h``; 1e-2 is meaningful (not perched at a failure boundary).
ZERO_NOISE_TOL = 1e-2

#: Factor by which the pre-WP1 real-only ``_channel_switch`` blows the
#: per-bin channel kernels ``|K_a|`` above the (switch-independent) total
#: amplification ``|F|`` at the near-cusp config.  The buggy rule hands a
#: still-merged channel to the divergent stationary-phase kernel, driving
#: ``max|K_a|`` to ~5e5 while ``|F| ~ 3`` (measured); the production
#: full-cluster switch keeps ``max|K_a|`` at O(|F|).  1e3 sits far above
#: the healthy O(1) ratio and far below the pathological ~1e5, so it both
#: confirms the blow-up and certifies the production switch stays bounded
#: -- pinning the WP1 switch fix as load-bearing (FINDINGS F008).
SWITCH_PATHOLOGY_FACTOR = 1e3

#: Conservative lower bound on the RB speed-up over the full-grid brute
#: force.  ``lnlike`` touches ``n_bins`` coarse nodes and
#: ``n_bins*kernel_subsamples`` amplification points; ``lnlike_bruteforce``
#: touches the full FFT band twice (waveform + amplification), so the win
#: is structural, not marginal, at the fixture length.
SPEEDUP_MIN = 3.0

#: Tiny lens mass [Msun] driving the engine's ``w -> 0`` macro limit
#: (``w ~ 1e-7`` in band).  NOTE this drives ``F -> 1`` ONLY when the
#: macro sector is trivial: operator.py's certified ``w -> 0`` limit is
#: the macro CONSTANT ``1/sqrt(1 - gamma**2)`` (at ``kappa = 0``), not
#: unity.  See `MACRO_SHEAR` / `UNLENSED_LIMIT_LENS`.
TINY_M_LENS = 1e-6

#: Source position shared by every tiny-mass candidate, so the zero-noise
#: floor pair and the macro-contrast anchor differ ONLY in the macro
#: sector -- the contrast is attributable to shear, nothing else.
TINY_Y = (0.12, 0.035)

#: Shear of the tiny-mass contrast candidate B (and of the loose noisy
#: factor gate's candidate).  At ``kappa = 0`` its certified ``w -> 0``
#: macro limit is ``1/sqrt(1 - 0.04) = 1.0206207...`` -- a 2.06e-2 offset
#: from unity, which is CORRECT PHYSICS, not an error.
MACRO_SHEAR = 0.20

#: ``(gamma, kappa)`` of the tiny-mass candidate whose macro sector is
#: TRIVIAL, so the certified ``w -> 0`` limit is genuinely ``F -> 1`` and
#: an unlensed reference is the right oracle.
UNLENSED_LIMIT_LENS = (0.0, 0.0)

#: Ceiling on ``max_f | |F(f)| - 1 |`` across the kernel sub-sample grid
#: for the macro-trivial tiny candidate.  The residual there is the
#: ``O(w)`` wave correction at ``w ~ 1e-7``, i.e. ~1e-7 and flat; 1e-5
#: sits two decades above it and two decades BELOW the 2.06e-2 macro
#: constant of the sheared candidate, so this gate distinguishes the two
#: macro sectors unambiguously.
FLAT_F_TOL = 1e-5

#: Relative tolerance on the analytically PREDICTED macro offset of the
#: sheared contrast candidate B.  The prediction is exact for a constant
#: real ``F = c``; the ``O(w)`` wave correction and the template-
#: construction ``delta-h`` leave a sub-percent residual.
MACRO_OFFSET_RTOL = 1e-2

#: Directory for diagnostic plots (created on demand).
OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

# Diagnostic plots are written to disk, never shown: a test must not open
# a GUI window (or fail on a headless CI box for want of a display).
plt.switch_backend('Agg')


def _reference_par_dic():
    """
    A deterministic precessing reference ``par_dic`` for `APPROXIMANT`.

    Explicit (not randomly drawn) so the fixture is reproducible; the
    keys are asserted to match ``waveform.WaveformGenerator.params`` in
    `LensedLikelihoodTestCase.setUpClass`, so a schema drift fails loudly
    rather than subtly.
    """
    return {
        'm1': 60.0, 'm2': 45.0,
        's1x_n': 0.20, 's1y_n': 0.10, 's1z': 0.30,
        's2x_n': -0.10, 's2y_n': 0.15, 's2z': -0.20,
        'l1': 0.0, 'l2': 0.0,
        'iota': 1.0, 'phi_ref': 1.2,
        'ra': 1.8, 'dec': -0.3, 'psi': 0.9,
        't_geocenter': 0.0, 'd_luminosity': 600.0,
        'f_ref': 50.0,
    }


def _make_noisy_event():
    """Seeded Gaussian-noise HLV event with the fiducial signal injected."""
    event_data = data.EventData.gaussian_noise(
        eventname='test_lensed', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0., seed=SEED)
    event_data.inject_signal(_reference_par_dic(), APPROXIMANT)
    return event_data


def _real_only_channel_switch(w, delays, real_mask):
    """
    The pre-WP1 (buggy) ``_channel_switch``: measure each real channel's
    delay separation against OTHER REAL channels only, excluding the
    parked virtual labels that the paper's Eq. (delay-separation) includes.

    This independently re-implements the WRONG rule (a mutation), used
    ONLY by the near-cusp canary to show the retired neighbourhood blows
    the channel kernels up where the production full-cluster switch keeps
    them bounded.  It is deliberately NOT imported from the module under
    test, so the pin is not circular.
    """
    switch = np.zeros((w.shape[0], channels._N_CHANNELS), dtype=float)
    real_ids = np.flatnonzero(real_mask)
    for channel in real_ids:
        others = real_ids[real_ids != channel]
        if others.size == 0:
            continue
        separation = float(np.min(np.abs(delays[channel] - delays[others])))
        # Shared primitives from _gauge/operator, never channels' own
        # switch — the module whose defect this reproduces (FINDINGS
        # F002; matches the channels-suite reproduction's discipline).
        switch[:, channel] = _gauge.smootherstep(
            w * separation, operator.RHO_START, operator.RHO_END)
    return switch


class LensedLikelihoodTestCase(TestCase):
    """
    Shared fixtures plus the anti-vacuity comparison tally.

    One SEEDED injected Gaussian-noise event, one waveform generator, one
    lensed likelihood on an explicit uniform bin grid, built once for the
    whole class.  `tearDown` fails a test that asserted nothing.
    """

    @classmethod
    def setUpClass(cls):
        """Inject a signal and build the lensed likelihood (all seeded)."""
        cls.par_dic_0 = _reference_par_dic()
        assert sorted(cls.par_dic_0) == waveform.WaveformGenerator.params, (
            'reference par_dic keys drifted from '
            'WaveformGenerator.params; update _reference_par_dic')

        cls.event_data = _make_noisy_event()

        cls.waveform_generator = waveform.WaveformGenerator.from_event_data(
            cls.event_data, APPROXIMANT)

        band = cls.event_data.frequencies[cls.event_data.fslice]
        cls.f_lo, cls.f_hi = float(band[0]), float(band[-1])
        cls.fbin = cls._uniform_fbin(DF_BIN)

        cls.like = LensedRelativeBinningLikelihood(
            cls.event_data, cls.waveform_generator, cls.par_dic_0,
            delta_t_max=DELTA_T_MAX, fbin=cls.fbin)

    @classmethod
    def _uniform_fbin(cls, df_bin):
        """Uniform bin edges spanning the analysis band."""
        edges = np.arange(cls.f_lo, cls.f_hi, df_bin)
        if edges[-1] < cls.f_hi:
            edges = np.append(edges, cls.f_hi)
        return edges

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _lens_dic(y1, y2, gamma, beta, kappa, m_lens=M_LENS_MSUN,
                  z_lens=Z_LENS):
        """Assemble the seven lens keys expected in ``par_dic``."""
        return {'m_lens_msun': m_lens, 'z_lens': z_lens,
                'y1': y1, 'y2': y2, 'gamma': gamma, 'beta': beta,
                'kappa': kappa}

    def _candidate(self, lens_dic, waveform_par=None):
        """Merge waveform params (default: the fiducial) with a lens."""
        base = dict(waveform_par if waveform_par is not None
                    else self.par_dic_0)
        base.update(lens_dic)
        return base

    def _tiny_candidate(self):
        """
        A negligible-mass lens with a SHEARED macro sector.

        As the mass vanishes ``F`` tends to the macro constant
        ``1/sqrt(1 - MACRO_SHEAR**2) = 1.0206207...``, NOT to unity.  This
        is the candidate for the LOOSE noisy factor gate, whose ``NORM_TOL
        = 0.1`` comfortably absorbs that 2.06e-2 macro offset while still
        catching a gross O(SNR^2) factor error.  It is deliberately NOT
        used by the tight zero-noise floor -- see
        `_unlensed_limit_candidate` -- and is reused as candidate B of
        `MacroSectorContrastTestCase`, where the offset is not absorbed
        but PREDICTED.
        """
        return self._candidate(
            self._lens_dic(*TINY_Y, MACRO_SHEAR, 0.0, 0.0,
                           m_lens=TINY_M_LENS))

    def _unlensed_limit_candidate(self):
        """
        A negligible-mass lens with a TRIVIAL macro sector: ``F -> 1``.

        Identical to `_tiny_candidate` except ``gamma = kappa = 0``, so
        the engine's certified ``w -> 0`` macro limit
        ``1/sqrt(1 - gamma**2)`` collapses to unity and an UNLENSED
        reference is the physically correct oracle.  Separate from the
        shared `_tiny_candidate` so the noisy factor gate's configuration
        is untouched.
        """
        gamma, kappa = UNLENSED_LIMIT_LENS
        return self._candidate(
            self._lens_dic(*TINY_Y, gamma, 0.0, kappa, m_lens=TINY_M_LENS))

    def _amplification_profile(self, like, candidate):
        """
        ``(f, |F(f)| - 1)`` on the likelihood's kernel sub-sample grid.

        The engine's switch-independent ``exact_total`` is the same |F|
        the brute-force oracle rides, so this diagnostic reads the macro
        sector the lnL comparison actually sees.

        Parameters
        ----------
        like : LensedRelativeBinningLikelihood
            The likelihood whose bin/sub-sample grid to evaluate on.
        candidate : dict
            Waveform + lens parameters.

        Returns
        -------
        tuple of np.ndarray
            Frequencies [Hz] and ``|F(f)| - 1`` at those frequencies.
        """
        *_, partition = like._amplification_coefficients(candidate)
        return like._kernel_dense_f, np.abs(partition.exact_total) - 1.0

    @staticmethod
    def _save_figure(fig, name):
        """Write ``fig`` to ``cogwheel/tests/output/<name>.png`` and close."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=120,
                    bbox_inches='tight')
        plt.close(fig)


#: ``(label, y1, y2, gamma, beta, kappa)`` covering the required regimes.
#: All positive-parity (``1 - kappa > |gamma|``) and, at the fixture
#: mass, well inside the engine's wave branch.
_LENS_CONFIGS = [
    ('two-image', 0.50, 0.00, 0.20, 0.0, 0.0),
    ('four-image', 0.08, 0.06, 0.20, 0.0, 0.0),
    ('near-cusp', -0.38, 0.00, 0.20, 0.0, 0.0),
    ('kappa', 0.30, 0.10, 0.112, 0.0, 0.30),
    ('rotated-shear', 0.25, 0.10, 0.20, 0.70, 0.0),
]

#: The near-cusp entry of ``_LENS_CONFIGS`` singled out for the pin.
_NEAR_CUSP = ('near-cusp', -0.38, 0.00, 0.20, 0.0, 0.0)


class BruteForceAgreementTestCase(LensedLikelihoodTestCase):
    """
    Relative-binning ``lnlike`` reproduces the exact ``lnlike_bruteforce``
    across the required parameter regimes -- the isolation of the binning
    approximation (both paths share the amplification and waveform).
    """

    def _assert_agrees(self, label, candidate):
        rb = self.like.lnlike(candidate)
        bf = self.like.lnlike_bruteforce(candidate)
        self.n_checks += 1
        self.assertTrue(np.isfinite(rb) and np.isfinite(bf),
                        f'{label}: non-finite lnl (rb={rb}, bf={bf})')
        tol = max(RB_ATOL, RB_RTOL * abs(bf))
        self.assertLessEqual(
            abs(rb - bf), tol,
            f'{label}: relative-binning lnl {rb:.6g} disagrees with '
            f'brute-force {bf:.6g} by {abs(rb - bf):.4g} > {tol:.4g}')

    def test_agreement_over_lens_regimes(self):
        """2-/4-image, near-cusp, kappa!=0 and beta!=0 configurations."""
        for label, y1, y2, gamma, beta, kappa in _LENS_CONFIGS:
            with self.subTest(config=label):
                candidate = self._candidate(
                    self._lens_dic(y1, y2, gamma, beta, kappa))
                self._assert_agrees(label, candidate)

    def test_agreement_with_waveform_offset_from_fiducial(self):
        """
        A candidate whose WAVEFORM differs from the fiducial exercises the
        residual-delay / linear-free time-shift machinery (``r_m != 1``,
        ``dt_linearfree != 0``); RB must still track brute force.
        """
        offset = dict(self.par_dic_0)
        offset['m1'] *= 1.03
        offset['m2'] *= 0.98
        offset['t_geocenter'] += 2.0e-3
        candidate = self._candidate(
            self._lens_dic(0.30, 0.05, 0.20, 0.0, 0.0),
            waveform_par=offset)
        self._assert_agrees('waveform-offset', candidate)

    def test_agreement_near_fold(self):
        """
        A near-fold configuration (source at 0.95x the tangential
        caustic) stresses the rapidly varying kernels; RB still tracks
        brute force.
        """
        caustic = geometry.critical_point(0.20, np.pi / 4.0, 0.0, 0.0).source
        y = 0.95 * np.asarray(caustic, dtype=float)
        candidate = self._candidate(
            self._lens_dic(float(y[0]), float(y[1]), 0.20, 0.0, 0.0))
        self._assert_agrees('near-fold', candidate)


class NearCuspRegressionPinTestCase(LensedLikelihoodTestCase):
    """
    Pin the near-cusp correctness fix and prove it is load-bearing.

    At the near-cusp source ``y = (-0.38, 0)`` a real image's true
    cluster mate is a virtual label parked at the critical point.  The
    production ``_channel_switch`` measures each channel's delay
    separation over the WHOLE cluster (real AND parked virtual labels,
    paper Eq. delay-separation), keeping a still-merged channel in the
    bounded artificial gauge.  The pre-WP1 rule measured it over REAL
    channels only, spuriously ramped the switch to one, and handed the
    channel to the divergent stationary-phase kernel -- flooding every
    per-bin channel kernel ``K_a``, which ``_norm_term`` then squares into
    a spurious ``(h|h)`` (historically ~6.4e8; FINDINGS F008 supersedes
    the earlier F006 edge-secant attribution).  This test:

    (a) pins the CORRECTED value: the production ``lnlike`` agrees with
        the exact ``lnlike_bruteforce`` within tolerance (and within 1
        nat absolutely), and
    (b) demonstrates the mechanism with a falsifying canary on the SAME
        event / generator / bins: monkeypatching the engine's
        module-global ``_channel_switch`` back to the buggy real-only rule
        blows the per-bin kernels ``max|K_a|`` to
        ``>= SWITCH_PATHOLOGY_FACTOR * |F|`` while the production switch
        keeps them O(|F|) -- so the WP1 switch fix, not a benign bin grid,
        is what makes the pin hold.
    """

    def _near_cusp_candidate(self):
        _, y1, y2, gamma, beta, kappa = _NEAR_CUSP
        return self._candidate(self._lens_dic(y1, y2, gamma, beta, kappa))

    def test_production_lnlike_pins_bruteforce_at_near_cusp(self):
        """Production ``lnlike`` matches brute force at the near-cusp config."""
        cand = self._near_cusp_candidate()
        rb = self.like.lnlike(cand)
        bf = self.like.lnlike_bruteforce(cand)
        self.n_checks += 1
        self.assertTrue(np.isfinite(rb) and np.isfinite(bf),
                        f'near-cusp: non-finite lnl (rb={rb}, bf={bf})')
        tol = max(RB_ATOL, RB_RTOL * abs(bf))
        self.assertLessEqual(
            abs(rb - bf), tol,
            f'near-cusp: production lnl {rb:.6g} disagrees with brute '
            f'force {bf:.6g} by {abs(rb - bf):.4g} > {tol:.4g}')
        self.assertLess(
            abs(rb - bf), 1.0,
            f'near-cusp: production lnl off brute force by '
            f'{abs(rb - bf):.4g} nat (>1); the near-cusp switch fix regressed')

    def test_real_only_switch_variant_blows_up_kernels(self):
        """
        The pre-WP1 real-only ``_channel_switch`` blows the per-bin channel
        kernels up at the near-cusp config, while the production
        full-cluster switch keeps them bounded -- on the SAME event /
        generator / bins, so the near-cusp pin above is demonstrably
        load-bearing on the WP1 switch fix, not a coincidence of a benign
        bin grid.
        """
        cand = self._near_cusp_candidate()

        # Production full-cluster switch: bounded kernels.  ``exact_total``
        # is the switch-INDEPENDENT total amplification |F| (the brute-force
        # oracle rides it), so it is the natural O(1) scale to compare to.
        _, k0_prod, _, part = self.like._amplification_coefficients(cand)
        max_k_prod = float(np.max(np.abs(k0_prod)))
        f_scale = float(np.max(np.abs(part.exact_total)))

        # Buggy real-only switch on the SAME inputs: unbounded kernels.
        # Patch the engine's module-global switch that ``evaluate`` calls.
        with mock.patch.object(channels, '_channel_switch',
                               _real_only_channel_switch):
            _, k0_bug, _, _ = self.like._amplification_coefficients(cand)
        max_k_bug = float(np.max(np.abs(k0_bug)))

        self.n_checks += 1
        self.assertGreater(f_scale, 0.0,
                           f'near-cusp |F| should be positive, got {f_scale}')
        # The buggy real-only switch drives the kernels far above |F| ...
        self.assertGreaterEqual(
            max_k_bug, SWITCH_PATHOLOGY_FACTOR * f_scale,
            f'real-only switch max|K|={max_k_bug:.4g} did not blow up to '
            f'>= {SWITCH_PATHOLOGY_FACTOR:g} * |F|={f_scale:.4g}; the WP1 '
            'full-cluster switch fix is not demonstrably load-bearing')
        # ... while the production full-cluster switch keeps them O(|F|) ...
        self.assertLess(
            max_k_prod, SWITCH_PATHOLOGY_FACTOR * f_scale,
            f'production switch max|K|={max_k_prod:.4g} is not bounded '
            f'below {SWITCH_PATHOLOGY_FACTOR:g} * |F|={f_scale:.4g}; the '
            'production kernels are not O(|F|)')
        # ... i.e. the switch fix changes the kernel scale by >= 1e3x.
        self.assertGreaterEqual(
            max_k_bug, SWITCH_PATHOLOGY_FACTOR * max_k_prod,
            f'real-only switch max|K|={max_k_bug:.4g} is not >= '
            f'{SWITCH_PATHOLOGY_FACTOR:g}x the production kernel scale '
            f'{max_k_prod:.4g}; the fix is not demonstrably load-bearing')


class ProductOfSummariesRegressionTestCase(LensedLikelihoodTestCase):
    """
    Structural regression that a product-of-summaries implementation
    fails.

    At a near-fold source two images have near-degenerate delays, so the
    cross term ``2 Re(K_a K_c^* exp(2 pi i f (dt_a - dt_c)))`` dominates
    ``(h_L|h_L)``.  Summarizing ``F`` and ``h`` separately and
    multiplying the summaries drops the in-bin structure of that cross
    term; the delay-continuous summaries keep the rapid x rapid product
    INSIDE the frequency sum.  Pinning ``lnlike`` to the exact
    ``lnlike_bruteforce`` here is the guard: only the correct
    (in-sum) contraction reproduces the exact cross term.
    """

    def test_cross_term_dominated_config_matches_bruteforce(self):
        caustic = geometry.critical_point(0.20, np.pi / 4.0, 0.0, 0.0).source
        y = 0.97 * np.asarray(caustic, dtype=float)
        candidate = self._candidate(
            self._lens_dic(float(y[0]), float(y[1]), 0.20, 0.0, 0.0))
        rb = self.like.lnlike(candidate)
        bf = self.like.lnlike_bruteforce(candidate)
        self.n_checks += 1
        self.assertTrue(np.isfinite(rb) and np.isfinite(bf),
                        f'non-finite lnl (rb={rb}, bf={bf})')
        tol = max(RB_ATOL, RB_RTOL * abs(bf))
        self.assertLessEqual(
            abs(rb - bf), tol,
            'near-degenerate cross-term config: relative-binning lnl '
            f'{rb:.6g} disagrees with brute-force {bf:.6g} by '
            f'{abs(rb - bf):.4g} > {tol:.4g}; a product-of-summaries '
            'contraction would fail here')


class NormalizationFactorGateTestCase(LensedLikelihoodTestCase):
    """
    LOOSE, independent, deterministic ``F -> 1`` factor anchor on the
    seeded NOISY fixture.

    With ``F ~ 1`` the lensed brute-force lnl equals the exact unlensed
    ``lnlike_fft`` (both exact inner products through the SAME
    normalization, but ``lnlike_fft`` lives outside the module under
    test).  This gate's job is only to catch a GROSS factor error (a
    stray ``4 df`` or factor of two would be O(SNR^2) = tens-hundreds); a
    sub-nat construction residual is intentionally NOT what it measures --
    the zero-noise floor does that.
    """

    def test_bruteforce_floor_matches_exact_unlensed(self):
        exact_unlensed = self.like.lnlike_fft(self.par_dic_0)
        lensed_bf = self.like.lnlike_bruteforce(self._tiny_candidate())
        self.n_checks += 1
        self.assertLessEqual(
            abs(lensed_bf - exact_unlensed), NORM_TOL,
            f'lensed brute-force at F~1 ({lensed_bf:.6g}) != exact '
            f'unlensed lnlike_fft ({exact_unlensed:.6g}); a gross '
            'normalization factor error would show here')


class ZeroNoiseAnchorTestCase(LensedLikelihoodTestCase):
    """
    Fixture-only base: the ZERO-NOISE anchor (``d == h0`` exactly).

    Carries no tests of its own; `NormalizationFloorZeroNoiseTestCase`
    and `MacroSectorContrastTestCase` share this one deterministic
    fixture, so the floor pair and the macro-contrast anchor are read off
    the SAME data vector and the same bin grid.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Zero-noise event carrying ONLY the injected fiducial signal:
        # start from a seeded draw, zero the strain (updating blued_strain
        # via the same setter the constructor uses), then inject.
        zero_event = data.EventData.gaussian_noise(
            eventname='test_lensed_zeronoise', duration=4,
            detector_names='HLV',
            asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.,
            seed=SEED)
        zero_event._set_strain(np.zeros_like(zero_event.strain))
        zero_event.inject_signal(cls.par_dic_0, APPROXIMANT)

        zero_generator = waveform.WaveformGenerator.from_event_data(
            zero_event, APPROXIMANT)
        # On noiseless data the ASD-drift estimator takes the std of an
        # (essentially) all-outlier matched-filter statistic, so its sample
        # variance is empty (numpy "Degrees of freedom <= 0") and the drift
        # comes back NaN.  Drift is a LOCAL-NOISE correction with no meaning
        # on a zero-noise anchor, so pin it to unity: it is then applied
        # identically to lnlike_fft, lnlike and lnlike_bruteforce, leaving
        # the F->1 floor comparison exact and NaN-free.  This changes no
        # tolerance -- ZERO_NOISE_TOL stays 1e-2.
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            cls.zero_like = LensedRelativeBinningLikelihood(
                zero_event, zero_generator, cls.par_dic_0,
                delta_t_max=DELTA_T_MAX, fbin=cls.fbin)
        cls.zero_like.asd_drift = np.ones(len(zero_event.detector_names))

    def _h0_norm(self):
        """
        The unlensed ``(h0|h0)`` read off the likelihood's OWN normalization.

        On the zero-noise anchor ``d == h0``, so
        ``lnlike_fft(par_dic_0) = (d|h0) - (h0|h0)/2 = (h0|h0)/2``.
        Taking the norm this way (rather than re-deriving an inner product
        in the test) means the macro-offset prediction is expressed in the
        engine's own units and cannot drift from them.

        Returns
        -------
        float
            ``(h0|h0)``, strictly positive.
        """
        return 2.0 * self.zero_like.lnlike_fft(self.par_dic_0)


class NormalizationFloorZeroNoiseTestCase(ZeroNoiseAnchorTestCase):
    """
    TIGHT, physically meaningful ``F -> 1`` FLOOR on a ZERO-NOISE anchor.

    The data vector is set to the pure fiducial signal (no noise), so
    ``d == h0`` exactly and the ``<noise|delta-h>`` overlap term that
    would otherwise scale with the noise draw VANISHES.  With a tiny-lens
    candidate (``F ~ 1`` in band), both the lensed brute force and the
    lensed RB must sit within ``ZERO_NOISE_TOL`` of the exact unlensed
    ``lnlike_fft(par_dic_0)``.  What remains is the ~1e-4 lensing residual
    plus a small deterministic template-construction ``delta-h`` -- whose
    origin is the reference ``_h0_edges`` (stalled ringdown /
    precession-forced) being assembled slightly differently from the
    candidate ratio, NOT a normalization error.  On NOISY data this same
    residual would instead scale with the noise draw (the original
    symptom of the retired brittle 0.1 floor); the zero-noise construction
    removes that tail and makes the tolerance meaningful (1e-2, not
    perched at a failure boundary).

    The candidate is `_unlensed_limit_candidate` (``gamma = kappa = 0``),
    NOT the sheared `_tiny_candidate`: only a trivial macro sector makes
    ``F -> 1`` -- and hence an unlensed reference -- correct.  See the
    per-test docstrings.
    """

    def _assert_floor(self, label, lensed_lnl):
        """Assert an ``F -> 1`` lnl sits within `ZERO_NOISE_TOL` of exact."""
        exact_unlensed = self.zero_like.lnlike_fft(self.par_dic_0)
        residual = abs(lensed_lnl - exact_unlensed)
        self.n_checks += 1
        self.assertLessEqual(
            residual, ZERO_NOISE_TOL,
            f'zero-noise lensed {label} at the F->1 macro-trivial '
            f'candidate ({lensed_lnl:.10g}) != exact unlensed lnlike_fft '
            f'({exact_unlensed:.10g}); measured residual '
            f'{residual:.4g} > {ZERO_NOISE_TOL}. An O(SNR^2) offset flags '
            'a normalization factor error; an offset near '
            '0.5*(1/sqrt(1-gamma**2) - 1)**2*(h0|h0) flags a macro sector '
            'that is not trivial (see MacroSectorContrastTestCase)')

    def test_amplification_is_flat_at_unity_for_macro_trivial_candidate(self):
        """
        DIAGNOSTIC + PREMISE CHECK for the floor pair below.

        The two floor tests are only meaningful if their candidate's
        ``F`` really does tend to unity in band.  Here we read the
        engine's own ``|F(f)| - 1`` across the kernel sub-sample grid and
        require it flat at ~1e-7 (the ``O(w)`` wave correction), i.e.
        below `FLAT_F_TOL` -- two decades under the 2.06e-2 macro
        constant that the SHEARED `_tiny_candidate` would show here.  If
        this premise ever breaks, the floor failures below are physics,
        not a bug.
        """
        freqs, f_minus_one = self._amplification_profile(
            self.zero_like, self._unlensed_limit_candidate())
        worst = float(np.max(np.abs(f_minus_one)))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(freqs, f_minus_one, lw=1.0,
                label=r'macro-trivial ($\gamma=\kappa=0$)')
        ax.axhline(1.0 / np.sqrt(1.0 - MACRO_SHEAR ** 2) - 1.0,
                   color='crimson', ls='--',
                   label=r'sheared macro constant $1/\sqrt{1-\gamma^2}-1$')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel(r'$|F(f)| - 1$')
        ax.set_yscale('symlog', linthresh=1e-12)
        ax.set_title(f'zero-noise F->1 premise: max||F|-1| = {worst:.3g}')
        ax.legend(fontsize=8)
        self._save_figure(
            fig, 'test_amplification_is_flat_at_unity_amplification_profile')

        self.n_checks += 1
        self.assertLess(
            worst, FLAT_F_TOL,
            f'macro-trivial tiny candidate has max||F|-1| = {worst:.4g} '
            f'> {FLAT_F_TOL}; its F does NOT tend to unity in band, so the '
            'unlensed reference is the wrong oracle for the floor tests')

    def test_bruteforce_floor_is_physically_tight(self):
        """
        Zero-noise brute force at the MACRO-TRIVIAL tiny candidate matches
        the exact unlensed ``lnlike_fft`` within `ZERO_NOISE_TOL`.

        PREMISE REPAIR.  This test previously used the SHEARED
        `_tiny_candidate` (``gamma = 0.20``) and failed by ~0.1214 nat.
        That failure was CORRECT PHYSICS beaten against the wrong oracle:
        operator.py's certified ``w -> 0`` limit is the macro constant
        ``1/sqrt(1 - gamma**2) = 1.0206207...`` (at ``kappa = 0``), not
        unity, so ``F ~ 1 in band`` was never true for that candidate and
        the offset was the engine correctly reporting a 2.06e-2 macro
        magnification.  The fix is to the CANDIDATE, not the tolerance:
        with ``gamma = kappa = 0`` the unlensed limit is genuinely
        ``F -> 1`` and the residual drops to the ~1e-11 numerical floor.
        `ZERO_NOISE_TOL` is unchanged at 1e-2, and the offset that used to
        fire here is now PREDICTED and pinned by
        `MacroSectorContrastTestCase`.
        """
        self._assert_floor(
            'brute force',
            self.zero_like.lnlike_bruteforce(self._unlensed_limit_candidate()))

    def _standard_unlensed_rb(self):
        """
        A STANDARD (unlensed) `RelativeBinningLikelihood` on the SAME
        zero-noise event, base waveform generator, fiducial and bin grid
        as the lensed fixture.

        This is the mature-package machinery whose own binning/stall floor
        the lensed engine INHERITS; building it here lets the RB test
        isolate the lensing-layer contribution by differencing against it
        (the shared inherited floor cancels).  ``asd_drift`` is pinned to
        unity for the same reason it is on the lensed fixture: the drift
        estimator is meaningless on noiseless data (it returns NaN), and
        pinning it applies the identical correction to both likelihoods so
        the difference is clean.

        Returns
        -------
        RelativeBinningLikelihood
            Unlensed RB on the zero-noise anchor, ``asd_drift = 1``.
        """
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            std_like = RelativeBinningLikelihood(
                self.zero_like.event_data,
                self.zero_like.waveform_generator,
                self.par_dic_0, fbin=self.fbin)
        std_like.asd_drift = np.ones(
            len(self.zero_like.event_data.detector_names))
        return std_like

    def test_relative_binning_isolates_lensing_layer_increment(self):
        """
        The lensed RB's departure from exact on this zero-noise anchor is
        an INHERITED standard-RB binning floor plus a small lensing-layer
        increment -- the increment is tight, and the total is a documented
        regression pin, NOT a physical-tightness claim.

        MECHANISM (measured 2026-07-17; do not re-derive).  The lensed RB
        at the macro-trivial (``gamma = kappa = 0``, ``F -> 1``) candidate
        reads 285.386763 while the exact unlensed ``lnlike_fft`` reads
        285.398401, a 1.164e-2 offset.  On the SAME event/generator/
        fiducial/bins the standard unlensed `RelativeBinningLikelihood`
        reads 285.389439 -- an 8.96e-3 floor of its OWN (binning / stalled
        ringdown-precession reference), inherited by the lensed engine and
        out of the lensing program's scope.  The offset therefore
        decomposes as 8.96e-3 inherited + 2.68e-3 lensing layer.

        PRIMARY -- the lensing-layer increment.  Differencing the lensed RB
        at the ``F -> 1`` candidate against the standard unlensed RB at the
        fiducial CANCELS the shared inherited floor, leaving only the
        lensing layer's own contribution.  Measured 2.676e-3; pinned
        <= 5e-3 (~2x margin).

        SECONDARY -- a documented regression PIN, explicitly NOT a
        physical-tightness claim.  The full lensed-RB-vs-``lnlike_fft``
        offset is held <= 1.5e-2 at its measured 1.164e-2 (8.96e-3
        inherited standard-RB stall floor + 2.68e-3 lensing layer).  The
        physical ``F -> 1`` tightness claim on this anchor is carried by
        `test_bruteforce_floor_is_physically_tight`, whose brute-force path
        has no binning floor (residual ~1e-11, well under `ZERO_NOISE_TOL`)
        and so needs no pin.
        """
        lensed_rb_lnl = self.zero_like.lnlike(
            self._unlensed_limit_candidate())
        standard_rb_lnl = self._standard_unlensed_rb().lnlike(self.par_dic_0)
        exact_fft = self.zero_like.lnlike_fft(self.par_dic_0)

        increment = abs(lensed_rb_lnl - standard_rb_lnl)
        self.n_checks += 1
        self.assertLessEqual(
            increment, 5e-3,
            f'lensing-layer RB increment {increment:.4g} > 5e-3: the '
            f'lensed RB ({lensed_rb_lnl:.10g}) at the F->1 candidate has '
            f'drifted from the standard unlensed RB ({standard_rb_lnl:.10g}) '
            'at the fiducial by more than the shared-inherited-floor-'
            'cancelling difference (measured 2.676e-3)')

        regression = abs(lensed_rb_lnl - exact_fft)
        self.n_checks += 1
        self.assertLessEqual(
            regression, 1.5e-2,
            f'lensed-RB-vs-lnlike_fft regression pin {regression:.4g} > '
            f'1.5e-2 (measured 1.164e-2 = 8.96e-3 inherited standard-RB '
            'stall floor + 2.68e-3 lensing layer). This is a REGRESSION '
            'PIN, not a physical-tightness claim; the physical F->1 claim '
            'is carried by test_bruteforce_floor_is_physically_tight')


class MacroSectorContrastTestCase(ZeroNoiseAnchorTestCase):
    """
    Pin the repaired zero-noise floor to its TRUE cause: the macro sector.

    `NormalizationFloorZeroNoiseTestCase` was repaired by swapping the
    sheared tiny candidate for a macro-trivial one.  On its own that
    repair is indistinguishable from a dodge -- deleting an inconvenient
    configuration -- so this test keeps the sheared candidate and shows
    its 0.1214 nat offset is not merely tolerated but PREDICTED.

    Two tiny-mass candidates identical except for shear:

    * A -- ``gamma = kappa = 0``: certified ``w -> 0`` limit ``F -> 1``,
      so A sits within `ZERO_NOISE_TOL` of the unlensed reference.
    * B -- ``gamma = MACRO_SHEAR``, ``kappa = 0``: certified limit is the
      macro CONSTANT ``c = 1/sqrt(1 - gamma**2) = 1.0206207...``, so B is
      offset from the reference by a computable amount.

    The prediction is derived INDEPENDENTLY in the test, from closed form
    rather than from any production path: for a constant real ``F = c``
    the candidate strain is ``c*h0``, and on data ``d == h0``

        lnL(c*h0) - lnL(h0) = [c*(h0|h0) - c**2*(h0|h0)/2]
                              - [(h0|h0) - (h0|h0)/2]
                            = -0.5*(c - 1)**2*(h0|h0),

    i.e. a strictly NEGATIVE offset quadratic in the macro magnification
    error.  Only ``(h0|h0)`` is taken from the likelihood (via
    `_h0_norm`, the engine's own normalization); ``c`` and the algebra are
    the test's.

    This goes red if anyone 'normalizes' the macro magnification out of
    the engine (B would collapse onto A) or reintroduces a small-``w``
    short-circuit forcing ``F = 1`` (likewise) -- the two regressions the
    premise repair would otherwise silently invite.
    """

    def test_macro_shear_offsets_lnlike_by_the_predicted_amount(self):
        """A matches the unlensed reference; B is offset as predicted."""
        reference = self.zero_like.lnlike_fft(self.par_dic_0)
        lnl_a = self.zero_like.lnlike_bruteforce(
            self._unlensed_limit_candidate())            # gamma = kappa = 0
        lnl_b = self.zero_like.lnlike_bruteforce(
            self._tiny_candidate())                      # gamma = MACRO_SHEAR

        # Independent closed-form prediction (see class docstring).
        macro_c = 1.0 / np.sqrt(1.0 - MACRO_SHEAR ** 2)
        h0_norm = self._h0_norm()
        predicted_offset = -0.5 * (macro_c - 1.0) ** 2 * h0_norm
        observed_offset = lnl_b - reference

        self._plot_contrast(reference, lnl_a, lnl_b, predicted_offset)

        # (i) The macro-trivial candidate A really does sit at the
        #     unlensed reference -- without this, an offset for B proves
        #     nothing about shear.
        self.n_checks += 1
        self.assertLessEqual(
            abs(lnl_a - reference), ZERO_NOISE_TOL,
            f'macro-trivial candidate A ({lnl_a:.10g}) does not sit at the '
            f'unlensed reference ({reference:.10g}); residual '
            f'{abs(lnl_a - reference):.4g} > {ZERO_NOISE_TOL}')

        # (ii) The sheared candidate B is offset by the PREDICTED macro
        #      amount -- pinning the historical 0.1214 nat to its cause.
        self.n_checks += 1
        self.assertGreater(
            abs(predicted_offset), 10.0 * ZERO_NOISE_TOL,
            f'predicted macro offset {predicted_offset:.4g} is not '
            'comfortably outside the floor tolerance; this test would not '
            'distinguish candidate B from candidate A')
        self.assertLessEqual(
            abs(observed_offset - predicted_offset),
            MACRO_OFFSET_RTOL * abs(predicted_offset),
            f'sheared candidate B is offset from the unlensed reference by '
            f'{observed_offset:.6g} nat, but the closed-form macro '
            f'prediction -0.5*(c-1)**2*(h0|h0) with c = {macro_c:.7f} and '
            f'(h0|h0) = {h0_norm:.6g} is {predicted_offset:.6g} nat '
            f'(relative miss '
            f'{abs(observed_offset / predicted_offset - 1.0):.3g} > '
            f'{MACRO_OFFSET_RTOL}). Either the macro magnification has been '
            'normalized out of the engine, a small-w short-circuit forces '
            'F = 1, or the w->0 macro limit has changed')

    def _plot_contrast(self, reference, lnl_a, lnl_b, predicted_offset):
        """Annotated bar chart of the three lnL values and the prediction."""
        labels = ['unlensed\nreference', r'A: $\gamma=\kappa=0$',
                  rf'B: $\gamma={MACRO_SHEAR}$']
        values = [reference, lnl_a, lnl_b]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, values, color=['0.6', 'tab:green', 'tab:red'],
               width=0.55)
        for idx, value in enumerate(values):
            ax.annotate(f'{value:.4f}', (idx, value), ha='center',
                        va='bottom', fontsize=8)
        # Arrow from the reference down to the predicted position of B.
        ax.annotate(
            '', xy=(2, reference + predicted_offset), xytext=(2, reference),
            arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1.4})
        ax.annotate(
            rf'predicted $-\frac{{1}}{{2}}(c-1)^2 (h_0|h_0)$ = '
            rf'{predicted_offset:.4f}',
            xy=(2, reference + 0.5 * predicted_offset), xytext=(6, 0),
            textcoords='offset points', va='center', fontsize=8)
        ax.axhline(reference, color='0.4', ls=':', lw=1.0)
        ax.set_ylabel(r'$\ln \mathcal{L}$ (zero-noise, $d = h_0$)')
        ax.set_title('macro-sector contrast: shear offsets lnL as predicted')
        span = max(abs(predicted_offset), 1e-3)
        ax.set_ylim(reference - 3.0 * span, reference + 1.2 * span)
        self._save_figure(fig, 'test_macro_shear_offsets_lnlike_contrast')


class BinGuardTestCase(LensedLikelihoodTestCase):
    """
    The lens-aware bin guard `LensedBinningError` actually fires -- both
    at construction (bins too coarse) and at evaluation (candidate delays
    exceed ``delta_t_max``).  Falsification, not prose.
    """

    def test_construction_rejects_bins_too_coarse_for_delta_t_max(self):
        """
        A ``delta_t_max`` too large for the bin widths violates
        ``pi * Delta_f_bin * delta_t_max < bin_delay_tol`` and raises at
        construction.  Uses the known-good fine grid with an oversized
        ``delta_t_max`` so the guard -- not base-class bin setup -- is
        what fires.
        """
        widest = float(np.max(np.diff(self.fbin)))
        big_delta_t = 0.05
        # Sanity: the config really does breach the criterion (default
        # bin_delay_tol is 0.5 rad).
        self.assertGreaterEqual(np.pi * widest * big_delta_t, 0.5)
        self.n_checks += 1
        with self.assertRaises(LensedBinningError):
            LensedRelativeBinningLikelihood(
                self.event_data, self.waveform_generator, self.par_dic_0,
                delta_t_max=big_delta_t, fbin=self.fbin)

    def test_evaluation_rejects_candidate_delays_beyond_max(self):
        """
        A likelihood built with a tiny ``delta_t_max`` (so construction
        passes) raises when a candidate presents image delays larger than
        that ``delta_t_max``.
        """
        tight = LensedRelativeBinningLikelihood(
            self.event_data, self.waveform_generator, self.par_dic_0,
            delta_t_max=1e-4, fbin=self.fbin)
        # Fixture-mass lens: image delays are milliseconds >> 1e-4 s,
        # while in-band w stays small so the engine returns normally.
        candidate = self._candidate(
            self._lens_dic(0.30, 0.05, 0.20, 0.0, 0.0))
        self.n_checks += 1
        with self.assertRaises(LensedBinningError):
            tight.lnlike(candidate)


class ContractionTimingTestCase(LensedLikelihoodTestCase):
    """
    Timing gates grounded in the additive ``M^2 + n_img^2`` design.

    Two claims are made, both with a best-of-N pattern (robust to
    scheduler jitter) and warm caches:

    (a) SPEED-UP -- relative-binning ``lnlike`` beats the full-grid
        matched filter ``lnlike_bruteforce`` by a conservative margin.
        This is the claim that RB is worth having.

    (b) ALGEBRA SUBDOMINANCE -- the pure ``_data_term`` + ``_norm_term``
        contraction is faster than the amplification-engine call
        ``_amplification_coefficients`` (the 1F1 special-function
        evaluation at ``n_bins * kernel_subsamples`` points), the
        unavoidable per-eval cost.  The contraction inputs are produced by
        the LIVE hot path (``_amplification_coefficients``, the
        sub-sample fit), NOT the retired edge secant, so the measured
        contraction matches production.

    We deliberately do NOT assert the contraction beats the coarse
    ``get_strain_at_detectors(fbin, ...)`` waveform call: that coarse
    strain is a per-eval CO-COST shared by the RB path, not its
    competitor.  RB's competitor is ``lnlike_bruteforce``, and the
    additive ``M^2 + n_img^2`` contraction is legitimately heavier than
    one coarse higher-mode strain call -- so 'subdominant' means
    subdominant to the amplification engine, and 'faster' means faster
    than brute force.
    """

    #: Repeats for the best-of timing (robust to scheduler jitter).
    _REPEATS = 7

    def _best_time(self, thunk):
        best = np.inf
        for _ in range(self._REPEATS):
            start = time.perf_counter()
            thunk()
            best = min(best, time.perf_counter() - start)
        return best

    def test_relative_binning_faster_than_bruteforce(self):
        """(a) SPEED-UP: ``lnlike`` beats ``lnlike_bruteforce``."""
        candidate = self._candidate(
            self._lens_dic(0.08, 0.06, 0.20, 0.0, 0.0))  # 4-image

        def rb():
            self.like.lnlike(candidate)

        def brute():
            self.like.lnlike_bruteforce(candidate)

        rb()
        brute()
        t_rb = self._best_time(rb)
        t_brute = self._best_time(brute)
        self.n_checks += 1
        self.assertGreater(
            t_brute, SPEEDUP_MIN * t_rb,
            f'relative-binning lnlike ({t_rb * 1e3:.3f} ms) is not at '
            f'least {SPEEDUP_MIN}x faster than brute force '
            f'({t_brute * 1e3:.3f} ms); the RB speed-up regressed')

    def test_contraction_subdominant_to_amplification_engine(self):
        """
        (b) ALGEBRA SUBDOMINANCE: the pure contraction is faster than the
        amplification-engine call that feeds it.  Reintroducing an FFT or
        a per-frequency Python loop on the hot path fails here.
        """
        candidate = self._candidate(
            self._lens_dic(0.08, 0.06, 0.20, 0.0, 0.0))  # 4-image

        # Precompute the contraction inputs once via the LIVE hot path
        # (setup cost, not timed).
        r0, r1, dt_lf = self.like._candidate_bin_ratios(candidate)
        rho0, rho1 = r0.conj(), r1.conj()
        delays, k0, k1, _ = self.like._amplification_coefficients(candidate)
        kbar0, kbar1 = k0.conj(), k1.conj()
        tau = delays - dt_lf
        f_center = self.like._f_center

        def contraction():
            _data_term(self.like._a_moments, rho0, rho1, kbar0, kbar1, tau,
                       f_center)
            _norm_term(self.like._b_moments, r0, r1, rho0, rho1, k0, k1,
                       kbar0, kbar1, delays, f_center)

        def amplification_engine():
            self.like._amplification_coefficients(candidate)

        # Warm up (JIT/import/caches) before timing.
        contraction()
        amplification_engine()

        t_contract = self._best_time(contraction)
        t_engine = self._best_time(amplification_engine)
        self.n_checks += 1
        self.assertLess(
            t_contract, t_engine,
            f'mode-then-image contraction ({t_contract * 1e3:.3f} ms) is '
            f'not subdominant to the amplification engine call '
            f'({t_engine * 1e3:.3f} ms); an FFT or per-frequency Python '
            'loop may have crept onto the hot path')


class MacroSaddleRejectionTestCase(LensedLikelihoodTestCase):
    """
    Macro-saddle (non-positive-parity) candidates raise
    `geometry.LensDomainError` from BOTH ``lnlike`` and
    ``lnlike_bruteforce`` -- the likelihood-path analogue of the
    generator-boundary rejection, propagated unswallowed.  Engine-refusal
    SYMMETRY: never one path returning a value while the other refuses.
    """

    #: ``1 - kappa <= |gamma|`` -- outside the positive-parity regime.
    BAD_CONFIGS = (
        ('boundary 0.5/0.5', dict(kappa=0.5, gamma=0.5)),
        ('interior 0.5/0.6', dict(kappa=0.5, gamma=0.6)),
    )

    def test_lnlike_paths_reject_macro_saddles(self):
        for label, bad in self.BAD_CONFIGS:
            with self.subTest(config=label):
                candidate = self._candidate(
                    self._lens_dic(0.20, 0.05, bad['gamma'], 0.0,
                                   bad['kappa']))
                self.n_checks += 1
                with self.assertRaises(geometry.LensDomainError):
                    self.like.lnlike(candidate)
                with self.assertRaises(geometry.LensDomainError):
                    self.like.lnlike_bruteforce(candidate)


class DeterminismTestCase(LensedLikelihoodTestCase):
    """
    Every stochastic input is seeded, so the fixtures and the hot path are
    exact pure functions of their inputs -- bit-identical across repeated
    constructions and repeated evaluations in one process.
    """

    def test_seeded_strain_is_bit_identical(self):
        """
        Two independently-constructed seeded events have EXACTLY equal
        strain.  A nonzero diff flags an unseeded input (recall
        ``gaussian_noise`` draws from its own ``default_rng(seed)``, so a
        bare ``np.random.seed`` would leave this nonzero).
        """
        event_a = data.EventData.gaussian_noise(
            eventname='det_a', duration=4, detector_names='HLV',
            asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.,
            seed=SEED)
        event_b = data.EventData.gaussian_noise(
            eventname='det_b', duration=4, detector_names='HLV',
            asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.,
            seed=SEED)
        for det in range(len(event_a.detector_names)):
            with self.subTest(detector=det):
                self.n_checks += 1
                self.assertTrue(
                    np.array_equal(event_a.strain[det], event_b.strain[det]),
                    f'seeded strain differs at detector {det}; an unseeded '
                    'input remains in the noise draw')

    def test_lnlike_is_exactly_repeatable(self):
        """
        ``lnlike`` and ``lnlike_bruteforce`` return bit-identical values on
        repeated calls with the same candidate (assertEqual, not
        almost-equal): the hot path and the oracle are pure functions, with
        no hidden RNG or order nondeterminism.
        """
        candidate = self._candidate(
            self._lens_dic(0.08, 0.06, 0.20, 0.0, 0.0))  # 4-image

        rb_first = self.like.lnlike(candidate)
        rb_second = self.like.lnlike(candidate)
        self.n_checks += 1
        self.assertEqual(rb_first, rb_second,
                         'lnlike is not exactly repeatable -- hidden RNG or '
                         'nondeterministic engine path on the hot path')

        bf_first = self.like.lnlike_bruteforce(candidate)
        bf_second = self.like.lnlike_bruteforce(candidate)
        self.n_checks += 1
        self.assertEqual(bf_first, bf_second,
                         'lnlike_bruteforce is not exactly repeatable -- '
                         'hidden RNG or nondeterministic oracle path')


class SelfFalsificationTestCase(LensedLikelihoodTestCase):
    """
    Prove the central brute-force-agreement gate can go red.

    A green agreement test is worth only as much as its ability to fail;
    a perturbed lnl must breach the tolerance.
    """

    def test_agreement_gate_detects_a_perturbed_lnl(self):
        """
        A candidate's exact ``lnlike_bruteforce`` differs from a value
        shifted by ``10 * RB_ATOL`` by more than the agreement tolerance
        -- so a contraction that was wrong by that much could not slip
        through `BruteForceAgreementTestCase`.
        """
        candidate = self._candidate(
            self._lens_dic(0.50, 0.00, 0.20, 0.0, 0.0))
        bf = self.like.lnlike_bruteforce(candidate)
        self.n_checks += 1
        self.assertTrue(np.isfinite(bf))
        perturbed = bf + 10.0 * RB_ATOL
        tol = max(RB_ATOL, RB_RTOL * abs(bf))
        self.assertGreater(
            abs(perturbed - bf), tol,
            'a 10*RB_ATOL perturbation slips the agreement tolerance; the '
            'brute-force gate would assert nothing')


if __name__ == '__main__':
    main()
