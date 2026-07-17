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

* A PRODUCT-OF-SUMMARIES structural regression: at a near-fold
  configuration two images have near-degenerate delays and the cross
  term dominates ``(h_L|h_L)``.  Summarizing ``F`` and ``h`` separately
  and multiplying the summaries drops that cross term's in-bin phase
  structure; pinning ``lnlike`` to the exact ``lnlike_bruteforce`` there
  is the guard.

* A NEAR-CUSP regression PIN with a falsifying canary.  At the near-cusp
  source the candidate channel kernels vary rapidly across a coarse bin.
  The production hot path reduces each kernel to a per-bin (value, slope)
  by a least-squares fit over ``kernel_subsamples`` sub-samples; a plain
  bin-edge secant (``kernel_subsamples=2``) ALIASES that variation and
  ``_norm_term`` squares the blown-up slope.  The pin holds ``lnlike`` to
  ``lnlike_bruteforce`` at the corrected value AND demonstrates, on the
  SAME event/generator/bins, that the ``kernel_subsamples=2`` path
  reproduces the pathology (disagreement orders of magnitude larger) --
  so the sub-sample reduction is load-bearing and the pin is non-vacuous.

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

import time
from unittest import TestCase, main

import numpy as np

from cogwheel import data, waveform
from cogwheel.lensing.chang_refsdal import geometry
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

#: Minimum ``|rb_secant - bf|`` the ``kernel_subsamples=2`` edge-secant
#: canary must exceed at the near-cusp config.  The aliased slope, once
#: squared by ``_norm_term``, blows the disagreement toward ~1e8; a
#: threshold of 1e3 is far above both the true binning floor and any
#: healthy value, so passing it certifies the pathology is reproduced.
SECANT_ALIAS_MIN = 1e3

#: Conservative lower bound on the RB speed-up over the full-grid brute
#: force.  ``lnlike`` touches ``n_bins`` coarse nodes and
#: ``n_bins*kernel_subsamples`` amplification points; ``lnlike_bruteforce``
#: touches the full FFT band twice (waveform + amplification), so the win
#: is structural, not marginal, at the fixture length.
SPEEDUP_MIN = 3.0

#: Tiny lens mass [Msun] driving ``F -> 1`` (``w ~ 1e-7`` in band).
TINY_M_LENS = 1e-6


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
        """A benign lens with negligible mass, so ``F ~ 1`` in band."""
        return self._candidate(
            self._lens_dic(0.12, 0.035, 0.20, 0.0, 0.0,
                           m_lens=TINY_M_LENS))


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
    Pin the F006 near-cusp fix and prove it is non-vacuous.

    At the near-cusp source ``y = (-0.38, 0)`` the candidate channel
    kernels vary rapidly within a coarse bin.  The production hot path
    reduces each kernel to a per-bin (value, slope) by least squares over
    ``kernel_subsamples`` (default 8) sub-samples; a plain bin-edge secant
    (``kernel_subsamples=2``) aliases that variation and ``_norm_term``
    squares the blown-up slope.  This test:

    (a) pins the CORRECTED value: the production ``lnlike`` agrees with
        the exact ``lnlike_bruteforce`` within tolerance (and within 1
        nat absolutely), and
    (b) demonstrates the mechanism with a falsifying canary on the SAME
        event/generator/bins: the ``kernel_subsamples=2`` likelihood's
        disagreement with brute force is orders of magnitude larger.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Same event / generator / bins, but the retired plain edge secant
        # (kernel_subsamples=2) instead of the production sub-sample fit.
        cls.like_secant = LensedRelativeBinningLikelihood(
            cls.event_data, cls.waveform_generator, cls.par_dic_0,
            delta_t_max=DELTA_T_MAX, fbin=cls.fbin, kernel_subsamples=2)

    def _near_cusp_candidate(self):
        _, y1, y2, gamma, beta, kappa = _NEAR_CUSP
        return self._candidate(self._lens_dic(y1, y2, gamma, beta, kappa))

    def test_production_lnlike_pins_bruteforce_at_near_cusp(self):
        """Production ``lnlike`` (kernel_subsamples=8) matches brute force."""
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
            f'{abs(rb - bf):.4g} nat (>1); the F006 fix regressed')

    def test_edge_secant_canary_reproduces_aliasing_pathology(self):
        """
        The ``kernel_subsamples=2`` edge secant blows up at the same config,
        confirming the sub-sample least-squares reduction is load-bearing
        (the pin above is non-vacuous, not a coincidence of a benign bin
        grid).
        """
        cand = self._near_cusp_candidate()
        bf = self.like.lnlike_bruteforce(cand)
        rb_secant = self.like_secant.lnlike(cand)
        self.n_checks += 1
        self.assertTrue(np.isfinite(bf), f'near-cusp: non-finite bf={bf}')
        self.assertGreater(
            abs(rb_secant - bf), SECANT_ALIAS_MIN,
            f'edge-secant lnl {rb_secant:.6g} is within '
            f'{abs(rb_secant - bf):.4g} of brute force {bf:.6g}; the '
            'aliasing pathology did not reproduce, so the near-cusp pin '
            'is not demonstrably load-bearing')


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


class NormalizationFloorZeroNoiseTestCase(LensedLikelihoodTestCase):
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
        cls.zero_like = LensedRelativeBinningLikelihood(
            zero_event, zero_generator, cls.par_dic_0,
            delta_t_max=DELTA_T_MAX, fbin=cls.fbin)

    def test_bruteforce_floor_is_physically_tight(self):
        exact_unlensed = self.zero_like.lnlike_fft(self.par_dic_0)
        lensed_bf = self.zero_like.lnlike_bruteforce(self._tiny_candidate())
        self.n_checks += 1
        self.assertLessEqual(
            abs(lensed_bf - exact_unlensed), ZERO_NOISE_TOL,
            f'zero-noise lensed brute-force at F~1 ({lensed_bf:.6g}) != '
            f'exact unlensed lnlike_fft ({exact_unlensed:.6g}) within '
            f'{ZERO_NOISE_TOL}; an O(SNR^2) offset flags a factor error')

    def test_relative_binning_floor_is_physically_tight(self):
        exact_unlensed = self.zero_like.lnlike_fft(self.par_dic_0)
        lensed_rb = self.zero_like.lnlike(self._tiny_candidate())
        self.n_checks += 1
        self.assertLessEqual(
            abs(lensed_rb - exact_unlensed), ZERO_NOISE_TOL,
            f'zero-noise lensed RB at F~1 ({lensed_rb:.6g}) != exact '
            f'unlensed lnlike_fft ({exact_unlensed:.6g}) within '
            f'{ZERO_NOISE_TOL}; the residual is template construction, '
            'not normalization -- an O(SNR^2) offset flags a factor error')


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
