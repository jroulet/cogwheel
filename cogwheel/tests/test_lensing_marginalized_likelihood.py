"""
Tests for `lensing.marginalized_likelihood` -- the coherent-score
extrinsic-marginalized likelihood `LensedMarginalizedExtrinsicLikelihood`
for Chang--Refsdal microlensed CBC signals (WP1), and its registered prior
+ `LensedPosterior` reuse wiring (WP2).

WHAT THIS SUITE PINS (crown gate: the amplification is folded correctly)
------------------------------------------------------------------------
`LensedMarginalizedExtrinsicLikelihood` reuses the stock coherent-score
machinery (`MarginalizedExtrinsicLikelihood`) but folds the total lensing
amplification ``F`` into the per-mode matched-filter timeseries: ``F*h`` in
the data term and ``|F|**2`` scaling the norm.  The value of the class is
nothing unless that fold is exact in the unlensed limit and matches an
INDEPENDENT reconstruction of ``F`` elsewhere.  Each oracle below is chosen
so the check is not circular:

* UNLENSED-LIMIT IDENTITY (spec 1).  At ``gamma = kappa = 0`` with a tiny
  lens mass, ``w -> 0`` and the certified operator limit is ``F == 1``
  EXACTLY (FINDINGS F009: only the trivial macro sector gives unity; a
  sheared config tends to ``1/sqrt(1-gamma**2) != 1``).  There the lensed
  ``_get_dh_hh_timeshift`` must reproduce the stock UNLENSED
  `MarginalizedExtrinsicLikelihood._get_dh_hh_timeshift` built on the same
  event / generator / bins / coherent score, bit-for-bit up to ``F``'s
  ``O(w) ~ 1e-7`` wave residual.  The stock class is a DIFFERENT code path
  (no lens engine at all), so this catches a spurious per-image time shift
  or a kernel-amplitude error in the fold.

* SINGLE-IMAGE TIME-SHIFT / AMPLITUDE (spec 2).  On a genuinely lensed
  config the lensed data timeseries is compared against a manual reference
  that folds the EXACT engine amplification -- obtained ONLY from
  `LensedWaveformGenerator.amplification` (the untouched
  ``ChangRefsdalChannels.exact_total`` path, NEVER the class under test) --
  into the same linear-free templates and contracts with the stock
  ``_d_h_weights``.  An `ast`/import guard forbids the module-under-test's
  names inside the oracle helper, so the oracle cannot silently reuse the
  production ``_edge_amplification``.  The residual is the class's
  linear-kernel edge reconstruction vs the exact total (~2e-4), so the gate
  is ``3e-3`` -- tight enough to catch an O(1) delay or kernel error, loose
  enough for the certified linear-model reconstruction.

* NORM-PATH ``|F|**2`` CROSS-CHECK (spec 7).  The ``|F(f_b)|**2`` that
  scales the marginalized norm (from the production ``_edge_amplification``)
  is cross-checked against ``|F_exact(f_b)|**2`` from the same independent
  `LensedWaveformGenerator.amplification` engine at the bin edges, ``3e-3``
  relative -- the fast-path interpolation tolerance.  A delay-difference or
  bin-guard bug shows as a per-bin ``|F|**2`` divergence.

* REFUSAL CONTRACT (spec 5).  A macro-saddle config raises
  `geometry.LensDomainError` and a cancellation-band config raises
  `SchwingerCertificationError`; the lensed ``_get_dh_hh_timeshift`` calls
  the engine BEFORE the coherent score, so the refusal must propagate with
  the coherent-score ``get_marginalization_info`` call-count at exactly
  ZERO, and `LensedPosterior` must map it to an exact ``-inf`` (no NaN).
  The mutation guard proves this is non-vacuous: swallowing the engine
  refusal (returning finite arrays) makes the coherent score run.

* BIN GUARD UNDER MARGINALIZATION (spec 9).  Constructing with a
  ``delta_t_max`` that breaches ``pi * Delta_f_bin * delta_t_max`` on the
  coarse bins raises `LensedBinningError` from inside ``_set_summary`` --
  the guard is active on the marginalized path, not just the plain engine.

* REGISTRATION / PAIRING / DETERMINISM / SERIALIZATION (spec 8).  The
  registered `LensedMarginalizedExtrinsicIASPrior` is present in
  ``gw_prior.prior_registry``, its ``standard_params`` equal the
  likelihood's ``params``, seeded prior draws yield finite-or-exact-``-inf``
  posteriors with zero NaN, the deterministic ``_get_dh_hh_timeshift`` is
  bit-repeatable, and a `JSONMixin` round-trip reproduces it bit-for-bit
  (engine rebuilt from primitives, not stored as an init arg).

* CONDITIONAL-DRAW CONSISTENCY (spec 6).  Extrinsic samples drawn from the
  marginalization info via ``gen_samples_from_marg_info`` land in the
  high-likelihood region: the plain `LensedRelativeBinningLikelihood.lnlike`
  evaluated at each drawn (intrinsic+lens+extrinsic) point is finite and
  its LOW percentile stays at or above ``lnL_marg`` (a biased conditional --
  wrong sky/distance/time -- would crater specific draws far below).  NOTE:
  in cogwheel's extrinsic-marginalization normalization the peak plain
  ``lnlike`` sits ABOVE ``lnL_marg`` by the extrinsic Occam factor (~tens of
  nats over 5-6 dims), so the meaningful, convention-robust direction is a
  LOWER bound; the raw spec's ``none > lnL_marg + 0.5`` upper bound encodes
  a different (peak==marginal) normalization that this codebase does not use
  and is deliberately not asserted.

TOLERANCES
----------
The reconstruction gates (specs 2, 7) use ``3e-3`` relative: the class's
per-bin linear kernel model reconstructs the exact engine amplification to
~2e-4 on these configs, and ``3e-3`` sits an order above that while still
failing on any O(1) delay/kernel error.  The unlensed-limit identity (spec
1) uses ``1e-6`` relative: only ``F``'s ``O(w) ~ 1e-7`` wave residual and
float32 template accumulation separate the two paths there.  The
conditional-draw lower bound (spec 6) uses a generous nats margin because
the coherent-score QMC estimate of ``lnL_marg`` is itself stochastic.

DETERMINISM AND ANTI-VACUITY
----------------------------
Every stochastic input is seeded (``EventData.gaussian_noise(seed=SEED)``
draws from its own ``np.random.default_rng``).  The marginalized ``lnlike``
is a Quasi-Monte-Carlo estimate and is NOT bit-repeatable (that
stochasticity is owned by the base coherent score); the DETERMINISTIC
``_get_dh_hh_timeshift`` IS, and that is what spec 8 pins with
``assertEqual``.  ``_MarginalizedLensTestCase.tearDown`` fails any test that
made zero comparisons, and `SelfFalsificationTestCase` proves the central
fold gate can go red.
"""
from __future__ import annotations

import ast
import functools
import inspect
import pathlib
import types
import warnings
import os
from unittest import TestCase, main, mock, skipUnless

import numpy as np
from matplotlib import pyplot as plt

import lal

from cogwheel import data, gw_prior, skyloc_angles, waveform
from cogwheel.prior import FixedPrior
from cogwheel.likelihood.marginalized_extrinsic import (
    MarginalizedExtrinsicLikelihood)
from cogwheel.likelihood.reference_waveform_finder import (
    ReferenceWaveformFinder)
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, LensedBinningError)
from cogwheel.lensing.marginalized_likelihood import (
    LensedMarginalizedExtrinsicLikelihood)
from cogwheel.lensing.posterior import LensedPosterior
from cogwheel.lensing.prior import LensedMarginalizedExtrinsicIASPrior
from cogwheel.lensing.waveform import (
    LensedWaveformGenerator, dimensionless_frequency)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError, W_CEILING_SCHWINGER)

#: Higher-mode precessing approximant, so the mode-pair (``M**2``) norm
#: contraction and the per-``|m|`` data fold are genuinely exercised
#: (|m| in {1, 2, 3, 4}), not the trivial 22-only case.
APPROXIMANT = 'IMRPhenomXPHM'

#: Fixed seed for every stochastic input.  ``EventData.gaussian_noise``
#: uses ``np.random.default_rng(seed)`` internally, so this -- not a bare
#: ``np.random.seed`` -- fixes the noise realization.
SEED = 20260718

#: Bin width [Hz] of the uniform relative-binning grid.  Chosen so the
#: lens-aware criterion ``pi * Delta_f_bin * DELTA_T_MAX = pi*4*0.02 =
#: 0.25`` clears the default 0.5 rad tolerance while staying fine enough
#: that the linear-in-bin kernel model is accurate.
DF_BIN = 4.0

#: Largest relative image delay [s] the main fixture's bins support.
DELTA_T_MAX = 0.02

#: Lens mass [Msun] / redshift for the well-conditioned main fixture.
M_LENS_MSUN = 90.0
Z_LENS = 0.4

#: Well-conditioned two-image lens config (moderate fringe, trivial
#: convergence): in-band ``w`` of order a few, image delays a few ms, far
#: from any cancellation refusal.  Seven standard lens keys.
MAIN_LENS = {'m_lens_msun': M_LENS_MSUN, 'z_lens': Z_LENS,
             'y1': 0.20, 'y2': 0.05, 'gamma': 0.10, 'beta': 0.0,
             'kappa': 0.0}

#: Trivial-macro tiny-mass config: ``gamma = kappa = 0`` so the certified
#: ``w -> 0`` operator limit is genuinely ``F == 1`` (F009); the ~1e-7
#: ``O(w)`` wave residual is all that separates it from the unlensed path.
UNLENSED_LIMIT_LENS = {'m_lens_msun': 1e-6, 'z_lens': Z_LENS,
                       'y1': 0.12, 'y2': 0.035, 'gamma': 0.0, 'beta': 0.0,
                       'kappa': 0.0}

#: Over-critical (Type III, ``1 - kappa <= 0``): both the engine and
#: generator raise `geometry.LensDomainError`.  (The former macro-saddle
#: pin at gamma=0.50/kappa=0.60 is a saddle INTERIOR that EVALUATES since
#: Build 7b, so the refusal-precedence contract is pinned at the
#: over-critical domain, which stays a named refusal.)
OVER_CRITICAL_LENS = {'m_lens_msun': M_LENS_MSUN, 'z_lens': Z_LENS,
                      'y1': 0.20, 'y2': 0.05, 'gamma': 0.50, 'beta': 0.0,
                      'kappa': 1.50}

#: HARD-CORE wave-branch config: a near-caustic 4-image positive-parity
#: source whose above-ceiling nodes (``w > 60``) are refused by BOTH
#: uniform arms (fold argument xi ~ 2.4 and Pearcey radius R ~ 2.6 both
#: too small to certify), so the engine raises the named
#: `SchwingerCertificationError`.  RE-BASELINE (Build 8e serving ladder):
#: the previous strong-shear ``CANCELLATION_LENS`` (gamma' = 0.94,
#: y = (0.20, 0.05)) is now ARM-SERVED at the engine, so its refusal moved
#: downstream to the bin guard (`LensedBinningError`) -- no longer a
#: wave-branch refusal.  This hard-core config keeps the refusal on the
#: engine's named wave-branch exit, which the refusal-precedence contract
#: needs.  The ``m_lens_msun`` x4 scale pushes the refusing nodes above
#: the ``w = 60`` Schwinger ceiling (engine refusal precedes the bin
#: guard, so the tiny-splitting delay is immaterial here).
CANCELLATION_LENS = {'m_lens_msun': M_LENS_MSUN * 4, 'z_lens': Z_LENS,
                     'y1': 0.10, 'y2': 0.10, 'gamma': 0.47, 'beta': 0.0,
                     'kappa': 0.0}

#: Relative tolerance on the unlensed-limit ``_get_dh_hh_timeshift``
#: identity (spec 1).  Only ``F``'s ``O(w) ~ 1e-7`` wave residual and
#: float32 accumulation separate the lensed and stock paths there.
UNLENSED_IDENTITY_RTOL = 1e-6

#: Relative tolerance on the amplification-reconstruction gates (specs 2,
#: 7): the production per-bin linear kernel model reconstructs the exact
#: engine total to ~2e-4 on these configs; ``3e-3`` sits an order above
#: that yet fails on any O(1) delay/kernel error.
RECONSTRUCTION_RTOL = 3e-3

#: Number of conditional extrinsic draws (spec 6).
N_CONDITIONAL_DRAWS = 50

#: Nats the LOW percentile of the plain ``lnlike`` at conditional draws may
#: fall below ``lnL_marg`` (spec 6).  Generous: ``lnL_marg`` is itself a
#: stochastic QMC estimate and the draws sit well above it (Occam factor).
CONDITIONAL_LOWER_MARGIN = 0.5

#: Percentile of the conditional-draw plain ``lnlike`` used for the lower
#: bound; the 10th percentile ignores a stray low-weight tail draw while
#: still cratering under a biased sky/distance/time conditional.
CONDITIONAL_LOW_PERCENTILE = 10.0

#: Nats the BEST conditional draw may fall below ``lnL_marg`` (spec 6).  The
#: coherent-score fold guarantees the conditional posterior mode reconstructs
#: the marginalized value up to the QMC estimator noise, so ``max`` must very
#: nearly reach (and typically exceed) ``lnL_marg``.
CONDITIONAL_MAX_MARGIN = 0.3

#: Number of seeded prior draws for the finite-or-(-inf) / no-NaN sweep
#: (spec 8c).  COST (F061): the sweep caps the box lens mass so EVERY draw
#: evaluates at ``w <= 60`` on the fast DOUBLE-DOUBLE Schwinger path
#: (~0.2 s/engine-eval); the dominant per-draw cost is the coherent-score
#: QMC marginalization at the ~90% of draws that are in-support (~1.6 s
#: each, measured end-to-end incl. lnposterior overhead), so 30 draws is
#: ~48 s -- inside the fast-tier <60 s single-test ceiling.  WITHOUT the
#: mass cap, the un-capped box (up to 3500 Msun) sends high-mass saddle
#: draws to ``w`` in ``(60, 150]`` -> the mpmath arbitrary-precision path
#: at ~100 s EACH, a build-killer.
N_PRIOR_DRAWS = 30

#: Target ceiling [dimensionless] on the maximum in-band lensing frequency
#: ``w = xi(m_lens, z_lens) * f`` any prior draw may reach, sitting a
#: comfortable margin below the mpmath dispatch threshold
#: `W_CEILING_SCHWINGER` (= 60).  The sweep reduces ONLY the box lens-mass
#: lever so ``w <= W_SWEEP_CEILING`` at the band top for the largest sampled
#: mass; the reduced mass is orthogonal to the certified/refused split,
#: which is a shear (``gamma'``) / source-position phenomenon (Professor
#: Ruling 4), so both finite and exact ``-inf`` outcomes survive.
#: DERIVED from `W_CEILING_SCHWINGER` (a fixed 5-unit margin below it) so
#: the sweep target follows the dispatch threshold instead of stranding.
W_SWEEP_CEILING = W_CEILING_SCHWINGER - 5.0

#: Names of the module under test that must NOT appear inside the spec-2
#: independent oracle helper (F002 oracle-tautology guard).
FORBIDDEN_ORACLE_NAMES = (
    'LensedMarginalizedExtrinsicLikelihood',
    'marginalized_likelihood',
    '_edge_amplification',
    '_get_dh_hh_timeshift',
    '_TWO_PI_I')

#: Directory for diagnostic plots (created on demand); never shown.
OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

# A test must never open a GUI window (headless CI has no display).
plt.switch_backend('Agg')


def _reference_par_dic() -> dict:
    """
    Deterministic precessing CBC reference ``par_dic`` for `APPROXIMANT`.

    Explicit (not randomly drawn) so the fixture is reproducible; keys are
    asserted against ``waveform.WaveformGenerator.params`` in the harness,
    so a schema drift fails loudly.
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


def _uniform_fbin(f_lo: float, f_hi: float, df_bin: float) -> np.ndarray:
    """Uniform relative-binning edges spanning ``[f_lo, f_hi]``."""
    edges = np.arange(f_lo, f_hi, df_bin)
    if edges[-1] < f_hi:
        edges = np.append(edges, f_hi)
    return edges


@functools.lru_cache(maxsize=1)
def _harness() -> types.SimpleNamespace:
    """
    Build (once) the shared event, generator, bins and likelihoods.

    Cached so the ~20 s XPHM injection and coherent-score summary build are
    paid a SINGLE time across every `TestCase`.  The stock unlensed
    `MarginalizedExtrinsicLikelihood` is built sharing the lensed
    likelihood's coherent score, so their summary weights are identical and
    the unlensed-limit identity (spec 1) isolates ONLY the ``F`` fold.
    """
    par_dic_cbc = _reference_par_dic()
    assert sorted(par_dic_cbc) == waveform.WaveformGenerator.params, (
        'reference par_dic keys drifted from WaveformGenerator.params; '
        'update _reference_par_dic')

    event_data = data.EventData.gaussian_noise(
        eventname='test_lensed_marg', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0., seed=SEED)
    event_data.inject_signal(par_dic_cbc, APPROXIMANT)

    wfg = waveform.WaveformGenerator.from_event_data(event_data, APPROXIMANT)

    band = event_data.frequencies[event_data.fslice]
    fbin = _uniform_fbin(float(band[0]), float(band[-1]), DF_BIN)

    par_dic_0 = {**par_dic_cbc, **MAIN_LENS}

    lensed_marg = LensedMarginalizedExtrinsicLikelihood(
        event_data, wfg, par_dic_0, delta_t_max=DELTA_T_MAX, fbin=fbin)

    # Stock unlensed marginalized likelihood on the SAME bins, sharing the
    # lensed likelihood's coherent score -> identical summary weights.
    unlensed_marg = MarginalizedExtrinsicLikelihood(
        event_data, wfg, par_dic_cbc, fbin=fbin,
        coherent_score=lensed_marg.coherent_score)

    # Registered marginalized prior + LensedPosterior (WP2 wiring).  The
    # prior samples only intrinsic CBC + lens, so its ``standard_params``
    # equal the marginalized likelihood's ``params``.
    rwf = ReferenceWaveformFinder(
        event_data, wfg, par_dic_cbc, pn_phase_tol=0.05)
    marg_prior = LensedMarginalizedExtrinsicIASPrior\
        .from_reference_waveform_finder(rwf)
    posterior = LensedPosterior(marg_prior, lensed_marg)

    return types.SimpleNamespace(
        par_dic_cbc=par_dic_cbc, par_dic_0=par_dic_0, event_data=event_data,
        waveform_generator=wfg, fbin=fbin, lensed_marg=lensed_marg,
        unlensed_marg=unlensed_marg, plain_engine=lensed_marg._engine,
        rwf=rwf, prior=marg_prior, posterior=posterior)


def _intrinsic_lens_point(marg, lens_dic: dict, **overrides) -> dict:
    """
    A ``par_dic`` over exactly ``marg.params`` (intrinsic CBC + 7 lens
    keys), from the CBC reference merged with ``lens_dic``.
    """
    full = {**_reference_par_dic(), **lens_dic, **overrides}
    return {key: full[key] for key in marg.params}


def _candidate_par_dic(lens_dic: dict, **overrides) -> dict:
    """Full CBC + lens ``par_dic`` for the plain engine (all engine params)."""
    return {**_reference_par_dic(), **lens_dic, **overrides}


def _exact_edge_amplification(lens_dic: dict, fbin: np.ndarray) -> np.ndarray:
    """
    Independent EXACT total amplification ``F`` at the bin edges ``fbin``.

    Oracle for specs 2 and 7: built ONLY from `LensedWaveformGenerator`,
    whose ``amplification`` returns the untouched
    ``ChangRefsdalChannels.exact_total`` -- a DIFFERENT code path from the
    production per-bin linear kernel reconstruction.  An `ast` guard
    (``test_oracle_is_independent``) forbids any module-under-test name
    here, so this cannot silently reuse the production edge fold.
    """
    dummy_generator = types.SimpleNamespace(
        get_hplus_hcross=lambda *a, **k: None,
        m_arr=np.array([2]))
    lens_wfg = LensedWaveformGenerator(
        dummy_generator, m_lens_msun=lens_dic['m_lens_msun'],
        z_lens=lens_dic['z_lens'], y=(lens_dic['y1'], lens_dic['y2']),
        gamma=lens_dic['gamma'], beta=lens_dic['beta'],
        kappa=lens_dic['kappa'])
    return lens_wfg.amplification(np.asarray(fbin, dtype=float))


class _MarginalizedLensTestCase(TestCase):
    """
    Shared base carrying the anti-vacuity guard.

    ``setUp`` zeroes a per-test comparison counter; every concrete test
    increments ``self.n_compared`` for each oracle comparison it actually
    runs.  ``tearDown`` FAILS if the counter is still zero, so a silently
    empty sweep or a skipped fixture cannot read green.
    """

    #: Set True only by tests that legitimately record no comparison.
    allow_zero_comparisons = False

    def setUp(self):
        self.n_compared = 0

    def tearDown(self):
        if not self.allow_zero_comparisons and self.n_compared == 0:
            self.fail(
                f'{self._testMethodName} made zero comparisons -- the test '
                'is vacuous (empty sweep or skipped fixture). A green result '
                'here would be a false pass.')


@skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class UnlensedLimitTimeseriesIdentityTestCase(_MarginalizedLensTestCase):
    """
    Spec 1 -- at ``gamma = kappa = 0`` with a tiny lens mass (``F == 1``,
    F009), the lensed ``_get_dh_hh_timeshift`` reproduces the stock
    UNLENSED `MarginalizedExtrinsicLikelihood` one built on the same
    event / generator / bins / coherent score, up to ``F``'s ``O(w)``
    residual.  A spurious per-image shift or kernel-amplitude bug fails.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()

    def _both_dh_hh(self):
        """(lensed, unlensed) ``(dh_mptd, hh_mppd)`` at the tiny-mass point."""
        intrinsic = _reference_par_dic()
        lensed_point = {**intrinsic, **UNLENSED_LIMIT_LENS}
        lensed_point = {k: lensed_point[k] for k in self.h.lensed_marg.params}
        unlensed_point = {k: intrinsic[k]
                          for k in self.h.unlensed_marg.params}

        dh_l, hh_l, _ = self.h.lensed_marg._get_dh_hh_timeshift(lensed_point)
        dh_u, hh_u, _ = self.h.unlensed_marg._get_dh_hh_timeshift(
            unlensed_point)
        return (dh_l, hh_l), (dh_u, hh_u)

    def test_data_timeseries_matches_unlensed(self):
        """Lensed ``dh_mptd`` equals the stock unlensed one (``F -> 1``)."""
        (dh_l, _), (dh_u, _) = self._both_dh_hh()
        self.assertEqual(dh_l.shape, dh_u.shape)
        rel = (np.max(np.abs(dh_l - dh_u))
               / np.max(np.abs(dh_u)))
        self.n_compared += 1
        self.assertLess(
            rel, UNLENSED_IDENTITY_RTOL,
            f'lensed dh_mptd deviates from unlensed by rel {rel:.2e} at the '
            'F->1 limit -- a spurious per-image time shift or kernel error.')

        self._plot_delta(dh_l, dh_u)

    def test_norm_matches_unlensed(self):
        """Lensed ``hh_mppd`` equals the stock unlensed one (``|F|**2 -> 1``)."""
        (_, hh_l), (_, hh_u) = self._both_dh_hh()
        self.assertEqual(hh_l.shape, hh_u.shape)
        rel = np.max(np.abs(hh_l - hh_u)) / np.max(np.abs(hh_u))
        self.n_compared += 1
        self.assertLess(
            rel, UNLENSED_IDENTITY_RTOL,
            f'lensed hh_mppd deviates from unlensed by rel {rel:.2e} at the '
            '|F|**2 -> 1 limit -- a norm-fold amplitude error.')

    def _plot_delta(self, dh_l, dh_u):
        """Heatmap of ``|dh_l - dh_u|`` over (time, detector) [spec 1]."""
        delta = np.abs(dh_l - dh_u)
        # Collapse the mode / polarization axes to the max per (t, det).
        # dh_mptd axes are (m, p, t, d); reduce all but t and d.
        reduced = delta.max(axis=tuple(
            i for i in range(delta.ndim) if i not in (delta.ndim - 2,
                                                      delta.ndim - 1)))
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        im = ax.imshow(np.atleast_2d(reduced).T, aspect='auto',
                       origin='lower')
        ax.set_xlabel('time index')
        ax.set_ylabel('detector index')
        ax.set_title('|delta dh_mptd| at F->1 (spec 1)')
        fig.colorbar(im, ax=ax)
        fig.savefig(
            OUTPUT_DIR / 'marg_spec1_unlensed_limit_dh_delta.png', dpi=80)
        plt.close(fig)


@skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class SingleImageTimeShiftTestCase(_MarginalizedLensTestCase):
    """
    Spec 2 -- on a genuinely lensed two-image config the production lensed
    ``dh_mptd`` matches a manual reference that folds the EXACT engine
    amplification (from `LensedWaveformGenerator`, never the class under
    test) into the same linear-free templates and contracts with the stock
    ``_d_h_weights``.  A delay error appears as a time offset, a kernel
    error as an amplitude/phase mismatch.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()
        cls.marg = cls.h.lensed_marg

    def _reference_dh(self, candidate: dict) -> np.ndarray:
        """
        Manual ``dh_mptd`` with the INDEPENDENT exact ``F`` folded in.

        Mirrors the production data-term contraction exactly except that
        the amplification comes from `_exact_edge_amplification` (the
        untouched channels engine), not the class under test.
        """
        amplification = _exact_edge_amplification(MAIN_LENS, self.h.fbin)
        h_mpb, _ = self.marg._get_linearfree_hplus_hcross_dt(
            dict(candidate) | self.marg._ref_dic, by_m=True)
        h_mpb = h_mpb.astype(np.complex64)
        h_lensed = (amplification[np.newaxis, np.newaxis, :]
                    * h_mpb).astype(np.complex64)
        dh_mptd = (self.marg._d_h_weights[:, np.newaxis]
                   @ h_lensed.conj()[:, :, np.newaxis, :, np.newaxis])[..., 0]
        dh_mptd *= self.marg.asd_drift.astype(np.float32) ** -2
        return dh_mptd

    def test_dh_matches_exact_amplification_reference(self):
        """Production lensed ``dh_mptd`` == exact-``F`` reference (3e-3)."""
        candidate = _intrinsic_lens_point(self.marg, MAIN_LENS)
        dh_prod, _, _ = self.marg._get_dh_hh_timeshift(candidate)
        dh_ref = self._reference_dh(candidate)
        self.assertEqual(dh_prod.shape, dh_ref.shape)
        rel = np.max(np.abs(dh_prod - dh_ref)) / np.max(np.abs(dh_ref))
        self.n_compared += 1
        self.assertLess(
            rel, RECONSTRUCTION_RTOL,
            f'lensed dh_mptd deviates from the exact-amplification reference '
            f'by rel {rel:.2e} -- a per-image delay or kernel-fold error.')
        self._plot_overlay(dh_prod, dh_ref)

    def test_oracle_is_independent(self):
        """
        F002 guard: the amplification oracle references no name of the
        module under test, so it cannot be gated against itself.
        """
        source = inspect.getsource(_exact_edge_amplification)
        tree = ast.parse(source)
        # Collect every REFERENCED identifier (call targets, attribute
        # accesses, bare names).  The function's own def name and its
        # docstring are not referenced identifiers, so an oracle whose
        # name merely shares a substring with a forbidden token is not
        # flagged; only an actual reference to the module under test is.
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        for forbidden in FORBIDDEN_ORACLE_NAMES:
            self.n_compared += 1
            self.assertNotIn(
                forbidden, identifiers,
                f'oracle helper references forbidden module-under-test name '
                f'{forbidden!r} -- the oracle is not independent.')

    def _plot_overlay(self, dh_prod, dh_ref):
        """Overlay Re/Im of a representative ``z_m(t)`` [spec 2 diagnostic]."""
        # Pick the mode/pol/detector slice of maximal reference amplitude.
        flat = np.abs(dh_ref)
        idx = np.unravel_index(np.argmax(flat), flat.shape)
        sl = idx[:-2] + (slice(None), idx[-1])
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(dh_prod[sl].real, label='prod Re')
        ax.plot(dh_ref[sl].real, '--', label='ref Re')
        ax.plot(dh_prod[sl].imag, label='prod Im')
        ax.plot(dh_ref[sl].imag, '--', label='ref Im')
        ax.set_xlabel('time index')
        ax.set_ylabel('z_m(t)')
        ax.set_title('lensed dh vs exact-F reference (spec 2)')
        ax.legend(fontsize=7)
        fig.savefig(
            OUTPUT_DIR / 'marg_spec2_single_image_dh_overlay.png', dpi=80)
        plt.close(fig)


@skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class NormPathAmplificationTestCase(_MarginalizedLensTestCase):
    """
    Spec 7 -- the ``|F(f_b)|**2`` that scales the marginalized norm (from
    the production edge fold) matches ``|F_exact(f_b)|**2`` from the
    independent `LensedWaveformGenerator` engine at the bin edges, to the
    fast-path interpolation tolerance.  A delay-difference / bin-guard bug
    shows as a per-bin ``|F|**2`` divergence.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()
        cls.marg = cls.h.lensed_marg

    def _production_edge_amplification(self, candidate: dict) -> np.ndarray:
        """Production ``F`` at bin edges, from the engine coefficients."""
        delays, k0, k1, _ = \
            self.marg._engine._amplification_coefficients(candidate)
        self.marg._engine._check_candidate_delays(delays)
        return self.marg._edge_amplification(delays, k0, k1)

    def test_norm_amplitude_matches_exact_engine(self):
        """Production ``|F|**2`` == exact-engine ``|F|**2`` per edge (3e-3)."""
        candidate = _intrinsic_lens_point(self.marg, MAIN_LENS)
        f_prod = self._production_edge_amplification(candidate)
        f_exact = _exact_edge_amplification(MAIN_LENS, self.h.fbin)
        self.assertEqual(f_prod.shape, f_exact.shape)

        prod_sq = np.abs(f_prod) ** 2
        exact_sq = np.abs(f_exact) ** 2
        rel = np.max(np.abs(prod_sq - exact_sq)) / np.max(exact_sq)
        self.n_compared += 1
        self.assertLess(
            rel, RECONSTRUCTION_RTOL,
            f'norm-path |F|**2 deviates from the exact engine by rel '
            f'{rel:.2e} -- a delay-difference or bin-guard problem.')
        self._plot_integrand(prod_sq, exact_sq)

    def test_complex_amplification_matches_exact_engine(self):
        """Production complex ``F`` == exact-engine ``F`` per edge (3e-3)."""
        candidate = _intrinsic_lens_point(self.marg, MAIN_LENS)
        f_prod = self._production_edge_amplification(candidate)
        f_exact = _exact_edge_amplification(MAIN_LENS, self.h.fbin)
        rel = np.max(np.abs(f_prod - f_exact)) / np.max(np.abs(f_exact))
        self.n_compared += 1
        self.assertLess(
            rel, RECONSTRUCTION_RTOL,
            f'complex edge amplification deviates by rel {rel:.2e} -- a '
            'kernel-reconstruction or image-delay-phase error.')

    def _plot_integrand(self, prod_sq, exact_sq):
        """Overlay per-edge ``|F|**2`` (production vs exact) [spec 7]."""
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(self.h.fbin, exact_sq, label='exact engine |F|^2')
        ax.plot(self.h.fbin, prod_sq, '--', label='marginalized |F|^2')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('|F|^2')
        ax.set_title('norm-path |F|^2 cross-check (spec 7)')
        ax.legend(fontsize=8)
        fig.savefig(
            OUTPUT_DIR / 'marg_spec7_norm_amplification.png', dpi=80)
        plt.close(fig)


class RefusalContractTestCase(_MarginalizedLensTestCase):
    """
    Spec 5 -- a macro-saddle (`LensDomainError`) and a cancellation-band
    (`SchwingerCertificationError`) config are refused by the lens engine
    BEFORE the
    extrinsic marginalization: the coherent-score
    ``get_marginalization_info`` call-count stays exactly 0 and
    `LensedPosterior` maps the refusal to an exact ``-inf`` (no NaN).  The
    mutation guard proves this is non-vacuous.
    """

    # ONLY structurally-anchored refusals belong here.  `over_critical` is
    # `kappa >= 1` -- an exact parity-boundary fact that no threshold can
    # move -- so it pins the refusal-precedence contract permanently.
    #
    # The `cancellation` entry was RETIRED 2026-08-13 after its THIRD drift,
    # not repointed a third time.  Its history is the argument: Build 8d
    # picked a cancellation-band config; Build 8e found it "now ARM-SERVED at
    # the engine, so it no longer refuses -- the spy premise died" and
    # repointed it at a "HARD-CORE near-caustic config whose above-ceiling
    # nodes NO arm certifies"; that one then stopped refusing too, and sat in
    # `.claude/sdk/known_failures.txt` from 2026-08-06.  Raising
    # `_MP_PANEL_ORDER` 32 -> 40 (certification now holds through w ~ 204)
    # moves the boundary again.
    #
    # A witness anchored to a CERTIFICATION THRESHOLD drifts every time the
    # serving ladder widens, which is continuously. A witness anchored to a
    # STRUCTURAL boundary does not. The contract under test -- refusal
    # precedes the extrinsic marginalization -- is fully exercised by
    # `over_critical`, which raises a NAMED refusal through the same
    # likelihood path and asserts the same zero spy call-count; nothing about
    # the contract was specific to WHICH named exception arrived.
    #
    # If a cancellation-band witness is ever wanted back, DERIVE it at test
    # time from the live certification boundary and assert the premise (that
    # it genuinely refuses) before using it -- do not pin a fourth literal.
    REFUSING_CONFIGS = (
        ('over_critical', OVER_CRITICAL_LENS, LensDomainError),)

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()
        cls.marg = cls.h.lensed_marg

    def test_refusal_precedes_coherent_score(self):
        """Each refusing config raises with ``get_marginalization_info``
        never called (refusal before the integral)."""
        cs = self.marg.coherent_score
        for label, lens, exc in self.REFUSING_CONFIGS:
            with self.subTest(config=label):
                candidate = _intrinsic_lens_point(self.marg, lens)
                with mock.patch.object(
                        cs, 'get_marginalization_info',
                        wraps=cs.get_marginalization_info) as spy:
                    with self.assertRaises(exc):
                        self.marg.lnlike(candidate)
                    self.n_compared += 1
                    self.assertEqual(
                        spy.call_count, 0,
                        f'{label}: coherent score was consulted despite the '
                        'engine refusal -- refusal did not precede the '
                        'extrinsic marginalization.')

    def _in_support_sampled_vec(self):
        """
        A sampled-coordinate vector inside the prior support (finite
        ``lnprior``), so patching the likelihood actually reaches the
        refusal-catch instead of short-circuiting at the prior.

        Derived DETERMINISTICALLY from the well-conditioned ``MAIN_LENS``
        fixture via ``prior.inverse_transform``.  Rejection-sampling
        random prior draws through the REAL ``lnposterior`` here became a
        TIMEOUT hazard with the ceiling-keyed band-split serve: a
        large-mass draw that formerly refused fast is now legitimately
        ADMITTED and served through the mpmath engine band
        (60 < w <= 150) at minutes per call.
        """
        posterior = self.h.posterior
        prior = posterior.prior
        point = _intrinsic_lens_point(self.marg, MAIN_LENS)
        # Conform to the prior's OWN fixed entries (e.g. ``z_lens``) so
        # the inversion premise derives from the live prior, not a
        # literal that can drift.
        for subprior in prior.subpriors:
            if isinstance(subprior, FixedPrior):
                point.update(subprior.standard_par_dic)
        sampled = prior.inverse_transform(**point)
        vec = [sampled[par] for par in prior.sampled_params]
        # Premise: PRIOR support only.  The consumers mock the likelihood
        # seam, so a finite lnprior is exactly what guarantees the
        # refusal-catch is reached (an out-of-support point would go
        # -inf vacuously, never consulting the likelihood).  Evaluating
        # the REAL lnposterior here would both re-open the slow-serve
        # timeout hazard and mutate the shared engine's adaptive state,
        # breaking the bit-for-bit JSON-roundtrip pin downstream.
        self.assertTrue(
            np.isfinite(prior.lnprior(*vec)),
            'premise: the deterministic MAIN_LENS point must be inside '
            'the prior support (finite lnprior); it drifted out.')
        return vec

    def test_posterior_maps_refusal_to_exact_neg_inf(self):
        """`LensedPosterior` returns exact ``-inf`` / ``None`` on refusal."""
        posterior = self.h.posterior
        vec = self._in_support_sampled_vec()

        for label, exc in (('macro_saddle', LensDomainError),
                           ('cancellation', SchwingerCertificationError)):
            with self.subTest(refusal=label):
                with mock.patch.object(
                        posterior.likelihood, 'lnlike_and_metadata',
                        side_effect=exc('injected refusal')):
                    lnpost, par_dic, metadata = \
                        posterior.lnposterior_pardic_and_metadata(*vec)
                self.n_compared += 1
                self.assertTrue(
                    np.isneginf(lnpost),
                    f'{label}: expected exact -inf, got {lnpost!r}.')
                self.assertFalse(np.isnan(lnpost))
                self.assertIsNone(metadata)
                self.assertIsInstance(par_dic, dict)

    def test_swallowing_mutation_would_consult_coherent_score(self):
        """
        Mutation guard: a swallowing ``_get_dh_hh_timeshift`` (returning
        finite arrays instead of propagating the refusal) makes the
        coherent score run -- so the call-count-0 assertion above CAN go
        red, i.e. it is non-vacuous.
        """
        cs = self.marg.coherent_score
        # Finite, valid data/norm arrays from a NON-refusing point.
        finite = self.marg._get_dh_hh_timeshift(
            _intrinsic_lens_point(self.marg, UNLENSED_LIMIT_LENS))
        refusing_candidate = _intrinsic_lens_point(
            self.marg, OVER_CRITICAL_LENS)

        with mock.patch.object(self.marg, '_get_dh_hh_timeshift',
                               new=lambda par_dic: finite):
            with mock.patch.object(
                    cs, 'get_marginalization_info',
                    wraps=cs.get_marginalization_info) as spy:
                # Must NOT raise now (refusal swallowed by the mutation).
                self.marg.lnlike(refusing_candidate)
                self.n_compared += 1
                self.assertGreaterEqual(
                    spy.call_count, 1,
                    'swallowing the refusal did not reach the coherent '
                    'score -- the refusal-contract test would be vacuous.')


class BinGuardTestCase(_MarginalizedLensTestCase):
    """
    Spec 9 -- constructing `LensedMarginalizedExtrinsicLikelihood` with a
    ``delta_t_max`` too large for the coarse bins
    (``pi * Delta_f_bin * delta_t_max`` breaching the bin delay tolerance)
    raises `LensedBinningError` from inside ``_set_summary``.  The guard is
    active on the MARGINALIZED path, not just the plain engine.
    """

    #: ``pi * DF_BIN * delta_t_max = pi*4*0.5 = 6.28 rad`` >> the 0.5 rad
    #: default tolerance -- comfortably over the guard.
    BAD_DELTA_T_MAX = 0.5

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()

    def test_construction_raises_lensed_binning_error(self):
        """A too-coarse ``delta_t_max`` is rejected at construction."""
        self.n_compared += 1
        with self.assertRaises(LensedBinningError):
            LensedMarginalizedExtrinsicLikelihood(
                self.h.event_data, self.h.waveform_generator, self.h.par_dic_0,
                delta_t_max=self.BAD_DELTA_T_MAX, fbin=self.h.fbin)

    def test_reference_construction_succeeds(self):
        """
        Positive control: the SAME bins accept the fixture's
        ``DELTA_T_MAX``, so the guard rejects the coarse case on merit,
        not because construction is universally broken.
        """
        self.n_compared += 1
        # The harness likelihood was built with DELTA_T_MAX on these bins
        # and is a live object -- its mere existence is the control.
        self.assertIsInstance(
            self.h.lensed_marg, LensedMarginalizedExtrinsicLikelihood)
        self.assertLess(
            np.pi * DF_BIN * DELTA_T_MAX, np.pi * DF_BIN * self.BAD_DELTA_T_MAX)


class RegistrationPairingSerializationTestCase(_MarginalizedLensTestCase):
    """
    Spec 8 -- the registered `LensedMarginalizedExtrinsicIASPrior` is in the
    prior registry, its ``standard_params`` equal the likelihood's
    ``params``, seeded prior draws yield finite-or-exact-``-inf`` posteriors
    with zero NaN, the DETERMINISTIC ``_get_dh_hh_timeshift`` is
    bit-repeatable, and a `JSONMixin` round-trip rebuilds the engine from
    primitives and reproduces that deterministic layer bit-for-bit.

    NOTE (spec 8d): the marginalized ``lnlike`` is a Quasi-Monte-Carlo
    estimate owned by the base coherent score and is deliberately NOT
    bit-repeatable; the WP1 code's deterministic contribution is
    ``_get_dh_hh_timeshift``, so bit-repeatability is pinned there (and
    across serialization), where it is a genuine property of this class.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()
        cls.marg = cls.h.lensed_marg

    def test_prior_is_registered(self):
        """The marginalized prior is discoverable in the registry."""
        self.n_compared += 1
        self.assertIn('LensedMarginalizedExtrinsicIASPrior',
                      gw_prior.prior_registry)
        self.assertIs(
            gw_prior.prior_registry['LensedMarginalizedExtrinsicIASPrior'],
            LensedMarginalizedExtrinsicIASPrior)

    def test_prior_default_likelihood_is_the_marginalized_class(self):
        """The registered prior defaults to this likelihood class."""
        self.n_compared += 1
        self.assertIs(
            LensedMarginalizedExtrinsicIASPrior.default_likelihood_class,
            LensedMarginalizedExtrinsicLikelihood)

    def test_prior_standard_params_equal_likelihood_params(self):
        """``prior.standard_params`` == ``likelihood.params`` (set)."""
        self.n_compared += 1
        self.assertEqual(set(self.h.prior.standard_params),
                         set(self.marg.params))

    def test_prior_draws_are_finite_or_exact_neg_inf(self):
        """
        No seeded prior draw yields NaN or ``+inf`` (spec 8c); every draw is
        finite or exactly ``-inf``, and BOTH outcomes are present.

        The sampling box's lens-mass lever is capped (mass axis ONLY) so
        every draw evaluates at ``w <= W_SWEEP_CEILING < 60`` on the fast
        DOUBLE-DOUBLE Schwinger path -- no draw reaches the mpmath band
        (F061: ~0.2 s fast double-double vs ~100 s mpmath).  Per Professor
        Ruling 4 the lens mass is orthogonal to the certified/refused split
        (a shear-``gamma'`` / source-position phenomenon), so keeping the
        shear range (0..1.6, spanning the ~0.5 ``F_op`` cancellation band)
        and the source box (straddling the ``r = 1`` caustic, inside/outside)
        at FULL extent preserves BOTH finite (low-shear interior) and exact
        ``-inf`` (cancellation / saddle / outside-caustic / prior-boundary)
        outcomes.  Only the finite-vs-exact-``-inf`` VALUE is asserted, never
        which band produced it (Professor Ruling 5).
        """
        posterior = self.h.posterior
        prior = posterior.prior

        # Reduce ONLY the lens-mass axis of the box so max-in-band w <= the
        # sweep ceiling.  ``z_lens`` is fixed at 0 by FixedLensGeometryPrior
        # and ``w = xi(m) * f`` is linear in mass, so the largest sampled
        # mass sets the band-top w; capping mass caps w, leaving shear and
        # source position (hence the certified/refused split) untouched.
        i_mass = prior.sampled_params.index('ln_m_lens_msun')
        f_top = float(self.h.fbin[-1])
        w_per_msun = float(dimensionless_frequency(f_top, 1.0, 0.0))
        ln_m_cap = np.log(W_SWEEP_CEILING / w_per_msun)
        cubemin = prior.cubemin.copy()
        cubesize = prior.cubesize.copy()
        ln_m_hi = min(prior.cubemin[i_mass] + prior.cubesize[i_mass], ln_m_cap)
        cubesize[i_mass] = ln_m_hi - prior.cubemin[i_mass]
        self.assertGreater(
            cubesize[i_mass], 0.0,
            'capped lens-mass axis collapsed to an empty range')

        rng = np.random.default_rng(SEED + 8)
        n_finite = n_neginf = 0
        w_maxes = np.empty(N_PRIOR_DRAWS)
        for idx in range(N_PRIOR_DRAWS):
            vec = cubemin + rng.uniform(0.0, 1.0, cubemin.shape) * cubesize
            w_maxes[idx] = dimensionless_frequency(
                f_top, float(np.exp(vec[i_mass])), 0.0)
            value = posterior.lnposterior(*vec)
            self.n_compared += 1
            self.assertFalse(np.isnan(value), f'NaN lnposterior at {vec!r}')
            self.assertFalse(np.isposinf(value), '+inf lnposterior')
            self.assertTrue(np.isfinite(value) or np.isneginf(value))
            n_finite += int(np.isfinite(value))
            n_neginf += int(np.isneginf(value))

        # Every draw stayed on the fast double-double path (F061): no engine
        # call saw ``w > 60``, so the mpmath arbitrary-precision path (the
        # ~100 s/draw build-killer) never fired.
        self.assertLessEqual(
            float(w_maxes.max()), W_CEILING_SCHWINGER,
            f'a draw reached w = {w_maxes.max()} > {W_CEILING_SCHWINGER}; the '
            'mass cap failed and the slow mpmath path would run.')

        # Non-vacuity: the sweep must exercise BOTH outcome classes, else it
        # proves nothing about finite evaluations (n_finite guard, unchanged)
        # OR about the exact ``-inf`` refusal mapping (n_neginf guard, added).
        self.assertGreater(
            n_finite, 0, 'no prior draw produced a finite posterior -- the '
            'no-NaN sweep would be vacuous.')
        self.assertGreater(
            n_neginf, 0, 'no prior draw produced an exact -inf posterior -- '
            'the finite/-inf sweep would not exercise the refusal mapping.')

        self._save_prior_sweep_diagnostic(w_maxes, n_finite, n_neginf)

    @staticmethod
    def _save_prior_sweep_diagnostic(w_maxes, n_finite, n_neginf):
        """
        Diagnostic: histogram of the per-draw max in-band ``w`` (all must
        sit below the `W_CEILING_SCHWINGER` mpmath threshold) beside a bar
        of the finite / exact-``-inf`` outcome counts.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.2))
        ax0.hist(w_maxes, bins=20, color='tab:blue', alpha=0.85)
        ax0.axvline(W_CEILING_SCHWINGER, color='tab:red', ls='--',
                    label=f'mpmath threshold w={W_CEILING_SCHWINGER:g}')
        ax0.set_xlabel('max in-band w per draw')
        ax0.set_ylabel('draw count')
        ax0.set_title('all draws on fast double-double path')
        ax0.legend(fontsize=8)
        ax1.bar(['finite', 'exact -inf'], [n_finite, n_neginf],
                color=['tab:green', 'tab:gray'])
        ax1.set_ylabel('draw count')
        ax1.set_title('outcome classes')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'prior_sweep_wmax_and_outcomes.png', dpi=110)
        plt.close(fig)

    def test_deterministic_timeshift_is_bit_repeatable(self):
        """``_get_dh_hh_timeshift`` is an exact pure function (spec 8d)."""
        point = _intrinsic_lens_point(self.marg, MAIN_LENS)
        dh1, hh1, ts1 = self.marg._get_dh_hh_timeshift(point)
        dh2, hh2, ts2 = self.marg._get_dh_hh_timeshift(point)
        self.n_compared += 1
        np.testing.assert_array_equal(dh1, dh2)
        np.testing.assert_array_equal(hh1, hh2)
        self.assertEqual(ts1, ts2)

    def test_json_roundtrip_reproduces_deterministic_layer(self):
        """
        `JSONMixin` round-trip rebuilds the engine from primitives and
        reproduces ``_get_dh_hh_timeshift`` bit-for-bit (spec 8e).
        """
        import tempfile
        from cogwheel import utils

        point = _intrinsic_lens_point(self.marg, MAIN_LENS)
        dh0, hh0, ts0 = self.marg._get_dh_hh_timeshift(point)

        with tempfile.TemporaryDirectory() as tmp:
            self.marg.to_json(tmp, overwrite=True)
            reloaded = utils.read_json(tmp)

        self.n_compared += 1
        self.assertIsInstance(reloaded, LensedMarginalizedExtrinsicLikelihood)
        # Engine rebuilt from primitives, not stored as an init arg.
        self.assertIsInstance(reloaded._engine, LensedRelativeBinningLikelihood)
        self.assertEqual(set(reloaded.params), set(self.marg.params))

        dh1, hh1, ts1 = reloaded._get_dh_hh_timeshift(point)
        np.testing.assert_array_equal(dh0, dh1)
        np.testing.assert_array_equal(hh0, hh1)
        self.assertEqual(ts0, ts1)


@skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class ConditionalDrawConsistencyTestCase(_MarginalizedLensTestCase):
    """
    Spec 6: conditional extrinsic draws round-trip against the plain engine.

    The coherent-score marginalization stores enough state in ``marg_info``
    to draw extrinsic samples ``(d_luminosity, dec, lon, phi_ref, psi,
    t_geocenter)`` distributed like the conditional posterior at the fixed
    intrinsic+lens point.  Re-evaluating the FULL plain
    ``LensedRelativeBinningLikelihood.lnlike`` at those draws must reconstruct
    the marginalized value: the best draw reaches ``lnL_marg`` (up to QMC
    noise) and no draw craters far below it.

    Convention note
    ---------------
    The raw spec listed an UPPER bound ("no draw exceeds ``lnL_marg`` + 0.5").
    In cogwheel's normalization the plain ``lnlike`` at a single extrinsic
    point sits systematically ABOVE the marginalized value by the extrinsic
    Occam factor (marginalizing 5-6 dimensions costs tens of nats), so an
    upper bound of +0.5 is unphysical and would fail on a CORRECT
    implementation.  The convention-robust, bug-catching direction is the
    LOWER bound: a biased sky / distance / time conditional would push the
    draws BELOW ``lnL_marg``.  That is what this suite gates.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()
        cls.marg = cls.h.lensed_marg

    def _draw_and_score(self, lens_dic, seed):
        """
        Marginalize at ``lens_dic``, draw conditional extrinsic samples, and
        score them with the INDEPENDENT plain engine.

        Returns ``(lnl_marg, lnl_full)`` with ``lnl_full`` a float array of
        length ``N_CONDITIONAL_DRAWS``.
        """
        point = _intrinsic_lens_point(self.marg, lens_dic)
        lnl_marg, marg_info = self.marg.lnlike_and_metadata(point)

        coherent_score = self.marg.coherent_score
        coherent_score._rng = np.random.default_rng(seed)
        samples = coherent_score.gen_samples_from_marg_info(
            marg_info, num=N_CONDITIONAL_DRAWS)

        gmst = lal.GreenwichMeanSiderealTime(self.h.event_data.tgps)
        ra = skyloc_angles.lon_to_ra(np.asarray(samples['lon']), gmst)
        dec = np.asarray(samples['dec'])

        lnl_full = np.empty(N_CONDITIONAL_DRAWS)
        for idx in range(N_CONDITIONAL_DRAWS):
            full = _candidate_par_dic(
                lens_dic,
                d_luminosity=float(samples['d_luminosity'][idx]),
                phi_ref=float(samples['phi_ref'][idx]),
                psi=float(samples['psi'][idx]),
                t_geocenter=float(samples['t_geocenter'][idx]),
                ra=float(ra[idx]),
                dec=float(dec[idx]))
            lnl_full[idx] = self.h.plain_engine.lnlike(full)
        return lnl_marg, lnl_full

    def test_conditional_draws_reconstruct_marginalized_value(self):
        """
        Best draw reaches ``lnL_marg`` (spec 6) and the low percentile does
        not crater below it -- a biased conditional would fail both.
        """
        for lens_dic, label in ((MAIN_LENS, 'main'),
                                 (UNLENSED_LIMIT_LENS, 'unlensed_limit')):
            with self.subTest(config=label):
                lnl_marg, lnl_full = self._draw_and_score(
                    lens_dic, SEED + 6)
                self.n_compared += 1

                self.assertTrue(np.all(np.isfinite(lnl_full)),
                                f'{label}: non-finite conditional draw')

                # Best draw reconstructs (or exceeds) the marginalized value.
                self.assertGreaterEqual(
                    float(np.max(lnl_full)),
                    lnl_marg - CONDITIONAL_MAX_MARGIN,
                    f'{label}: best conditional draw {np.max(lnl_full):.3f} '
                    f'fell >{CONDITIONAL_MAX_MARGIN} below lnL_marg '
                    f'{lnl_marg:.3f} -- the fold under-reconstructs.')

                # Bulk of the draws sit at or above lnL_marg (Occam factor);
                # the low percentile catches a biased sky/distance/time draw.
                low = float(np.percentile(lnl_full, CONDITIONAL_LOW_PERCENTILE))
                self.assertGreaterEqual(
                    low, lnl_marg - CONDITIONAL_LOWER_MARGIN,
                    f'{label}: {CONDITIONAL_LOW_PERCENTILE:.0f}th-pct draw '
                    f'{low:.3f} craters below lnL_marg {lnl_marg:.3f} -- '
                    'conditional draws are biased.')

                self._plot_histogram(label, lnl_marg, lnl_full)

    def _plot_histogram(self, label, lnl_marg, lnl_full):
        """Histogram of plain lnlike at the draws, with lnL_marg marked."""
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        ax.hist(lnl_full, bins=15, color='C0', alpha=0.8)
        ax.axvline(lnl_marg, color='k', linestyle='--',
                   label=f'lnL_marg = {lnl_marg:.2f}')
        ax.set_xlabel('plain lnlike at conditional draw')
        ax.set_ylabel('count')
        ax.set_title(f'spec 6 conditional draws ({label})')
        ax.legend()
        fig.savefig(
            OUTPUT_DIR / f'marg_spec6_conditional_draws_{label}.png', dpi=80)
        plt.close(fig)


@skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class SelfFalsificationTestCase(_MarginalizedLensTestCase):
    """
    Proof-of-teeth: the central numerical gates go RED when fed a
    deliberately corrupted quantity.  A numerical suite that cannot be made
    to fail is indistinguishable from one that asserts nothing; these tests
    re-run the exact comparison logic of specs 1, 2 and 7 against a mutated
    input and REQUIRE an ``AssertionError``.

    None of these tests touch production code -- they corrupt only a local
    copy of an array, then confirm the suite's own tolerances reject it.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = _harness()
        cls.marg = cls.h.lensed_marg

    def _production_edge_amplification(self, candidate: dict) -> np.ndarray:
        """Production ``F`` at bin edges (mirrors the spec-7 helper)."""
        delays, k0, k1, _ = \
            self.marg._engine._amplification_coefficients(candidate)
        self.marg._engine._check_candidate_delays(delays)
        return self.marg._edge_amplification(delays, k0, k1)

    def test_reconstruction_gate_rejects_scaled_amplification(self):
        """
        Spec-7 gate fires when the exact oracle is multiplied by a factor
        just past the tolerance -- proves the ``rtol`` has teeth.
        """
        candidate = _intrinsic_lens_point(self.marg, MAIN_LENS)
        f_prod = self._production_edge_amplification(candidate)
        f_exact = _exact_edge_amplification(MAIN_LENS, self.h.fbin)

        # Corrupt the oracle by 10x the tolerance -- a genuine bug of this
        # size must not slip through the |F|**2 comparison.
        f_corrupt = f_exact * (1.0 + 10.0 * RECONSTRUCTION_RTOL)
        prod_sq = np.abs(f_prod) ** 2
        corrupt_sq = np.abs(f_corrupt) ** 2
        rel = np.max(np.abs(prod_sq - corrupt_sq)) / np.max(corrupt_sq)

        self.n_compared += 1
        with self.assertRaises(AssertionError):
            self.assertLess(rel, RECONSTRUCTION_RTOL)

    def test_identity_gate_rejects_shifted_timeseries(self):
        """
        Spec-1 identity gate fires when the unlensed ``dh_mptd`` is given a
        spurious per-detector time shift (a rolled timeseries) -- proves the
        ``1e-6`` identity would catch a per-image delay leak.
        """
        intrinsic = _reference_par_dic()
        lensed_point = {k: {**intrinsic, **UNLENSED_LIMIT_LENS}[k]
                        for k in self.marg.params}
        unlensed_point = {k: intrinsic[k]
                          for k in self.h.unlensed_marg.params}
        dh_l, _, _ = self.marg._get_dh_hh_timeshift(lensed_point)
        dh_u, _, _ = self.h.unlensed_marg._get_dh_hh_timeshift(unlensed_point)

        # A one-sample circular roll along the time axis: a delay bug of the
        # kind the identity gate is meant to catch.
        dh_u_shifted = np.roll(dh_u, 1, axis=-2)
        rel = np.max(np.abs(dh_l - dh_u_shifted)) \
            / np.max(np.abs(dh_l))

        self.n_compared += 1
        with self.assertRaises(AssertionError):
            self.assertLess(rel, UNLENSED_IDENTITY_RTOL)

    def test_oracle_guard_rejects_a_forbidden_reference(self):
        """
        The spec-2 AST oracle-independence walk itself has teeth: a source
        snippet that DOES reference the module under test is flagged.
        """
        tainted = (
            'def bad_oracle(lens_dic, fbin):\n'
            '    return LensedMarginalizedExtrinsicLikelihood._edge_'
            'amplification(lens_dic, fbin)\n')
        tree = ast.parse(tainted)
        referenced = {node.id for node in ast.walk(tree)
                      if isinstance(node, ast.Name)}
        referenced |= {node.attr for node in ast.walk(tree)
                       if isinstance(node, ast.Attribute)}
        hits = [name for name in FORBIDDEN_ORACLE_NAMES
                if name in referenced]
        self.n_compared += 1
        self.assertTrue(
            hits, 'the AST oracle-independence walk failed to flag a source '
            'that references the module under test -- the spec-2 guard is '
            'toothless.')


if __name__ == '__main__':
    main()
