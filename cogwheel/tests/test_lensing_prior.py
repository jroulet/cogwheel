"""
Tests for the microlensing sampling layer -- `lensing.prior` (WP1: the
reduced-coordinate lens subpriors and their folding), `lensing.posterior`
(WP2: `LensedIASPrior` composed with `LensedPosterior`'s named-refusal
net), and the WP3 fork/pickle-safe fiducial cache exercised implicitly by
the sampling smoke test.

WHAT THIS SUITE PINS (the sampling-boundary contract)
-----------------------------------------------------
The lens subpriors are cheap coordinate maps, but a sign error in a
Jacobian, a clamped boundary, an asymmetric fold, or a leaky refusal net
silently biases every downstream posterior.  Each gate here uses an
oracle chosen so the check is not circular:

* ROUND-TRIP (C1): ``transform`` then ``inverse_transform`` is the
  identity on every lens coordinate to ~machine precision.  A boundary
  clamp in ``Y(m)`` or the log map shows up only when the draw sits a
  nanometre inside an endpoint, so the sweep deliberately includes those
  neighbourhoods.

* JACOBIAN (C2): the analytic ``ln_jacobian_determinant`` is checked
  against a *central-difference* log-determinant of the SAME transform
  direction ``lnprior_and_transform`` consumes (``standard -> sampled``,
  differentiating ``inverse_transform``).  An independent numerical
  derivative catches a wrong sign or a dropped ``+ln m`` / ``+2 ln Y``
  term that an algebra-vs-algebra check would miss.

* DOMAIN SAFETY (C3): over a 10^4-point sweep of the WHOLE sampled box
  the emitted standard parameters stay strictly inside the engine's
  certified region -- positive parity ``1 - kappa > |gamma|``, the
  frequency ceiling ``w_max <= 450`` (margin below the certified 500),
  and the product ceiling ``w * sqrt(s) <= 58`` (margin below 60).  The
  ceilings are written out from the physical constant, not read back from
  the engine.

* FOLDING (C4): the astroid quadrant symmetry is verified at the ENGINE
  (``|F|`` invariant under ``u -> -u`` reflections, C4a) and at the
  POSTERIOR (unfold-sum consistency, C4b); C4c pins that NO
  constant-lens-phase fold is injected, because that degeneracy is
  22-mode-only and must not be assumed for IMRPhenomXPHM higher modes.

* MASS-SHEET (C5): the eliminated ``kappa`` direction is an exact
  degeneracy.  The twin is built from the closed-form mass-sheet identity
  ``F_{kappa,gamma}(w,y) = (1/lam) exp[i w(...)] F_{0,gamma/lam}(w,
  y/sqrt(lam))`` (professor/microlensing_chang_refsdal), verified against
  the engine at the amplification level, and the brute-force ``lnlike`` is
  shown invariant along it.

* REFUSAL NET (C6): an in-support proposal that trips a NAMED engine
  refusal returns exactly ``-inf`` through the posterior while the raw
  likelihood still raises; a mutation that narrows the ``except`` clause
  turns the ``-inf`` gate red, proving it non-vacuous.

* SMOKE (C7): a seeded prior draw evaluated through the posterior never
  raises, every value is finite-or-``-inf``, and the injected truth
  out-scores every random draw.

TOLERANCE PROVENANCE (why these numbers, not tighter/looser)
------------------------------------------------------------
* ``ROUNDTRIP_ATOL = ROUNDTRIP_RTOL = 1e-12`` -- the maps are ``exp``/
  ``log`` and a constant rescale; the only error is float64 rounding, so
  1e-12 sits a few orders above the ~1e-16 floor and fails on a real
  clamp, not on noise.
* ``JAC_TOL = 1e-5`` -- a central difference with relative step
  ``H_REL = 1e-7`` has truncation error ``O(H_REL**2) ~ 1e-14`` but a
  round-off floor ``~ eps / H_REL ~ 1e-9``; 1e-5 clears that floor with
  margin while still catching a sign flip (which is O(1)).
* ``W_MAX_CEILING = 450`` / ``WSQRTS_CEILING = 58`` -- deliberately inside
  the engine's certified ``500`` / ``60`` so the prior box has headroom;
  a too-wide mass or source range pushes a tail across these first.
* ``REFLECT_RTOL = 1e-9`` -- the reflection is an exact source-plane
  symmetry; the engine reproduces it to ~1e-15 here, so 1e-9 is a wide
  but still-meaningful band.
* ``FOLD_BRUTE_TOL = 1e-6`` vs ``FOLD_RB_TOL = 0.5`` -- the brute path is
  the tight gate (only float64 reflection round-off differs between the
  unfold machinery and a hand-built quadrant sum); the relative-binning
  path is loose because a nanometre ``u`` difference can snap two
  reflected images to different fiducial-lattice cells, a per-image error
  bounded near the ``4e-3``-currency but showing up as up to a few tenths
  of a nat after log-sum-exp.
* ``MASSSHEET_BRUTE_TOL = 0.01`` -- the twin's strain is bit-identical up
  to the ``1/lam`` amplitude that ``d_L/lam`` cancels, so the residual is
  pure float64 (measured ~1e-13); 0.01 is the spec gate.
* ``C7_MIN_FINITE_FRACTION = 0.05`` (hard, non-vacuity) and the aspirational
  ``0.90`` carried as an ``expectedFailure`` -- the prior box extends into
  the ``gamma ~ 0.5`` cancellation band, so ~59% of uniform draws are
  legitimately refused to ``-inf`` here; that is a prior-width shortfall
  the net handles correctly, documented rather than papered over.

Every numerical `TestCase` inherits `_LensSuiteTestCase`, whose
``tearDown`` FAILS if a test made zero comparisons (a silently-skipping
suite must not read green), and `SelfFalsificationTestCase` proves the
harness can go red.
"""
from __future__ import annotations

import functools
import itertools
import os
import pathlib
import types
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from cogwheel import data, waveform
from cogwheel.likelihood.reference_waveform_finder import ReferenceWaveformFinder
from cogwheel.gw_prior import IASPrior
from cogwheel.lensing import posterior as posterior_module
from cogwheel.lensing.likelihood import (LensedBinningError,
                                         LensedRelativeBinningLikelihood)
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)
from cogwheel.lensing.posterior import LensedPosterior
from cogwheel.lensing.prior import (
    LensedIASPrior, UniformLensMassPrior, UniformReducedShearPrior,
    UniformSourcePositionPrior,
    _source_scale, _LN_M_LENS_RANGE, _Y_SCALE, _Y_SCALE_CAP)
from cogwheel.lensing.waveform import (
    LensedWaveformGenerator, dimensionless_frequency, _EIGHT_PI_MTSUN_S)
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError

# ---------------------------------------------------------------------------
# Fixed configuration (all stochastic inputs are seeded from here).
# ---------------------------------------------------------------------------

#: Seed for every stochastic input (noise realization, parameter sweeps).
#: ``EventData.gaussian_noise`` and each sweep use
#: ``np.random.default_rng(SEED)``, so the fixtures are reproducible.
SEED = 20260718

#: Higher-mode approximant for the crown-config event, so C4c genuinely
#: exercises the ``|m| > 2`` regime where the constant-lens-phase fold
#: must NOT be applied.
APPROXIMANT_HM = 'IMRPhenomXPHM'

#: Uniform relative-binning bin width [Hz].  Chosen with `DELTA_T_MAX` so
#: the lens bin-delay criterion ``pi * DF_BIN * DELTA_T_MAX = 0.25`` clears
#: the 0.5 rad default tolerance at construction.
DF_BIN = 4.0

#: Largest relative image delay [s] the fixture's bins support.
DELTA_T_MAX = 0.02

# ---------------------------------------------------------------------------
# Tolerances (provenance in the module docstring).
# ---------------------------------------------------------------------------

#: Round-trip identity tolerance on every lens sampled coordinate (C1).
ROUNDTRIP_ATOL = 1e-12
ROUNDTRIP_RTOL = 1e-12

#: Relative finite-difference step for the numerical Jacobian (C2).
H_REL = 1e-7

#: Tolerance on ``|ln_J_analytic - ln_J_numeric|`` (C2).
JAC_TOL = 1e-5

#: Domain-safety ceilings (C3), inside the engine's certified 500 / 60.
W_MAX_CEILING = 450.0
WSQRTS_CEILING = 58.0

#: Upper edge [Hz] of the analysis band used to form ``w_max = xi * f_max``
#: (C3).  The crown event is 4 s at 2048 Hz Nyquist -> 1024 Hz band edge.
F_MAX_HZ = 1024.0

#: Relative tolerance on ``|F|`` invariance under source reflection (C4a).
REFLECT_RTOL = 1e-9

#: Fold unfold-sum agreement: tight brute-force gate / loose RB gate (C4b).
FOLD_BRUTE_TOL = 1e-6
FOLD_RB_TOL = 0.5

#: Mass-sheet lnlike invariance: brute-force gate / informational RB gate (C5).
MASSSHEET_BRUTE_TOL = 0.01
MASSSHEET_RB_TOL = 0.5

#: Mass-sheet MAGNITUDE identity (C5, convention-free anchor): the engine's
#: ``|F_{kappa,gamma}(w,y)|`` must equal ``(1/lam) |F_{0,gamma/lam}(w,
#: y/sqrt(lam))|`` exactly (the phase factors have unit modulus and the
#: per-config ``t_min`` re-reference is a pure phase), so the amplitudes are
#: tied by the professor closed form with no CBC time convention involved.
#: The engine reproduces it to ~1e-13 here; 1e-9 is a wide but meaningful band.
MASSSHEET_MAG_RTOL = 1e-9

#: (kappa, gamma, y1, y2, m_lens_msun) mass-sheet configs (C5).  Each has
#: positive parity in BOTH frames (``1 - kappa > |gamma|`` and
#: ``1 > |gamma/lam|``) and a moderate ``w`` window (moderate mass, small
#: source) so both the original and the twin brute-force lnlikes are finite
#: and certified.
MASSSHEET_CONFIGS = (
    (0.15, 0.15, 0.10, 0.05, 90.0),
    (0.25, 0.20, 0.08, 0.06, 110.0),
    (0.30, 0.18, 0.06, 0.04, 120.0),
)

#: ``w`` grid for the C5 magnitude identity (inside the certified band for
#: every `MASSSHEET_CONFIGS` entry at the fixture masses).
MASSSHEET_W_GRID = np.linspace(1.5, 15.0, 40)

#: Max in-support seeded draws scanned to collect NAMED-refusal proposals
#: for C6.  The prior box (gamma up to 0.45, lens mass up to 3500) overlaps
#: a NAMED wave refusal (post-8d dominated by the Schwinger ceiling: any
#: draw whose w grid crosses w = 60 refuses on its first such node), so
#: such proposals are dense; this budget finds several well before exhausting.
C6_SEARCH_BUDGET = 800

#: Number of named-refusal proposals C6 pins (posterior -> -inf, raw
#: path -> raise, and the mutation check).
C6_N_REFUSALS = 3

#: Reference near-unlensed lens config for the C7 peak-vs-truth sanity: the
#: LIGHTEST allowed lens (``m_lens = 11 Msun``, smallest ``w``), a small
#: shear, and a source held WELL OFF the caustic centre (``|y| ~ 0.9``, not
#: ``y = 0`` where the on-axis magnification diverges) make ``F -> 1``,
#: matching the UNLENSED injection.  This is the closest the forced-lens
#: model gets to the true (unlensed) peak, so no noise-fitting random draw
#: should out-score it -- verified at lnposterior ~ 260 vs a best draw ~ 18.
C7_REFERENCE_LENS = {'m_lens_msun': 11.0, 'z_lens': 0.0, 'y1': 0.9,
                     'y2': 0.0, 'gamma': 0.05, 'beta': 0.0, 'kappa': 0.0}

#: Nats by which the best random draw may exceed the near-truth reference
#: lnposterior (C7).  A random draw beating a near-truth fit by >50 nats
#: would signal a coordinate/range bug placing the peak away from the
#: injection.
C7_PEAK_MARGIN_NATS = 50.0

#: End-to-end smoke fractions (C7).  The hard non-vacuity floor; the
#: aspirational 0.90 is carried separately as an expected failure.
C7_N_DRAWS = 500
C7_MIN_FINITE_FRACTION = 0.05
C7_ASPIRATIONAL_FINITE_FRACTION = 0.90

#: Directory for diagnostic plots (created on demand); never shown.
OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

plt.switch_backend('Agg')

#: Lens sampled coordinates whose round-trip / support the suite pins.
_LENS_SAMPLED = ('ln_m_lens_msun', 'gamma', 'u1', 'u2')


def _reference_par_dic() -> dict:
    """
    Deterministic precessing CBC reference (crown config), keys per
    ``waveform.WaveformGenerator.params``.  Explicit, not random, so the
    fixture is reproducible; asserted against the generator schema in
    the harness so a drift fails loudly.
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
    Build (once) the crown-config event, prior, likelihood and posterior.

    Cached so the ~20 s XPHM injection and reference-waveform fit are paid
    a single time across every `TestCase` in the module.
    """
    par_dic_0 = _reference_par_dic()
    assert sorted(par_dic_0) == waveform.WaveformGenerator.params, (
        'reference par_dic keys drifted from WaveformGenerator.params; '
        'update _reference_par_dic')

    event_data = data.EventData.gaussian_noise(
        eventname='test_lensed_prior', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0., seed=SEED)
    event_data.inject_signal(par_dic_0, APPROXIMANT_HM)

    wfg = waveform.WaveformGenerator.from_event_data(event_data, APPROXIMANT_HM)

    band = event_data.frequencies[event_data.fslice]
    f_lo, f_hi = float(band[0]), float(band[-1])
    fbin = _uniform_fbin(f_lo, f_hi, DF_BIN)

    rwf = ReferenceWaveformFinder(event_data, wfg, par_dic_0, pn_phase_tol=0.05)
    prior = LensedIASPrior.from_reference_waveform_finder(rwf)

    like = LensedRelativeBinningLikelihood(
        event_data, wfg, par_dic_0, delta_t_max=DELTA_T_MAX, fbin=fbin)

    post = LensedPosterior(prior, like)

    assert set(prior.standard_params) == set(like.params), (
        'prior.standard_params and likelihood.params are incompatible')

    return types.SimpleNamespace(
        par_dic_0=par_dic_0, event_data=event_data, waveform_generator=wfg,
        prior=prior, likelihood=like, posterior=post, rwf=rwf,
        f_lo=f_lo, f_hi=f_hi)


def _random_sampled_point(prior, rng) -> np.ndarray:
    """Draw one uniform point inside the sampled hypercube of ``prior``."""
    return prior.cubemin + rng.uniform(0.0, 1.0, prior.cubemin.shape) \
        * prior.cubesize


def _standard_lens_dic(base_par_dic: dict, **lens) -> dict:
    """
    Merge a CBC ``base_par_dic`` with a full set of standard lens params.

    ``lens`` must supply ``m_lens_msun, z_lens, y1, y2, gamma, beta,
    kappa`` (the seven the likelihood consumes); the CBC keys come from
    ``base_par_dic``.
    """
    return {**base_par_dic, **lens}


class _LensSuiteTestCase(unittest.TestCase):
    """
    Shared base carrying the anti-vacuity guard.

    ``setUp`` zeroes a per-test comparison counter; every concrete test
    must increment ``self.n_compared`` for each oracle comparison it
    actually runs.  ``tearDown`` FAILS if the counter is still zero, so a
    test that silently skipped every case (an empty sweep, a fixture that
    never entered the loop) cannot read green.
    """

    #: Set True by tests that legitimately record no comparison (none do
    #: at present, but the hook keeps the guard explicit).
    allow_zero_comparisons = False

    def setUp(self):
        self.n_compared = 0

    def tearDown(self):
        if not self.allow_zero_comparisons and self.n_compared == 0:
            self.fail(
                f'{self._testMethodName} made zero comparisons -- the test '
                'is vacuous (empty sweep or skipped fixture). A green result '
                'here would be a false pass.')


class RoundTripIdentityTestCase(_LensSuiteTestCase):
    """C1 -- ``transform`` then ``inverse_transform`` is the identity."""

    @classmethod
    def setUpClass(cls):
        cls.prior = _harness().prior
        cls.lens_inds = [cls.prior.sampled_params.index(name)
                         for name in _LENS_SAMPLED]

    def _roundtrip_error(self, sampled_vec: np.ndarray) -> np.ndarray:
        """Return abs round-trip error on the lens coordinates only."""
        prior = self.prior
        standard = prior.transform(*sampled_vec)
        recovered = prior.inverse_transform(**standard)
        errors = np.array([
            abs(recovered[name] - sampled_vec[idx])
            for name, idx in zip(_LENS_SAMPLED, self.lens_inds)])
        return errors

    def test_interior_sweep_roundtrips_to_machine_precision(self):
        """~2000 interior draws round-trip on every lens coordinate."""
        prior = self.prior
        rng = np.random.default_rng(SEED)
        max_errors = []
        boundary_distances = []
        for _ in range(2000):
            sampled = _random_sampled_point(prior, rng)
            errors = self._roundtrip_error(sampled)
            lens_vals = sampled[self.lens_inds]
            lo = prior.cubemin[self.lens_inds]
            hi = lo + prior.cubesize[self.lens_inds]
            dist = float(np.min(np.minimum(lens_vals - lo, hi - lens_vals)))
            max_errors.append(float(errors.max()))
            boundary_distances.append(dist)
            for name, err, val in zip(_LENS_SAMPLED, errors,
                                      lens_vals):
                with self.subTest(coord=name):
                    self.assertLessEqual(
                        err,
                        ROUNDTRIP_ATOL + ROUNDTRIP_RTOL * abs(val))
                self.n_compared += 1
        self._plot_roundtrip(boundary_distances, max_errors)

    def test_boundary_neighborhoods_roundtrip(self):
        """Draws a nanometre inside each lens endpoint still round-trip."""
        prior = self.prior
        rng = np.random.default_rng(SEED + 1)
        lo = prior.cubemin
        hi = prior.cubemin + prior.cubesize
        eps = 1e-9
        for idx in self.lens_inds:
            for edge in (lo[idx] + eps, hi[idx] - eps):
                sampled = _random_sampled_point(prior, rng)
                sampled[idx] = edge
                errors = self._roundtrip_error(sampled)
                with self.subTest(coord=prior.sampled_params[idx], edge=edge):
                    self.assertLessEqual(
                        float(errors.max()),
                        ROUNDTRIP_ATOL + ROUNDTRIP_RTOL * abs(edge))
                self.n_compared += 1

    def _plot_roundtrip(self, distances, errors):
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        ax.scatter(distances, np.maximum(errors, 1e-18), s=4, alpha=0.4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axhline(ROUNDTRIP_ATOL, color='r', ls='--',
                   label=f'atol={ROUNDTRIP_ATOL:g}')
        ax.set_xlabel('distance to nearest lens-coordinate boundary')
        ax.set_ylabel('max abs round-trip error')
        ax.set_title('C1 round-trip error vs boundary distance')
        ax.legend()
        fig.savefig(OUTPUT_DIR / 'test_lensing_prior_c1_roundtrip.png',
                    dpi=80, bbox_inches='tight')
        plt.close(fig)


class JacobianConsistencyTestCase(_LensSuiteTestCase):
    """C2 -- analytic ``ln_jacobian_determinant`` vs central difference."""

    def test_mass_prior_jacobian_matches_finite_difference(self):
        """``-log m`` matches d(ln m)/dm from ``inverse_transform``."""
        rng = np.random.default_rng(SEED + 2)
        ln_lo, ln_hi = _LN_M_LENS_RANGE
        for _ in range(500):
            m_lens = float(np.exp(rng.uniform(ln_lo, ln_hi)))
            analytic = UniformLensMassPrior.ln_jacobian_determinant(m_lens)
            numeric = self._numeric_logdet_mass(m_lens)
            with self.subTest(m_lens_msun=m_lens):
                self.assertLess(abs(analytic - numeric), JAC_TOL)
            self.n_compared += 1

    def test_source_prior_jacobian_matches_finite_difference(self):
        """``-2 log Y(m)`` matches log|det d(u)/d(y)| centrally."""
        rng = np.random.default_rng(SEED + 3)
        ln_lo, ln_hi = _LN_M_LENS_RANGE
        for _ in range(500):
            m_lens = float(np.exp(rng.uniform(ln_lo, ln_hi)))
            scale = _source_scale(m_lens)
            y1 = float(rng.uniform(-1.0, 1.0)) * scale
            y2 = float(rng.uniform(-1.0, 1.0)) * scale
            analytic = UniformSourcePositionPrior.ln_jacobian_determinant(
                y1, y2, m_lens)
            numeric = self._numeric_logdet_source(y1, y2, m_lens)
            with self.subTest(m_lens_msun=m_lens):
                self.assertLess(abs(analytic - numeric), JAC_TOL)
            self.n_compared += 1

    @staticmethod
    def _numeric_logdet_mass(m_lens: float) -> float:
        """Central-difference ``log|d(ln_m_lens_msun)/d(m_lens_msun)|``."""
        step = H_REL * m_lens
        plus = UniformLensMassPrior.inverse_transform(
            m_lens + step)['ln_m_lens_msun']
        minus = UniformLensMassPrior.inverse_transform(
            m_lens - step)['ln_m_lens_msun']
        derivative = (plus - minus) / (2.0 * step)
        return float(np.log(abs(derivative)))

    @staticmethod
    def _numeric_logdet_source(y1: float, y2: float, m_lens: float) -> float:
        """Central-difference ``log|det d(u1,u2)/d(y1,y2)|``."""
        jac = np.empty((2, 2))
        base = (y1, y2)
        for col, _ in enumerate(base):
            step = H_REL * max(abs(base[col]), 1.0)
            hi_args = list(base)
            lo_args = list(base)
            hi_args[col] += step
            lo_args[col] -= step
            hi = UniformSourcePositionPrior.inverse_transform(
                *hi_args, m_lens)
            lo = UniformSourcePositionPrior.inverse_transform(
                *lo_args, m_lens)
            jac[0, col] = (hi['u1'] - lo['u1']) / (2.0 * step)
            jac[1, col] = (hi['u2'] - lo['u2']) / (2.0 * step)
        sign, logdet = np.linalg.slogdet(jac)
        assert sign != 0.0, 'degenerate numerical Jacobian'
        return float(logdet)


class BothParityDomainSafetyTestCase(_LensSuiteTestCase):
    """C3 -- the whole sampled box maps into the certified-or-named-refuse
    engine domain.

    The standard lens params are recomputed here from the SAME production
    maps the subpriors use (``exp`` for the mass, `_source_scale` for the
    source box, the REAL ``UniformReducedShearPrior.range_dic`` for the
    shear) rather than routed through ``CombinedPrior.transform``, so
    the sweep is a fast vectorized 10^4-point scan; correctness of those
    maps themselves is pinned by C1/C2.  The ceilings are written out from
    the physical constant ``_EIGHT_PI_MTSUN_S`` (via
    `dimensionless_frequency`), never read back from the engine.

    Since Build 7b the gamma range spans BOTH parities (positive parity
    ``gamma < 1`` and macro saddles ``gamma > 1``), so the old
    positive-parity assertion is replaced by the both-parity domain
    contract: every draw is either inside a parity interior (evaluable,
    certified-or-named-refuse downstream) or on the measure-zero
    ``gamma == 1`` boundary (named refusal) -- never in the over-critical
    Type III region, which the ``kappa = 0`` sampled space cannot reach.
    """

    def test_full_box_stays_in_certified_domain(self):
        """Both-parity gamma box, w_max <= 450, w*sqrt(s) <= 58, 10^4 draws."""
        rng = np.random.default_rng(SEED + 4)
        n_points = 10_000
        ln_lo, ln_hi = _LN_M_LENS_RANGE
        ln_m = rng.uniform(ln_lo, ln_hi, n_points)
        gamma_lo, gamma_hi = UniformReducedShearPrior.range_dic['gamma']
        gamma = rng.uniform(gamma_lo, gamma_hi, n_points)
        u1 = rng.uniform(-1.0, 1.0, n_points)
        u2 = rng.uniform(-1.0, 1.0, n_points)

        m_lens = np.exp(ln_m)
        scale = np.minimum(_Y_SCALE / m_lens, _Y_SCALE_CAP)
        y1 = u1 * scale
        y2 = u2 * scale
        s = y1 ** 2 + y2 ** 2
        # z_lens is fixed to 0 by FixedLensGeometryPrior.  Production
        # `dimensionless_frequency` only accepts scalar masses, so the
        # 10^4-point scan is vectorized straight from the physical constant
        # ``_EIGHT_PI_MTSUN_S``; a scalar cross-check ties that expression
        # back to the shipped function so we are not inventing the map.
        w_max = _EIGHT_PI_MTSUN_S * m_lens * (1.0 + 0.0) * F_MAX_HZ
        probe_m = float(m_lens[0])
        np.testing.assert_allclose(
            w_max[0], dimensionless_frequency(F_MAX_HZ, probe_m, 0.0),
            rtol=1e-12, atol=0.0,
            err_msg='vectorized w_max diverged from dimensionless_frequency')

        # (a) both-parity domain: with kappa = 0 every draw is a parity
        # INTERIOR (gamma != 1: positive parity below, macro saddle
        # above -- both evaluable, certified-or-named-refuse downstream)
        # and never over-critical (1 - kappa = 1 > 0 always).  The
        # measure-zero gamma == 1 boundary is a named det-A = 0 refusal;
        # a uniform draw hits it with probability 0, asserted exactly.
        with self.subTest(check='both_parity_domain'):
            self.assertTrue(np.all(gamma != 1.0),
                            'a draw landed exactly on the det-A = 0 '
                            'parity boundary')
            self.assertTrue(np.any(gamma < 1.0) and np.any(gamma > 1.0),
                            'the sampled box no longer spans both '
                            'parities -- the range_dic drifted')
        # (b) frequency ceiling.
        with self.subTest(check='w_max_ceiling'):
            self.assertLessEqual(
                float(w_max.max()), W_MAX_CEILING,
                f'w_max tail {w_max.max():.2f} exceeds {W_MAX_CEILING}')
        # (c) product ceiling.
        product = w_max * np.sqrt(s)
        with self.subTest(check='w_sqrt_s_ceiling'):
            self.assertLessEqual(
                float(product.max()), WSQRTS_CEILING,
                f'w*sqrt(s) tail {product.max():.2f} exceeds {WSQRTS_CEILING}')

        self.n_compared += n_points
        self._plot_domain(w_max, product)

    def _plot_domain(self, w_max, product):
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].hist(w_max, bins=60)
        axes[0].axvline(W_MAX_CEILING, color='r', ls='--')
        axes[0].set_xlabel('w_max = xi * f_max')
        axes[0].set_title('C3 w_max')
        axes[1].hist(product, bins=60)
        axes[1].axvline(WSQRTS_CEILING, color='r', ls='--')
        axes[1].set_xlabel('w_max * sqrt(s)')
        axes[1].set_title('C3 product ceiling')
        fig.savefig(OUTPUT_DIR / 'test_lensing_prior_c3_domain.png',
                    dpi=80, bbox_inches='tight')
        plt.close(fig)


class ReflectionSymmetryTestCase(_LensSuiteTestCase):
    """C4a -- ``|F|`` is invariant under shear-frame source reflection.

    The reflected inputs ``(+-y1, +-y2)`` are built from RAW coordinates
    and fed to a FRESH `ChangRefsdalChannels` per quadrant (no shared
    label state, no channels-derived expected value), so the oracle is the
    astroid source-plane symmetry itself, not a value the pipeline already
    computed (avoids the F002 oracle tautology).
    """

    #: (gamma, y1, y2) configs kept inside the certified domain over the
    #: ``w`` window below (2- and 4-image, near-fold), so no quadrant
    #: raises and the comparison is genuine.
    _CONFIGS = ((0.20, 0.10, 0.05), (0.25, 0.12, 0.08),
                (0.15, 0.30, 0.20), (0.30, 0.10, 0.06))

    _W_GRID = np.linspace(2.0, 25.0, 40)

    def test_amplification_and_delays_invariant_under_reflection(self):
        """``|F|`` and the sorted real delays are reflection-invariant."""
        for gamma, y1, y2 in self._CONFIGS:
            partitions = {}
            for sign_x, sign_y in itertools.product((+1, -1), (+1, -1)):
                channels = ChangRefsdalChannels(self._W_GRID)
                partitions[(sign_x, sign_y)] = channels.evaluate(
                    gamma=gamma, y=(sign_x * y1, sign_y * y2),
                    beta=0.0, kappa=0.0)
            base = partitions[(+1, +1)]
            base_abs = np.abs(base.exact_total)
            base_delays = np.sort(base.delays[base.real_mask])
            for key in ((-1, +1), (+1, -1), (-1, -1)):
                part = partitions[key]
                rel = (np.max(np.abs(np.abs(part.exact_total) - base_abs))
                       / np.max(base_abs))
                with self.subTest(config=(gamma, y1, y2), reflection=key,
                                  quantity='abs_F'):
                    self.assertLess(rel, REFLECT_RTOL)
                delays = np.sort(part.delays[part.real_mask])
                with self.subTest(config=(gamma, y1, y2), reflection=key,
                                  quantity='delays'):
                    self.assertEqual(delays.shape, base_delays.shape)
                    np.testing.assert_allclose(delays, base_delays,
                                               rtol=1e-9, atol=1e-12)
                self.n_compared += 1


@unittest.skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class FoldUnfoldConsistencyTestCase(_LensSuiteTestCase):
    """C4b -- the folded posterior equals a hand-built quadrant sum.

    A folded point is unfolded two ways: (i) by the production folding
    machinery (`prior.unfold`), restricted to the four rows that differ
    only in ``(u1, u2)``; (ii) by reflecting ``(u1, u2) -> (-u1, -u2)``
    from RAW coordinates.  ``logsumexp`` over the posterior at each set is
    compared -- tight for the brute-force likelihood (only float64
    reflection round-off differs), loose for relative binning (a
    nanometre ``u`` shift can snap a reflected image to a different
    fiducial-lattice cell).
    """

    @classmethod
    def setUpClass(cls):
        harness = _harness()
        cls.prior = harness.prior
        cls.posterior = harness.posterior
        cls.likelihood = harness.likelihood
        cls.par_dic_0 = harness.par_dic_0
        cls.u1_idx = cls.prior.sampled_params.index('u1')
        cls.u2_idx = cls.prior.sampled_params.index('u2')
        cls.other_folded = [idx for idx in cls.prior._folded_inds
                            if idx not in (cls.u1_idx, cls.u2_idx)]

    def _benign_folded_point(self, m_lens, gamma, y1, y2) -> np.ndarray:
        """A folded sampled vector at a benign (finite-lnL) lens config."""
        standard = _standard_lens_dic(
            self.par_dic_0, m_lens_msun=m_lens, z_lens=0.0, y1=y1, y2=y2,
            gamma=gamma, beta=0.0, kappa=0.0)
        sampled = self.prior.inverse_transform(**standard)
        return self.prior.fold(**sampled)

    def _machinery_quadrant(self, folded_point) -> np.ndarray:
        """Four `prior.unfold` rows that vary only in ``(u1, u2)``."""
        rows = self.prior.unfold(folded_point)
        keep = np.all(
            np.isclose(rows[:, self.other_folded],
                       folded_point[self.other_folded], atol=1e-12), axis=1)
        quadrant = rows[keep]
        self.assertEqual(quadrant.shape[0], 4,
                         'expected exactly four u1/u2 unfoldings')
        return quadrant

    def _hand_quadrant(self, folded_point) -> np.ndarray:
        """Four quadrant images built by raw ``u -> -u`` reflection."""
        u1_f = folded_point[self.u1_idx]
        u2_f = folded_point[self.u2_idx]
        images = []
        for u1_val, u2_val in itertools.product((u1_f, -u1_f), (u2_f, -u2_f)):
            row = folded_point.copy()
            row[self.u1_idx] = u1_val
            row[self.u2_idx] = u2_val
            images.append(row)
        return np.array(images)

    def _brute_lnposterior(self, sampled_vec) -> float:
        """lnprior + brute-force lnlike (independent of the RB path)."""
        lnprior, standard = self.prior.lnprior_and_transform(*sampled_vec)
        if np.isneginf(lnprior):
            return -np.inf
        return lnprior + self.likelihood.lnlike_bruteforce(standard)

    def test_unfold_reflection_matches_raw_quadrant(self):
        """Machinery ``u`` unfoldings are exactly the raw ``+-u`` set."""
        point = self._benign_folded_point(90.0, 0.20, 0.10, 0.05)
        machinery = self._machinery_quadrant(point)
        u1_f = point[self.u1_idx]
        u2_f = point[self.u2_idx]
        machinery_u1 = np.sort(machinery[:, self.u1_idx])
        machinery_u2 = np.sort(machinery[:, self.u2_idx])
        np.testing.assert_allclose(
            machinery_u1, np.sort([u1_f, -u1_f, u1_f, -u1_f]), atol=1e-12)
        np.testing.assert_allclose(
            machinery_u2, np.sort([u2_f, u2_f, -u2_f, -u2_f]), atol=1e-12)
        self.n_compared += 1

    def test_brute_force_unfold_sum_consistency(self):
        """Brute-force folded lnposterior matches the raw quadrant sum."""
        configs = ((90.0, 0.20, 0.10, 0.05), (120.0, 0.25, 0.08, 0.04))
        for m_lens, gamma, y1, y2 in configs:
            point = self._benign_folded_point(m_lens, gamma, y1, y2)
            machinery = self._machinery_quadrant(point)
            hand = self._hand_quadrant(point)
            lse_machinery = logsumexp(
                [self._brute_lnposterior(row) for row in machinery])
            lse_hand = logsumexp(
                [self._brute_lnposterior(row) for row in hand])
            with self.subTest(config=(m_lens, gamma, y1, y2)):
                self.assertLess(abs(lse_machinery - lse_hand), FOLD_BRUTE_TOL)
            self.n_compared += 1

    def test_relative_binning_unfold_sum_consistency(self):
        """RB folded lnposterior matches within the looser snapping band."""
        configs = ((90.0, 0.20, 0.10, 0.05), (120.0, 0.25, 0.08, 0.04))
        for m_lens, gamma, y1, y2 in configs:
            point = self._benign_folded_point(m_lens, gamma, y1, y2)
            machinery = self._machinery_quadrant(point)
            hand = self._hand_quadrant(point)
            lse_machinery = logsumexp(
                [self.posterior.lnposterior(*row) for row in machinery])
            lse_hand = logsumexp(
                [self.posterior.lnposterior(*row) for row in hand])
            with self.subTest(config=(m_lens, gamma, y1, y2)):
                self.assertLess(abs(lse_machinery - lse_hand), FOLD_RB_TOL)
            self.n_compared += 1


class PhaseFoldAbsenceTestCase(_LensSuiteTestCase):
    """C4c -- the lens layer adds only the ``u1/u2`` reflection fold.

    No constant-lens-phase ~ orbital-phase fold is injected: that
    degeneracy is 22-mode-only (professor/priors_and_coordinates), and the
    crown event here uses IMRPhenomXPHM higher modes, where it must not be
    assumed.  The folding is a structural property of ``prior_classes``,
    identical for a 22-only or a higher-mode generator; the contrast is
    made against the base `IASPrior`.
    """

    @classmethod
    def setUpClass(cls):
        harness = _harness()
        cls.lensed = harness.prior
        cls.ias = IASPrior.from_reference_waveform_finder(harness.rwf)

    def test_lens_layer_adds_exactly_u1_u2_reflection(self):
        """Reflected set gains only ``u1, u2`` over the base IAS prior."""
        added_reflected = (set(self.lensed.folded_reflected_params)
                           - set(self.ias.folded_reflected_params))
        self.assertEqual(added_reflected, {'u1', 'u2'})
        self.n_compared += 1

    def test_no_phase_fold_injected_by_lens_layer(self):
        """Shifted (phase-like) folds are unchanged by the lens layer."""
        self.assertEqual(set(self.lensed.folded_shifted_params),
                         set(self.ias.folded_shifted_params))
        self.n_compared += 1

    def test_no_lens_phase_parameter_is_folded(self):
        """No folded parameter is a constant-lens-phase coordinate."""
        for name in self.lensed.folded_params:
            with self.subTest(param=name):
                self.assertNotIn('lens_phase', name)
                # The only lens folds are the source-position reflections.
                if name not in self.ias.folded_params:
                    self.assertIn(name, ('u1', 'u2'))
            self.n_compared += 1


def _dimensionless_xi(m_lens_msun: float, z_lens: float) -> float:
    """Return ``xi`` [s] such that ``w = xi * f`` (``= w`` at ``f = 1 Hz``)."""
    return float(dimensionless_frequency(1.0, m_lens_msun, z_lens))


def _massheet_dt_ms(par_dic: dict) -> float:
    """
    Mass-sheet constant time shift ``dt_ms`` [s] (professor closed form).

    ``dt_ms = xi/(4*pi) * (-kappa * s' + ln(lam))`` with ``lam = 1 - kappa``
    and ``s' = |y|^2 / lam`` (from the exact identity phase
    ``w(ln(lam)/2 - kappa|y|^2/(2 lam))`` and ``w = xi * f``).
    """
    kappa = par_dic['kappa']
    lam = 1.0 - kappa
    s_prime = (par_dic['y1'] ** 2 + par_dic['y2'] ** 2) / lam
    xi = _dimensionless_xi(par_dic['m_lens_msun'], par_dic['z_lens'])
    return xi / (4.0 * np.pi) * (-kappa * s_prime + np.log(lam))


#: Two-point probe grid for reading a config's minimum image delay.  The
#: minimum Fermat delay ``t_min`` is a geometry scalar independent of ``w``,
#: but `ChangRefsdalChannels` requires a >=2-point strictly-increasing
#: positive ``w`` grid, so any such grid returns the same ``t_min``.
_MASSSHEET_PROBE_W = np.array([1.0, 2.0])


def _massheet_min_delay(gamma: float, y1: float, y2: float,
                        kappa: float) -> float:
    """Minimum absolute image Fermat delay ``t_min`` (dimensionless).

    Read straight off `ChangRefsdalPartition.t_min`, the carrier the engine
    subtracts from its ``exact_total`` (``exp(-1j w t_min)``); it is a
    config-level scalar, so the probe ``w`` grid is irrelevant.
    """
    part = ChangRefsdalChannels(_MASSSHEET_PROBE_W).evaluate(
        gamma=gamma, y=(y1, y2), beta=0.0, kappa=kappa)
    return float(part.t_min)


def _massheet_twin(par_dic: dict) -> dict:
    """
    Return the ``kappa -> 0`` mass-sheet twin of a lensed ``par_dic``.

    Applies the professor mapping ``lam = 1 - kappa``, ``gamma_B =
    gamma/lam``, ``y_B = y/sqrt(lam)``, ``d_L_B = d_L*lam`` and a constant
    time shift.  The shift has TWO pieces, both fixed by the waveform's
    frequency-domain convention ``exp(-2j*pi*f*t_geocenter)``
    (``waveform.WaveformGenerator._get_shifts``):

    * ``dt_ms`` -- the professor closed-form phase ``w(ln(lam)/2 -
      kappa|y|^2/(2 lam))`` re-expressed as a time (``_massheet_dt_ms``);
    * ``dt_ref`` -- a bookkeeping correction for the engine's
      minimum-image carrier.  ``ChangRefsdalPartition.exact_total`` is
      referenced to each config's OWN ``t_min`` (``exp(-1j w t_min)``,
      ``channels._exact_total``), so the twin's different ``t_min`` shifts
      its carrier by ``dt_ref = xi*(t_min_B - t_min_A)/(2 pi)``.

    The strain must cancel the net amplification phase, giving ``t_B - t_c =
    -(dt_ms + dt_ref)``.  (Empirically ``dt_ref = -dt_ms`` to machine
    precision -- the min-image referencing exactly absorbs the mass-sheet
    constant phase -- but the correction is computed from the engine's
    reported ``t_min`` rather than assuming that identity.)
    """
    kappa = par_dic['kappa']
    lam = 1.0 - kappa
    gamma_b = par_dic['gamma'] / lam
    y1_b = par_dic['y1'] / np.sqrt(lam)
    y2_b = par_dic['y2'] / np.sqrt(lam)
    dt_ms = _massheet_dt_ms(par_dic)
    xi = _dimensionless_xi(par_dic['m_lens_msun'], par_dic['z_lens'])
    t_min_a = _massheet_min_delay(par_dic['gamma'], par_dic['y1'],
                                  par_dic['y2'], kappa)
    t_min_b = _massheet_min_delay(gamma_b, y1_b, y2_b, 0.0)
    dt_ref = xi * (t_min_b - t_min_a) / (2.0 * np.pi)
    twin = dict(par_dic)
    twin['kappa'] = 0.0
    twin['gamma'] = gamma_b
    twin['y1'] = y1_b
    twin['y2'] = y2_b
    twin['d_luminosity'] = par_dic['d_luminosity'] * lam
    twin['t_geocenter'] = par_dic['t_geocenter'] - dt_ms - dt_ref
    return twin


@unittest.skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — exact '
    'path ~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')
class MassSheetDegeneracyTestCase(_LensSuiteTestCase):
    """C5 -- the eliminated ``kappa`` direction is an exact degeneracy.

    Two independent gates.  (i) A CONVENTION-FREE magnitude identity: the
    engine's ``|F_{kappa,gamma}(w,y)|`` equals ``(1/lam)
    |F_{0,gamma/lam}(w, y/sqrt(lam))|`` (professor closed form, evaluated
    from RAW coordinates through a fresh `ChangRefsdalChannels` per config,
    so the oracle is the analytic mass-sheet rescaling, not a value the
    pipeline reuses).  This is the amplitude content ``F`` carries on TOP of
    ``1/d_luminosity``: ``d_luminosity`` is the PHYSICAL distance and the
    apparent distance an unlensed amplitude fit would report is
    ``d_luminosity / sqrt(mu_macro)``.  (ii) The spec's
    brute-force ``lnlike`` invariance along the twin built by
    `_massheet_twin`, tying the amplitude to ``d_L`` and the residual phase
    to a constant ``t_c`` shift.
    """

    @classmethod
    def setUpClass(cls):
        harness = _harness()
        cls.likelihood = harness.likelihood
        cls.par_dic_0 = harness.par_dic_0

    def _lensed_par_dic(self, kappa, gamma, y1, y2, m_lens) -> dict:
        """Merge the CBC truth with a full standard lens config."""
        return _standard_lens_dic(
            self.par_dic_0, m_lens_msun=m_lens, z_lens=0.0, y1=y1, y2=y2,
            gamma=gamma, beta=0.0, kappa=kappa)

    def test_amplification_magnitude_identity(self):
        """``|F_kappa(w,y)| == (1/lam) |F_0(w, y/sqrt(lam))|`` on the grid."""
        for kappa, gamma, y1, y2, _m in MASSSHEET_CONFIGS:
            lam = 1.0 - kappa
            original = ChangRefsdalChannels(MASSSHEET_W_GRID).evaluate(
                gamma=gamma, y=(y1, y2), beta=0.0, kappa=kappa)
            twin = ChangRefsdalChannels(MASSSHEET_W_GRID).evaluate(
                gamma=gamma / lam,
                y=(y1 / np.sqrt(lam), y2 / np.sqrt(lam)),
                beta=0.0, kappa=0.0)
            abs_original = np.abs(original.exact_total)
            abs_predicted = np.abs(twin.exact_total) / lam
            with self.subTest(config=(kappa, gamma, y1, y2)):
                np.testing.assert_allclose(
                    abs_original, abs_predicted,
                    rtol=MASSSHEET_MAG_RTOL, atol=0.0,
                    err_msg='mass-sheet magnitude identity violated')
            self.n_compared += 1

    def test_bruteforce_lnlike_invariant_along_kappa(self):
        """Brute-force ``lnlike`` is invariant original -> mass-sheet twin."""
        deltas = []
        kappas = []
        for kappa, gamma, y1, y2, m_lens in MASSSHEET_CONFIGS:
            original = self._lensed_par_dic(kappa, gamma, y1, y2, m_lens)
            twin = _massheet_twin(original)
            ln_a = self.likelihood.lnlike_bruteforce(original)
            ln_b = self.likelihood.lnlike_bruteforce(twin)
            deltas.append(ln_a - ln_b)
            kappas.append(kappa)
            with self.subTest(config=(kappa, gamma, y1, y2), path='brute'):
                self.assertLess(abs(ln_a - ln_b), MASSSHEET_BRUTE_TOL)
            self.n_compared += 1
        self._plot_masssheet(kappas, deltas)

    def test_relative_binning_lnlike_invariant_informational(self):
        """RB ``lnlike`` tracks the twin within the looser informational band."""
        for kappa, gamma, y1, y2, m_lens in MASSSHEET_CONFIGS:
            original = self._lensed_par_dic(kappa, gamma, y1, y2, m_lens)
            twin = _massheet_twin(original)
            ln_a = self.likelihood.lnlike(original)
            ln_b = self.likelihood.lnlike(twin)
            with self.subTest(config=(kappa, gamma, y1, y2), path='rb'):
                self.assertLess(abs(ln_a - ln_b), MASSSHEET_RB_TOL)
            self.n_compared += 1

    def _plot_masssheet(self, kappas, deltas):
        OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(kappas, deltas, 'o-')
        ax.axhline(0.0, color='k', lw=0.5)
        ax.axhspan(-MASSSHEET_BRUTE_TOL, MASSSHEET_BRUTE_TOL,
                   color='g', alpha=0.15, label=f'+-{MASSSHEET_BRUTE_TOL} nat')
        ax.set_xlabel('kappa (eliminated direction)')
        ax.set_ylabel('lnlike(original) - lnlike(twin)  [nat]')
        ax.set_title('C5 mass-sheet lnlike invariance')
        ax.legend()
        fig.savefig(OUTPUT_DIR / 'test_lensing_prior_c5_masssheet.png',
                    dpi=80, bbox_inches='tight')
        plt.close(fig)


#: Gated at CLASS level, not per method: the cost is in `setUpClass`, whose
#: `_collect_cancellation_proposals` scan evaluates configs above the
#: Schwinger double-double ceiling (``w > 60``) at ~85-120 s per engine call
#: on the mpmath path instead of ~0.2 s (F061). A method-level skip does NOT
#: prevent `setUpClass` from running, so every test in the class would still
#: pay the scan and time out -- measured: 5 errors at setup, 906 s.
@unittest.skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — the '
    'setUpClass refusal scan evaluates above the Schwinger ceiling '
    '(w > 60), ~85-120 s per engine call on the mpmath path (F061)')
class RefusalNetTestCase(_LensSuiteTestCase):
    """C6 -- the posterior maps NAMED engine refusals to exactly ``-inf``.

    In-support proposals (``gamma <= 0.45``, ``kappa = 0`` so positive
    parity always holds) can still trip `SchwingerCertificationError` when
    source curvature drives the effective shear into the uncertifiable
    band.  The suite collects real such proposals by a seeded
    scan of the sampled box, then pins that (a) the posterior returns the
    ``(-inf, dict, None)`` triple with NO exception escaping, while (b) the
    raw likelihood on the SAME standard config still raises the named
    refusal.  A MUTATION that removes the refusal the box actually produces
    from the net's ``except`` clause (by patching the module global the
    clause resolves) must turn (a) red -- the standing proof the ``-inf``
    gate is not vacuous.
    """

    @classmethod
    def setUpClass(cls):
        harness = _harness()
        cls.prior = harness.prior
        cls.likelihood = harness.likelihood
        cls.posterior = harness.posterior
        cls.refusals = cls._collect_cancellation_proposals()

    @classmethod
    def _collect_cancellation_proposals(cls):
        """Seeded in-support draws that trip a NAMED wave refusal.

        Returns ``(sampled_vec, standard_par_dic, exc_class)`` triples
        whose raw likelihood raises a named engine refusal.  The
        reachable in-support vocabulary is `SchwingerCertificationError`
        (dominant -- any draw whose ``w`` grid crosses the ceiling
        refuses on its first such node) and `LensedBinningError`
        (wide-delay saddle images).  This class pins the posterior net
        for whatever named refusals the box actually reaches; the
        remaining branches keep their falsification through the
        injection tests below (reachability not required).
        ``LensDomainError`` draws (the measure-zero boundary) are
        skipped.
        """
        rng = np.random.default_rng(SEED + 6)
        found = []
        for _ in range(C6_SEARCH_BUDGET):
            sampled = _random_sampled_point(cls.prior, rng)
            standard = cls.prior.transform(*sampled)
            try:
                cls.likelihood.lnlike(standard)
            except (SchwingerCertificationError,
                    LensedBinningError) as exc:
                found.append((sampled, standard, type(exc)))
                if len(found) >= C6_N_REFUSALS:
                    break
            except LensDomainError:
                pass
        return found

    def test_named_refusals_were_found_in_support(self):
        """The seeded scan actually located in-support named refusals."""
        self.assertGreaterEqual(
            len(self.refusals), C6_N_REFUSALS,
            f'only {len(self.refusals)} named-refusal proposals found in '
            f'{C6_SEARCH_BUDGET} in-support draws; C6 cannot proceed')
        self.n_compared += 1

    def test_posterior_maps_refusal_to_neginf(self):
        """The posterior returns ``(-inf, dict, None)`` and never raises."""
        for sampled, _standard, _exc in self.refusals:
            result = self.posterior.lnposterior_pardic_and_metadata(*sampled)
            with self.subTest(sampled=tuple(np.round(sampled, 4))):
                self.assertEqual(len(result), 3)
                self.assertTrue(np.isneginf(result[0]))
                self.assertIsInstance(result[1], dict)
                self.assertIsNone(result[2])
            self.n_compared += 1

    def test_raw_likelihood_still_raises_named_refusal(self):
        """The raw likelihood on the same config raises the named refusal."""
        for _sampled, standard, exc_class in self.refusals:
            with self.subTest(config='raw'):
                with self.assertRaises(exc_class):
                    self.likelihood.lnlike(standard)
            self.n_compared += 1

    def test_mutation_narrowing_except_turns_neginf_red(self):
        """Dropping the collected refusal class from the net re-raises."""
        class _UnrelatedRefusal(Exception):
            """Stand-in that does NOT match the real refusal class."""

        sampled, _standard, exc_class = self.refusals[0]
        # Control: the unmutated net swallows the refusal.
        self.assertTrue(np.isneginf(
            self.posterior.lnposterior_pardic_and_metadata(*sampled)[0]))
        # Mutation: patch the module global the ``except`` tuple resolves
        # (the CLASS of the refusal this config actually raises) so it is
        # no longer caught -- the falsification path must stay reachable
        # for the vocabulary the box actually produces (F010).
        with mock.patch.object(posterior_module, exc_class.__name__,
                               _UnrelatedRefusal):
            with self.assertRaises(exc_class):
                self.posterior.lnposterior_pardic_and_metadata(*sampled)
        self.n_compared += 1

    def test_domain_error_branch_is_also_caught(self):
        """A `LensDomainError` from the engine is likewise mapped to -inf.

        ``LensDomainError`` is unreachable from the ``kappa = 0`` sampled
        box, so it is injected at the likelihood boundary (a synthetic raise)
        to exercise the OTHER named branch of the net's ``except`` clause.
        """
        sampled, _standard, _exc = self.refusals[0]
        with mock.patch.object(
                self.likelihood, 'lnlike_and_metadata',
                side_effect=LensDomainError('injected macro saddle')):
            result = self.posterior.lnposterior_pardic_and_metadata(*sampled)
        self.assertTrue(np.isneginf(result[0]))
        self.assertIsNone(result[2])
        self.n_compared += 1


@unittest.skipUnless(
    os.environ.get('COGWHEEL_BRUTE_ACCURACY'),
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 — each C7 '
    'draw rebuilds a full exact envelope (seconds/draw post-8d)')
class SamplingSmokeTestCase(_LensSuiteTestCase):
    """C7 -- a seeded prior draw runs clean through the posterior.

    ``C7_N_DRAWS`` uniform draws over the sampled box are evaluated through
    `LensedPosterior.lnposterior`: none may raise (the refusal net catches
    every named engine refusal), every value must be finite or exactly
    ``-inf``, the finite fraction must clear the hard non-vacuity floor, and
    the best random draw must not beat a NEAR-TRUTH reference by more than
    `C7_PEAK_MARGIN_NATS` (the peak sits near the injection).  The
    aspirational 0.90 finite fraction the spec anticipates is carried as an
    ``expectedFailure``: the prior box deliberately extends into the
    ``gamma ~ 0.5`` cancellation band, so a large minority of uniform draws
    are legitimately refused -- a documented prior-width property, not a bug.
    """

    @classmethod
    def setUpClass(cls):
        harness = _harness()
        cls.prior = harness.prior
        cls.posterior = harness.posterior
        cls.par_dic_0 = harness.par_dic_0

        rng = np.random.default_rng(SEED + 7)
        cls.lnposts = np.array([
            cls.posterior.lnposterior(
                *_random_sampled_point(cls.prior, rng))
            for _ in range(C7_N_DRAWS)])
        cls.finite_mask = np.isfinite(cls.lnposts)
        cls.finite_fraction = float(np.mean(cls.finite_mask))

        standard_ref = _standard_lens_dic(cls.par_dic_0, **C7_REFERENCE_LENS)
        sampled_ref = cls.prior.inverse_transform(**standard_ref)
        vec_ref = [sampled_ref[name] for name in cls.prior.sampled_params]
        cls.lnpost_reference = cls.posterior.lnposterior(*vec_ref)

    def test_no_result_is_nan_or_positive_infinity(self):
        """Every draw is finite or exactly ``-inf`` (never NaN / ``+inf``)."""
        for value in self.lnposts:
            with self.subTest(value=value):
                self.assertTrue(np.isfinite(value) or np.isneginf(value))
            self.n_compared += 1

    def test_finite_fraction_clears_hard_floor(self):
        """A meaningful minority of draws land in the certified domain."""
        self.assertGreaterEqual(self.finite_fraction, C7_MIN_FINITE_FRACTION)
        self.n_compared += 1
        self._plot_smoke()

    def test_reference_near_truth_is_finite(self):
        """The near-unlensed reference config yields a finite lnposterior."""
        self.assertTrue(np.isfinite(self.lnpost_reference))
        self.n_compared += 1

    def test_best_draw_does_not_beat_truth_by_a_wide_margin(self):
        """No random draw out-scores the near-truth reference by >50 nats."""
        best = float(np.max(self.lnposts[self.finite_mask]))
        self.assertLessEqual(
            best, self.lnpost_reference + C7_PEAK_MARGIN_NATS,
            f'best random draw {best:.2f} exceeds near-truth reference '
            f'{self.lnpost_reference:.2f} by more than {C7_PEAK_MARGIN_NATS} '
            'nats; the posterior peak may be away from the injection')
        self.n_compared += 1

    @unittest.expectedFailure
    def test_aspirational_ninety_percent_finite(self):
        """
        The spec's aspirational 0.90 finite fraction is NOT met: the prior
        box overlaps the ``gamma ~ 0.5`` cancellation band, so a large
        minority of uniform draws are refused.  Carried as an expected
        failure so it flips to a loud unexpected-success the day the prior
        box is tightened to the certified domain.
        """
        # Count the comparison BEFORE the (expected-to-fail) assertion so the
        # anti-vacuity tearDown does not itself ERROR under expectedFailure.
        self.n_compared += 1
        self.assertGreaterEqual(self.finite_fraction,
                                C7_ASPIRATIONAL_FINITE_FRACTION)

    def _plot_smoke(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        finite = self.lnposts[self.finite_mask]
        fig, ax = plt.subplots()
        ax.hist(finite, bins=40)
        ax.axvline(self.lnpost_reference, color='r', ls='--',
                   label='near-truth reference')
        ax.set_xlabel('lnposterior (finite draws)')
        ax.set_title(
            f'C7 smoke: {self.finite_fraction:.0%} finite of {C7_N_DRAWS}')
        ax.legend()
        fig.savefig(OUTPUT_DIR / 'test_lensing_prior_c7_smoke.png',
                    dpi=80, bbox_inches='tight')
        plt.close(fig)


class SelfFalsificationTestCase(_LensSuiteTestCase):
    """Prove the suite can go RED: each gate fires under an injected fault.

    Every method here is GREEN because it asserts a fault IS detected.  A
    suite whose gates could not distinguish a broken pipeline from a correct
    one would read green forever, so this class is the standing proof that
    the round-trip gate, the mass-sheet magnitude gate, and the anti-vacuity
    ``tearDown`` are all discriminating.  (The refusal-net ``-inf`` gate's
    mutation proof lives in `RefusalNetTestCase`.)
    """

    def test_anti_vacuity_teardown_fails_on_zero_comparisons(self):
        """The base `tearDown` fails a test that made zero comparisons."""
        probe = _LensSuiteTestCase(methodName='setUp')
        probe.setUp()
        self.n_compared += 1
        with self.assertRaises(probe.failureException):
            probe.tearDown()

    def test_roundtrip_gate_rejects_a_boundary_clamp(self):
        """A 1e-6 clamp on a recovered coordinate exceeds the round-trip tol.

        The C1 tolerance ``ROUNDTRIP_ATOL + ROUNDTRIP_RTOL*|val|`` must sit
        far below a realistic clamp, else the round-trip identity is vacuous.
        """
        prior = _harness().prior
        rng = np.random.default_rng(SEED + 99)
        sampled = _random_sampled_point(prior, rng)
        standard = prior.transform(*sampled)
        recovered = prior.inverse_transform(**standard)
        idx = prior.sampled_params.index('u1')
        clean_err = abs(recovered['u1'] - sampled[idx])
        tol = ROUNDTRIP_ATOL + ROUNDTRIP_RTOL * abs(sampled[idx])
        bugged_err = clean_err + 1e-6
        self.assertLessEqual(clean_err, tol,
                             'control round-trip should already pass')
        self.assertGreater(bugged_err, tol,
                           'round-trip gate cannot see a 1e-6 clamp (vacuous)')
        self.n_compared += 1

    def test_magnitude_identity_gate_rejects_a_wrong_lambda(self):
        """A 1% error in ``lam`` breaks the mass-sheet magnitude identity."""
        kappa, gamma, y1, y2, _m = MASSSHEET_CONFIGS[0]
        lam = 1.0 - kappa
        original = ChangRefsdalChannels(MASSSHEET_W_GRID).evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=kappa)
        twin = ChangRefsdalChannels(MASSSHEET_W_GRID).evaluate(
            gamma=gamma / lam, y=(y1 / np.sqrt(lam), y2 / np.sqrt(lam)),
            beta=0.0, kappa=0.0)
        abs_original = np.abs(original.exact_total)
        clean = np.max(np.abs(abs_original - np.abs(twin.exact_total) / lam)
                       / abs_original)
        bugged = np.max(
            np.abs(abs_original - np.abs(twin.exact_total) / (lam * 1.01))
            / abs_original)
        self.assertLess(clean, MASSSHEET_MAG_RTOL,
                        'control magnitude identity should already pass')
        self.assertGreater(bugged, MASSSHEET_MAG_RTOL,
                           'magnitude gate cannot see a 1% lambda error')
        self.n_compared += 1


if __name__ == '__main__':
    unittest.main()
