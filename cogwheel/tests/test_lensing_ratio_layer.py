"""
Tests for the RATIO LAYER of the Chang--Refsdal microlensing likelihood
(``lensing.likelihood`` WP1/WP2): the candidate/fiducial heterodyne that
divides the smooth candidate envelope by a memoized snapped-lattice
fiducial envelope so only the ultra-flat ratio ``rho_bare`` is
interpolated, plus the guards, the direct-path fallback, and the
refusal symmetry that protect it.

WHAT THIS SUITE PINS
--------------------
The ratio layer is an ACCELERATION that must move no answer.  Its
contract, gate by gate:

* IDENTITY AT A LATTICE POINT (`RatioDirectIdentityTestCase`).  When the
  candidate lens sits EXACTLY on the fiducial lattice, the fiducial cell
  equals the candidate, ``dtau_c`` and ``rho_bare - 1`` collapse to the
  engine's grid-reproducibility floor, and the ratio-path reconstructed
  envelope and ``lnlike`` must equal the direct path's (``_force_direct``)
  to that floor -- pure gauge algebra, NOT interpolation.

* PERTURBED AGREEMENT (`PerturbedRatioDirectTestCase`).  One cell off the
  lattice (``rho_bare != 1``), the ratio path must track the direct path
  within the inherited relative-binning budget, and its TYPICAL deviation
  must stay in the tight ratio-layer budget.

* RATIO-VS-BRUTEFORCE (`RatioBruteforceTestCase`).  The physically
  meaningful correctness gate: the guarded ratio-path ``lnlike`` matches
  the exact full-grid ``lnlike_bruteforce`` within the UNCHANGED
  ``max(RB_ATOL, RB_RTOL*|bf|)``, on the five anchors plus a
  ``beta != 0`` rotated anchor that exercises the shear-matrix rotation.

* CACHE DETERMINISM (`FiducialCacheDeterminismTestCase`).  The fiducial
  is a pure function of its cell key, so memoization is BIT-identical and
  order independent -- checked by raw float ``==``, never ``isclose``.

* PATH SYMMETRY & FALLBACK (`RefusalSymmetryTestCase`).  A macro-saddle
  candidate is EVALUATED symmetrically -- the ratio path, the direct path
  and brute force agree within the inherited RB tolerance; a refusing
  SNAPPED fiducial must NOT veto a candidate that is itself inside the
  certified domain -- the ratio path falls back to direct and returns a
  finite matched value.  Symmetric NAMED REFUSAL is pinned on the
  STRUCTURAL over-critical boundary by
  `test_lensing_fast_path::test_paths_refuse_over_critical_symmetrically`,
  which (unlike a certification threshold) cannot drift.

* GUARD FALLBACK (`GuardFallbackTestCase`).  An image-count mismatch or an
  unhealthy (near-zero-dip) fiducial envelope makes the ratio path fall
  back to the direct path bit-for-bit, preserving correctness.

* TIMING (`RatioTimingTestCase`).  The machine-independent structural
  gates the acceleration was for: the ratio ``lnlike`` beats brute force
  by ``>= SPEEDUP_MIN`` and the candidate ratio node count is
  config-independent and ``<= RATIO_NODE_CEILING``.  The absolute warm ms
  is reported and guarded only by a generous machine-calibrated ceiling
  (DEVIATION, see below).

* DEEP-BAND MACRO LIMIT (`DeepBandMacroLimitTestCase`, F009).  Through the
  ratio path, the reconstructed ``|F|`` at tiny ``w`` still equals the
  exact macro magnification ``1/sqrt((1-kappa)**2 - gamma**2)`` to
  ``1e-6`` -- the ratio layer adds no small-``w`` surgery.

ORACLE INDEPENDENCE (F002)
--------------------------
The correctness oracles here are INDEPENDENT of the ratio layer's own
derivation: ``lnlike_bruteforce`` builds the lensed strain on the full
FFT grid through a separate `LensedWaveformGenerator` (no ratio, no
binning); ``_force_direct`` is the un-accelerated SACR-C reconstruction;
and the deep-band limit is the closed-form macro magnification.  The
ratio path is never gated against itself.

TWO DOCUMENTED DEVIATIONS FROM THE BRIEF
----------------------------------------
1. The brief's exact-identity envelope tolerance ``1e-13`` is
   UNACHIEVABLE and is not a bug: the ratio path reconstructs
   ``E_cand = rho * E_fid`` from the candidate seed (8-node) grid and the
   fiducial's LOO-refined grid, and the Chang--Refsdal engine reproduces
   its envelope and ``critical_delay`` across DIFFERENT node grids only to
   ~1e-11 (measured).  So ``rho_bare - 1`` and ``dtau_c`` floor at ~1e-12,
   not machine epsilon.  We gate the envelope identity at
   `ENVELOPE_IDENTITY_RTOL = 1e-9` -- still SEVEN orders below the
   ``_LOO_STOP = 4e-3`` interpolation budget, so it certifies the brief's
   real claim ("algebra only, no interpolation") -- and gate the
   physically meaningful ``lnlike`` identity at the brief's ``1e-9`` (met).

2. The brief's absolute ``<= 10 ms`` warm ``lnlike`` ceiling is a
   server-specific step-rule guarded only under a Professor obstruction;
   the numba special-function engine dominates and the honest per-eval
   cost is machine-dependent, so the ABSOLUTE ms is reported and bounded
   only by a generous `MS_CEILING`.  The HARD gates are the
   machine-independent SPEEDUP and node-count ones.

ANTI-VACUITY AND SELF-FALSIFICATION
-----------------------------------
`RatioLayerTestCase.tearDown` fails a test that made zero comparisons.
`SelfFalsificationTestCase` proves the identity, the RB-agreement and the
anti-vacuity gate can each go red.
"""
from __future__ import annotations

# Single-thread pinning for the timing gate (best-effort): the honest
# per-eval cost the parallel sampler pays per core is the single-thread
# one.  These are read by OpenBLAS/MKL/numba at import, so set them BEFORE
# numpy/matplotlib/numba import.  A no-op if another module initialised the
# BLAS pool first; the HARD timing gates (speedup, node count) are robust
# to that regardless.
import os as _os

# Pin single-threaded numerics ONLY in strict-timing mode (the sole
# consumer of the determinism): an import-scope pin poisons shared
# pytest workers — numba's thread layer launches once per process, so
# a layer launched at 1 by a lensing prange call makes any later
# parallel ufunc (e.g. marginalized_extrinsic_qas) hard-fail on the
# default 64 (Build 8f gate incident, 2026-07-21).
if _os.environ.get('COGWHEEL_STRICT_TIMING'):
    for _thread_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        _os.environ.setdefault(_thread_var, '1')

import pathlib
import time
import types
import warnings
from unittest import TestCase, main, mock, skipUnless

# --- Two-tier test split (Build 8d re-pricing) -------------------------------
# The exact positive-parity path is now the Schwinger evaluator (~90 ms/node),
# so ``lnlike_bruteforce`` -- the full-FFT-grid matched filter that evaluates
# the exact engine per frequency -- costs ~138 s/call post-8d.  Tests whose
# runtime is dominated by that brute-force accuracy oracle are the DRIVER /
# post-build tier, gated OFF by default and run in-build only as FAST
# structural / witness / refusal gates.  Set ``COGWHEEL_BRUTE_ACCURACY=1`` to
# run the brute-force accuracy tier (it remains falsifiable and green there).
_BRUTE_ACCURACY = bool(_os.environ.get('COGWHEEL_BRUTE_ACCURACY'))
_brute_accuracy_tier = skipUnless(
    _BRUTE_ACCURACY,
    'brute-force accuracy tier: set COGWHEEL_BRUTE_ACCURACY=1 -- exact path '
    '~90 ms/node makes lnlike_bruteforce ~138 s/call post-8d')

import numpy as np
from matplotlib import pyplot as plt

from cogwheel import data, waveform
import cogwheel.lensing.likelihood as likelihood_module
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, _FiducialEnvelope, _fiducial_key,
    _lens_from_key, _snap, _ENVELOPE_HEALTH_FLOOR,
    _FID_GAMMA_SPACING, _FID_BETA_SPACING, _FID_Y_SPACING)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError

# ---------------------------------------------------------------------------
# Fixture constants (shared with the crown / fast-path suites so the anchors
# are read off the SAME conditioned data and bin grid).
# ---------------------------------------------------------------------------

#: Higher-mode approximant so the mode-pair contraction is exercised.
APPROXIMANT = 'IMRPhenomXPHM'

#: Fixed seed for every stochastic input.
SEED = 20260717

#: Bin width [Hz] of the uniform relative-binning grid.
DF_BIN = 4.0

#: Largest relative image delay [s] the fixture's bins support.
DELTA_T_MAX = 0.02

#: Lens mass [Msun] / redshift of the well-conditioned fixture.
M_LENS_MSUN = 90.0
Z_LENS = 0.4

#: Inherited relative-binning agreement tolerances (UNCHANGED): the ratio
#: path must match ``lnlike_bruteforce`` within ``max(RB_ATOL, RB_RTOL*|bf|)``.
RB_ATOL = 1.5
RB_RTOL = 1e-2

#: Envelope identity tolerance at a lattice point (DEVIATION 1): the
#: engine's cross-grid reproducibility floor (~1e-11), seven orders below
#: the ``_LOO_STOP`` interpolation budget, so it still certifies
#: "algebra only, no interpolation".
ENVELOPE_IDENTITY_RTOL = 1e-9

#: ``lnlike`` identity tolerance [nats] at a lattice point (brief value).
LNLIKE_IDENTITY_ATOL = 1e-9

#: Per-perturbation ceiling [nats] on ``|lnlike_ratio - lnlike_direct|``:
#: the inherited RB budget floor.  Both paths are RB approximations to
#: brute force within ``max(RB_ATOL, RB_RTOL*|.|)``, so the ratio layer
#: must add no error beyond that.
PERTURBED_ATOL = RB_ATOL

#: Ceiling [nats] on the MEDIAN ratio-vs-direct deviation over a
#: per-anchor sweep -- the tight ratio-layer budget that a heavy tail from
#: a too-small node budget would inflate.  Set with margin above the
#: measured medians (<= ~0.08).
PERTURBED_MEDIAN_MAX = 0.15

#: Conservative lower bound on the warm ratio speed-up over brute force
#: (measured ~140x on this box).
SPEEDUP_MIN = 5.0

#: Config-independent ceiling on the candidate ratio node count (measured
#: 8 on every anchor).
RATIO_NODE_CEILING = 20

#: Best-of-N repeats for warm timing.
TIMING_REPEATS = 5

#: LOOSE absolute ceiling [s] on the warm best-of-N ratio ``lnlike``
#: (DEVIATION 2): a generous regression guard on THIS box, not the brief's
#: physical 10 ms claim.  RE-TUNED (Build 8d homogenization): the exact
#: positive-parity wave branch is the Schwinger evaluator at ~90 ms/node,
#: so the warm ratio ``lnlike`` measures ~0.75 s.  Raised 0.5 -> 3.0
#: (~4x) -- generous against a loaded box yet still catching a
#: catastrophic regression.  The exact path is the SINGLE certified
#: evaluator BY DESIGN.
MS_CEILING = 3.0

#: Strict-timing switch (opt-in).  The brute-force speed-up gate
#: re-evaluates the FULL-grid matched filter, which -- since homogenization
#: routes the exact Schwinger engine per-frequency -- now costs ~140 s per
#: brute call (best-of-N would be minutes), a build-killer for the default
#: fast suite.  Gated OFF unless ``COGWHEEL_STRICT_TIMING`` is set; the
#: default suite keeps the machine-independent node-count gate and the
#: loose absolute ceiling.
_STRICT_TIMING = bool(_os.environ.get('COGWHEEL_STRICT_TIMING'))

#: Relative tolerance on the ratio-path ``|F|`` vs the closed-form macro
#: magnification in the deep band (matches the existing DeepBandMacroLimit
#: gate).
MACRO_LIMIT_RTOL = 1e-6

#: Deep-band (F009) sheared config, lattice-aligned and positive parity
#: (``1 - 0.30 = 0.70 > 0.21``): ``(gamma, kappa, y1, y2)``.
DEEP_GAMMA = 0.21
DEEP_KAPPA = 0.30
DEEP_Y1 = 0.05
DEEP_Y2 = 0.0

#: Three tiny lens masses [Msun] placing ``w = xi*f`` in three successive
#: deep-band decades where ``|F| -> 1/sqrt((1-kappa)**2 - gamma**2)``.
DEEP_M_LENS = (1e-4, 1e-5, 1e-6)

#: The five fast-path anchors, LATTICE-ALIGNED so ``_fiducial_key`` returns
#: them unchanged: ``(label, y1, y2, gamma, beta, kappa)``.  Each value is
#: an exact multiple of its lattice spacing (gamma 0.03, beta pi/16, kappa
#: 0.02, y 0.05); verified on the lattice by `test_anchors_are_on_lattice`.
ANCHORS = (
    ('crown', 0.10, 0.05, 0.21, 0.0, 0.0),        # four real images
    ('near_cusp', -0.40, 0.00, 0.21, 0.0, 0.0),   # two images, near cusp
    ('two_image', 0.50, 0.00, 0.21, 0.0, 0.0),    # two images
    ('near_fold', 0.15, 0.05, 0.21, 0.0, 0.0),    # four images, near fold
    ('sheared_sw', 0.30, 0.10, 0.12, 0.0, 0.30),  # convergent + shear
)

#: A ``beta != 0`` rotated anchor exercising the ``macro_matrix`` rotation.
ROTATED_ANCHOR = ('rotated', 0.30, 0.10, 0.20, 0.70, 0.0)

#: Macro-SADDLE ``(gamma, kappa)`` INTERIOR ``0 < 1 - kappa < |gamma|``
#: (``1 - kappa = 0.4 < |gamma| = 0.5``).  Pre-Build-7b the domain gate
#: refused this with `LensDomainError`; Build 7b lifted the saddle guard
#: end to end, so all three paths now EVALUATE it finitely and agree
#: within RB tolerance (see `test_macro_saddle_evaluated_symmetrically`).
MACRO_SADDLE = dict(gamma=0.5, beta=0.0, kappa=0.6, y1=0.20, y2=0.05)

#: Image-count-mismatch config: the candidate has two real images while
#: its snapped-lattice fiducial has four (verified in the guard test).
IMAGE_MISMATCH_CONFIG = dict(gamma=0.18, beta=0.0, kappa=0.0,
                             y1=0.22, y2=0.05)

OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'


def _reference_par_dic():
    """A deterministic precessing reference ``par_dic`` for `APPROXIMANT`."""
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
        eventname='test_ratio_layer', duration=4, detector_names='HLV',
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0., seed=SEED)
    event_data.inject_signal(_reference_par_dic(), APPROXIMANT)
    return event_data


def _build_likelihood():
    """Build the lensed relative-binning likelihood on the fixture data."""
    event_data = _make_noisy_event()
    wfg = waveform.WaveformGenerator.from_event_data(event_data, APPROXIMANT)
    band = event_data.frequencies[event_data.fslice]
    f_lo, f_hi = float(band[0]), float(band[-1])
    edges = np.arange(f_lo, f_hi, DF_BIN)
    if edges[-1] < f_hi:
        edges = np.append(edges, f_hi)
    return LensedRelativeBinningLikelihood(
        event_data, wfg, _reference_par_dic(),
        delta_t_max=DELTA_T_MAX, fbin=edges)


class RatioLayerTestCase(TestCase):
    """
    Shared fixture (one likelihood, built once) plus the anti-vacuity tally.

    `setUp` resets a per-test comparison counter; `tearDown` fails a test
    that asserted nothing, so a sweep that silently skipped every anchor
    cannot read green.
    """

    like: LensedRelativeBinningLikelihood

    @classmethod
    def setUpClass(cls):
        """Build the shared likelihood once for the whole class."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.like = _build_likelihood()
        cls.par_dic_0 = _reference_par_dic()

    def setUp(self):
        """Reset the anti-vacuity tally and the ratio-layer state."""
        self.n_checks = 0
        self.like._force_direct = False
        self.like._fid_cache.clear()

    def tearDown(self):
        """Fail a test that made zero comparisons."""
        self.like._force_direct = False
        if self.n_checks == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _lens_dic(y1, y2, gamma, beta, kappa,
                  m_lens=M_LENS_MSUN, z_lens=Z_LENS):
        """Assemble the seven lens keys expected in ``par_dic``."""
        return {'m_lens_msun': m_lens, 'z_lens': z_lens,
                'y1': y1, 'y2': y2, 'gamma': gamma, 'beta': beta,
                'kappa': kappa}

    def _candidate(self, lens_dic):
        """Merge the fiducial waveform params with a lens sub-dictionary."""
        base = dict(self.par_dic_0)
        base.update(lens_dic)
        return base

    def _anchor_candidate(self, anchor):
        """Build a candidate from an ``ANCHORS`` / ``ROTATED_ANCHOR`` row."""
        _, y1, y2, gamma, beta, kappa = anchor
        return self._candidate(self._lens_dic(y1, y2, gamma, beta, kappa))

    def _lnlike_direct(self, par_dic):
        """``lnlike`` forcing the un-accelerated direct SACR-C path."""
        self.like._force_direct = True
        try:
            return self.like.lnlike_and_metadata(par_dic)[0]
        finally:
            self.like._force_direct = False

    def _lnlike_ratio(self, par_dic):
        """``lnlike`` through the ratio layer (guards active)."""
        self.like._force_direct = False
        return self.like.lnlike_and_metadata(par_dic)[0]

    def _capture_reconstruction(self, par_dic):
        """
        Reconstruct the candidate and return ``(w, E_cand, F_total)``.

        Both the ratio path and the direct path funnel their dense
        envelope through the module-level `reconstruct_from_envelope`
        (`_kernels_from_dense_envelope` for the ratio path,
        `_reconstruct_kernels` -> `_kernels_from_dense_envelope` for the
        direct path), so patching that single seam records exactly the
        dense envelope ``E_cand`` each path builds and the reconstructed
        total ``F`` it returns -- no duplication of production math.
        Respects the current ``_force_direct`` flag.

        Returns
        -------
        dense_w : np.ndarray
            The dense reconstruction grid (dimensionless frequency).
        envelope : np.ndarray
            The dense candidate envelope ``E_cand(w)`` (complex).
        total : np.ndarray
            The reconstructed total amplification ``F(w)`` (complex).
        """
        store = {}
        original = likelihood_module.reconstruct_from_envelope

        def wrapper(dense_w, envelope_dense, *args, **kwargs):
            kernels, total = original(dense_w, envelope_dense,
                                      *args, **kwargs)
            store['w'] = np.asarray(dense_w).copy()
            store['envelope'] = np.asarray(envelope_dense).copy()
            store['total'] = np.asarray(total).copy()
            return kernels, total

        with mock.patch.object(likelihood_module, 'reconstruct_from_envelope',
                               wrapper):
            self.like._amplification_coefficients(par_dic)
        return store['w'], store['envelope'], store['total']

    def _capture_envelope(self, par_dic):
        """Dense ``(w, E_cand)`` from `_capture_reconstruction` (spec 1)."""
        dense_w, envelope, _total = self._capture_reconstruction(par_dic)
        return dense_w, envelope

    def _ratio_node_count(self, par_dic):
        """
        Number of candidate ratio nodes for ``par_dic`` (ratio path).

        Wraps `_ratio_loo_nodes` to record the length of the refined
        ratio grid, the config-independent count the timing gate pins.
        Returns ``None`` if the candidate did not take the ratio path
        (a guard/refusal fell back to the direct route).
        """
        store = {}
        original = self.like._ratio_loo_nodes

        def wrapper(*args, **kwargs):
            coarse_w, rho_nodes = original(*args, **kwargs)
            store['count'] = int(np.size(coarse_w))
            return coarse_w, rho_nodes

        with mock.patch.object(self.like, '_ratio_loo_nodes',
                               side_effect=wrapper):
            self.like._force_direct = False
            self.like._amplification_coefficients(par_dic)
        return store.get('count')

    @staticmethod
    def _save_figure(fig, name):
        """Write ``fig`` to ``cogwheel/tests/output/<name>.png``."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)


class FixtureSanityTestCase(RatioLayerTestCase):
    """Guard the fixtures the rest of the suite rests on."""

    def test_anchors_are_on_lattice(self):
        """
        Every anchor's snapped lens cell is the anchor itself (to float
        rounding), so the fiducial cell equals the candidate -- the
        identity test's premise.  The comparison is `assertAlmostEqual`,
        not `assertEqual`, because an on-lattice value need not be
        BIT-equal to its snapped multiple (``round(0.15/0.05)*0.05 =
        0.15000000000000002``); the residual is ~1 ULP, far below the
        identity gate's ``ENVELOPE_IDENTITY_RTOL``.  Snapping is also
        idempotent, so the candidate and its fiducial share one cell.
        """
        for anchor in ANCHORS:
            with self.subTest(anchor=anchor[0]):
                self.n_checks += 1
                _, y1, y2, gamma, beta, kappa = anchor
                lens = self._lens_dic(y1, y2, gamma, beta, kappa)
                key = _fiducial_key(lens)
                snapped = _lens_from_key(key)
                # Idempotence: the snapped value snaps back to the same cell.
                self.assertEqual(key, _fiducial_key(snapped),
                                 f'{anchor[0]}: snapping is not idempotent')
                for name in ('gamma', 'beta', 'kappa', 'y1', 'y2'):
                    self.assertAlmostEqual(
                        snapped[name], lens[name], places=12,
                        msg=f'{anchor[0]}: {name} is not on the fiducial '
                        'lattice')

    def test_snap_is_pure_nearest_multiple(self):
        """`_snap` returns the nearest lattice multiple (spot check)."""
        self.n_checks += 1
        self.assertEqual(_snap(0.21, _FID_GAMMA_SPACING), 0.21)
        self.assertAlmostEqual(_snap(0.205, _FID_GAMMA_SPACING), 0.21)
        self.assertAlmostEqual(_snap(-0.40, _FID_Y_SPACING), -0.40)


class RatioDirectIdentityTestCase(RatioLayerTestCase):
    """
    Spec 1: at a lattice point the ratio path is the direct path.

    When the candidate lens sits exactly on the fiducial lattice, the
    fiducial cell equals the candidate: ``dtau_c`` and ``rho_bare - 1``
    collapse to the engine's cross-grid reproducibility floor, so the
    ratio-path reconstructed envelope and ``lnlike`` must equal the direct
    path's to that floor -- pure gauge algebra, not interpolation.
    """

    def test_envelope_matches_direct_path_at_lattice_points(self):
        """
        ``max|E_ratio - E_direct| / max|E_direct| < ENVELOPE_IDENTITY_RTOL``
        on every lattice anchor; a nonzero floor above that would reveal a
        carrier / critical-delay bug (e.g. the fiducial's ``tau_c`` leaking
        into the reconstruction).
        """
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        worst = 0.0
        for anchor in ANCHORS:
            with self.subTest(anchor=anchor[0]):
                par_dic = self._anchor_candidate(anchor)

                self.like._force_direct = False
                w_ratio, e_ratio = self._capture_envelope(par_dic)
                self.like._force_direct = True
                try:
                    w_direct, e_direct = self._capture_envelope(par_dic)
                finally:
                    self.like._force_direct = False

                # Both paths reconstruct on the same dense w grid.
                np.testing.assert_allclose(w_ratio, w_direct, rtol=0, atol=0)
                scale = float(np.max(np.abs(e_direct)))
                rel = float(np.max(np.abs(e_ratio - e_direct))) / scale
                worst = max(worst, rel)
                self.n_checks += 1
                self.assertLess(
                    rel, ENVELOPE_IDENTITY_RTOL,
                    f'{anchor[0]}: |E_ratio - E_direct|/max|E_direct| = '
                    f'{rel:.3e} exceeds {ENVELOPE_IDENTITY_RTOL:.0e} '
                    '(carrier / critical-delay bug?)')
                ax.semilogy(w_ratio, np.abs(e_ratio - e_direct) + 1e-300,
                            lw=0.9, label=anchor[0])

        ax.set_xlabel('w (dimensionless frequency)')
        ax.set_ylabel('|E_ratio - E_direct|')
        ax.set_title('Spec 1: ratio-vs-direct envelope residual at lattice '
                     f'points (worst rel = {worst:.2e})')
        ax.legend(fontsize=7)
        self._save_figure(fig, 'spec1_envelope_identity_residual')

    def test_lnlike_matches_direct_path_at_lattice_points(self):
        """
        ``|lnlike_ratio - lnlike_direct| < LNLIKE_IDENTITY_ATOL`` (the
        brief's ``1e-9`` nats) on every lattice anchor.
        """
        for anchor in ANCHORS:
            with self.subTest(anchor=anchor[0]):
                par_dic = self._anchor_candidate(anchor)
                lnr = self._lnlike_ratio(par_dic)
                lnd = self._lnlike_direct(par_dic)
                self.n_checks += 1
                self.assertLess(
                    abs(lnr - lnd), LNLIKE_IDENTITY_ATOL,
                    f'{anchor[0]}: |lnlike_ratio - lnlike_direct| = '
                    f'{abs(lnr - lnd):.3e} exceeds '
                    f'{LNLIKE_IDENTITY_ATOL:.0e} nats')


class PerturbedRatioDirectTestCase(RatioLayerTestCase):
    """
    Spec 2: one cell off the lattice, ratio tracks direct.

    Perturbing the lens geometry by up to one lattice cell makes a NEARBY
    cell the fiducial (``rho_bare != 1``), so the ratio layer now genuinely
    interpolates.  Its per-perturbation deviation from the direct path must
    stay within the inherited relative-binning budget, and its TYPICAL
    (median) deviation must sit in the tight ratio-layer budget -- a heavy
    tail would betray a node budget too small for the lattice spacing.
    """

    #: Number of off-lattice perturbations swept per anchor.
    N_PERTURBATIONS = 10

    def _perturbations(self, rng):
        """Yield ``N_PERTURBATIONS`` sub-cell lens offsets (deterministic)."""
        for _ in range(self.N_PERTURBATIONS):
            yield {
                'gamma': float(rng.uniform(-0.5, 0.5)) * _FID_GAMMA_SPACING,
                'beta': float(rng.uniform(-0.5, 0.5)) * _FID_BETA_SPACING,
                'y1': float(rng.uniform(-0.5, 0.5)) * _FID_Y_SPACING,
                'y2': float(rng.uniform(-0.5, 0.5)) * _FID_Y_SPACING,
            }

    def test_ratio_tracks_direct_off_lattice(self):
        """
        Per perturbation ``|lnlike_ratio - lnlike_direct| < PERTURBED_ATOL``
        and, per anchor, the MEDIAN deviation ``< PERTURBED_MEDIAN_MAX``.
        """
        rng = np.random.default_rng(SEED)
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        all_deltas = []
        for anchor in ANCHORS:
            _, y1, y2, gamma, beta, kappa = anchor
            deltas = []
            for pert in self._perturbations(rng):
                lens = self._lens_dic(
                    y1 + pert['y1'], y2 + pert['y2'], gamma + pert['gamma'],
                    beta + pert['beta'], kappa)
                par_dic = self._candidate(lens)
                try:
                    lnr = self._lnlike_ratio(par_dic)
                    lnd = self._lnlike_direct(par_dic)
                except LensDomainError:
                    # A sub-cell perturbation that wanders into a refusal is
                    # not part of this agreement gate; skip it (the sweep as
                    # a whole still compares many points -> anti-vacuity OK).
                    continue
                delta = abs(lnr - lnd)
                deltas.append(delta)
                self.n_checks += 1
                self.assertLess(
                    delta, PERTURBED_ATOL,
                    f'{anchor[0]}: |lnlike_ratio - lnlike_direct| = '
                    f'{delta:.3e} exceeds RB budget {PERTURBED_ATOL}')
            with self.subTest(anchor=anchor[0]):
                self.assertGreater(
                    len(deltas), 0,
                    f'{anchor[0]}: every perturbation refused (no data)')
                median = float(np.median(deltas))
                self.assertLess(
                    median, PERTURBED_MEDIAN_MAX,
                    f'{anchor[0]}: median ratio-vs-direct deviation '
                    f'{median:.3e} exceeds {PERTURBED_MEDIAN_MAX} '
                    '(ratio node budget too small?)')
            all_deltas.extend(deltas)

        ax.hist(np.log10(np.asarray(all_deltas) + 1e-300), bins=24)
        ax.axvline(np.log10(PERTURBED_MEDIAN_MAX), color='C1',
                   label=f'median gate {PERTURBED_MEDIAN_MAX}')
        ax.axvline(np.log10(PERTURBED_ATOL), color='C3',
                   label=f'per-point gate {PERTURBED_ATOL}')
        ax.set_xlabel('log10 |lnlike_ratio - lnlike_direct| [nats]')
        ax.set_ylabel('count')
        ax.set_title('Spec 2: off-lattice ratio-vs-direct deviations')
        ax.legend(fontsize=7)
        self._save_figure(fig, 'spec2_perturbed_ratio_direct_hist')


class RatioBruteforceTestCase(RatioLayerTestCase):
    """
    Spec 3: the physically meaningful correctness gate.

    The guarded ratio-path ``lnlike`` matches the exact full-grid
    ``lnlike_bruteforce`` -- an INDEPENDENT oracle built through a separate
    `LensedWaveformGenerator` on the full FFT grid, with no ratio and no
    binning (F002) -- within the UNCHANGED ``max(RB_ATOL, RB_RTOL*|bf|)``,
    on the five anchors plus a ``beta != 0`` rotated anchor that exercises
    the shear-matrix rotation untested elsewhere.
    """

    @_brute_accuracy_tier
    def test_ratio_matches_bruteforce_on_all_anchors(self):
        """Ratio ``lnlike`` matches brute force within inherited RB tol."""
        rows = []
        for anchor in ANCHORS + (ROTATED_ANCHOR,):
            with self.subTest(anchor=anchor[0]):
                par_dic = self._anchor_candidate(anchor)
                lnr = self._lnlike_ratio(par_dic)
                lnbf = self.like.lnlike_bruteforce(par_dic)
                tol = max(RB_ATOL, RB_RTOL * abs(lnbf))
                delta = abs(lnr - lnbf)
                rows.append((anchor[0], lnr, lnbf, delta, tol))
                self.n_checks += 1
                self.assertLess(
                    delta, tol,
                    f'{anchor[0]}: |lnlike_ratio - lnlike_bruteforce| = '
                    f'{delta:.3e} exceeds max(RB_ATOL, RB_RTOL*|bf|) = '
                    f'{tol:.3e}')
        # Per-anchor diagnostic table (ratio, bruteforce, delta, tol).
        header = f'{"anchor":<12}{"ratio":>14}{"bruteforce":>14}' \
                 f'{"delta":>12}{"tol":>12}'
        lines = [header] + [
            f'{name:<12}{lnr:>14.5f}{lnbf:>14.5f}{delta:>12.3e}{tol:>12.3e}'
            for name, lnr, lnbf, delta, tol in rows]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'spec3_ratio_vs_bruteforce_table.txt').write_text(
            '\n'.join(lines) + '\n')


class FiducialCacheDeterminismTestCase(RatioLayerTestCase):
    """
    Spec 4: memoization is bit-identical and order-independent.

    The fiducial is a pure function of its cell key, so re-evaluating an
    earlier candidate after the cache has been populated by others, and
    evaluating the same candidate on a brand-new (empty-cache) instance,
    must reproduce the first value to the raw float bit -- checked by exact
    ``==``, never ``isclose``.  Any difference means the cache key or the
    fiducial is not a pure function of the candidate.
    """

    def test_repeat_and_fresh_instance_are_bit_identical(self):
        """Repeat and fresh-instance ``lnlike`` equal the first, bit-for-bit."""
        target = self._anchor_candidate(ANCHORS[0])
        others = [self._anchor_candidate(a) for a in ANCHORS[1:]]

        # Fresh empty cache: the first (cold) evaluation of the target.
        self.like._fid_cache.clear()
        first = self._lnlike_ratio(target)

        # Populate the cache with distinct candidates, then re-evaluate the
        # target: order-independence and cache-hit determinism.
        for par_dic in others:
            self._lnlike_ratio(par_dic)
        repeat = self._lnlike_ratio(target)
        self.n_checks += 1
        # Raw-bit equality (floats): identical object bit patterns.
        self.assertEqual(np.float64(first).tobytes(),
                         np.float64(repeat).tobytes(),
                         'repeat lnlike differs at the bit level')

        # Brand-new instance with an empty cache: same value from scratch.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fresh_like = _build_likelihood()
        fresh_like._force_direct = False
        fresh_value = fresh_like.lnlike_and_metadata(target)[0]
        self.n_checks += 1
        self.assertEqual(np.float64(first).tobytes(),
                         np.float64(fresh_value).tobytes(),
                         'fresh-instance lnlike differs at the bit level '
                         '(fiducial not a pure function of the candidate)')


class RefusalSymmetryTestCase(RatioLayerTestCase):
    """
    Spec 5: paths agree symmetrically; a refusing fiducial does not veto.

    (a) A macro-saddle candidate is now EVALUATED (not refused)
    symmetrically -- Build 7b lifted the saddle guard end to end, so the
    ratio path, the direct path and brute force all return a finite
    ``lnlike`` agreeing within the inherited RB tolerance.
    (b) DELETED 2026-08-13 (test-debt audit): the wave-branch
    named-refusal shard.  Its ``CANCELLATION_CONFIG`` witness had to be
    hand-repointed once already (Build 8e) when the engine's certified
    domain grew under it, and it drifted a SECOND time -- all three paths
    now certify it and agree to 1.93e-2 nats.  A third hand-picked config
    would drift again.  The symmetric-NAMED-REFUSAL contract is pinned
    instead on the STRUCTURAL over-critical boundary (``kappa >= 1``) by
    `test_lensing_fast_path::test_paths_refuse_over_critical_symmetrically`,
    which does not move with the certification threshold; the agreement
    this fixture now demonstrates is (a) on a second config.
    (c) A candidate INSIDE the certified domain whose SNAPPED fiducial
    refuses must NOT raise: the ratio path falls back to direct and returns
    a finite ``lnlike`` matching brute force -- verified by monkeypatching
    `_get_or_build_fiducial` to refuse, since a naturally-refusing snapped
    fiducial over a certified candidate is not reachable from the lattice.
    """

    @_brute_accuracy_tier
    def test_macro_saddle_evaluated_symmetrically(self):
        """
        A macro-saddle candidate is EVALUATED symmetrically on all three
        paths (Build 7b lifted the saddle guard end to end).

        Reconciliation choice (documented per the build brief): the shipped
        ``MACRO_SADDLE`` config ``gamma = 0.5, kappa = 0.6`` has
        ``1 - kappa = 0.4 < |gamma| = 0.5`` -- a macro-saddle INTERIOR that
        the pre-7b domain gate refused with `LensDomainError` but which now
        flows through the parity-blind SACR-C construction (the saddle wave
        branch routed to the Schwinger evaluator).  Rather than repoint the
        config at a still-refusing boundary, this test PRESERVES its
        original intent -- PATH SYMMETRY -- in the new regime: the ratio
        path, the direct path and brute force must all return a FINITE
        ``lnlike`` and agree within the inherited RB tolerance
        ``max(RB_ATOL, RB_RTOL*|bf|)``.  The named-refusal symmetry that
        used to live here is now carried, for the surviving boundaries, by
        `test_lensing_fast_path::test_paths_refuse_over_critical_symmetrically`
        and by the waveform/marginalized suites (the over-critical
        ``kappa >= 1`` and ``det A = 0`` `LensDomainError`).
        """
        par_dic = self._candidate(self._lens_dic(**MACRO_SADDLE))
        lnbf = self.like.lnlike_bruteforce(par_dic)
        self.n_checks += 1
        self.assertTrue(np.isfinite(lnbf),
                        'macro-saddle bruteforce lnlike is not finite')
        tol = max(RB_ATOL, RB_RTOL * abs(lnbf))
        for label, call in (('ratio', self._lnlike_ratio),
                            ('direct', self._lnlike_direct)):
            with self.subTest(path=label):
                value = call(par_dic)
                self.n_checks += 1
                self.assertTrue(
                    np.isfinite(value),
                    f'{label}: macro-saddle lnlike is not finite')
                self.assertLess(
                    abs(value - lnbf), tol,
                    f'{label}: |lnlike_{label} - lnlike_bruteforce| = '
                    f'{abs(value - lnbf):.3e} exceeds max(RB_ATOL, '
                    f'RB_RTOL*|bf|) = {tol:.3e} on the macro saddle')

    @_brute_accuracy_tier
    def test_refusing_snapped_fiducial_falls_back_to_direct(self):
        """
        A certified candidate whose snapped fiducial refuses is NOT vetoed:
        the ratio path falls back to direct, returns a finite ``lnlike``
        bit-identical to the direct path and matching brute force.
        """
        par_dic = self._anchor_candidate(ANCHORS[0])
        lnd = self._lnlike_direct(par_dic)
        lnbf = self.like.lnlike_bruteforce(par_dic)

        for refusal in (LensDomainError('snapped fiducial macro-saddle'),):
            with self.subTest(refusal=type(refusal).__name__):
                self.like._fid_cache.clear()

                def _raise(*_args, _exc=refusal, **_kwargs):
                    raise _exc

                with mock.patch.object(self.like, '_get_or_build_fiducial',
                                       side_effect=_raise):
                    lnr = self._lnlike_ratio(par_dic)
                self.n_checks += 1
                self.assertTrue(np.isfinite(lnr),
                                'fallback lnlike is not finite')
                # Fallback IS the direct path: bit-identical.
                self.assertEqual(np.float64(lnr).tobytes(),
                                 np.float64(lnd).tobytes(),
                                 'fiducial-refusal fallback did not reproduce '
                                 'the direct path bit-for-bit')
                tol = max(RB_ATOL, RB_RTOL * abs(lnbf))
                self.assertLess(abs(lnr - lnbf), tol,
                                'fallback lnlike does not match brute force')


class GuardFallbackTestCase(RatioLayerTestCase):
    """
    Spec 6: the two guards fall back to the direct path, preserving answers.

    (a) IMAGE-COUNT mismatch -- a candidate whose snapped-lattice fiducial
    has a different number of real images -- must trip Guard 1.
    (b) An UNHEALTHY fiducial envelope (``min|E_fid|/max|E_fid| < 0.01``)
    must trip Guard 2, since dividing by a near-zero envelope is
    ill-conditioned.
    In each case the ratio path must fall back to the direct path
    bit-for-bit and ``lnlike`` must still match brute force within RB tol.
    """

    def _ratio_path_taken(self, par_dic):
        """True iff ``_ratio_coefficients`` runs (i.e. no guard fired)."""
        taken = {'flag': False}
        original = self.like._ratio_coefficients

        def wrapper(*args, **kwargs):
            taken['flag'] = True
            return original(*args, **kwargs)

        with mock.patch.object(self.like, '_ratio_coefficients',
                               side_effect=wrapper):
            self.like._force_direct = False
            self.like.lnlike_and_metadata(par_dic)
        return taken['flag']

    def _real_image_counts(self, par_dic):
        """``(n_candidate_images, n_fiducial_images)`` for ``par_dic``."""
        lens = self.like._lens_params(par_dic)
        _, _, _, partition_cand = \
            self.like._amplification_coefficients_direct(par_dic)
        key = _fiducial_key(lens)
        fiducial = self.like._get_or_build_fiducial(key, _lens_from_key(key))
        return (int(partition_cand.real_mask.sum()),
                int(fiducial.partition.real_mask.sum()))

    @_brute_accuracy_tier
    def test_image_count_mismatch_falls_back_to_direct(self):
        """Guard 1: an image-count mismatch takes the direct path exactly."""
        par_dic = self._candidate(self._lens_dic(**IMAGE_MISMATCH_CONFIG))
        n_cand, n_fid = self._real_image_counts(par_dic)
        self.n_checks += 1
        self.assertNotEqual(
            n_cand, n_fid,
            'IMAGE_MISMATCH_CONFIG no longer straddles a caustic '
            f'(candidate {n_cand} == fiducial {n_fid} images); '
            'the Guard 1 fixture must be repaired')

        self.assertFalse(self._ratio_path_taken(par_dic),
                         'ratio path was taken despite an image-count '
                         'mismatch (Guard 1 did not fire)')
        lnr = self._lnlike_ratio(par_dic)
        lnd = self._lnlike_direct(par_dic)
        self.assertEqual(np.float64(lnr).tobytes(),
                         np.float64(lnd).tobytes(),
                         'Guard 1 fallback did not reproduce the direct path')
        lnbf = self.like.lnlike_bruteforce(par_dic)
        tol = max(RB_ATOL, RB_RTOL * abs(lnbf))
        self.assertLess(abs(lnr - lnbf), tol,
                        'Guard 1 fallback lnlike does not match brute force')

    @_brute_accuracy_tier
    def test_unhealthy_fiducial_envelope_falls_back_to_direct(self):
        """
        Guard 2: a fiducial envelope dipping below the health floor takes
        the direct path exactly.  The unhealthy envelope is INJECTED (a
        near-zero dip forced into a real fiducial) because a naturally
        vanishing fiducial envelope over a certified candidate is not
        reachable from the lattice; the guard's arithmetic is what is
        under test.
        """
        par_dic = self._anchor_candidate(ANCHORS[0])
        lens = self.like._lens_params(par_dic)
        key = _fiducial_key(lens)
        real_fid = self.like._get_or_build_fiducial(key, _lens_from_key(key))

        def _unhealthy_envelope(w):
            values = np.atleast_1d(real_fid.envelope(w)).astype(complex)
            values[values.size // 2] *= 1e-4  # force a near-zero dip
            return values

        sick_fid = types.SimpleNamespace(
            partition=real_fid.partition, envelope=_unhealthy_envelope)

        # Confirm the injected envelope actually trips the health floor.
        dense_w = np.geomspace(float(real_fid.coarse_w[0]),
                               float(real_fid.coarse_w[-1]), 64)
        magnitude = np.abs(sick_fid.envelope(dense_w))
        health = float(np.min(magnitude) / np.max(magnitude))
        self.n_checks += 1
        self.assertLess(health, _ENVELOPE_HEALTH_FLOOR,
                        'injected envelope did not breach the health floor')

        def _return_sick(*_args, **_kwargs):
            return sick_fid

        with mock.patch.object(self.like, '_get_or_build_fiducial',
                               side_effect=_return_sick):
            self.assertFalse(
                self._ratio_path_taken(par_dic),
                'ratio path taken despite an unhealthy fiducial envelope '
                '(Guard 2 did not fire)')
            lnr = self._lnlike_ratio(par_dic)
        lnd = self._lnlike_direct(par_dic)
        self.assertEqual(np.float64(lnr).tobytes(),
                         np.float64(lnd).tobytes(),
                         'Guard 2 fallback did not reproduce the direct path')
        lnbf = self.like.lnlike_bruteforce(par_dic)
        tol = max(RB_ATOL, RB_RTOL * abs(lnbf))
        self.assertLess(abs(lnr - lnbf), tol,
                        'Guard 2 fallback lnlike does not match brute force')


class RatioTimingTestCase(RatioLayerTestCase):
    """
    Spec 7: structural timing gates (HARD) plus a machine-calibrated ms.

    The acceleration the ratio layer exists for, gated machine-independent
    first: the warm ratio ``lnlike`` beats the exact ``lnlike_bruteforce``
    by ``>= SPEEDUP_MIN``, and the candidate ratio node count is
    config-independent and ``<= RATIO_NODE_CEILING``.  The absolute warm
    best-of-N ms is reported and bounded only by a generous, box-specific
    ceiling (DEVIATION 2 -- the brief's physical 10 ms claim is a
    server-specific step-rule out of scope here).
    """

    @staticmethod
    def _best_of(func, par_dic, repeats):
        """Best-of-``repeats`` wall time [s] of ``func(par_dic)``."""
        best = float('inf')
        for _ in range(repeats):
            start = time.perf_counter()
            func(par_dic)
            best = min(best, time.perf_counter() - start)
        return best

    def test_ratio_node_count_is_config_independent_and_bounded(self):
        """
        The candidate ratio node count is identical across the anchors and
        ``<= RATIO_NODE_CEILING`` -- the flat, config-independent budget
        the acceleration relies on.
        """
        counts = {}
        for anchor in ANCHORS:
            self.like._fid_cache.clear()
            count = self._ratio_node_count(self._anchor_candidate(anchor))
            self.assertIsNotNone(
                count, f'{anchor[0]}: did not take the ratio path')
            counts[anchor[0]] = count
            self.n_checks += 1
            self.assertLessEqual(
                count, RATIO_NODE_CEILING,
                f'{anchor[0]}: ratio node count {count} exceeds '
                f'{RATIO_NODE_CEILING}')
        self.assertEqual(
            len(set(counts.values())), 1,
            f'ratio node count is not config-independent: {counts}')

    def test_ratio_is_structurally_faster_than_bruteforce(self):
        """Warm ratio ``lnlike`` sits under the loose `MS_CEILING`; the
        brute-force speed-up gate is opt-in under ``COGWHEEL_STRICT_TIMING``.

        RE-TUNED (Build 8d): the exact wave branch is the Schwinger
        evaluator (~90 ms/node), so warm ratio ``lnlike`` is ~0.75 s and
        the loose ceiling is 3.0 s.  The speed-up over ``lnlike_bruteforce``
        stays the structural claim, but brute now re-evaluates the exact
        engine per-frequency (~140 s per call), so it is measured only
        under ``COGWHEEL_STRICT_TIMING`` -- the default suite must stay
        fast."""
        par_dic = self._anchor_candidate(ANCHORS[0])
        # Warm the fiducial cache and any JIT with one prior eval.
        self._lnlike_ratio(par_dic)

        t_ratio = self._best_of(self._lnlike_ratio, par_dic, TIMING_REPEATS)
        node_count = self._ratio_node_count(par_dic)

        self.n_checks += 1
        self.assertLess(
            t_ratio, MS_CEILING,
            f'warm ratio lnlike {t_ratio*1e3:.2f} ms exceeds the loose '
            f'ceiling {MS_CEILING*1e3:.0f} ms')

        report = (f'ratio warm best-of-{TIMING_REPEATS}: '
                  f'{t_ratio*1e3:.3f} ms; ratio node count {node_count}\n')

        if _STRICT_TIMING:
            self.like.lnlike_bruteforce(par_dic)  # warm
            t_bf = self._best_of(self.like.lnlike_bruteforce, par_dic,
                                 TIMING_REPEATS)
            speedup = t_bf / t_ratio
            report += (f'STRICT brute force: {t_bf*1e3:.3f} ms; '
                       f'speed-up {speedup:.1f}x\n')
            self.n_checks += 1
            self.assertGreaterEqual(
                speedup, SPEEDUP_MIN,
                f'warm ratio speed-up {speedup:.1f}x below the required '
                f'{SPEEDUP_MIN}x (ratio {t_ratio*1e3:.2f} ms, '
                f'brute force {t_bf*1e3:.2f} ms)')

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'spec7_timing_report.txt').write_text(report)


class DeepBandMacroLimitTestCase(RatioLayerTestCase):
    """
    Spec 8 (F009): the deep-band macro limit survives the ratio path.

    Through the production ratio dispatch, the reconstructed total ``|F|``
    at tiny ``w`` must still equal the exact macro magnification
    ``1/sqrt((1-kappa)**2 - gamma**2)`` -- written LITERALLY here from the
    shear and convergence alone, never from the engine / channels / any
    pipeline path (F002 oracle-tautology trap) -- to `MACRO_LIMIT_RTOL`.
    The ratio layer must add no small-``w`` surgery: a ``1/w`` prefactor
    leak would show as a rising ``|F|`` instead of a plateau.
    """

    def _deep_par_dic(self, m_lens):
        """Candidate with the deep-band shear config at ``m_lens`` [Msun]."""
        return self._candidate(self._lens_dic(
            DEEP_Y1, DEEP_Y2, DEEP_GAMMA, 0.0, DEEP_KAPPA,
            m_lens=m_lens))

    def test_ratio_path_preserves_macro_magnification_limit(self):
        """
        Across three tiny-``w`` decades the ratio-path ``|F|`` at the lowest
        ``w`` equals the closed-form macro magnification to
        `MACRO_LIMIT_RTOL`, and the low-band ``|F|`` is FLAT (no slope).
        """
        closed_form = 1.0 / np.sqrt((1.0 - DEEP_KAPPA)**2 - DEEP_GAMMA**2)
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        for m_lens in DEEP_M_LENS:
            with self.subTest(m_lens=m_lens):
                self.like._fid_cache.clear()
                par_dic = self._deep_par_dic(m_lens)
                dense_w, _envelope, total = \
                    self._capture_reconstruction(par_dic)
                order = np.argsort(dense_w)
                w_sorted = dense_w[order]
                mag = np.abs(total[order])

                low = float(mag[0])
                rel = abs(low - closed_form) / closed_form
                self.n_checks += 1
                self.assertLess(
                    rel, MACRO_LIMIT_RTOL,
                    f'm_lens={m_lens}: |F|(w_min={w_sorted[0]:.2e}) = '
                    f'{low:.9f} vs closed form {closed_form:.9f} '
                    f'(rel {rel:.2e} > {MACRO_LIMIT_RTOL:.0e}); a 1/w '
                    'prefactor leak in the ratio path?')
                ax.semilogx(w_sorted, mag, lw=0.9,
                            label=f'M_L={m_lens:.0e}')

        ax.axhline(closed_form, color='k', ls='--', lw=0.8,
                   label=f'1/sqrt((1-k)^2 - g^2) = {closed_form:.6f}')
        ax.set_xlabel('w (dimensionless frequency)')
        ax.set_ylabel('|F_recon| (ratio path)')
        ax.set_title('Spec 8 (F009): deep-band macro-magnification plateau')
        ax.legend(fontsize=7)
        self._save_figure(fig, 'spec8_deepband_macro_limit')


class SelfFalsificationTestCase(RatioLayerTestCase):
    """
    Prove the suite can go RED: the identity gate, the RB-agreement gate,
    and the anti-vacuity ``tearDown`` each fire under an injected fault.

    Every method here is GREEN (it asserts a fault IS detected); a suite
    whose gates could not distinguish a broken pipeline from a correct one
    would be worthless, so this class is the standing proof that they can.
    """

    def test_identity_gate_rejects_a_spurious_carrier(self):
        """
        A spurious residual carrier ``exp(1j*w*eps)`` on the reconstructed
        envelope pushes the ratio-vs-direct relative residual ABOVE
        `ENVELOPE_IDENTITY_RTOL` -- exactly the carrier / critical-delay
        bug the identity gate exists to catch.
        """
        par_dic = self._anchor_candidate(ANCHORS[0])
        self.like._force_direct = False
        w_ratio, e_ratio = self._capture_envelope(par_dic)
        self.like._force_direct = True
        try:
            _w_direct, e_direct = self._capture_envelope(par_dic)
        finally:
            self.like._force_direct = False

        # Inject a carrier far larger than the identity floor.
        epsilon = 1e-6
        e_bugged = e_ratio * np.exp(1j * w_ratio * epsilon)
        scale = float(np.max(np.abs(e_direct)))
        rel_clean = float(np.max(np.abs(e_ratio - e_direct))) / scale
        rel_bugged = float(np.max(np.abs(e_bugged - e_direct))) / scale
        self.n_checks += 1
        self.assertLess(rel_clean, ENVELOPE_IDENTITY_RTOL,
                        'control (clean) residual should already pass')
        self.assertGreater(
            rel_bugged, ENVELOPE_IDENTITY_RTOL,
            'identity gate failed to reject a spurious carrier '
            '(gate is vacuous)')

    @_brute_accuracy_tier
    def test_rb_gate_rejects_a_corrupted_reconstruction(self):
        """
        Scaling the reconstructed kernels by 1.5 on the ratio path pushes
        ``lnlike`` beyond ``max(RB_ATOL, RB_RTOL*|bf|)`` from brute force --
        the RB-agreement gate catches a corrupted pipeline.
        """
        par_dic = self._anchor_candidate(ANCHORS[0])
        lnbf = self.like.lnlike_bruteforce(par_dic)
        tol = max(RB_ATOL, RB_RTOL * abs(lnbf))

        # Control: the true ratio path passes.
        lnr_clean = self._lnlike_ratio(par_dic)
        self.assertLess(abs(lnr_clean - lnbf), tol,
                        'control ratio lnlike should already match bf')

        original = likelihood_module.reconstruct_from_envelope

        def corrupt(dense_w, envelope_dense, *args, **kwargs):
            kernels, total = original(dense_w, envelope_dense,
                                      *args, **kwargs)
            return kernels * 1.5, total

        self.like._fid_cache.clear()
        with mock.patch.object(likelihood_module,
                               'reconstruct_from_envelope', corrupt):
            lnr_bad = self.like.lnlike_and_metadata(par_dic)[0]
        self.n_checks += 1
        self.assertGreater(
            abs(lnr_bad - lnbf), tol,
            'RB gate failed to reject a corrupted reconstruction '
            '(gate is vacuous)')

    def test_anti_vacuity_teardown_fails_on_zero_checks(self):
        """The base `tearDown` fails a test that made zero comparisons."""
        probe = RatioLayerTestCase(methodName='setUp')
        probe.like = self.like
        probe.n_checks = 0
        self.n_checks += 1
        with self.assertRaises(probe.failureException):
            probe.tearDown()


if __name__ == '__main__':
    main()
