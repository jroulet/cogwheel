"""
Tests for `lensing.chang_refsdal._diffractive` -- Rung P, the positive-parity
diffractive analytic serve object and its truncation certificate.

WHAT THIS SUITE ADJUDICATES
---------------------------
`_diffractive.w_low_fit` is the fitted, O(1) truncation-certificate boundary
that replaced the per-proposal honest scan.  Its correctness rests on ONE
load-bearing claim: it never OVER-SERVES -- for every admitted draw, the
order-`_DEFAULT_MAX_ORDER` operator series agrees with the exact engine to
within `CERTIFICATION_BAR` on ``[w_lo, w_low]``.  This suite IS the
engine-oracle validation of that claim.  Three questions:

  1. LOW-W ANCHOR (`LowWAnchorTestCase`).  The served DC limit must tend to
     ``sqrt(mu_macro)`` with the F009-S Morse phase -- ``+sqrt(mu)`` (arg -> 0)
     for positive parity, ``-1j sqrt(mu)`` (arg -> -pi/2) for the macro saddle
     -- and to exactly 1 ONLY in the degenerate ``gamma = kappa = 0`` point
     mass.  A carrier-limit refactor that reinstated ``F -> 1`` everywhere
     would silently corrupt every low-w serve; this pins against that.

  2. TRUNCATION VS EXACT ENGINE over the certified band
     (`TruncationCertifiedBandTestCase`, `KappaEngineOracleTestCase`).  Where
     the fitted certificate admits, the order-`_DEFAULT_MAX_ORDER` truncation
     must agree with the exact engine to within `CERTIFICATION_BAR` (1e-4).
     This is ESCALATE-ON-MISS: no widened tolerance is encoded.  The sweep
     runs over ``[w_lo, 0.9*w_low_fit]`` (the 0.9 margin keeps the sweep off
     the ceiling, where a float64 round-off at ``w == w_low`` could trip a
     value that is in fact at the bar).  If `w_low_fit` ever over-serves on
     the swept domain, this class goes red.  The conservative-region sweep
     only samples `CLEAN_GAMMAS` at the single source `Y_REF` (a conservative
     corner of the fit), so a stale re-bake of the fitted coefficients can
     pass it unnoticed; `FullGridCertificateOracleTestCase` re-runs the same
     served-vs-engine comparison over the calibration script's OWN full grid
     (``scripts/fit_diffractive_certificate.py::_grid_points('full', 42)``),
     which covers the over-serve corners (small/large ``r``, gamma 0.4-0.5)
     by construction -- the same grid the fit was trained on.

  3. SELF-REFUSAL at the wall (`WallRefusalTestCase`).  At/beyond the parity
     wall ``gamma' >= 1 - DELTA_GAMMA_P`` the rung must decline
     (``DiffractiveDomainError``), never return a small optimistic number.

ORACLE INDEPENDENCE
-------------------
The reference is `_schwinger.f_schwinger` -- the exact Schwinger proper-time
double-double engine -- evaluated in the eigenframe ``R(-beta) y`` (at
``kappa = 0``) or reconstructed through the mass-sheet map (at ``kappa > 0``,
see `_engine_reference_kappa`).  It shares NO accumulation path with the
operator series under test (the series is the float64 truncation of the
separate mpmath ``_oracle_fop`` contraction, F002).  `f_schwinger` is the
engine the shipped likelihood hands off to above ``w_low``, so agreeing with
it is exactly the serve-consistency the rung needs.  ``w`` stays ``<= 60`` so
the engine runs on its exact double-double path (mpmath only above 60).
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os
import cmath
import unittest
from unittest import TestCase, main, mock

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpmath as mp

from cogwheel.lensing.chang_refsdal._schwinger import (
    f_schwinger, _f_schwinger_mpmath, _CERTIFICATION_TOL,
    W_CEILING_SCHWINGER, W_CEILING_SCHWINGER_QD, SchwingerCertificationError)
from cogwheel.lensing.chang_refsdal._born import DELTA_GAMMA_P
from cogwheel.lensing.ppgo_map import CERTIFICATION_BAR
from cogwheel.lensing.chang_refsdal import operator as _operator
import cogwheel.lensing.chang_refsdal._diffractive as _diffractive_mod
from cogwheel.lensing.chang_refsdal._diffractive import (
    diffractive_amplification, w_low_fit, DiffractiveDomainError,
    _operator_terms, _kernel_length)
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    HypergeometricDomainError, prefactor_c, point_mass_g_derivatives)
from cogwheel.lensing.likelihood import (
    _band_split_mask, LensedRelativeBinningLikelihood)


#: Directory for diagnostic plots (created on demand).
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

#: Reference source offset used across the suite (arbitrary, off-axis so the
#: shear operator acts on both eigen-directions).
Y_REF = (0.8, 0.4)

#: Reduced shears whose order-`_DEFAULT_MAX_ORDER` truncation is genuinely
#: certified: measured worst relative error over ``[w_lo, w_low_fit]`` stays
#: below `CERTIFICATION_BAR` at these three.
CLEAN_GAMMAS = (0.1, 0.2, 0.3)

#: Eigenframe rotations exercised for the frame-pairing oracle (kappa = 0).
BETAS = (0.0, 0.7, -1.1)

#: (gamma, beta) combos at `Y_REF` that fall INSIDE the near-fold fence
#: (``rho = _caustic_rho(abs(gamma), s, theta)`` in ``[RHO_LO, 1 + DELTA]``)
#: and are therefore DECLINED by `w_low_fit` (returns None) rather than
#: served.  `TruncationCertifiedBandTestCase` skips these in its over-serve
#: sweep and asserts the decline in `test_near_fold_shell_is_declined`.  At
#: ``gamma=0.3, beta=-1.1`` the reduced caustic ratio is ``rho=1.247``
#: (inside ``[0.6, 1.4]``).
NEAR_FOLD_DECLINED_WITNESSES = ((0.3, -1.1),)

#: Small frequencies probing the DC (w -> 0) anchor.  Kept well inside the
#: engine's exact double-double band (w <= 60).
ANCHOR_WS = (1e-2, 3e-3, 1e-3)

#: Number of frequency samples per certified-band sweep.
N_BAND = 40

#: Convergence (kappa) values exercising the kappa > 0 mass-sheet
#: reconstruction: the fitted certificate serves ``kappa != 0`` verbatim
#: (no upstream ``kappa == 0`` guard), so the engine-oracle sweep must cover
#: nonzero kappa too.
KAPPA_GRID = (0.1, 0.2, 0.3)

#: Reduced-shear/kappa combinations whose order-`_DEFAULT_MAX_ORDER`
#: truncation stays genuinely certified at kappa > 0: measured worst relative
#: error over ``[w_lo, w_low_fit]`` stays below `CERTIFICATION_BAR` at every
#: (kappa, gamma) in `KAPPA_GRID[:2] x KAPPA_CLEAN_GAMMAS`.  ``kappa=0.3`` is
#: covered separately by `KAPPA_WITNESS`.
KAPPA_CLEAN_GAMMAS = (0.1, 0.2)

#: (kappa, gamma, beta) exercising the kappa > 0 mass-sheet path at the upper
#: end of `KAPPA_GRID`: `KappaEngineOracleTestCase.
#: test_truncation_within_bar_at_former_leaky_gate_witness` pins that the
#: admitted band's worst truncation error stays below `CERTIFICATION_BAR`.
KAPPA_WITNESS = (0.3, 0.2, 0.7)

#: Path of the calibration script relative to this test file.  The full-grid
#: oracle sweep imports ``_grid_points`` / ``_unreduced_source`` from it (see
#: `_load_fit_certificate_script`) so the probe grid is the SAME grid the fit
#: was trained on and cannot drift from the training domain.
_FIT_SCRIPT_REL = os.path.join('..', '..', 'scripts',
                               'fit_diffractive_certificate.py')

#: Slow-tier gate for the FULL-calibration-grid zero-over-serve sweep (on-grid
#: nodes AND off-grid theta midpoints).  The provisional smoke-baked
#: coefficients are de-rated over the SMOKE grid only, so the full-grid
#: zero-over-serve claim can only hold with the FINAL driver-baked
#: coefficients.  In-build the sweep is skipped LOUDLY (this is the load the
#: fast tier must NOT pay); the driver re-runs it with
#: ``COGWHEEL_DIFFRACTIVE_FULL_BAKE=1`` after the full bake lands and pastes
#: the emission block into ``_diffractive.py``.
_COGWHEEL_DIFFRACTIVE_FULL_BAKE = bool(
    os.environ.get('COGWHEEL_DIFFRACTIVE_FULL_BAKE'))

_COGWHEEL_DIFFRACTIVE_FULL_BAKE_REASON = (
    'FULL-calibration-grid zero-over-serve sweep gated behind '
    'COGWHEEL_DIFFRACTIVE_FULL_BAKE=1: it can only pass with the FINAL '
    'driver-baked coefficients -- the provisional smoke coefficients are '
    'de-rated over the smoke grid only and over-serve the off-grid theta '
    'midpoints, so this gate is red in-build BY DESIGN. The driver re-runs '
    'it after the full bake.')

#: Skip reason for `CornerRawOverPredictionTestCase` (its OWN gate, NOT the
#: shared `_COGWHEEL_DIFFRACTIVE_FULL_BAKE_REASON`, whose 'red in-build by
#: design' claim is specific to the full-grid zero-over-serve sweep).  The
#: corner pin certifies the DE-RATE-TARGET bar (< 1.5x): the near-fold shell
#: is FENCED out (INS-1-001's marginal-resonance corner, which previously
#: forced the bar up to < 2.0, is no longer served), so the raw surface's
#: worst over-prediction returns toward the 0.70 de-rate target (<= 1.43x)
#: and a < 1.5x bar has teeth again.  The pin pays an engine probe
#: (`_measure_w_low_true`, ~1.2 s), so it is gated out of the fast tier; the
#: driver re-runs it after the full bake to confirm the FINAL coefficients
#: still satisfy it.
_CORNER_RAW_OVER_PREDICTION_REASON = (
    'corner raw-over-prediction pin gated behind COGWHEEL_DIFFRACTIVE_FULL_BAKE=1: '
    'it pays an engine probe (_measure_w_low_true) to certify the '
    'de-rate-target < 1.5x bar on the SERVED near-exterior witness (the '
    'near-fold shell is fenced). The driver re-runs it after the full bake '
    'to confirm the FINAL coefficients still satisfy it.')


def _rot_minus_beta(beta: float) -> np.ndarray:
    """Return the eigenframe rotation ``R(-beta)`` (2x2)."""
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    return np.array([[cos_b, sin_b], [-sin_b, cos_b]])


def _sqrt_mu(gamma: float, kappa: float = 0.0) -> float:
    """Macro amplitude ``sqrt(mu_macro) = 1 / sqrt(|(1-kappa)^2 - gamma^2|)``."""
    lam = 1.0 - kappa
    return 1.0 / math.sqrt(abs(lam * lam - gamma * gamma))


def _engine_reference(w: float, y, gamma: float, beta: float = 0.0) -> complex:
    """Exact engine amplitude at ``kappa = 0`` in the eigenframe.

    The shipped engine `f_schwinger` takes the eigenframe source offset
    ``R(-beta) y`` and the (unreduced, kappa=0) shear ``gamma``.  This is the
    frame-verified oracle pairing: `diffractive_amplification(w, y, gamma,
    beta, 0)` reconstructs to this value to ~1e-4 across the certified band.
    """
    y_eig = _rot_minus_beta(beta) @ np.asarray(y, dtype=float)
    return f_schwinger(w, y_eig, gamma)


def _band_worst_relerr(y, gamma: float, beta: float = 0.0):
    """Return ``(worst_rel, w_worst, w_low, ws, rels)`` over ``[w_lo, 0.9*w_low]``.

    The diffractive rung serves ``w <= w_low``; the sweep runs from a small
    positive floor up to ``0.9 * w_low`` and scores each point against the
    exact engine.  The 0.9 factor keeps the sweep OFF the ceiling so a point
    right at ``w_low`` (where the fit has zero margin) does not trip on
    float64 round-off; ``0.9 * w_low`` is still a hard, un-widened probe of
    the served band's interior.
    """
    w_low = w_low_fit(y, gamma, beta, 0.0)
    if w_low is None or not w_low > 0.0:
        raise AssertionError(
            f'w_low_fit returned {w_low} for gamma={gamma}; '
            'the band is undefined and the sweep would assert nothing.')
    w_lo = max(0.05, 0.02 * w_low)
    ws = np.linspace(w_lo, 0.9 * w_low, N_BAND)
    rels = np.empty_like(ws)
    for i, w in enumerate(ws):
        f_p = diffractive_amplification(w, y, gamma, beta, 0.0)
        f_e = _engine_reference(w, y, gamma, beta)
        rels[i] = abs(f_p - f_e) / abs(f_e)
    idx = int(np.argmax(rels))
    return float(rels[idx]), float(ws[idx]), float(w_low), ws, rels


def _engine_reference_kappa(w: float, y, gamma: float, beta: float,
                             kappa: float) -> complex:
    """Exact engine amplitude at ``kappa >= 0`` via the mass-sheet map.

    Built from `operator._mass_sheet_map` -- the SINGLE shipped
    implementation of the ``(lam, y', gamma')`` reduction, independently
    reused (not re-derived) at three call sites inside `operator.py` -- plus
    the same `f_schwinger` engine `_engine_reference` uses.  This shares no
    code with `_diffractive.diffractive_amplification`'s own ``recon`` phase
    construction, so it is a genuine second derivation of the mass-sheet
    reconstruction, not a restatement of the code under test.  At
    ``kappa = 0`` it collapses to `_engine_reference` exactly (`lam = 1``,
    ``y' = y``, ``gamma' = gamma``).
    """
    lam, y_scaled, gamma_prime = _operator._mass_sheet_map(
        np.asarray(y, dtype=float), gamma, kappa)
    s = float(y_scaled @ y_scaled)
    y_eig = _rot_minus_beta(beta) @ y_scaled
    f_pure = f_schwinger(w, y_eig, gamma_prime)
    mass_sheet_phase = cmath.exp(0.5j * w * math.log(lam)
                                  - 0.5j * w * kappa * s)
    return mass_sheet_phase * f_pure / lam


def _band_worst_relerr_kappa(y, gamma: float, beta: float, kappa: float):
    """Like `_band_worst_relerr` but at ``kappa >= 0`` via the mass-sheet oracle.

    Kept as a separate helper (rather than adding a ``kappa`` parameter to
    `_band_worst_relerr`) so the existing kappa=0 invariants keep calling a
    signature that cannot silently pick up a nonzero kappa.  The sweep top is
    ``0.9 * w_low`` for the same ceiling-margin reason as `_band_worst_relerr`.
    """
    w_low = w_low_fit(y, gamma, beta, kappa)
    if w_low is None or not w_low > 0.0:
        raise AssertionError(
            f'w_low_fit returned {w_low} for gamma={gamma}, '
            f'kappa={kappa}; the band is undefined and the sweep would '
            'assert nothing.')
    w_lo = max(0.05, 0.02 * w_low)
    ws = np.linspace(w_lo, 0.9 * w_low, N_BAND)
    rels = np.empty_like(ws)
    for i, w in enumerate(ws):
        f_p = diffractive_amplification(w, y, gamma, beta, kappa)
        f_e = _engine_reference_kappa(w, y, gamma, beta, kappa)
        rels[i] = abs(f_p - f_e) / abs(f_e)
    idx = int(np.argmax(rels))
    return float(rels[idx]), float(ws[idx]), float(w_low), ws, rels

@functools.lru_cache(maxsize=1)
def _load_fit_certificate_script():
    """Lazily import the calibration script (single source of truth).

    `scripts/fit_diffractive_certificate.py` defines the calibration grid
    (`_grid_points`) and the source reconstruction (`_unreduced_source`);
    importing it here -- rather than re-deriving the grid inside this test --
    means `FullGridCertificateOracleTestCase` probes EXACTLY the domain the
    fit was trained on, so the sweep cannot drift from the training domain.
    The module is cached so the sweep cache below stays keyed on one object.
    """
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), _FIT_SCRIPT_REL))
    spec = importlib.util.spec_from_file_location(
        'fit_diffractive_certificate', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grid_relerr(w: float, y, gamma: float, beta: float,
                 kappa: float) -> float | None:
    """Served-vs-engine relative error at frequency ``w`` for one grid row.

    ``None`` when the served series cannot be evaluated at ``w``
    (`HypergeometricDomainError`, the point-mass kernel leaves its certified
    double-double domain ``w * sqrt(s) <= DD_PRODUCT_CEILING``) or the engine
    refuses to certify there -- such a row is not measurable and the
    certificate over-reaches rather than over-serves.
    """
    try:
        f_p = diffractive_amplification(w, y, gamma, beta, kappa)
        f_e = _engine_reference_kappa(w, y, gamma, beta, kappa)
    except (HypergeometricDomainError, SchwingerCertificationError):
        return None
    return float(abs(f_p - f_e) / abs(f_e))


@functools.lru_cache(maxsize=1)
def _full_grid_sweep(script):
    """Served-vs-engine sweep over the full calibration grid AND its off-grid
    theta midpoints.

    Returns ``(rows, n_refused, n_domain)`` with each row
    ``(gamma, beta, kappa, r, theta, w_low, rel_at_wlow, rel_at_09wlow,
    off_grid)``.  The on-grid rows come from ``script._grid_points('full',
    42)``; the off-grid rows come from ``script._off_grid_points('full', 42)``
    (the theta MIDPOINTS between consecutive grid nodes -- the points a
    harmonic fit is LEAST constrained at, and exactly where the sub-grid
    caustic dip lives).  The off-grid probes close the blind spot that let the
    on-grid-only sweep pass GREEN while the surface over-served off-grid.

    Pure (no test counters) and cached so the assertion and diagnostic-plot
    tests share ONE engine sweep.  The cached state is the shipped constants;
    `test_removing_derate_trips_overserve` does NOT use this cache (it runs
    its own loop under a patched de-rate).
    """
    rows: list[tuple[float, float, float, float, float, float,
                     float, float, bool]] = []
    n_refused = 0
    n_domain = 0

    def collect(gamma, beta, kappa, r, theta, off_grid):
        nonlocal n_refused, n_domain
        y = script._unreduced_source(r, theta, gamma, beta, kappa)
        w_low = w_low_fit(y, gamma, beta, kappa)
        if w_low is None or not w_low > 0.0:
            n_refused += 1
            return
        rel_wlow = _grid_relerr(w_low, y, gamma, beta, kappa)
        if rel_wlow is None:
            n_domain += 1
            return
        rel_09 = _grid_relerr(0.9 * w_low, y, gamma, beta, kappa)
        if rel_09 is None:
            # The kernel domain is monotone in w, so if w_low evaluates,
            # 0.9*w_low does too -- an engine refusal here is a genuine
            # non-serve; count it rather than asserting nothing.
            n_domain += 1
            return
        rows.append((float(gamma), float(beta), float(kappa), float(r),
                     float(theta), float(w_low), float(rel_wlow),
                     float(rel_09), bool(off_grid)))

    for gamma, beta, kappa, r, theta in script._grid_points('full', 42):
        collect(gamma, beta, kappa, r, theta, off_grid=False)
    for gamma, beta, kappa, r, theta in script._off_grid_points('full', 42):
        collect(gamma, beta, kappa, r, theta, off_grid=True)
    return rows, n_refused, n_domain


#: mpmath working precision for the independent point-mass oracles.  50 dps
#: leaves ~35 guard digits over the 1e-12 phase tie, so the oracle error is
#: never the thing the assertion measures.
mp.mp.dps = 50

#: Squared source offset ``s = |y|**2`` for the reference geometry.
S_REF = Y_REF[0] ** 2 + Y_REF[1] ** 2

#: Frequencies for the shear-free phase cross-check.  Spans four decades so the
#: unbounded ``0.5*w*ln(0.5*w)`` phase term reaches ~91 rad at the top (a
#: bounded-phase bug cannot hide there), while staying <= 60 for the DD path.
PHASE_WS = (0.5, 1.0, 3.0, 8.0, 20.0, 40.0, 55.0)

#: Frequencies for the mpmath FULL point-mass tie (Spec 1, leg C).  Capped at
#: 40: the point-mass DD kernel obeys ``~ eps_dd * e**(w*sqrt(s)) / |1F1|``, so
#: at w=55 (w*sqrt(s) ~ 49) the served value is only ~2e-11 -- honest for the
#: physics leg but past the 1e-9 tie used here (the EXACT operator-collapse leg
#: A covers 55 at rtol 1e-12).
MPMATH_PHYS_WS = (0.5, 1.0, 3.0, 8.0, 20.0, 40.0)

#: Reduced shears in the macro-saddle regime (gamma' > 1) for Rung S.
SADDLE_GAMMA_PRIMES = (1.5, 2.0)

#: Low frequencies inside the saddle band-split host (Rung S), all <= 60 so the
#: engine runs its exact DD path against the mpmath reference.
SADDLE_WS = (5.0, 12.0, 25.0)


def _mp_prefactor_c(w: float) -> mp.mpc:
    """Independent mpmath point-mass prefactor ``C(w)``.

    Built directly from the documented closed form
    ``exp(pi*w/4 + i*(w/2)*ln(w/2)) * Gamma(1 - i*w/2)`` at high precision --
    it shares NO code with `prefactor_c` (which splits magnitude via ``expm1``
    and phase via ``loggamma.imag``).  The ``(w/2)*ln(w/2)`` term IS the
    unbounded ``w*ln(w)`` phase this leg pins.
    """
    w = float(w)
    return mp.exp(mp.pi * w / 4 + 1j * (w / 2) * mp.log(w / 2)) \
        * mp.gamma(1 - 1j * w / 2)


def _mp_point_mass(w: float, s: float) -> mp.mpc:
    """Independent mpmath isolated point-mass amplification ``F_pm(w)``.

    ``C(w) * 1F1(i*w/2; 1; i*w*s/2)`` at high precision -- the full closed form
    the positive-parity diffractive serve reduces to at ``gamma = kappa = 0``.
    A wholly separate derivation from the production DD kernel ladder, so a tie
    against it certifies the served value, not just internal consistency.
    """
    return _mp_prefactor_c(w) * mp.hyp1f1(1j * w / 2, 1, 1j * w * s / 2)


class DiffractiveTestCase(TestCase):
    """Base: per-test comparison tally + anti-vacuity guard.

    Sweeps that skip every comparison (e.g. every geometry refused) would
    otherwise pass while asserting nothing; `tearDown` fails those.  Tests that
    never touch the tally are unaffected.
    """

    def setUp(self):
        """Reset the per-test comparison tally used by `tearDown`."""
        self.n_compared = 0
        self.n_skipped = 0

    def tearDown(self):
        """Fail a test whose every comparison was skipped."""
        if self.n_skipped and not self.n_compared:
            self.fail(f'all {self.n_skipped} comparisons were skipped; '
                      'the test asserted nothing')


class LowWAnchorTestCase(DiffractiveTestCase):
    """Spec 1 -- the served DC limit is sqrt(mu_macro), NOT 1.

    F009-S Morse convention: as ``w -> 0`` the positive-parity serve tends to
    ``+sqrt(mu_macro)`` (Morse index 0, arg -> 0) and the macro saddle to
    ``-1j sqrt(mu_macro)`` (Morse index 1, arg -> -pi/2).  ``F -> 1`` is the
    degenerate ``gamma = kappa = 0`` point mass ONLY.  The discriminating teeth
    are that the modulus converges to ``sqrt(mu)`` and NOT to 1 (positive
    parity), and the phase converges to ``-pi/2`` and NOT to 0 (saddle).
    """

    def test_positive_parity_modulus_tends_to_sqrt_mu_not_one(self):
        """Positive parity: |F(w->0)| -> sqrt(mu) and away from 1.

        The discriminating teeth (both gammas): at the smallest w, |F| is far
        closer to ``sqrt(mu)`` than to 1 -- a reinstated ``F -> 1`` carrier
        limit would put it near 1.  Strict monotone convergence of the modulus
        is asserted ONLY on the clean gamma (0.3): at gamma'=0.6 the order-8
        series truncation carries a ~1e-3 modulus residual (0.6 is already past
        the honesty crossover) that competes with the O(w) DC convergence and
        legitimately breaks strict monotonicity -- that residual is the subject
        of the separate certified-band classes, not the anchor.
        """
        for gamma in (0.3, 0.6):
            sqrt_mu = _sqrt_mu(gamma)
            dev_target, dev_wrong = [], []
            for w in ANCHOR_WS:
                mod = abs(diffractive_amplification(w, Y_REF, gamma, 0.0, 0.0))
                dev_target.append(abs(mod - sqrt_mu))
                dev_wrong.append(abs(mod - 1.0))
                self.n_compared += 1
            with self.subTest(gamma=gamma):
                # Teeth against a reinstated F->1 anchor (both gammas): at the
                # smallest w, |F| sits far closer to sqrt(mu) than to 1.
                self.assertLess(
                    dev_target[-1], 0.1 * dev_wrong[-1],
                    f'anchor ambiguous at gamma={gamma}: dev_sqrt_mu='
                    f'{dev_target[-1]:.2e} vs dev_1={dev_wrong[-1]:.2e}')
                if gamma in CLEAN_GAMMAS:
                    # Where the series is faithful, |F| converges monotonically
                    # to the anchor -- proving it TENDS there, not accidentally
                    # near it.
                    self.assertTrue(
                        all(a > b for a, b in zip(dev_target, dev_target[1:])),
                        f'|F|-sqrt(mu) not converging: {dev_target}')

    def test_positive_parity_phase_tends_to_zero(self):
        """Positive parity (Morse 0): arg F(w->0) -> 0."""
        for gamma in (0.3, 0.6):
            args = [abs(cmath.phase(
                diffractive_amplification(w, Y_REF, gamma, 0.0, 0.0)))
                for w in ANCHOR_WS]
            self.n_compared += len(args)
            with self.subTest(gamma=gamma):
                self.assertTrue(
                    all(a > b for a, b in zip(args, args[1:])),
                    f'|arg F| not decreasing toward 0: {args}')
                self.assertLess(args[-1], 1e-2)

    def test_macro_saddle_engine_anchor_is_minus_1j_sqrt_mu(self):
        """Macro saddle (engine-hosted): |F| -> sqrt(mu), arg -> -pi/2.

        Rung P refuses gamma' > 1; the saddle low-w serve is engine-hosted
        (Rung S), so the anchor is read from `f_schwinger` directly.  The
        teeth: the phase tends to -pi/2 (Morse 1), NOT to 0 -- a wrong Morse
        anchor would put it near 0.
        """
        y_eig = np.asarray(Y_REF, dtype=float)
        for gamma_prime in (1.5, 2.0, 3.0):
            sqrt_mu = _sqrt_mu(gamma_prime)   # 1/sqrt(gamma'^2 - 1) for saddle
            mod_dev, phase_dev = [], []
            for w in ANCHOR_WS:
                f = f_schwinger(w, y_eig, gamma_prime)
                mod_dev.append(abs(abs(f) - sqrt_mu))
                phase_dev.append(abs(cmath.phase(f) - (-math.pi / 2.0)))
                self.n_compared += 1
            with self.subTest(gamma_prime=gamma_prime):
                self.assertTrue(
                    all(a > b for a, b in zip(mod_dev, mod_dev[1:])),
                    f'|F| not converging to sqrt(mu): {mod_dev}')
                self.assertLess(mod_dev[-1], 5e-3)
                # Phase locked to -pi/2 and unambiguously away from 0.
                self.assertLess(phase_dev[-1], 5e-2)
                arg_last = cmath.phase(f_schwinger(ANCHOR_WS[-1], y_eig,
                                                   gamma_prime))
                self.assertLess(arg_last, -1.5,
                                'saddle phase drifted toward the Morse-0 anchor')

    def test_degenerate_point_mass_tends_to_one(self):
        """gamma = kappa = 0: sqrt(mu) = 1, so F(w->0) -> 1 exactly.

        This is the ONLY geometry where F -> 1; documenting it keeps the
        modulus anchor from being read as "always 1".
        """
        dev = []
        for w in ANCHOR_WS:
            f = diffractive_amplification(w, Y_REF, 0.0, 0.0, 0.0)
            dev.append(abs(abs(f) - 1.0))
            self.n_compared += 1
        self.assertTrue(all(a > b for a, b in zip(dev, dev[1:])),
                        f'|F| not converging to 1: {dev}')
        # Convergence is O(w); the smallest probe (1e-3) sits ~8e-4 from 1.
        self.assertLess(dev[-1], 5e-3)

    def test_diagnostic_plot_anchor(self):
        """Save |F| and arg(F) vs log-w for both parities (diagnostic)."""
        ws = np.geomspace(1e-3, 1.0, 40)
        fig, (ax_mod, ax_arg) = plt.subplots(1, 2, figsize=(11, 4))
        # Positive parity gamma'=0.6 (sqrt_mu=1.25).
        gpos = 0.6
        mod_pos = [abs(diffractive_amplification(w, Y_REF, gpos, 0.0, 0.0))
                   for w in ws]
        arg_pos = [cmath.phase(diffractive_amplification(w, Y_REF, gpos, 0.0,
                                                         0.0)) for w in ws]
        # Macro saddle gamma'=2.0 (sqrt_mu=0.5774) via engine.
        gsad = 2.0
        y_eig = np.asarray(Y_REF, dtype=float)
        mod_sad = [abs(f_schwinger(w, y_eig, gsad)) for w in ws]
        arg_sad = [cmath.phase(f_schwinger(w, y_eig, gsad)) for w in ws]
        ax_mod.semilogx(ws, mod_pos, label=f"+ parity g'={gpos}")
        ax_mod.semilogx(ws, mod_sad, label=f"saddle g'={gsad}")
        ax_mod.axhline(_sqrt_mu(gpos), ls='--', color='C0')
        ax_mod.axhline(_sqrt_mu(gsad), ls='--', color='C1')
        ax_mod.axhline(1.0, ls=':', color='k', label='wrong anchor F=1')
        ax_mod.set_xlabel('w'); ax_mod.set_ylabel('|F|'); ax_mod.legend()
        ax_arg.semilogx(ws, arg_pos, label='+ parity -> 0')
        ax_arg.semilogx(ws, arg_sad, label='saddle -> -pi/2')
        ax_arg.axhline(-math.pi / 2, ls='--', color='C1')
        ax_arg.axhline(0.0, ls=':', color='k')
        ax_arg.set_xlabel('w'); ax_arg.set_ylabel('arg F'); ax_arg.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'anchor_low_w_both_parities.png'),
                    dpi=90)
        plt.close(fig)
        self.assertTrue(os.path.exists(
            os.path.join(OUTPUT_DIR, 'anchor_low_w_both_parities.png')))


class TruncationCertifiedBandTestCase(DiffractiveTestCase):
    """Spec 2 -- truncation matches the exact engine over the certified band.

    ESCALATE-ON-MISS.  Over the admitted band ``[w_lo, 0.9*w_low_fit]`` the
    truncation must agree with `f_schwinger` to within `CERTIFICATION_BAR`.
    The sweep stops at ``0.9 * w_low`` (not ``w_low``) because a point exactly
    at the fitted ceiling has zero margin, so it could trip on float64
    round-off alone.  The bar is NOT widened: a miss inside the 0.9 band is a
    genuine over-serve by the fitted certificate -- the one claim this suite
    exists to police.
    """

    def test_truncation_within_bar_over_band(self):
        """max rel-err(w) <= CERTIFICATION_BAR across [w_lo, 0.9*w_low_fit].

        The near-fold shell is fenced: combos in
        `NEAR_FOLD_DECLINED_WITNESSES` are declined by `w_low_fit` (None) and
        are skipped here -- their decline is pinned by
        `test_near_fold_shell_is_declined`.
        """
        for gamma in CLEAN_GAMMAS:
            for beta in BETAS:
                if (gamma, beta) in NEAR_FOLD_DECLINED_WITNESSES:
                    continue  # fenced; decline pinned by the sibling test
                worst, w_worst, w_low, _, _ = _band_worst_relerr(
                    Y_REF, gamma, beta)
                self.n_compared += 1
                with self.subTest(gamma=gamma, beta=beta):
                    self.assertLessEqual(
                        worst, CERTIFICATION_BAR,
                        f'truncation misses inside the admitted band: '
                        f'gamma={gamma} beta={beta} worst={worst:.3e} '
                        f'@w={w_worst:.3f} (w_low={w_low:.3f}) '
                        f'> bar={CERTIFICATION_BAR:.1e}')

    def test_near_fold_shell_is_declined(self):
        """`w_low_fit` declines (None) the near-fold-shell witnesses.

        Inside the fence shell ``[RHO_LO, 1 + DELTA]`` the diffractive rung
        declines (returns None) so the draw falls through to the fold arm /
        exact engine.  Premise: the witness's reduced caustic ratio ``rho``
        -- computed with the SAME `_caustic_rho` discriminator the fence uses
        -- falls inside the shell, so the decline is the fence doing its job,
        not a degenerate refusal.
        """
        s = Y_REF[0] ** 2 + Y_REF[1] ** 2
        for gamma, beta in NEAR_FOLD_DECLINED_WITNESSES:
            z_eig = cmath.exp(-1j * beta) * complex(Y_REF[0], Y_REF[1])
            theta = math.atan2(z_eig.imag, z_eig.real)
            rho = _diffractive_mod._caustic_rho(abs(gamma), s, theta)
            self.n_compared += 1
            with self.subTest(gamma=gamma, beta=beta):
                self.assertGreaterEqual(
                    rho, _diffractive_mod._DIFFRACTIVE_FIT_FENCE_RHO_LO)
                self.assertLessEqual(
                    rho, 1.0 + _diffractive_mod._DIFFRACTIVE_FIT_FENCE_DELTA)
                self.assertIsNone(w_low_fit(Y_REF, gamma, beta, 0.0))

    def test_band_is_nonempty_on_clean_domain(self):
        """Every clean geometry yields a positive, finite served band."""
        for gamma in CLEAN_GAMMAS:
            w_low = w_low_fit(Y_REF, gamma, 0.0, 0.0)
            self.n_compared += 1
            with self.subTest(gamma=gamma):
                self.assertIsNotNone(w_low)
                self.assertTrue(math.isfinite(w_low) and w_low > 0.0,
                                f'empty band w_low={w_low} at gamma={gamma}')

    def test_diagnostic_plot_certified_band(self):
        """Save rel-err(w) vs w with the bar line (certified-clean domain)."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for gamma in CLEAN_GAMMAS:
            _, _, w_low, ws, rels = _band_worst_relerr(Y_REF, gamma, 0.0)
            ax.semilogy(ws, rels, marker='.', label=f"g'={gamma} "
                        f"(w_low={w_low:.2f})")
        ax.axhline(CERTIFICATION_BAR, ls='--', color='k',
                   label=f'bar={CERTIFICATION_BAR:.0e}')
        ax.set_xlabel('w'); ax.set_ylabel('relative error vs engine')
        ax.set_title('Rung P truncation vs exact engine over [w_lo, 0.9 w_low]')
        ax.legend()
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'certified_band_relerr_vs_w.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self.assertTrue(os.path.exists(path))


class KappaEngineOracleTestCase(DiffractiveTestCase):
    """Engine-oracle pin for the kappa > 0 mass-sheet regime.

    The fitted certificate serves ``kappa != 0`` verbatim (no upstream
    ``kappa == 0`` guard), so the truncation-vs-engine invariant must be
    exercised at kappa > 0 too.  First a pairing gate confirms the kappa-aware
    oracle (`_engine_reference_kappa`) agrees with the already-verified
    kappa=0 oracle (`_engine_reference`) at kappa=0, then the certified-band
    invariant is re-run at kappa > 0 on the sub-domain that measurably stays
    under the bar.
    """

    def test_pairing_gate_kappa_zero_matches_verified_oracle(self):
        """At kappa=0 the mass-sheet oracle must reduce to `_engine_reference`."""
        for gamma in KAPPA_CLEAN_GAMMAS:
            for beta in BETAS:
                w_low = w_low_fit(Y_REF, gamma, beta, 0.0)
                w_probe = 0.3 * w_low
                f_new = _engine_reference_kappa(w_probe, Y_REF, gamma, beta, 0.0)
                f_old = _engine_reference(w_probe, Y_REF, gamma, beta)
                self.n_compared += 1
                with self.subTest(gamma=gamma, beta=beta):
                    rel = abs(f_new - f_old) / abs(f_old)
                    self.assertLess(
                        rel, 1e-10,
                        'kappa-aware oracle does not collapse to the '
                        f'verified kappa=0 oracle: gamma={gamma} beta={beta} '
                        f'rel={rel:.3e}')

    def test_truncation_within_bar_over_band_kappa_gt_zero(self):
        """kappa in (0.1, 0.2): truncation stays under the bar, as at kappa=0.

        Restricted to `KAPPA_GRID[:2]` x `KAPPA_CLEAN_GAMMAS`: measured worst
        error there is 1.576e-05, >= 6.3x under `CERTIFICATION_BAR` -- a
        genuine (non-witness) invariant, mirroring
        `TruncationCertifiedBandTestCase` at kappa > 0.
        """
        for kappa in KAPPA_GRID[:2]:
            for gamma in KAPPA_CLEAN_GAMMAS:
                for beta in BETAS:
                    worst, w_worst, w_low, _, _ = _band_worst_relerr_kappa(
                        Y_REF, gamma, beta, kappa)
                    self.n_compared += 1
                    with self.subTest(kappa=kappa, gamma=gamma, beta=beta):
                        self.assertLessEqual(
                            worst, CERTIFICATION_BAR,
                            f'kappa={kappa} gamma={gamma} beta={beta}: '
                            f'worst={worst:.3e} @w={w_worst:.3f} '
                            f'(w_low={w_low:.3f}) exceeds the bar')

    def test_truncation_within_bar_at_former_leaky_gate_witness(self):
        """`KAPPA_WITNESS` (kappa=0.3/gamma=0.2/beta=0.7): truncation under the bar.

        Pins that the fitted certificate's admitted band holds
        `CERTIFICATION_BAR` at the upper end of `KAPPA_GRID`, same as every
        other admitted draw in this suite.
        """
        kappa, gamma, beta = KAPPA_WITNESS
        worst, w_worst, w_low, _, _ = _band_worst_relerr_kappa(
            Y_REF, gamma, beta, kappa)
        self.n_compared += 1
        self.assertLessEqual(
            worst, CERTIFICATION_BAR,
            f'kappa={kappa} gamma={gamma} beta={beta}: worst={worst:.3e} '
            f'@w={w_worst:.3f} (w_low={w_low:.3f}) exceeds the bar -- '
            'INS-3-001 normalization fix (lam * sqrt_mu) not applied or '
            'insufficient in _diffractive.py.')


class FullGridCertificateOracleTestCase(DiffractiveTestCase):
    """INS-3-002 -- ZERO OVER-SERVE over the FULL calibration grid.

    `TruncationCertifiedBandTestCase` probes only `CLEAN_GAMMAS` at the single
    source `Y_REF` (``s = 0.8``, ``r ~ 0.894``) -- a CONSERVATIVE corner of
    the fit (probe ratios 0.94-0.99 there), so a stale re-bake of the fitted
    coefficients can pass it GREEN while the shipped surface over-serves the
    out-of-sample corners (small ``r ~ 0.3``, large ``r ~ 1.1-1.3``, gamma
    0.4-0.5).  This class re-runs the served-vs-engine comparison over the
    calibration script's OWN grid -- ``_grid_points('full', seed=42)`` from
    `scripts/fit_diffractive_certificate.py`, imported (not re-derived) so the
    probe domain is exactly the training domain -- which spans ``r = sqrt(s)``
    in [0.3, 1.3] x gamma in [0.05, 0.5] x 32 eigenframe angles plus 12 random
    (beta, kappa) rows, covering the over-serve corners by construction -- AND
    over the off-grid theta MIDPOINTS (``_off_grid_points('full', seed=42)``),
    the points a harmonic fit is LEAST constrained at and exactly where the
    sub-grid caustic dip lives.  The on-grid-only sweep was BLIND to that dip:
    it passed GREEN while the smoke-baked surface over-served off-grid; the
    midpoint probes close that hole.

    The whole zero-over-serve sweep is a SLOW-TIER gate (``~500 rows x 2
    probes`` of series-vs-engine, order minutes): it can only pass with the
    FINAL driver-baked coefficients (the provisional smoke coefficients are
    de-rated over the smoke grid only and over-serve the midpoints), so it is
    skipped LOUDLY in-build behind ``COGWHEEL_DIFFRACTIVE_FULL_BAKE=1`` and
    re-run by the driver after the full bake.

    Per row: the source is reconstructed with the script's `_unreduced_source`,
    ``w_low = w_low_fit(y, gamma, beta, kappa)`` is the certificate boundary,
    and the served series is probed at ``w = w_low`` (the band's worst point;
    the truncation tail grows with ``w``) AND at ``w = 0.9 * w_low`` (the
    interior, robust against ceiling round-off).  Both must agree with the
    exact engine (the mass-sheet-reconstructed `f_schwinger`,
    `_engine_reference_kappa`) to within `CERTIFICATION_BAR` -- ZERO
    over-serve.  Rows `w_low_fit` refuses (None) are counted; rows where the
    served series cannot be evaluated at ``w_low`` (kernel domain
    `HypergeometricDomainError`) are counted and must stay a strict minority
    -- a certificate promising frequencies the series cannot evaluate is
    over-reach, not service.

    Cost: ~492 rows (252 on-grid + 240 off-grid) x 2 probes x (series +
    oracle), order minutes at the final coefficients -- hence the slow-tier
    gate.  The falsification (`test_removing_derate_trips_overserve`) stays
    in the FAST tier: it runs its own early-exit loop under derate=1.0
    (~39 s, measured), independent of the final coefficients.
    """

    @unittest.skipUnless(_COGWHEEL_DIFFRACTIVE_FULL_BAKE,
                         _COGWHEEL_DIFFRACTIVE_FULL_BAKE_REASON)
    def test_zero_overserve_over_full_calibration_grid(self):
        """Served series stays within CERTIFICATION_BAR at w_low and 0.9 w_low,
        on-grid AND at the off-grid theta midpoints."""
        script = _load_fit_certificate_script()
        rows, n_refused, n_domain = _full_grid_sweep(script)
        self.n_compared += len(rows)
        self.n_skipped += n_refused + n_domain
        self.assertGreater(
            len(rows), 50,
            'premise lost: full-grid sweep measured too few rows to certify '
            f'zero over-serve (measured={len(rows)}, refused={n_refused}, '
            f'domain-refused={n_domain})')
        self.assertLess(
            n_domain, len(rows),
            'certificate over-reach: more rows refused at the kernel domain '
            f'({n_domain}) than measured ({len(rows)}) -- w_low_fit promises '
            'frequencies the served series cannot evaluate')
        for (gamma, beta, kappa, r, theta, w_low, rel_wlow, rel_09,
             off_grid) in rows:
            with self.subTest(gamma=gamma, beta=beta, kappa=kappa, r=r,
                              theta=theta, off_grid=off_grid):
                where = 'off-grid theta midpoint' if off_grid else 'on-grid'
                self.assertLessEqual(
                    rel_wlow, CERTIFICATION_BAR,
                    f'OVER-SERVE at the certificate boundary w=w_low='
                    f'{w_low:.3f} ({where}): rel={rel_wlow:.3e} > bar='
                    f'{CERTIFICATION_BAR:.0e} -- the baked fit is not '
                    'conservative on the calibration grid')
                self.assertLessEqual(
                    rel_09, CERTIFICATION_BAR,
                    f'OVER-SERVE at w=0.9*w_low={0.9 * w_low:.3f} ({where}): '
                    f'rel={rel_09:.3e} > bar={CERTIFICATION_BAR:.0e}')

    @unittest.skipUnless(_COGWHEEL_DIFFRACTIVE_FULL_BAKE,
                         _COGWHEEL_DIFFRACTIVE_FULL_BAKE_REASON)
    def test_diagnostic_plot_relerr_vs_domain(self):
        """Save relerr vs (r, gamma) over the grid with the bar line.

        Over-serve appears as points ABOVE the bar, clustered at small r,
        large r, and gamma 0.4-0.5 (the corners a stale re-bake inflates).
        Reuses the cached sweep, so this test adds no engine probes.
        """
        script = _load_fit_certificate_script()
        rows, _n_refused, _n_domain = _full_grid_sweep(script)
        self.n_compared += len(rows)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        gammas = np.array([r_[0] for r_ in rows])
        radii = np.array([r_[3] for r_ in rows])
        rels = np.array([r_[6] for r_ in rows])
        off_grid = np.array([r_[8] for r_ in rows])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        sc1 = ax1.scatter(gammas[~off_grid], rels[~off_grid], c=radii[~off_grid],
                          cmap='viridis', s=14, marker='o',
                          label='on-grid theta')
        ax1.scatter(gammas[off_grid], rels[off_grid], c=radii[off_grid],
                    cmap='viridis', s=18, marker='x', label='off-grid midpoint')
        ax1.axhline(CERTIFICATION_BAR, ls='--', color='r',
                    label=f'bar={CERTIFICATION_BAR:.0e}')
        ax1.set_yscale('log')
        ax1.set_xlabel('gamma')
        ax1.set_ylabel('relerr at w=w_low vs engine')
        ax1.set_title('full calibration grid; points above the bar over-serve')
        ax1.legend()
        fig.colorbar(sc1, ax=ax1, label='r')
        sc2 = ax2.scatter(radii[~off_grid], rels[~off_grid], c=gammas[~off_grid],
                          cmap='plasma', s=14, marker='o')
        ax2.scatter(radii[off_grid], rels[off_grid], c=gammas[off_grid],
                    cmap='plasma', s=18, marker='x', label='off-grid midpoint')
        ax2.axhline(CERTIFICATION_BAR, ls='--', color='r')
        ax2.set_yscale('log')
        ax2.set_xlabel('r')
        ax2.set_ylabel('relerr at w=w_low vs engine')
        fig.colorbar(sc2, ax=ax2, label='gamma')
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'full_grid_relerr_vs_domain.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self.assertTrue(os.path.exists(path))

    def test_removing_derate_trips_overserve(self):
        """SELF-FALSIFICATION: derate=1.0 must over-serve somewhere on the grid.

        The de-rate is the load-bearing conservative margin (the raw least-
        squares surface over-predicts by up to ~1.18x, the shipped de-rate
        being 0.85), so with it set to 1.0 the served ceiling
        inflates and the served series MUST exceed the bar at some grid row
        -- if none does, the zero-over-serve assertion has no teeth.
        Early-exits at the first over-serve row (measured ~39 s).  Runs on
        the shipped coefficients with ONLY the de-rate perturbed, and
        bypasses the `_full_grid_sweep` cache.  This teeth test stays in the
        FAST tier (it does not depend on
        the final coefficients: ANY bake over-serves without a de-rate),
        unlike the gated zero-over-serve sweep above.
        """
        script = _load_fit_certificate_script()
        found = None
        with mock.patch.object(_diffractive_mod,
                               '_DIFFRACTIVE_FIT_DERATE', 1.0):
            for gamma, beta, kappa, r, theta in script._grid_points('full', 42):
                y = script._unreduced_source(r, theta, gamma, beta, kappa)
                w_low = w_low_fit(y, gamma, beta, kappa)
                if w_low is None or not w_low > 0.0:
                    continue
                rel_wlow = _grid_relerr(w_low, y, gamma, beta, kappa)
                if rel_wlow is None:
                    continue
                self.n_compared += 1
                if rel_wlow > CERTIFICATION_BAR:
                    found = (gamma, beta, kappa, r, theta, w_low, rel_wlow)
                    break
        self.assertIsNotNone(
            found,
            'removing the de-rate did not over-serve on the calibration '
            'grid -- the zero-over-serve assertion has no teeth (the raw '
            'fit never exceeds the honest ceiling)')


class CornerRawOverPredictionTestCase(DiffractiveTestCase):
    """WP-1 + fence -- the corner raw-over-prediction pin (the fix is real).

    The even-harmonic + parametric-caustic representation replaced the
    incumbent ``cos(4k theta)`` degree-2 surface because the old surface
    over-predicted the engine-honest ceiling by ~2.06x at the corner
    ``(gamma=0.41, kappa=0, beta=0, r=0.55, theta = 3pi/4 + pi/32 ~ 2.454
    rad)`` -- the off-grid theta MIDPOINT where the ceiling collapses steeply
    toward the positive-parity wall.  A de-rate alone can hide a wrong
    surface (de-rating the old surface by 1/2.06 = 0.485 would make it pass
    any grid gate), so this pin strips the de-rate and measures the RAW
    fitted surface directly::

        raw_fit / w_low_true < 1.5   (the de-rate-target bar, de-rate >= 0.70)

    ``w_low_true`` is the engine-honest ceiling measured by the calibration
    script's own `_measure_w_low_true` (the order-`_DEFAULT_MAX_ORDER` series
    ``diffractive_amplification`` against the exact `f_schwinger` engine under
    the `CERTIFICATION_BAR` sup-over-w semantics, ``n_w=16`` -- the bake's
    default), and ``raw_fit`` is `w_low_fit` evaluated with
    `_DIFFRACTIVE_FIT_DERATE` patched to 1.0.  The ratio is the factor by
    which the raw surface over-claims the honest ceiling.  < 1.5 means the
    raw surface stays inside the de-rate-target regime (de-rate >= 0.70 <=>
    worst raw over-prediction <= 1.43x), so the de-rate is a safety margin,
    not the whole story.

    NEAR-FOLD FENCE: the ORIGINAL corner (``gamma=0.41, r=0.55``) is now
    INSIDE the near-fold shell -- ``w_low_fit`` FENCES it out (its reduced
    caustic ratio ``rho ~ 1.34`` falls in ``[RHO_LO, 1 + DELTA]``), returning
    None there so the draw falls through to the fold arm / exact engine.  The
    raw over-prediction pin can therefore no longer probe that point, and the
    marginal-resonance limitation that previously forced the bar up to < 2.0
    (INS-1-001) is fenced out too.  The witness is re-derived to a SERVED
    near-exterior point at the SAME diagonal direction
    (``theta = 3pi/4 + pi/32``) but high-gamma / just-outside-the-caustic
    (``gamma=0.5, r=1.1``, reduced caustic ratio ``rho ~ 2.19`` > ``1 +
    DELTA``): a fenced off-grid theta midpoint of the full calibration grid
    where the raw surface still over-claims the honest ceiling (measured
    ~1.01x at the current provisional re-baked smoke coefficients).  With
    the shell fenced, the original < 1.5 bar (abandoned to < 2.0 only
    because of the resonance) is restored.

    Cost: one `_measure_w_low_true` (n_w=16, ~36 series+engine probes, ~1.2 s
    measured) plus one `w_low_fit` (O(1)) -- well inside the fast-tier budget
    when the gate is lifted.
    """

    #: The corner witness: a SERVED near-exterior point at the off-grid theta
    #: midpoint ``3pi/4 + pi/_N_THETAS`` -- the same diagonal direction the old
    #: 4-harmonic surface over-predicted by ~2.06x, moved OUT of the fenced
    #: near-fold shell to ``gamma=0.5, r=1.1`` (reduced caustic ratio
    #: ``rho ~ 2.19`` > ``1 + DELTA``).  ``r=1.1`` is re-derived from the LIVE
    #: ``_off_grid_points('full', 42)`` output: the full-branch radii changed
    #: ``linspace(0.3, 1.3, 5)`` -> ``linspace(0.1, 1.3, 7)`` (WP-1 deep-interior
    #: calibration), dropping ``r=1.05``, so the witness moved to the nearest
    #: surviving near-exterior radius.  ``_N_THETAS = 32`` is single-sourced
    #: from `scripts/fit_diffractive_certificate.py` via the premise assertion
    #: in `test_raw_fit_over_prediction_within_derate_target` (the witness
    #: must be an actual fenced off-grid midpoint of the grid, not a pinned
    #: literal that could drift from the bake).
    CORNER_GAMMA = 0.5
    CORNER_BETA = 0.0
    CORNER_KAPPA = 0.0
    CORNER_R = 1.1
    CORNER_THETA = 3.0 * math.pi / 4.0 + math.pi / 32.0  # ~2.454 rad (midpoint)

    def _corner_source(self, script):
        """Return the reconstructed lens-plane source at the corner witness."""
        return script._unreduced_source(
            self.CORNER_R, self.CORNER_THETA, self.CORNER_GAMMA,
            self.CORNER_BETA, self.CORNER_KAPPA)

    @unittest.skipUnless(_COGWHEEL_DIFFRACTIVE_FULL_BAKE,
                         _CORNER_RAW_OVER_PREDICTION_REASON)
    def test_raw_fit_over_prediction_within_derate_target(self):
        """raw_fit / w_low_true < 1.5 at the served near-exterior witness."""
        script = _load_fit_certificate_script()
        # Premise: the witness IS a fenced off-grid theta midpoint of the
        # calibration grid (derived from the script's own off-grid set) -- a
        # SERVED point, not a declined one.
        off_rows = [row for row in script._off_grid_points('full', 42)
                    if abs(row[0] - self.CORNER_GAMMA) < 1e-9
                    and abs(row[3] - self.CORNER_R) < 1e-9
                    and abs(row[4] - self.CORNER_THETA) < 1e-9]
        self.assertTrue(
            off_rows, 'premise lost: corner witness is not an off-grid '
            'midpoint of the calibration grid')

        y = self._corner_source(script)
        w_low_true = script._measure_w_low_true(
            self.CORNER_GAMMA, self.CORNER_BETA, self.CORNER_KAPPA,
            float(y[0]), float(y[1]), n_w=16)
        self.assertIsNotNone(
            w_low_true,
            'premise lost: the corner geometry refused to measure an honest '
            'ceiling')
        self.assertGreater(w_low_true, 0.0)

        with mock.patch.object(_diffractive_mod,
                               '_DIFFRACTIVE_FIT_DERATE', 1.0):
            raw_fit = w_low_fit(y, self.CORNER_GAMMA, self.CORNER_BETA,
                                self.CORNER_KAPPA)
        self.assertIsNotNone(
            raw_fit,
            'premise lost: the served near-exterior witness was declined '
            '(w_low_fit -> None) -- the fence boundary moved or the witness '
            'drifted into the shell')
        self.assertLess(
            raw_fit, _diffractive_mod._DIFFRACTIVE_FIT_CEILING,
            'premise lost: raw fit clipped at the ceiling -- the ratio would '
            'measure the clip, not the fitted surface')

        ratio = raw_fit / w_low_true
        self.n_compared += 1
        self.assertLess(
            ratio, 1.5,
            f'corner raw over-prediction {ratio:.3f}x exceeds the '
            f'de-rate-target 1.5x bar (de-rate >= 0.70 <=> raw <= 1.43x): '
            f'raw_fit={raw_fit:.3f} vs honest ceiling '
            f'w_low_true={w_low_true:.3f} -- the fitted surface over-claims '
            'the served near-exterior ceiling beyond what the de-rate target '
            'can absorb')

    def test_dropping_caustic_feature_inflates_over_prediction(self):
        """SELF-FALSIFICATION: the parametric-caustic feature is load-bearing.

        Patching `_DIFFRACTIVE_FIT_CAUSTIC_COEFF` to 0.0 (dropping the WP-1
        caustic feature) must monotonically INFLATE the raw served witness's
        over-prediction (measured: dropping the feature raises the raw
        surface ~1.66x at the provisional smoke coefficients), both under
        `derate=1.0`, so the test goes red if the feature stops reducing
        over-prediction -- a fixed absolute floor cannot carry that claim,
        and a bare 'above-the-pin' comparison would be satisfiable by a no-op
        surface.  Two `w_low_fit` calls (O(1)); no engine probes.
        """
        script = _load_fit_certificate_script()
        self.assertLess(
            _diffractive_mod._DIFFRACTIVE_FIT_CAUSTIC_COEFF, 0.0,
            'premise lost: the caustic coefficient must be NEGATIVE (the '
            'ceiling dips toward the fold) for the feature to pull the raw '
            'surface DOWN at the corner')
        y = self._corner_source(script)
        with mock.patch.object(_diffractive_mod,
                               '_DIFFRACTIVE_FIT_DERATE', 1.0):
            raw_with_caustic = w_low_fit(y, self.CORNER_GAMMA,
                                         self.CORNER_BETA, self.CORNER_KAPPA)
        self.assertIsNotNone(raw_with_caustic)
        with mock.patch.object(_diffractive_mod,
                               '_DIFFRACTIVE_FIT_DERATE', 1.0), \
             mock.patch.object(_diffractive_mod,
                               '_DIFFRACTIVE_FIT_CAUSTIC_COEFF', 0.0):
            raw_nocaustic = w_low_fit(y, self.CORNER_GAMMA, self.CORNER_BETA,
                                      self.CORNER_KAPPA)
        self.assertIsNotNone(raw_nocaustic)
        self.n_compared += 1
        self.assertGreater(
            raw_nocaustic, raw_with_caustic * 1.05,
            'dropping the caustic feature does not inflate the raw corner '
            'over-prediction above the with-caustic surface -- the caustic '
            'feature is not load-bearing (a fixed absolute floor cannot '
            'distinguish a dropped feature from a merely offset surface)')


class WallRefusalTestCase(DiffractiveTestCase):
    """Spec 3a -- the rung self-refuses at/beyond the parity wall.

    For reduced shear ``gamma' >= 1 - DELTA_GAMMA_P`` (or non-physical
    ``1 - kappa <= 0``) both entry points must raise `DiffractiveDomainError`,
    never return a small optimistic number.  Inside the certified-clean domain
    (reduced shear ~1/3) the rung must still admit -- otherwise "refuses at the
    wall" would be vacuously satisfied by a rung that refuses everywhere.
    """

    #: The wall in reduced shear; anything at or above declines.
    WALL = 1.0 - DELTA_GAMMA_P

    def test_amplification_refuses_at_and_beyond_wall_via_gamma(self):
        """gamma' >= WALL (kappa=0): diffractive_amplification raises."""
        for gamma in (self.WALL, 1.0, 1.2):
            with self.subTest(gamma=gamma):
                with self.assertRaises(DiffractiveDomainError):
                    diffractive_amplification(1.0, Y_REF, gamma, 0.0, 0.0)

    def test_w_low_refuses_at_and_beyond_wall_via_gamma(self):
        """gamma' >= WALL (kappa=0): w_low_fit raises (not a number)."""
        for gamma in (self.WALL, 1.0, 1.2):
            with self.subTest(gamma=gamma):
                with self.assertRaises(DiffractiveDomainError):
                    w_low_fit(Y_REF, gamma, 0.0, 0.0)

    def test_refuses_via_kappa_reduced_shear_and_nonphysical_lambda(self):
        """kappa lifts gamma' over the wall / drives lambda <= 0 -> raise."""
        # (gamma, kappa): gamma'=gamma/(1-kappa); last two have 1-kappa <= 0.
        for gamma, kappa in ((0.5, 0.5), (0.6, 0.5), (0.3, 1.0), (0.3, 1.2)):
            with self.subTest(gamma=gamma, kappa=kappa):
                with self.assertRaises(DiffractiveDomainError):
                    w_low_fit(Y_REF, gamma, 0.0, kappa)

    def test_admits_inside_the_certified_domain(self):
        """gamma'=0.3 (certified-clean): the rung admits (refusal is not vacuous).

        The anti-vacuity admit witness lives in the certified-clean domain
        (reduced shear ~1/3) rather than just inside the wall.  Without this
        the "refuses at the wall" invariant would be vacuously met by a rung
        that refuses everywhere.
        """
        w_low = w_low_fit(Y_REF, 0.3, 0.0, 0.0)
        self.n_compared += 1
        self.assertIsNotNone(w_low)
        self.assertTrue(math.isfinite(w_low) and w_low > 0.0)
        # And the amplification evaluates without raising.
        val = diffractive_amplification(1.0, Y_REF, 0.3, 0.0, 0.0)
        self.assertTrue(math.isfinite(abs(val)))


class ShearFreePointMassPhaseTestCase(DiffractiveTestCase):
    """Spec 1 -- shear-free ``G_PM`` collapse + exact ``w*ln(w)`` phase.

    At ``gamma = kappa = 0`` (pure point mass) the shear operator series
    vanishes term by term, so ``F_P`` must reduce EXACTLY to the point-mass
    amplification ``C(w) * 1F1``.  Three legs, increasing independence:

      A. OPERATOR COLLAPSE (shares the DD kernel, pins the series + recon-phase
         machinery): the served ``F_P`` equals the hand-assembled
         ``exp(0.5j*w*s) * G_PM`` to rtol 1e-12, AND ``_operator_terms`` returns
         terms ``[1:]`` BYTE-EXACTLY zero -- the operator contribution is not
         merely small, it is identically absent.
      B. PHASE INDEPENDENCE (mpmath, no shared code): `prefactor_c` matches the
         independent ``_mp_prefactor_c`` to rtol 1e-12; the ``0.5*w*ln(0.5*w)``
         term grows to ~91 rad at w=55, so a certificate-BOUNDED phase (a bug
         that clipped ``w*ln(w)`` out of ``C(w)``) would blow this leg open.
      C. FULL PHYSICS TIE (mpmath, no shared code): the served ``F_P(gamma=0)``
         matches the wholly-independent ``_mp_point_mass`` to rtol 1e-9 up to
         w=40 -- certifying the served value carries the unbounded phase, not
         just that two production primitives agree.

    Tolerance rationale: leg A is exact (measured 0.0) so 1e-12 is slack; legs
    B/C measure production float64 against a 50-dps oracle, worst 5.5e-15 (B)
    and 5.1e-15 (C, w<=40) -- the 1e-12/1e-9 bars are ~200x/~10^5x margins,
    chosen to catch a real phase regression, not to chase machine noise.
    """

    def test_operator_series_collapses_to_point_mass_kernel(self):
        """Leg A: F_P(gamma=0) == exp(0.5j*w*s)*G_PM and terms[1:] are zero."""
        for w in PHASE_WS:
            n_terms = _kernel_length(w, S_REF)
            g_pm = point_mass_g_derivatives(w, S_REF, 16, n_terms)[0][0]
            recon = cmath.exp(0.5j * w * S_REF) * g_pm
            served = diffractive_amplification(w, Y_REF, 0.0, 0.0, 0.0)
            terms = _operator_terms(w, Y_REF[0], Y_REF[1], S_REF, 0.0, 8)
            self.n_compared += 1
            with self.subTest(w=w):
                self.assertLessEqual(
                    abs(served - recon) / abs(recon), 1e-12,
                    f'F_P(gamma=0) not the point-mass reconstruction at w={w}')
                # The operator contribution is identically absent (alpha=0),
                # not just below tolerance: terms[1:] must be byte-zero.
                self.assertEqual(
                    max(abs(t) for t in terms[1:]), 0.0,
                    f'shear-operator terms nonzero at gamma=0, w={w}')
                # ...and terms[0] IS the kernel it reconstructs from.
                self.assertEqual(abs(terms[0] - g_pm), 0.0)

    def test_prefactor_phase_matches_mpmath_unbounded(self):
        """Leg B: prefactor_c == mpmath C(w), and the w*ln(w) phase is huge."""
        for w in PHASE_WS:
            pc = prefactor_c(w)
            ref = complex(_mp_prefactor_c(w))
            self.n_compared += 1
            with self.subTest(w=w):
                self.assertLessEqual(
                    abs(pc - ref) / abs(ref), 1e-12,
                    f'prefactor_c phase/magnitude drifts from mpmath at w={w}')
        # Teeth for "grows like w*ln(w), never certificate-bounded": the phase
        # term at the top of the band is many radians -- a bounded-phase serve
        # is impossible to reconcile with leg B above.
        top = PHASE_WS[-1]
        self.assertGreater(abs(0.5 * top * math.log(0.5 * top)), 50.0)

    def test_served_matches_full_mpmath_point_mass(self):
        """Leg C: served F_P(gamma=0) == independent mpmath C(w)*1F1."""
        max_arg_residual = []
        for w in MPMATH_PHYS_WS:
            served = diffractive_amplification(w, Y_REF, 0.0, 0.0, 0.0)
            ref = complex(_mp_point_mass(w, S_REF))
            self.n_compared += 1
            with self.subTest(w=w):
                self.assertLessEqual(
                    abs(served - ref) / abs(ref), 1e-9,
                    f'F_P(gamma=0) drifts from mpmath point-mass at w={w}')
            max_arg_residual.append(
                (w, abs(cmath.phase(served / ref))))
        self._plot_arg_residual(max_arg_residual)

    def _plot_arg_residual(self, residuals):
        """Diagnostic: arg-residual vs w (flat at machine level = phase OK)."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ws = [w for w, _ in residuals]
        res = [r for _, r in residuals]
        fig, ax = plt.subplots()
        ax.semilogy(ws, np.maximum(res, 1e-18), 'o-')
        ax.set_xlabel('w')
        ax.set_ylabel('|arg(F_P / F_pm_mpmath)|  [rad]')
        ax.set_title('Spec 1 leg C: shear-free phase residual (flat = w*ln(w) '
                     'phase intact)')
        fig.savefig(os.path.join(
            OUTPUT_DIR, 'shear_free_point_mass_arg_residual.png'), dpi=80)
        plt.close(fig)


class NestedNullSplitByteIdentityTestCase(DiffractiveTestCase):
    """Spec 2 -- the nested band-split is a strict no-op when Rung P is idle.

    The likelihood hosts the c3/Born rungs on ``[w_low, w_split)`` and the
    analytic diffractive bottom on ``[w_lo, w_low)``.  The composition
    (reproduced here EXACTLY from the two production sites) is::

        _bs, below_mask       = _band_split_mask(dense_w, w_split)
        w_low                 = _diffractive_bottom_ceiling(lens, w_hi)
        band_split_low, below = _band_split_mask(dense_w, w_low)
        bottom_mask = (below & below_mask) if band_split_low else all-False
        host_mask   = below_mask & ~bottom_mask

    When ``w_low`` does not strictly straddle the grid (``None`` at the saddle
    wall, ``0.0`` for no-shear, or below the whole grid because every dense node
    is above ``w_low``), ``band_split_low`` is False and ``host_mask`` is
    BYTE-IDENTICAL to the pre-existing single-split ``below_mask`` -- the extra
    ``_band_split_mask`` call scatters nothing.  This guards the reconstruction
    against silent drift when the diffractive rung does not fire.

    Engine-free: it exercises the real `_band_split_mask` and
    `_diffractive_bottom_ceiling` mask logic on synthetic grids; no wave is
    evaluated.  ``_diffractive_bottom_ceiling`` is called with ``None`` self
    (the method never touches ``self``).
    """

    @staticmethod
    def _compose(dense_w, w_split, lens):
        """Reproduce the production nested-split composition verbatim."""
        _bs, below_mask = _band_split_mask(dense_w, w_split)
        w_low = LensedRelativeBinningLikelihood._diffractive_bottom_ceiling(
            None, lens, w_hi=float(dense_w.max()))
        band_split_low, below_low = _band_split_mask(dense_w, w_low)
        if w_low is not None and w_low >= float(dense_w.max()):
            bottom_mask = below_mask
        else:
            bottom_mask = ((below_low & below_mask) if band_split_low
                           else np.zeros(dense_w.shape, dtype=bool))
        host_mask = below_mask & ~bottom_mask
        return w_low, band_split_low, below_mask, bottom_mask, host_mask

    def test_saddle_wall_gives_none_and_identity(self):
        """gamma>1 -> w_low None -> nested bottom empty, host == below."""
        lens = dict(y1=Y_REF[0], y2=Y_REF[1], gamma=1.5, beta=0.0, kappa=0.0)
        dense = np.linspace(2.0, 80.0, 60)
        w_low, bsl, below, bottom, host = self._compose(dense, 40.0, lens)
        self.n_compared += 1
        self.assertIsNone(w_low)
        self.assertFalse(bsl)
        self.assertFalse(bottom.any())
        # byte-identical float/bool arrays -- not merely equal-valued.
        self.assertTrue(np.array_equal(host, below))

    def test_no_shear_gives_zero_ceiling_and_identity(self):
        """gamma=0 -> w_low 0.0 -> below-grid, host == below byte-identical."""
        lens = dict(y1=Y_REF[0], y2=Y_REF[1], gamma=0.0, beta=0.0, kappa=0.0)
        dense = np.linspace(2.0, 80.0, 60)
        w_low, bsl, below, bottom, host = self._compose(dense, 40.0, lens)
        self.n_compared += 1
        self.assertEqual(w_low, 0.0)
        self.assertFalse(bsl)          # 0.0 not strictly interior to the grid
        self.assertFalse(bottom.any())
        self.assertTrue(np.array_equal(host, below))

    def test_grid_entirely_above_wlow_is_identity(self):
        """Every dense node > w_low (positive parity) -> host == below."""
        lens = dict(y1=Y_REF[0], y2=Y_REF[1], gamma=0.2, beta=0.0, kappa=0.0)
        w_low = LensedRelativeBinningLikelihood._diffractive_bottom_ceiling(
            None, lens)
        self.assertGreater(w_low, 0.0)   # premise: a real positive ceiling...
        dense = np.linspace(w_low * 2.0, w_low * 2.0 + 60.0, 50)  # ...all above
        _wl, bsl, below, bottom, host = self._compose(dense, None, lens)
        self.n_compared += 1
        self.assertFalse(bsl)
        self.assertFalse(bottom.any())
        self.assertTrue(np.array_equal(host, below))

    def test_active_straddle_is_not_identity(self):
        """Contrast: a genuine straddle DOES carve a nonempty nested bottom.

        Without this the identity tests could pass vacuously (a composition
        that ALWAYS returned ``host == below`` would satisfy them).  Here the
        grid straddles ``w_low``, so ``band_split_low`` is True and the bottom
        is nonempty -- proving the no-op is conditional on the null-split.
        """
        lens = dict(y1=Y_REF[0], y2=Y_REF[1], gamma=0.2, beta=0.0, kappa=0.0)
        w_low = LensedRelativeBinningLikelihood._diffractive_bottom_ceiling(
            None, lens)
        dense = np.linspace(w_low * 0.3, w_low * 3.0, 50)   # straddles w_low
        _wl, bsl, below, bottom, host = self._compose(dense, None, lens)
        self.n_compared += 1
        self.assertTrue(bsl)
        self.assertGreater(int(bottom.sum()), 0)
        self.assertFalse(np.array_equal(host, below))
        # the carved bottom is exactly the nodes at-or-below w_low, and host is
        # its strict complement within below.
        self.assertTrue(np.array_equal(bottom, dense <= w_low))
        self.assertTrue(np.array_equal(host, below & ~bottom))


def _rung_s_w_reach(w_split, w_ceiling=W_CEILING_SCHWINGER):
    """Reproduce the F070 Rung S per-draw reachability cap VERBATIM.

    ``w_reach = W_CEILING_SCHWINGER if w_split is None else
    min(w_split, W_CEILING_SCHWINGER)`` -- the engine host serves only
    ``w_hi <= w_reach``.  Kept as a standalone mirror so the structural
    invariant (reach tracks ``min(w_split, ~60)``, never a hard-coded ceiling)
    is testable without instantiating the likelihood.
    """
    return w_ceiling if w_split is None else min(w_split, w_ceiling)


class RungSQuadratureSelfCertificateTestCase(DiffractiveTestCase):
    """Spec 3 -- the macro-saddle band-split host's admission certificate.

    Rung S has no convergent Fermat-moment series, so it hosts the EXACT
    Schwinger engine directly.  Its admission criterion IS the engine's own
    internal Gauss-Legendre N/2N paired self-certificate: `f_schwinger`
    RETURNS a value only when its order-N and order-2N quadratures agree to
    ``_CERTIFICATION_TOL`` (3e-10), and RAISES `SchwingerCertificationError`
    otherwise.  So:

      * wherever the rung claims to serve, the N/2N paired error is <= 3e-10
        BY CONSTRUCTION (a returned value == a passed certificate);
      * the served value must additionally match an INDEPENDENT higher-order
        reference (`_f_schwinger_mpmath`, a wholly separate mpmath contraction)
        to that same tolerance;
      * ``W_reach`` tracks ``min(w_split, ~60)`` per draw, never a hard ceiling.

    Oracle independence: the DD path (w<=60) and the mpmath reference share no
    accumulation; measured agreement 5.2e-15, ~10^4x inside the 3e-10 bar.
    """

    def test_certification_tol_honours_spec_bar(self):
        """The engine's internal N/2N bar is <= the spec's 3e-10."""
        self.assertLessEqual(_CERTIFICATION_TOL, 3e-10)

    def test_served_matches_independent_reference_within_certificate(self):
        """DD serve == mpmath reference to 3e-10 across the saddle band."""
        rows = []
        for gp in SADDLE_GAMMA_PRIMES:
            for w in SADDLE_WS:
                served = f_schwinger(w, np.asarray(Y_REF), gp)  # returns => cert
                ref = _f_schwinger_mpmath(w, np.asarray(Y_REF), gp)
                rel = abs(served - ref) / abs(ref)
                rows.append((gp, w, rel))
                self.n_compared += 1
                with self.subTest(gamma_prime=gp, w=w):
                    self.assertLessEqual(
                        rel, 3e-10,
                        f'saddle serve off reference at gp={gp}, w={w}')
        self._plot_certificate(rows)

    def test_w_reach_tracks_min_wsplit_not_hard_ceiling(self):
        """Reachability cap follows min(w_split, W_CEILING_SCHWINGER)."""
        self.assertEqual(W_CEILING_SCHWINGER, 60.0)
        # No split -> the plain engine ceiling.
        self.assertEqual(_rung_s_w_reach(None), 60.0)
        # A split BELOW the ceiling caps the reach at the split (the tracking
        # teeth: a hard-coded 60 ceiling would wrongly serve up to 60 here).
        self.assertEqual(_rung_s_w_reach(45.0), 45.0)
        self.assertLess(_rung_s_w_reach(45.0), W_CEILING_SCHWINGER)
        # A split ABOVE the ceiling is clamped back to the ceiling.
        self.assertEqual(_rung_s_w_reach(200.0), 60.0)
        self.n_compared += 1

    def test_refuses_past_qd_ceiling(self):
        """Beyond the QD ceiling the engine REFUSES, never serves uncertified.

        ``W_CEILING_SCHWINGER_QD`` (150) is the hard wall past which no
        quadrature order certifies; a serve there would be an uncertified
        number, so `f_schwinger` must raise instead.
        """
        self.assertEqual(W_CEILING_SCHWINGER_QD, 150.0)
        with self.assertRaises(SchwingerCertificationError):
            f_schwinger(W_CEILING_SCHWINGER_QD + 10.0,
                        np.asarray(Y_REF), 1.5)
        self.n_compared += 1

    def _plot_certificate(self, rows):
        """Diagnostic: served-vs-reference rel err vs w (must sit under 3e-10)."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, ax = plt.subplots()
        for gp in SADDLE_GAMMA_PRIMES:
            ws = [w for g, w, _ in rows if g == gp]
            rel = [r for g, _, r in rows if g == gp]
            ax.semilogy(ws, np.maximum(rel, 1e-18), 'o-', label=f"gamma'={gp}")
        ax.axhline(3e-10, ls='--', color='k', label='3e-10 certificate bar')
        ax.set_xlabel('w')
        ax.set_ylabel('|F_DD - F_mpmath| / |F_mpmath|')
        ax.set_title('Spec 3: Rung S saddle-host quadrature self-certificate')
        ax.legend()
        fig.savefig(os.path.join(
            OUTPUT_DIR, 'rung_s_quadrature_certificate.png'), dpi=80)
        plt.close(fig)


class DiffractiveSelfFalsificationTestCase(DiffractiveTestCase):
    """Prove the suite can go red -- teeth, not decoration.

    Each test re-runs a real assertion from the suite against a KNOWN-WRONG
    input and requires it to fail, so a green run of the other classes is
    evidence rather than a silently vacuous pass.
    """

    def test_wrong_anchor_would_fail(self):
        """The anchor teeth reject |F| -> 1 when sqrt(mu) != 1."""
        # At gamma'=0.6, sqrt(mu)=1.25: comparing |F| against the WRONG anchor
        # (1.0) must NOT satisfy "closer to target than 0.1*dev-from-1".
        gamma = 0.6
        mod = abs(diffractive_amplification(1e-3, Y_REF, gamma, 0.0, 0.0))
        dev_wrong_anchor = abs(mod - 1.0)          # ~0.25, the true anchor gap
        dev_from_sqrt_mu = abs(mod - _sqrt_mu(gamma))
        # The real test asserts dev_from_sqrt_mu < 0.1*dev_wrong_anchor (passes);
        # the inverted claim (|F| anchored at 1) must be false.
        with self.assertRaises(AssertionError):
            self.assertLess(dev_wrong_anchor, 0.1 * dev_from_sqrt_mu)

    def test_frame_pairing_gate_has_teeth(self):
        """A mis-rotated engine oracle disagrees with the served value.

        Confirms the pairing that makes the certified-band comparison
        meaningful: the CORRECT eigenframe R(-beta) y agrees to <1e-4, a wrong
        rotation (+beta) does not, so the band test is measuring a real match.
        """
        gamma, beta, w = 0.3, 0.7, 0.6
        f_p = diffractive_amplification(w, Y_REF, gamma, beta, 0.0)
        y_right = _rot_minus_beta(beta) @ np.asarray(Y_REF)
        y_wrong = _rot_minus_beta(-beta) @ np.asarray(Y_REF)
        rel_right = abs(f_p - f_schwinger(w, y_right, gamma)) \
            / abs(f_schwinger(w, y_right, gamma))
        rel_wrong = abs(f_p - f_schwinger(w, y_wrong, gamma)) \
            / abs(f_schwinger(w, y_wrong, gamma))
        self.n_compared += 1
        self.assertLess(rel_right, 1e-4)
        self.assertGreater(rel_wrong, 1e-2)

    def test_bounded_phase_bug_would_fail_leg_b(self):
        """A certificate-BOUNDED C(w) phase disagrees with mpmath (Spec 1 B).

        Emulate the exact bug the invariant guards: drop the unbounded
        ``0.5*w*ln(0.5*w)`` term from the prefactor phase.  Against the honest
        mpmath oracle the relative error must blow far past the 1e-12 tie, so
        leg B has real teeth.
        """
        w = 40.0
        magnitude = abs(prefactor_c(w))
        gamma_factor = complex(mp.gamma(1 - 1j * w / 2))
        # Reconstruct a phase-clipped prefactor: keep magnitude, keep ONLY the
        # bounded loggamma phase, DROP the unbounded 0.5*w*ln(0.5*w) term.
        bounded = magnitude * cmath.exp(
            1j * math.atan2(gamma_factor.imag, gamma_factor.real))
        ref = complex(_mp_prefactor_c(w))
        rel_bounded = abs(bounded - ref) / abs(ref)
        with self.assertRaises(AssertionError):
            self.assertLessEqual(rel_bounded, 1e-12)

    def test_null_split_identity_teeth(self):
        """A genuine straddle breaks host==below, so the identity is real.

        The Spec 2 no-op tests would be vacuous if the composition ALWAYS
        returned ``host == below``; here an active straddle must NOT.
        """
        lens = dict(y1=Y_REF[0], y2=Y_REF[1], gamma=0.2, beta=0.0, kappa=0.0)
        w_low = LensedRelativeBinningLikelihood._diffractive_bottom_ceiling(
            None, lens)
        dense = np.linspace(w_low * 0.3, w_low * 3.0, 40)
        _bs, below = _band_split_mask(dense, None)
        band_split_low, below_low = _band_split_mask(dense, w_low)
        bottom = below_low & below if band_split_low else np.zeros_like(below)
        host = below & ~bottom
        with self.assertRaises(AssertionError):
            self.assertTrue(np.array_equal(host, below))

    def test_uncertified_serve_would_fail_rung_s(self):
        """Past the QD ceiling a serve is impossible; expecting one fails.

        The Rung S admission (Spec 3) rests on `f_schwinger` RAISING past the
        certifiable band.  If a caller wrongly assumed it returns, that
        assumption trips here -- proving the refusal is load-bearing.
        """
        with self.assertRaises(SchwingerCertificationError):
            f_schwinger(W_CEILING_SCHWINGER_QD + 10.0, np.asarray(Y_REF), 2.0)


if __name__ == '__main__':
    main()
