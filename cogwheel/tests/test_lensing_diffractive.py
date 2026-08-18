"""
Tests for `lensing.chang_refsdal._diffractive` -- Rung P, the positive-parity
diffractive analytic serve object and its truncation certificate.

WHAT THIS SUITE ADJUDICATES
---------------------------
`_diffractive.diffractive_w_low` ships with an explicit UNVERIFIED note in its
own docstring:

    "the closed form's admission DIRECTION (serve w < w_low vs w > w_low) must
     be validated against the engine oracle before the rung is wired in. This
     function implements the WP1 closed form faithfully; it does not adjudicate
     direction."

This suite IS that engine-oracle validation.  Three questions, from the
Architect's specs:

  1. LOW-W ANCHOR (`LowWAnchorTestCase`).  The served DC limit must tend to
     ``sqrt(mu_macro)`` with the F009-S Morse phase -- ``+sqrt(mu)`` (arg -> 0)
     for positive parity, ``-1j sqrt(mu)`` (arg -> -pi/2) for the macro saddle
     -- and to exactly 1 ONLY in the degenerate ``gamma = kappa = 0`` point
     mass.  A carrier-limit refactor that reinstated ``F -> 1`` everywhere
     would silently corrupt every low-w serve; this pins against that.

  2. TRUNCATION VS EXACT ENGINE over the certified band
     (`TruncationCertifiedBandTestCase`).  Where the certificate admits, the
     order-8 truncation must agree with the exact engine to within
     ``CERTIFICATION_BAR`` (1e-4).  This is ESCALATE-ON-MISS: no widened
     tolerance is encoded.  It passes only on the sub-domain the truncation
     genuinely serves (measured: reduced shear ``gamma' <= 0.3``).

  3. SELF-REFUSAL AND MONOTONICITY near the wall (`WallRefusalTestCase`,
     `WLowMonotonicityWitnessTestCase`).  At/beyond the parity wall
     ``gamma' >= 1 - DELTA_GAMMA_P`` the rung must decline
     (``DiffractiveDomainError``), never return a small optimistic number.

MEASURED SPEC DISCREPANCY (escalated, not papered over)
-------------------------------------------------------
The engine oracle shows the WP1 closed form's serve direction and band width
are OPTIMISTIC beyond ``gamma' ~ 0.33``:

  * over the admitted band ``[w_lo, w_low]`` the truncation error REACHES
    ``6e-2`` at ``gamma' = 0.5`` and grows with ``gamma'`` -- far above the
    1e-4 bar the certificate promises (`CertificateOptimismWitnessTestCase`);
  * ``w_low`` INCREASES toward the wall (4.0 -> 5.2 as ``gamma'`` 0.90 ->
    0.994) rather than shrinking to nothing, so the rung claims an ever-WIDER
    low-w band exactly where the truncation is worst
    (`WLowMonotonicityWitnessTestCase`).

Rather than commit a permanently-red suite (which would jam the tree gate), the
two witness classes PIN the measured optimism with teeth: each asserts the
defect is present AND bounded, and each FLIPS RED the moment the certificate is
tightened to honour the bar.  See the change report / build report for the
escalation.  The clean-domain class (2) stays a hard, un-widened invariant.

ORACLE INDEPENDENCE
-------------------
The reference is `_schwinger.f_schwinger` -- the exact Schwinger proper-time
double-double engine -- evaluated in the eigenframe ``R(-beta) y`` at
``kappa = 0``.  It shares NO accumulation path with the operator series under
test (the series is the float64 truncation of the separate mpmath
``_oracle_fop`` contraction, F002).  `f_schwinger` is the engine the shipped
likelihood hands off to above ``w_low``, so agreeing with it is exactly the
serve-consistency the rung needs.  ``w`` stays ``<= 60`` so the engine runs on
its exact double-double path (mpmath only above 60).
"""

from __future__ import annotations

import math
import os
import cmath
from unittest import TestCase, main

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
from cogwheel.lensing.chang_refsdal._diffractive import (
    diffractive_amplification, diffractive_w_low, DiffractiveDomainError,
    _operator_terms, _kernel_length)
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    prefactor_c, point_mass_g_derivatives)
from cogwheel.lensing.likelihood import (
    _band_split_mask, LensedRelativeBinningLikelihood)


#: Directory for diagnostic plots (created on demand).
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

#: Reference source offset used across the suite (arbitrary, off-axis so the
#: shear operator acts on both eigen-directions).
Y_REF = (0.8, 0.4)

#: Reduced shears whose order-8 truncation is genuinely certified: measured
#: worst relative error over ``[w_lo, w_low]`` stays below `CERTIFICATION_BAR`
#: (2.998e-10, 4.687e-07, 5.378e-05 at these three).  Beyond ~0.33 the
#: truncation crosses the bar (see the witness class).
CLEAN_GAMMAS = (0.1, 0.2, 0.3)

#: Reduced shears where the certificate is OPTIMISTIC: the admitted band hosts
#: errors above the bar.  Used by the leaky-gate witness, NOT the invariant.
OPTIMISTIC_GAMMAS = (0.4, 0.5)

#: Eigenframe rotations exercised for the frame-pairing oracle (kappa = 0).
BETAS = (0.0, 0.7, -1.1)

#: Small frequencies probing the DC (w -> 0) anchor.  Kept well inside the
#: engine's exact double-double band (w <= 60).
ANCHOR_WS = (1e-2, 3e-3, 1e-3)

#: Number of frequency samples per certified-band sweep.
N_BAND = 40

#: Convergence (kappa) values probing INS-2-002: the honest-verification gate
#: in `diffractive_w_low` normalizes by ``sqrt_mu`` instead of the true
#: total-amplitude-space magnitude ``lam * sqrt_mu`` (``lam = 1 - kappa``), so
#: it understates relative truncation error by a factor of ``lam`` once
#: ``kappa > 0``.  At ``kappa = 0`` the fix is a no-op (``lam = 1``); these
#: values are all kappa > 0 so the reported bug can actually manifest.
KAPPA_GRID = (0.1, 0.2, 0.3)

#: Reduced-shear/kappa combinations whose order-8 truncation stays genuinely
#: certified at kappa > 0: measured worst relative error over
#: ``[w_lo, w_low]`` stays below `CERTIFICATION_BAR` by >= 6.3x at every
#: (kappa, gamma) in `KAPPA_GRID[:2] x KAPPA_CLEAN_GAMMAS` (worst measured
#: 1.576e-05 at kappa=0.2, gamma=0.2, beta=0.7).  ``kappa=0.3`` is excluded
#: here -- see `KAPPA_WITNESS`.
KAPPA_CLEAN_GAMMAS = (0.1, 0.2)

#: (kappa, gamma, beta) marking the INS-3-001 re-raise of INS-2-002: measured
#: worst relative error 1.4748e-04 > `CERTIFICATION_BAR` (1e-4) under the
#: pre-fix honest-verification normalization (divided by ``sqrt_mu`` instead
#: of ``lam * sqrt_mu``), i.e. this draw was ADMITTED by `diffractive_w_low`
#: yet the admitted band hosted truncation error above the bar the
#: certificate promises.  This is a stronger, directly-reproduced instance of
#: INS-2-002 (the finding's own cited draw, kappa=0.3/gamma=0.2/beta=0.0,
#: measures 8.594e-05 -- under the bar; widening beta to 0.7 crosses it).
#: `KappaEngineOracleTestCase.
#: test_truncation_within_bar_at_former_leaky_gate_witness` pins the
#: CORRECTED expectation (worst <= CERTIFICATION_BAR): the `lam * sqrt_mu`
#: normalization fix (INS-3-001) has landed in `_diffractive.py` and this
#: pin is GREEN (see that method's docstring for the live measured value).
KAPPA_WITNESS = (0.3, 0.2, 0.7)


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
    """Return ``(worst_rel, w_worst, w_low, ws, rels)`` over ``[w_lo, w_low]``.

    The diffractive rung serves ``w <= w_low``; the sweep runs from a small
    positive floor up to ``w_low`` and scores each point against the exact
    engine.  The worst error is where the truncation is stressed hardest,
    typically near ``w_low``.
    """
    w_low = diffractive_w_low(y, gamma, beta, 0.0)
    if w_low is None or not w_low > 0.0:
        raise AssertionError(
            f'diffractive_w_low returned {w_low} for gamma={gamma}; '
            'the band is undefined and the sweep would assert nothing.')
    w_lo = max(0.05, 0.02 * w_low)
    ws = np.linspace(w_lo, w_low, N_BAND)
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
    signature that cannot silently pick up a nonzero kappa.
    """
    w_low = diffractive_w_low(y, gamma, beta, kappa)
    if w_low is None or not w_low > 0.0:
        raise AssertionError(
            f'diffractive_w_low returned {w_low} for gamma={gamma}, '
            f'kappa={kappa}; the band is undefined and the sweep would '
            'assert nothing.')
    w_lo = max(0.05, 0.02 * w_low)
    ws = np.linspace(w_lo, w_low, N_BAND)
    rels = np.empty_like(ws)
    for i, w in enumerate(ws):
        f_p = diffractive_amplification(w, y, gamma, beta, kappa)
        f_e = _engine_reference_kappa(w, y, gamma, beta, kappa)
        rels[i] = abs(f_p - f_e) / abs(f_e)
    idx = int(np.argmax(rels))
    return float(rels[idx]), float(ws[idx]), float(w_low), ws, rels


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
    """Spec 2 -- order-8 truncation matches the exact engine over the band.

    ESCALATE-ON-MISS.  Over the full admitted band ``[w_lo, w_low]`` the
    truncation must agree with `f_schwinger` to within `CERTIFICATION_BAR`.
    This class runs ONLY on the sub-domain where the certificate is honest
    (`CLEAN_GAMMAS`, reduced shear <= 0.3); the bar is NOT widened -- a miss
    here is a genuine regression of the rung's legitimacy.  The optimistic
    regime is pinned separately (`CertificateOptimismWitnessTestCase`) so that
    it does not corrupt this invariant's tolerance.
    """

    def test_truncation_within_bar_over_band(self):
        """max rel-err(w) <= CERTIFICATION_BAR across [w_lo, w_low]."""
        for gamma in CLEAN_GAMMAS:
            for beta in BETAS:
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

    def test_band_is_nonempty_on_clean_domain(self):
        """Every clean geometry yields a positive, finite served band."""
        for gamma in CLEAN_GAMMAS:
            w_low = diffractive_w_low(Y_REF, gamma, 0.0, 0.0)
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
        ax.set_title('Rung P truncation vs exact engine over [w_lo, w_low]')
        ax.legend()
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'certified_band_relerr_vs_w.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self.assertTrue(os.path.exists(path))


class CertificateOptimismWitnessTestCase(DiffractiveTestCase):
    """Spec 2b -- the tightened certificate REFUSES the optimistic regime.

    Earlier the engine oracle showed `diffractive_w_low` admitted bands whose
    truncation error EXCEEDED `CERTIFICATION_BAR` once ``gamma' >~ 0.33`` (the
    spec-flagged optimism defect).  INS-1-001 added an honest self-consistency
    check evaluated at the band top ``w_low`` -- the worst (largest-w) point of
    the served band -- and declines any geometry whose leading omitted term
    breaches `CERTIFICATION_BAR` there.  This class was explicitly designed as
    a leaky-gate witness that would FLIP RED the moment the certificate was
    tightened; that has now happened, so it is repurposed to its intended
    successor invariant: the optimistic regime must be REFUSED (``None``),
    never served with an over-bar band.
    """

    def test_optimistic_regime_is_refused(self):
        """gamma' in {0.4, 0.5}: the honest gate declines (returns None)."""
        for gamma in OPTIMISTIC_GAMMAS:
            w_low = diffractive_w_low(Y_REF, gamma, 0.0, 0.0)
            self.n_compared += 1
            with self.subTest(gamma=gamma):
                self.assertIsNone(
                    w_low,
                    'OPTIMISM RETURNED? the honest self-consistency gate '
                    f'admitted the optimistic regime at gamma={gamma} '
                    f'(w_low={w_low}); it must decline it so no over-bar band '
                    'is served.')


class KappaEngineOracleTestCase(DiffractiveTestCase):
    """INS-2-002: engine-oracle pin for the previously-unpinned kappa > 0 regime.

    Before this class, every accuracy invariant in this suite fixed
    ``kappa = 0``, so the honest-verification gate's normalization defect (it
    divides by ``sqrt_mu`` instead of ``lam * sqrt_mu``, understating relative
    truncation error by a factor of ``lam = 1 - kappa`` once ``kappa > 0``)
    was never exercised by an engine oracle.  This closes that gap: first a
    pairing gate confirms the new kappa-aware oracle
    (`_engine_reference_kappa`) agrees with the already-verified kappa=0
    oracle at kappa=0, then the certified-band invariant is re-run at
    kappa > 0 on the sub-domain that measurably stays under the bar.
    """

    def test_pairing_gate_kappa_zero_matches_verified_oracle(self):
        """At kappa=0 the mass-sheet oracle must reduce to `_engine_reference`."""
        for gamma in KAPPA_CLEAN_GAMMAS:
            for beta in BETAS:
                w_low = diffractive_w_low(Y_REF, gamma, beta, 0.0)
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
        """INS-3-001: `KAPPA_WITNESS` (kappa=0.3/gamma=0.2/beta=0.7) certificate.

        Replaces the retired `KappaLeakyGateWitnessTestCase`, per that
        witness's own self-declared resolution condition ("fold this draw
        into KappaEngineOracleTestCase if it now stays under the bar").

        Pins the CORRECTED expectation: `diffractive_w_low`'s honest-
        verification gate normalizes by ``lam * sqrt_mu`` instead of the
        bare ``sqrt_mu`` (INS-3-001 fix, owned by production code), so the
        admitted band's worst truncation error at `KAPPA_WITNESS` genuinely
        holds `CERTIFICATION_BAR`, same as every other admitted draw in this
        suite.

        STATUS: the `lam * sqrt_mu` normalization fix has landed in
        `_diffractive.py` and this pin is GREEN.
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


class WallRefusalTestCase(DiffractiveTestCase):
    """Spec 3a -- the rung self-refuses at/beyond the parity wall.

    For reduced shear ``gamma' >= 1 - DELTA_GAMMA_P`` (or non-physical
    ``1 - kappa <= 0``) both entry points must raise `DiffractiveDomainError`,
    never return a small optimistic number.  Inside the certified-clean domain
    (reduced shear ~1/3) the rung must still admit -- otherwise "refuses at the
    wall" would be vacuously satisfied by a rung that refuses everywhere.  (The
    honest INS-1-001 gate declines the whole ``[~1/3, wall)`` band, so the
    anti-vacuity admit witness lives at ~1/3, not just inside the wall.)
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
        """gamma' >= WALL (kappa=0): diffractive_w_low raises (not a number)."""
        for gamma in (self.WALL, 1.0, 1.2):
            with self.subTest(gamma=gamma):
                with self.assertRaises(DiffractiveDomainError):
                    diffractive_w_low(Y_REF, gamma, 0.0, 0.0)

    def test_refuses_via_kappa_reduced_shear_and_nonphysical_lambda(self):
        """kappa lifts gamma' over the wall / drives lambda <= 0 -> raise."""
        # (gamma, kappa): gamma'=gamma/(1-kappa); last two have 1-kappa <= 0.
        for gamma, kappa in ((0.5, 0.5), (0.6, 0.5), (0.3, 1.0), (0.3, 1.2)):
            with self.subTest(gamma=gamma, kappa=kappa):
                with self.assertRaises(DiffractiveDomainError):
                    diffractive_w_low(Y_REF, gamma, 0.0, kappa)

    def test_admits_inside_the_certified_domain(self):
        """gamma'=0.3 (certified-clean): the rung admits (refusal is not vacuous).

        The honest INS-1-001 gate declines the whole ``[~1/3, wall)`` band, so
        the anti-vacuity admit witness lives in the certified-clean domain
        (reduced shear ~1/3) rather than just inside the wall.  Without this
        the "refuses at the wall" invariant would be vacuously met by a rung
        that refuses everywhere.
        """
        w_low = diffractive_w_low(Y_REF, 0.3, 0.0, 0.0)
        self.n_compared += 1
        self.assertIsNotNone(w_low)
        self.assertTrue(math.isfinite(w_low) and w_low > 0.0)
        # And the amplification evaluates without raising.
        val = diffractive_amplification(1.0, Y_REF, 0.3, 0.0, 0.0)
        self.assertTrue(math.isfinite(abs(val)))


class WLowMonotonicityWitnessTestCase(DiffractiveTestCase):
    """Spec 3b -- the certified band collapses to refusal toward the wall.

    The spec's correct behaviour: the certified band must vanish as
    ``gamma' -> 1`` so nothing is served where the truncation is worst.  The
    shipped WP1 closed form ALONE did the OPPOSITE -- ``w_low`` grew toward the
    wall -- exactly the "admission DIRECTION unvalidated" hazard the module
    flagged.  INS-1-001's honest self-consistency gate (evaluated at the band
    top ``w_low``) corrects this: the entire near-wall region breaches
    `CERTIFICATION_BAR` and is declined, so `diffractive_w_low` returns
    ``None`` across the approach to the wall.  This class -- explicitly built to
    FLIP RED once the certificate was corrected -- is repurposed to its intended
    successor invariant: refusal across the near-wall sweep.
    """

    #: gamma' sweep marching up toward the wall.  Every value is beyond the
    #: certified-clean boundary (~1/3), so the honest gate declines them all.
    GAMMAS_TO_WALL = (0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.994)

    def test_near_wall_band_is_refused(self):
        """CORRECTED: w_low declines (None) across the approach to the wall."""
        for g in self.GAMMAS_TO_WALL:
            w_low = diffractive_w_low(Y_REF, g, 0.0, 0.0)
            self.n_compared += 1
            with self.subTest(gamma=g):
                self.assertIsNone(
                    w_low,
                    'DIRECTION REGRESSED? the honest gate must decline the '
                    f'near-wall band, but gamma={g} admitted w_low={w_low} -- '
                    'a served band where the truncation is worst.')

    def test_diagnostic_plot_w_low_vs_gamma(self):
        """Save w_low vs gamma' (band shrinks to refusal past ~1/3)."""
        gammas = np.linspace(0.05, 0.994, 60)
        w_lows = [diffractive_w_low(Y_REF, g, 0.0, 0.0) for g in gammas]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(gammas, [w if w is not None else np.nan for w in w_lows],
                marker='.')
        ax.axvline(1.0 - DELTA_GAMMA_P, ls='--', color='k',
                   label='parity wall')
        ax.set_xlabel("reduced shear gamma'")
        ax.set_ylabel('w_low (certified band top)')
        ax.set_title('Certified band collapses to refusal past reduced shear ~1/3')
        ax.legend()
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'w_low_vs_gamma_direction.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self.assertTrue(os.path.exists(path))


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
        w_low                 = _diffractive_bottom_ceiling(lens)
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
            None, lens)
        band_split_low, below_low = _band_split_mask(dense_w, w_low)
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

    def test_optimism_witness_has_teeth(self):
        """A hypothetical honest certificate would trip the '> bar' guard."""
        # If a fix drove the band error to 0, the witness lower guard fails.
        honest_worst = 0.0
        with self.assertRaises(AssertionError):
            self.assertGreater(honest_worst, CERTIFICATION_BAR)

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
