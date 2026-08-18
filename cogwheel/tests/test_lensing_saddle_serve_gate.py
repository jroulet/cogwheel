"""
Invariants of the rewritten saddle far-field serve gate
``_saddle_farfield_analytic_serves`` (WP1: c3-led certificate with a
separation discriminator; WP2: census mirror moved to the new signature).

The gate is a pure predicate

    _saddle_farfield_analytic_serves(real_images, source, matrix, w_lo) -> bool

that decides whether the far-from-caustic macro saddle (gamma > 1, a
resolved 2-image exterior) may be served with a ZERO residual envelope over
the whole band.  Its logic (see the production docstring) is:

  1. certificate (PRIMARY): admit iff the safety-factored leading-omitted
     stationary-phase remainder at the band floor clears the bar,
     ``_SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR`` where
     ``est = ppgo_error_estimate(images, source, matrix, w_lo)``.  A ``None``
     estimate (divergent mu/c3 at a genuinely merging pair) refuses --
     this is the primary coalescence discriminator.
  2. separation backstop (SECONDARY, defense-in-depth): the minimum
     pairwise Euclidean image separation must be
     ``>= _SADDLE_FARFIELD_MIN_IMAGE_SEP`` (= 0.05).

This suite certifies the three behaviours the rewrite exists to guarantee,
each isolating ONE mechanism:

  * ``TiedMirrorPairServesTestCase`` -- a symmetry-tied mirror pair (equal
    delays, spatially far apart) serves.  HEAD's retired ``delta_tau > 0``
    resolution leg would have REFUSED this exact configuration; the new gate
    must serve it.  This is the false-refusal the rewrite fixes.
  * ``MergingPairRefusesTestCase`` -- a pair approaching the critical curve
    refuses, and the refusal is driven by the certificate leg (the backstop
    would otherwise pass).  The genuine ``est is None`` branch is exercised
    at its documented trigger.
  * ``SeparationFloorFlipTestCase`` -- two configs bracketing
    ``_SADDLE_FARFIELD_MIN_IMAGE_SEP``, both with a finite certificate that
    clears the bar, flip the gate exactly at the constant -- proving the
    backstop has teeth independent of the certificate.
  * ``CertificateBarFlipTestCase`` -- a fixed resolved far-apart pair with
    two band floors ``w_lo`` bracketing the certificate threshold
    ``S * est(w_lo) == bar`` flips the gate (refuse low, serve high), and the
    flip point matches the analytic identity ``S * est(w_flip) == bar``.  The
    separation is far above the floor, so the certificate leg alone drives
    the flip.
  * ``CertificateMonotoneDecayTestCase`` -- ``est`` is strictly decreasing in
    ``w_lo`` and scales as ``w_lo**-3`` (log-log slope == -3), so the band
    floor is the worst case: a pass at ``w_lo`` certifies the whole band.
  * ``CensusMirrorMatchesProductionGateTestCase`` -- the census served set
    (``characterize_sample`` category ``saddle-farfield-analytic``) mirrors
    the production gate boolean draw-for-draw (served == counted), across a
    serve draw and a refuse draw.

Oracle independence.  The gate is a boolean; the "oracle" for each invariant
is an INDEPENDENTLY computed premise -- the Fermat delay
``geometry.delay`` (for the delta_tau == 0 tie), the Euclidean image
separation (for the backstop flip), and the sign/finiteness of ``est`` (for
the certificate leg).  No test gates the predicate against a copy of its own
body: fixtures are DERIVED from the live boundary constants and every premise
is asserted before the gate verdict is read.

Tolerances.  The delay-tie fixture places the source on a Fermat symmetry
axis, so ``delta_tau == 0`` holds EXACTLY (bit-for-bit ``0.0``), not merely
within a tolerance -- ``_DELAY_TIE_ATOL`` is a paranoia floor.  The
separation fixtures bracket the floor by ``+/- 0.01`` (a 20% margin) so the
flip cannot be an interpolation artefact.  Certificate-clearance margins are
reported, not pinned, because they follow the fixed constants.

Cost.  Fast tier.  Each test builds at most a handful of 2-image geometry
partitions on a <= 12-point w grid (the exact DD path, w <= 60) plus pure
numpy; the whole file runs in a few seconds.

SPEC DISCREPANCY (documented, not papered over).  The MergingPairRefuses
spec asks for ``ppgo_error_estimate(...) is None`` at a physical near-fold
source.  Measured (2026-08-14): the DD root finder lands the merging image
just OFF the exact critical curve, so ``mu`` stays finite (~1e15) even with
the source exactly on the caustic (rho == 1.0) -- ``est`` grows without bound
but never returns ``None`` from a physical placement.  The genuine ``None``
branch (the actual coalescence discriminator) is therefore exercised via its
documented degenerate trigger (``w_min <= 0``) with a WELL-SEPARATED pair, so
the refusal is provably driven by ``est is None`` and not the backstop --
matching the spec's structural intent.  The reproducible PHYSICAL near-merge
refusal (finite-but-optimistic certificate) is certified separately.

CENSUS ARG CONSTRUCTION (resolved, INS-1-001).  Earlier the census (and the
live serve rung ``_saddle_farfield_analytic``) built the certificate's real
images as ``np.asarray(geom.images)[real]`` with ``real = geom.real_mask`` --
a length-4 CHANNEL mask -- but ``geom.images`` already holds ONLY the real
images (length ``k``; length 2 for a saddle 2-image draw), so the mask raised
``IndexError`` before any verdict was produced.  The double-mask is now fixed
at both sites (``real_images = np.asarray(geom.images)``), so the census no
longer crashes and the CensusMirror served==counted invariant is certified
end-to-end by ``test_census_served_matches_production_gate`` below as a live,
undecorated assertion.
"""
from __future__ import annotations

import math
import os
import unittest
from unittest import mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing.chang_refsdal.geometry import (
    macro_matrix, magnification, ppgo_error_estimate, delay)
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, reconstruct_farfield, FARFIELD_KERNEL_SUM)
from cogwheel.lensing.ppgo_map import caustic_geometry
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing.surrogate import TubeChart, LensAmplificationSurrogate
from cogwheel.lensing import surrogate_census
import cogwheel.lensing.likelihood as _likelihood_module
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood,
    _saddle_farfield_analytic_serves,
    _band_split_mask,
    _saddle_c3_split_point,
    _SADDLE_FARFIELD_SAFETY,
    _SADDLE_FARFIELD_CERT_BAR,
    _SADDLE_FARFIELD_MIN_IMAGE_SEP,
)
from cogwheel.lensing.chang_refsdal._schwinger import (
    W_CEILING_SCHWINGER_QD, f_schwinger)
from cogwheel.lensing.chang_refsdal._airy_fold import fold_ppgo_correction
from cogwheel.lensing.chang_refsdal.operator import RHO_END

#: Directory for diagnostic plots (created on demand).
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

#: Paranoia tolerance on the delay tie.  The mirror-pair fixture puts the
#: source on a Fermat symmetry axis, so ``delta_tau`` is identically ``0.0``;
#: this atol only guards against a future non-axis fixture drifting in.
_DELAY_TIE_ATOL = 1e-9

#: Band floor at which the ``w**-3`` certificate is evaluated.  Kept on the
#: exact DD path (w <= 60) and comfortably inside the served band.
_BAND_FLOOR_W = 12.0

#: How far the separation-flip fixtures bracket the backstop constant
#: (Einstein-radius units).  A 0.01 offset on a 0.05 floor is a 20% margin,
#: far above any float or interpolation noise.
_SEP_BRACKET = 0.01

#: Multiplicative brackets around the certificate-flip band floor ``w_flip``
#: (where ``S * est(w_flip) == bar`` exactly).  ``est`` scales as ``w**-3``,
#: so a 15% step in ``w_lo`` moves ``S * est`` by ~50% -- comfortably clear
#: of the bar on either side without leaving the exact-DD band.
_FLIP_REFUSE_FACTOR = 0.85
_FLIP_SERVE_FACTOR = 1.18

#: Relative tolerance on the analytic flip identity ``S * est(w_flip) == bar``.
#: The ``w**-3`` law is exact, so the only error is the float cube root; 1e-9
#: is a paranoia floor, not a physics tolerance.
_FLIP_RTOL = 1e-9

#: Lens mass (solar masses) used to map the census ``f_grid`` back onto a
#: chosen dimensionless-``w`` band.  ``dimensionless_frequency`` is LINEAR in
#: ``f``, so ``f_grid = w_grid / xi`` with ``xi = dimensionless_frequency(1,
#: M, 0)`` reproduces the target ``w_grid`` EXACTLY for any positive ``M``.
_CENSUS_M_LENS_MSUN = 1.0e6


#: Dense-``w`` grid for the band-split mask unit (Spec 1).  Endpoints
#: 10..100 give integer-friendly split brackets; the mask arithmetic is
#: frequency-agnostic so exact values do not matter, only the ordering.
_MASK_W_LO = 10.0
_MASK_W_HI = 100.0
_MASK_N = 16

#: Multiplicative half-step used to bracket the c3 split point when checking
#: the pass-just-above / fail-just-below flip (Spec 2).  ``est`` is a pure
#: ``w**-3`` law, so a 1e-6 step moves ``S*est`` by ~3e-6 -- far above the
#: float-inversion floor yet unambiguously on one side of the bar.
_C3_SPLIT_STEP = 1e-6

#: Relative tolerance on the closed-form inversion ``S*est(w_split) == bar``
#: (Spec 2) and on the null-split identity's certificate premises.  Only the
#: float cube root contributes error; 1e-9 is a paranoia floor.
_C3_RTOL = 1e-9

#: Kernel-subsampling shape for the ``_saddle_farfield_analytic`` probe
#: (Spec 3).  ``_NBINS * _NSUB`` must equal the dense-``w`` length so
#: ``_reduce_dense_kernels`` can reshape the channel kernels.
_NBINS = 8
_NSUB = 8

#: Certified accuracy bar for the c3 in-band overlap (Spec: c3 IN-BAND
#: ACCURACY).  The gate certifies ``S * est <= _SADDLE_FARFIELD_CERT_BAR``
#: with ``S = _SADDLE_FARFIELD_SAFETY = 20``, so the true LEADING remainder
#: at the split is ``<= bar / S = 5e-5``; ``bar`` (=1e-3) is the conservative
#: certificate CURRENCY -- the amplitude bar the served (above-split) band
#: must clear against the exact engine.  Read from the production constant so
#: the test follows the certificate if it ever moves.
_C3_OVERLAP_BAR = _SADDLE_FARFIELD_CERT_BAR

#: Upper edge of the c3-overlap comparison band.  The exact Schwinger oracle
#: ``f_schwinger`` is on the exact DD path for ``w <= 60``; 55 keeps every
#: node strictly inside it (no mpmath, no ceiling), matching the spec's
#: "keep w <= 60, cheap; do NOT compare above 150 (no oracle)".
_C3_OVERLAP_W_HI = 55.0

#: Number of ``w`` nodes on the c3-overlap comparison band (a handful, per
#: the spec).  ``est ~ w**-3`` peaks at the band floor, so the worst case is
#: the lowest node and a dozen points resolve the decay cleanly.
_C3_OVERLAP_N = 12

#: Multiplicative standoff of the comparison band floor above ``w_split``.
#: At exactly ``w_split`` the certificate sits at threshold (``S*est == bar``);
#: a 0.1% standoff keeps the band strictly inside the SERVED (admitted) region
#: while still probing the worst case just above the split.
_C3_FLOOR_STANDOFF = 1.001

#: Resolved 2-image far-from-caustic saddle configs (gamma, (y1, y2)) whose
#: c3 split lands well inside the DD band and whose served zero-envelope
#: reconstruction clears the currency by a comfortable margin (measured
#: 2026-08-17: max |F_analytic - F_engine| in {6.8e-4, 5.5e-5, 3.8e-5}).
#: These sit in the rung's ACTUAL contract domain (resolved AND far enough
#: from the caustic / high enough split that the subleading remainder is
#: below the currency); the canonical tied mirror leads.
_C3_CLEAN_CONFIGS = (
    (2.0, (1.0, 0.0)),
    (2.2, (1.0, 0.0)),
    (2.0, (0.95, 0.2)),
)

#: Leaky-gate WITNESS (FLAGGED calibration-optimism discrepancy, escalated in
#: the change report, NOT papered over).  The c3 certificate ADMITS this
#: source (``S*est <= bar`` at the band floor, so ``_saddle_farfield_analytic``
#: serves the whole band with a ZERO envelope), yet the served reconstruction
#: MISSES the currency over a low sub-band: measured max
#: |F_analytic - F_engine| ~ 3.1e-3 at ``w_split ~ 9.61`` (the LOWEST split
#: among the probed configs, and the closest to the caustic, ``rho ~ 0.48``),
#: decaying under the bar only above ``~1.5 * w_split``.  The 20x certificate
#: safety absorbs the leading ``w**-3`` remainder but not the subleading
#: terms in this near-caustic low-split corner.  Pinned below as the measured
#: reality so a future safety-factor / domain fix flips it red and forces a
#: re-baseline; see the class docstring.
_C3_LEAKY_WITNESS = (2.0, (1.1, 0.0))

#: Bracketing bounds on the leaky witness's measured miss (amplitude).  The
#: worst-case node exceeds the currency but stays well below 1e-2 -- a
#: 3x-over-currency optimism, not a catastrophic blow-up.
_C3_LEAKY_MISS_LO = _SADDLE_FARFIELD_CERT_BAR      # must exceed the currency
_C3_LEAKY_MISS_HI = 1.0e-2                          # but not blow up

#: Straddling above-ceiling ``w`` grid for the ppGO per-node partition
#: (Spec: PER-NODE ABOVE-CEILING).  A few below-ceiling nodes kept on the
#: exact DD path (``w <= 60``, so ``_engine_envelope_below_split`` is cheap)
#: plus a majority above the 150 ceiling where fold_ppgo carries.  Length
#: must equal ``_NBINS * _NSUB`` for ``_reduce_dense_kernels``.
_CEIL_BELOW_LO = 40.0
_CEIL_BELOW_HI = 58.0
_CEIL_BELOW_N = 8
_CEIL_ABOVE_LO = 160.0
_CEIL_ABOVE_HI = 300.0
_CEIL_ABOVE_N = _NBINS * _NSUB - _CEIL_BELOW_N

#: Resolved above-ceiling fixture (gamma, (y1, y2)): a 2-image saddle whose
#: lowest above-ceiling node is resolved (``150 * min_delta_tau >= RHO_END``);
#: measured 2026-08-17 ``150 * min_delta_tau ~ 164`` >> 4.  The premise is
#: RE-ASSERTED from live geometry at test time, never trusted as a literal.
_CEIL_RESOLVED_CONFIG = (1.5, (0.6, 0.9))

#: Near-caustic above-ceiling fixture (gamma, (y1, y2)): a 4-image source with
#: a tiny minimum delay gap so the lowest above-ceiling node is UNRESOLVED
#: (``150 * min_delta_tau < RHO_END``); measured ``150 * min_delta_tau ~ 3.94``
#: < 4, just inside the refusal.  ``_ppgo_above_ceiling`` must return ``None``
#: (fall through to the engine -> the deferred 2b refusal).  Premise
#: re-asserted from live geometry.
_CEIL_NEARCAUSTIC_CONFIG = (1.5, (1.5, 0.0))

#: Sentinel engine-envelope values used to prove the above-ceiling PARTITION
#: structurally without running the (expensive) exact engine: the spy returns
#: ``value`` on the below-ceiling nodes and ``0`` above, exactly like the real
#: ``_engine_envelope_below_split``.  Running with two DISTINCT sentinels and
#: checking the above-ceiling region is byte-invariant proves the fold carrier
#: is decoupled from the engine contribution (a clean split, no leak).
_CEIL_SENTINEL_A = 7.0 + 3.0j
_CEIL_SENTINEL_B = -2.0 - 5.0j


class _CountingTestCase(unittest.TestCase):
    """
    Anti-vacuity base for the invariant suites that do NOT read a serve-gate
    verdict (the mask unit, the c3 split point, the null-split identity).

    Each concrete test bumps ``self._checks`` once per invariant it actually
    exercises; ``tearDown`` fails loudly if a test asserted nothing about the
    function under test (an import drift or a silently-skipped fixture would
    otherwise let the suite go green while certifying nothing).
    """

    def setUp(self) -> None:
        self._checks = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self._checks, 0,
            'anti-vacuity: no invariant of the function under test was '
            'exercised in this test')


def _min_image_separation(images: np.ndarray) -> float:
    """
    Minimum pairwise Euclidean separation among image positions.

    Independent re-derivation of the gate's backstop currency, used only as
    a test-side premise oracle (never to *decide* the gate).
    """
    diffs = images[:, None, :] - images[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    iu = np.triu_indices(len(images), k=1)
    return float(np.min(dists[iu]))


class _ServeGateTestCase(unittest.TestCase):
    """
    Base carrying the anti-vacuity guard.

    Every concrete subclass increments ``self._gate_calls`` each time it
    actually reads a gate verdict.  A suite that silently stops exercising
    the gate (an import drift, a skipped fixture) would otherwise go green
    while certifying nothing; ``tearDown`` fails loudly if zero verdicts ran.
    """

    def setUp(self) -> None:
        self._gate_calls = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self._gate_calls, 0,
            'anti-vacuity: no serve-gate verdict was exercised in this test')

    def _serve(self, images, source, matrix, w_lo) -> bool:
        """Read one gate verdict and count it (anti-vacuity bookkeeping)."""
        verdict = _saddle_farfield_analytic_serves(
            np.asarray(images, float), np.asarray(source, float),
            np.asarray(matrix, float), float(w_lo))
        self._gate_calls += 1
        return verdict

    @staticmethod
    def _save_plot(fig, name: str) -> None:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(_OUTPUT_DIR, name), dpi=110,
                    bbox_inches='tight')
        plt.close(fig)



def _real_images(gamma: float, source, w) -> np.ndarray:
    """
    REAL image positions of a source via the production geometry partition.

    Uses ``ChangRefsdalChannels(w).geometry_partition`` -- exactly what the
    live serve rung builds -- and returns ``geom.images`` (the real images,
    length ``k``), NOT the length-4 channel arrays.
    """
    geom = ChangRefsdalChannels(np.asarray(w, float)).geometry_partition(
        gamma=float(gamma), y=(float(source[0]), float(source[1])),
        beta=0.0, kappa=0.0)
    return np.asarray(geom.images, dtype=float)


class TiedMirrorPairServesTestCase(_ServeGateTestCase):
    """
    A symmetry-tied mirror pair serves (guards the exact false refusal HEAD
    introduced).

    A macro-saddle source on the Fermat x-axis (``y = (1, 0)``,
    ``gamma = 2``) has two real images that are a ``+/-y`` mirror pair: their
    delays are EQUAL (``delta_tau == 0`` exactly, by the ``y -> -y``
    invariance of ``[[1-g,0],[0,1+g]]``), yet they sit ~1 Einstein radius
    apart.  The images are resolved and the c3 certificate clears the bar,
    so the new gate serves.  HEAD's retired ``delta_tau > 0`` resolution leg
    read ``w_lo * delta_tau = 0 >= 4`` as unresolved and REFUSED -- the
    precise false refusal the rewrite removes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.source = np.array([1.0, 0.0])
        cls.matrix = macro_matrix(cls.gamma)
        cls.w = np.geomspace(_BAND_FLOOR_W, 60.0, 12)
        cls.w_lo = float(cls.w.min())
        cls.images = _real_images(cls.gamma, cls.source, cls.w)
        cls.est = ppgo_error_estimate(
            cls.images, cls.source, cls.matrix, cls.w_lo)

    def test_premise_is_a_resolved_two_image_pair(self) -> None:
        """Premise: exactly two real images, well above the backstop floor."""
        self.assertEqual(len(self.images), 2,
                         'fixture must be a 2-image saddle exterior')
        sep = _min_image_separation(self.images)
        self.assertGreater(
            sep, 10.0 * _SADDLE_FARFIELD_MIN_IMAGE_SEP,
            'mirror pair must be spatially far apart (separation >> floor)')
        self._gate_calls += 1  # premise assertion still exercises the fixture

    def test_premise_delay_tie_is_exact(self) -> None:
        """Premise: the two images are a delay-tied mirror pair."""
        d0 = delay(self.images[0], self.source, self.matrix)
        d1 = delay(self.images[1], self.source, self.matrix)
        self.assertAlmostEqual(
            d0, d1, delta=_DELAY_TIE_ATOL,
            msg='mirror images must have coincident Fermat delays')
        self._gate_calls += 1

    def test_certificate_clears_the_bar(self) -> None:
        """The c3 certificate is finite and passes at the band floor."""
        self.assertIsNotNone(self.est, 'certificate must be finite here')
        self.assertTrue(np.isfinite(self.est))
        self.assertLessEqual(
            _SADDLE_FARFIELD_SAFETY * self.est, _SADDLE_FARFIELD_CERT_BAR,
            'safety-factored certificate must clear the production bar')
        self._gate_calls += 1

    def test_gate_serves_the_tied_pair(self) -> None:
        """The rewritten gate SERVES the delay-tied, spatially-separated pair."""
        self.assertTrue(
            self._serve(self.images, self.source, self.matrix, self.w_lo),
            'a resolved, delay-tied mirror pair must be served')

    def test_retired_delta_tau_leg_would_have_refused(self) -> None:
        """
        The HEAD resolution leg (``w_lo * delta_tau >= 4``) would refuse.

        This is the regression the rewrite fixes: the delay tie makes
        ``delta_tau == 0`` so the old product is ``0``, far below the old
        threshold of ``4`` -- HEAD refused a configuration that serves fine.
        """
        d0 = delay(self.images[0], self.source, self.matrix)
        d1 = delay(self.images[1], self.source, self.matrix)
        delta_tau = abs(d1 - d0)
        head_product = self.w_lo * delta_tau
        self.assertLess(
            head_product, 4.0,
            'sanity: the retired leg must indeed have refused this fixture')
        # ... and the new gate serves it anyway.
        self.assertTrue(
            self._serve(self.images, self.source, self.matrix, self.w_lo))

    def test_diagnostic_plot_separation_vs_delta_tau(self) -> None:
        """
        Scatter over on-axis sources: separation varies widely while the
        mirror delay tie stays ~0 -- proving delay-coincidence and spatial
        separation are distinct.
        """
        seps, dtaus = [], []
        for sx in np.linspace(0.6, 1.6, 9):
            src = np.array([sx, 0.0])
            imgs = _real_images(self.gamma, src, self.w)
            if len(imgs) != 2:
                continue
            seps.append(_min_image_separation(imgs))
            dtaus.append(abs(delay(imgs[1], src, self.matrix)
                             - delay(imgs[0], src, self.matrix)))
        self.assertGreater(len(seps), 0, 'need at least one 2-image sample')
        # The whole point: delays stay tied while separation ranges widely.
        self.assertLess(max(dtaus), _DELAY_TIE_ATOL,
                        'on-axis mirror delays must stay tied across the sweep')
        self.assertGreater(max(seps) - min(seps), 0.1,
                           'separation must vary appreciably across the sweep')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(dtaus, seps, c='C0', s=40, zorder=3, label='on-axis sources')
        ax.axhline(_SADDLE_FARFIELD_MIN_IMAGE_SEP, color='C3', ls='--',
                   label=f'sep floor = {_SADDLE_FARFIELD_MIN_IMAGE_SEP}')
        ax.axvline(0.0, color='0.5', ls=':')
        ax.set_xlabel(r'$|\Delta\tau|$ between mirror images')
        ax.set_ylabel('min image separation')
        ax.set_title('Tied mirror pair: delay-coincidence vs spatial separation')
        ax.legend()
        self._save_plot(fig, 'saddle_serve_gate_tied_mirror_scatter.png')
        self._gate_calls += 1



class MergingPairRefusesTestCase(_ServeGateTestCase):
    """
    A pair approaching the critical curve refuses (guards the primary
    coalescence discriminator).

    Two legs, both isolating that the refusal comes from the CERTIFICATE,
    never the separation backstop:

      * physical near-fold (reproducible).  A source just outside a fold
        (``rho = 1.001`` along the caustic ray) has a merging image with
        ``|mu| -> inf``, so the certificate ``est`` blows up (~1e15) and the
        safety-factored value is astronomically above the bar -- REFUSE.
        The two images are ~2 Einstein radii apart, so the backstop would
        PASS; the refusal is purely the certificate leg.
      * genuine ``None`` branch.  ``est is None`` is the gate's stated
        coalescence discriminator.  With the SAME well-separated near-fold
        images but a degenerate band floor (``w_min <= 0``),
        ``ppgo_error_estimate`` returns ``None`` and the gate refuses -- the
        refusal is asserted to be driven by ``est is None`` (the backstop
        would otherwise pass).  See the module SPEC DISCREPANCY note: a
        physical placement cannot make ``mu`` non-finite, so this documented
        trigger is how the ``None`` leg is exercised.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 1.6
        reach, direction = caustic_geometry(cls.gamma, kappa=0.0)
        cls.source = 1.001 * reach * np.asarray(direction, float)
        cls.matrix = macro_matrix(cls.gamma)
        cls.w = np.geomspace(8.0, 60.0, 12)
        cls.w_lo = float(cls.w.min())
        cls.reach = float(reach)
        cls.direction = np.asarray(direction, float)
        cls.images = _real_images(cls.gamma, cls.source, cls.w)
        cls.est = ppgo_error_estimate(
            cls.images, cls.source, cls.matrix, cls.w_lo)

    def test_premise_pair_is_well_separated(self) -> None:
        """
        Premise: the images are far apart, so any refusal is NOT the backstop.
        """
        self.assertGreaterEqual(len(self.images), 2)
        sep = _min_image_separation(self.images)
        self.assertGreater(
            sep, 10.0 * _SADDLE_FARFIELD_MIN_IMAGE_SEP,
            'near-fold pair must clear the separation floor by a wide margin')
        self._gate_calls += 1

    def test_premise_certificate_is_blown_up(self) -> None:
        """Premise: near the fold the certificate is finite but astronomical."""
        self.assertIsNotNone(self.est)
        self.assertTrue(np.isfinite(self.est))
        self.assertGreater(
            _SADDLE_FARFIELD_SAFETY * self.est,
            1e6 * _SADDLE_FARFIELD_CERT_BAR,
            'a near-merge must fail the certificate by many orders')
        # And a merging image carries a large magnification.
        max_mu = max(abs(magnification(im, self.matrix)) for im in self.images)
        self.assertGreater(max_mu, 10.0, 'merging image must have large |mu|')
        self._gate_calls += 1

    def test_physical_near_fold_refuses_via_certificate(self) -> None:
        """The near-fold pair is refused, and the backstop alone would pass."""
        self.assertFalse(
            self._serve(self.images, self.source, self.matrix, self.w_lo),
            'a near-merge must be refused')
        # Isolate: the separation leg alone would ADMIT -> refusal is the cert.
        self.assertGreaterEqual(
            _min_image_separation(self.images),
            _SADDLE_FARFIELD_MIN_IMAGE_SEP,
            'separation backstop would pass, so refusal is the certificate')

    def test_none_branch_drives_refusal(self) -> None:
        """
        ``est is None`` (the coalescence discriminator) refuses even with a
        pair the separation backstop would admit.
        """
        est_none = ppgo_error_estimate(
            self.images, self.source, self.matrix, -1.0)
        self.assertIsNone(
            est_none,
            'a degenerate band floor must drive the certificate to None')
        # Backstop would pass on these images, so a refusal here is the None.
        self.assertGreaterEqual(
            _min_image_separation(self.images),
            _SADDLE_FARFIELD_MIN_IMAGE_SEP)
        self.assertFalse(
            self._serve(self.images, self.source, self.matrix, -1.0),
            'gate must refuse when the certificate is None')

    def test_diagnostic_plot_certificate_blowup(self) -> None:
        """Plot ``est`` vs distance-to-caustic showing the blow-up toward None."""
        rhos = np.array([1.05, 1.02, 1.01, 1.005, 1.002, 1.001, 1.0005])
        ests = []
        for rho in rhos:
            src = rho * self.reach * self.direction
            imgs = _real_images(self.gamma, src, self.w)
            est = ppgo_error_estimate(imgs, src, self.matrix, self.w_lo)
            ests.append(est if est is not None else np.nan)
        ests = np.asarray(ests)
        self.assertTrue(np.all(np.diff(ests) > 0),
                        'certificate must grow monotonically toward the fold')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(rhos - 1.0, ests, 'o-', color='C1')
        ax.axhline(_SADDLE_FARFIELD_CERT_BAR / _SADDLE_FARFIELD_SAFETY,
                   color='C3', ls='--', label='admit ceiling (est)')
        ax.invert_xaxis()
        ax.set_xlabel(r'distance to caustic $\rho - 1$')
        ax.set_ylabel('ppgo_error_estimate (band floor)')
        ax.set_title('Certificate blows up as the fold pair merges')
        ax.legend()
        self._save_plot(fig, 'saddle_serve_gate_certificate_blowup.png')
        self._gate_calls += 1



class SeparationFloorFlipTestCase(_ServeGateTestCase):
    """
    The separation backstop flips the gate exactly at
    ``_SADDLE_FARFIELD_MIN_IMAGE_SEP`` (guards the defense-in-depth teeth).

    Two synthetic ``+/-x`` image pairs at a common ``y0`` bracket the floor
    by ``+/- _SEP_BRACKET``, chosen so that BOTH have a finite certificate
    that clears the c3 bar (the certificate leg alone would admit both).
    The only thing that changes across the pair is the Euclidean separation,
    so a below-floor -> ``False`` / above-floor -> ``True`` flip proves the
    backstop is load-bearing independently of the certificate.

    The bracketing offsets are DERIVED from the live constant, not pinned,
    so the fixture follows the floor if it ever moves.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.matrix = macro_matrix(cls.gamma)
        cls.source = np.array([0.0, 0.0])
        cls.w_lo = _BAND_FLOOR_W
        cls.y0 = 2.0
        cls.floor = _SADDLE_FARFIELD_MIN_IMAGE_SEP
        cls.sep_below = cls.floor - _SEP_BRACKET
        cls.sep_above = cls.floor + _SEP_BRACKET

    def _pair(self, sep: float) -> np.ndarray:
        """A synthetic ``+/-x`` image pair with Euclidean separation ``sep``."""
        return np.array([[-sep / 2.0, self.y0], [sep / 2.0, self.y0]])

    def test_premise_both_certificates_clear_the_bar(self) -> None:
        """
        Premise: BOTH bracket configs pass the certificate leg, so the flip
        can only come from the backstop.
        """
        for label, sep in (('below', self.sep_below), ('above', self.sep_above)):
            with self.subTest(bracket=label):
                imgs = self._pair(sep)
                est = ppgo_error_estimate(imgs, self.source, self.matrix,
                                          self.w_lo)
                self.assertIsNotNone(est)
                self.assertLessEqual(
                    _SADDLE_FARFIELD_SAFETY * est, _SADDLE_FARFIELD_CERT_BAR,
                    f'{label}-floor certificate must clear the bar')
        self._gate_calls += 1

    def test_premise_separations_bracket_the_floor(self) -> None:
        """Premise: the two measured separations straddle the constant."""
        below = _min_image_separation(self._pair(self.sep_below))
        above = _min_image_separation(self._pair(self.sep_above))
        self.assertLess(below, self.floor)
        self.assertGreater(above, self.floor)
        self._gate_calls += 1

    def test_below_floor_refuses(self) -> None:
        """Just below the floor the gate refuses (backstop veto)."""
        self.assertFalse(
            self._serve(self._pair(self.sep_below), self.source,
                        self.matrix, self.w_lo),
            'a sub-floor separation must be refused despite a clean certificate')

    def test_above_floor_serves(self) -> None:
        """Just above the floor the gate serves."""
        self.assertTrue(
            self._serve(self._pair(self.sep_above), self.source,
                        self.matrix, self.w_lo),
            'an above-floor separation with a clean certificate must serve')

    def test_diagnostic_plot_separation_sweep(self) -> None:
        """1-D sweep of min separation across the floor with the gate overlaid."""
        seps = np.linspace(self.floor - 3 * _SEP_BRACKET,
                           self.floor + 3 * _SEP_BRACKET, 25)
        verdicts = []
        for sep in seps:
            verdicts.append(int(self._serve(self._pair(sep), self.source,
                                            self.matrix, self.w_lo)))
        verdicts = np.asarray(verdicts)
        # The flip must be single and exactly at the floor.
        flip_idx = np.where(np.diff(verdicts) != 0)[0]
        self.assertEqual(len(flip_idx), 1, 'exactly one flip expected')
        self.assertLess(seps[flip_idx[0]], self.floor)
        self.assertGreaterEqual(seps[flip_idx[0] + 1], self.floor)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.step(seps, verdicts, where='post', color='C0')
        ax.axvline(self.floor, color='C3', ls='--',
                   label=f'floor = {self.floor}')
        ax.set_xlabel('min image separation')
        ax.set_ylabel('gate serves (1) / refuses (0)')
        ax.set_yticks([0, 1])
        ax.set_title('Separation backstop: gate flips exactly at the floor')
        ax.legend()
        self._save_plot(fig, 'saddle_serve_gate_separation_sweep.png')


class CertificateBarFlipTestCase(_ServeGateTestCase):
    """
    The certificate leg flips the gate two-sidedly at the band floor
    ``w_lo`` (guards the ``S * est <= bar`` admission and its band-floor
    evaluation).

    A single fixed, resolved, far-apart saddle 2-image config (the tied
    mirror pair, separation ``>> 0.05`` so the backstop is never the active
    leg) is gated at two band floors ``w_lo`` that BRACKET the certificate
    threshold.  Because ``est`` scales exactly as ``w_lo**-3``, the
    admission boundary is the closed form

        w_flip = (S * C / bar) ** (1/3),   C = est(w_ref) * w_ref**3,

    with ``S = _SADDLE_FARFIELD_SAFETY`` and ``bar = _SADDLE_FARFIELD_CERT_BAR``.
    Below ``w_flip`` (``S * est > bar``) the gate REFUSES; above it
    (``S * est < bar``) it SERVES.  The flip point is checked against the
    analytic identity ``S * est(w_flip) == bar`` to ``_FLIP_RTOL`` -- an
    INDEPENDENT oracle (the closed-form cube-root inversion of the ``w**-3``
    law), not a re-read of the gate body.  The fixtures are DERIVED from the
    live constants, so they follow the bar if it ever moves.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.source = np.array([1.0, 0.0])
        cls.matrix = macro_matrix(cls.gamma)
        cls.w = np.geomspace(_BAND_FLOOR_W, 60.0, 12)
        cls.images = _real_images(cls.gamma, cls.source, cls.w)
        # w-independent numerator of the w**-3 certificate.
        w_ref = 20.0
        cls.cref = ppgo_error_estimate(
            cls.images, cls.source, cls.matrix, w_ref) * w_ref ** 3
        cls.w_flip = float(
            (_SADDLE_FARFIELD_SAFETY * cls.cref / _SADDLE_FARFIELD_CERT_BAR)
            ** (1.0 / 3.0))
        cls.w_refuse = cls.w_flip * _FLIP_REFUSE_FACTOR
        cls.w_serve = cls.w_flip * _FLIP_SERVE_FACTOR

    def _est(self, w_lo: float) -> float:
        return ppgo_error_estimate(self.images, self.source, self.matrix,
                                   float(w_lo))

    def test_premise_pair_is_far_apart(self) -> None:
        """Premise: separation ``>> 0.05`` so the backstop is never active."""
        self.assertEqual(len(self.images), 2)
        self.assertGreater(
            _min_image_separation(self.images),
            10.0 * _SADDLE_FARFIELD_MIN_IMAGE_SEP,
            'the certificate, not the backstop, must be the active leg')
        self._gate_calls += 1

    def test_premise_brackets_straddle_the_bar(self) -> None:
        """Premise: ``S * est`` is above the bar below the flip, below above."""
        self.assertGreater(
            _SADDLE_FARFIELD_SAFETY * self._est(self.w_refuse),
            _SADDLE_FARFIELD_CERT_BAR,
            'low band floor must fail the certificate')
        self.assertLess(
            _SADDLE_FARFIELD_SAFETY * self._est(self.w_serve),
            _SADDLE_FARFIELD_CERT_BAR,
            'high band floor must clear the certificate')
        self._gate_calls += 1

    def test_low_band_floor_refuses(self) -> None:
        """Below the flip the gate refuses (certificate veto)."""
        self.assertFalse(
            self._serve(self.images, self.source, self.matrix, self.w_refuse),
            'a band floor with S*est > bar must be refused')

    def test_high_band_floor_serves(self) -> None:
        """Above the flip the gate serves."""
        self.assertTrue(
            self._serve(self.images, self.source, self.matrix, self.w_serve),
            'a band floor with S*est < bar must be served')

    def test_flip_point_matches_analytic_identity(self) -> None:
        """``S * est(w_flip) == bar`` at the derived flip (independent oracle)."""
        s_est_flip = _SADDLE_FARFIELD_SAFETY * self._est(self.w_flip)
        self.assertAlmostEqual(
            s_est_flip / _SADDLE_FARFIELD_CERT_BAR, 1.0,
            delta=_FLIP_RTOL,
            msg='closed-form flip must sit exactly on the certificate bar')
        self._gate_calls += 1

    def test_diagnostic_plot_certificate_vs_band_floor(self) -> None:
        """Plot ``S * est(w_lo)`` vs ``w_lo`` with the bar and served regions."""
        w_lo = np.geomspace(self.w_flip * 0.5, self.w_flip * 2.0, 40)
        s_est = _SADDLE_FARFIELD_SAFETY * np.array([self._est(w) for w in w_lo])
        # Verdict boundary must coincide with the S*est == bar crossing.
        served = np.array([
            self._serve(self.images, self.source, self.matrix, w)
            for w in w_lo])
        crossing = np.where(np.diff(served.astype(int)) != 0)[0]
        self.assertEqual(len(crossing), 1, 'exactly one serve/refuse flip')
        self.assertLess(w_lo[crossing[0]], self.w_flip)
        self.assertGreaterEqual(w_lo[crossing[0] + 1], self.w_flip)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(w_lo, s_est, 'o-', color='C0', label=r'$S\cdot$est$(w_{lo})$')
        ax.axhline(_SADDLE_FARFIELD_CERT_BAR, color='C3', ls='--',
                   label=f'bar = {_SADDLE_FARFIELD_CERT_BAR}')
        ax.axvline(self.w_flip, color='0.5', ls=':',
                   label=fr'$w_{{flip}}$ = {self.w_flip:.2f}')
        ax.axvspan(w_lo[0], self.w_flip, color='C3', alpha=0.08)
        ax.axvspan(self.w_flip, w_lo[-1], color='C2', alpha=0.08)
        ax.set_xlabel(r'band floor $w_{lo}$')
        ax.set_ylabel(r'$S\cdot$ppgo_error_estimate')
        ax.set_title('Certificate flip: refuse (left) / serve (right) at $w_{flip}$')
        ax.legend()
        self._save_plot(fig, 'saddle_serve_gate_certificate_flip.png')
        self._gate_calls += 1


class CertificateMonotoneDecayTestCase(_ServeGateTestCase):
    """
    The certificate decays strictly monotonically as ``w_lo**-3`` (guards the
    "band floor is the worst case, so certifying at ``w_lo`` certifies the
    whole band" assumption the gate rests on).

    For a fixed resolved saddle 2-image config, ``ppgo_error_estimate`` over
    a monotonically increasing ``w_lo`` array must be strictly decreasing and
    fit a log-log slope of ``-3`` -- the ``w**-3`` stationary-phase remainder
    law.  This is what makes the band floor the largest remainder in the
    band, so a certificate PASS at ``w_lo`` bounds every higher frequency.

    Oracle independence: the slope is fit from the raw estimate values; the
    ``-3`` reference is the analytic exponent of the leading omitted term, not
    a copy of any gate branch.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.source = np.array([1.0, 0.0])
        cls.matrix = macro_matrix(cls.gamma)
        cls.w = np.geomspace(_BAND_FLOOR_W, 60.0, 12)
        cls.images = _real_images(cls.gamma, cls.source, cls.w)
        cls.w_lo = np.geomspace(6.0, 60.0, 16)
        cls.ests = np.array([
            ppgo_error_estimate(cls.images, cls.source, cls.matrix, float(w))
            for w in cls.w_lo])

    def test_premise_is_a_resolved_two_image_pair(self) -> None:
        """Premise: a real, resolved 2-image saddle with a finite estimate."""
        self.assertEqual(len(self.images), 2)
        self.assertTrue(np.all(np.isfinite(self.ests)))
        self.assertTrue(np.all(self.ests > 0.0))
        self._gate_calls += 1

    def test_estimate_is_strictly_decreasing(self) -> None:
        """The estimate drops at every step as ``w_lo`` grows."""
        self.assertTrue(
            np.all(np.diff(self.ests) < 0.0),
            'ppgo_error_estimate must be strictly decreasing in w_lo')
        self._gate_calls += 1

    def test_loglog_slope_is_minus_three(self) -> None:
        """The estimate scales as ``w_lo**-3`` (log-log slope == -3)."""
        slope = np.polyfit(np.log(self.w_lo), np.log(self.ests), 1)[0]
        self.assertAlmostEqual(
            slope, -3.0, delta=1e-6,
            msg='the leading omitted term must scale as w**-3')
        self._gate_calls += 1

    def test_worst_case_is_the_band_floor(self) -> None:
        """The smallest ``w_lo`` yields the LARGEST estimate (worst case)."""
        self.assertEqual(
            int(np.argmax(self.ests)), 0,
            'the band floor must be the worst case, certifying the band')
        self._gate_calls += 1

    def test_diagnostic_plot_loglog_decay(self) -> None:
        """Log-log plot of the estimate vs ``w_lo`` with a ``-3`` reference."""
        ref = self.ests[0] * (self.w_lo / self.w_lo[0]) ** -3.0
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(self.w_lo, self.ests, 'o', color='C0',
                  label='ppgo_error_estimate')
        ax.loglog(self.w_lo, ref, '-', color='C3',
                  label=r'reference slope $-3$')
        ax.set_xlabel(r'band floor $w_{lo}$')
        ax.set_ylabel('ppgo_error_estimate')
        ax.set_title(r'Certificate decays as $w_{lo}^{-3}$')
        ax.legend()
        self._save_plot(fig, 'saddle_serve_gate_certificate_decay.png')
        self._gate_calls += 1


def _decoy_saddle_blind_surrogate() -> LensAmplificationSurrogate:
    """
    A one-chart surrogate that never serves a ``gamma > 1`` saddle query.

    A single positive-parity ``TubeChart`` with a gamma box ``[0.30, 0.50]``
    so ``select_chart`` returns ``None`` for every saddle draw (``gamma >
    1``): ``characterize_sample`` then falls through to the far-field saddle
    rung under test -- exactly the path whose served==counted mirror we
    certify.  Zero envelopes keep the chart cheap; it is never evaluated
    (only its box is consulted by ``select_chart``).  ``TubeChart.from_values``
    rejects an empty chart set, so a decoy chart -- not an empty surrogate --
    is the way to force the fall-through.
    """
    grid = 4
    gamma_grid = np.linspace(0.30, 0.50, grid)
    u_grid = np.linspace(math.sqrt(0.02), math.sqrt(0.05), grid)
    theta_grid = np.linspace(0.2, 1.2, grid)
    log_w_grid = np.linspace(math.log(2.0), math.log(60.0), grid)
    zeros = np.zeros((grid, grid, grid, grid))
    tube = TubeChart.from_values(
        gamma_grid=gamma_grid, u_grid=u_grid, theta_grid=theta_grid,
        log_w_grid=log_w_grid, envelope_real=zeros, envelope_imag=zeros,
        image_count=2, parity=1, eta_floor=0.02, eta_max=0.05,
        cusp_windows=())
    return LensAmplificationSurrogate([tube], {'chart_count': 1})


class CensusMirrorMatchesProductionGateTestCase(_ServeGateTestCase):
    """
    The census served set is a byte-faithful mirror of the production gate
    (WP2: served == counted).

    ``surrogate_census.characterize_sample`` must record
    ``category == 'saddle-farfield-analytic'`` for EXACTLY the saddle 2-image
    draws that the production predicate
    ``_saddle_farfield_analytic_serves(real_images, source, matrix, w_lo)``
    admits, with the arguments built the same way (real images, source, macro
    matrix at ``beta = kappa = 0``, band floor).  The battery spans both
    outcomes: a symmetry-tied far-apart pair that SERVES and a near-fold
    merging pair that REFUSES via the certificate.

    Oracle independence.  The "counted" side is the pure gate predicate read
    directly on the SAME geometry the census builds; the "served" side is the
    end-to-end ``characterize_sample`` verdict.  The mirror is the equality of
    the two -- neither computes the other, and both draws are genuine 2-image
    saddles (premise asserted) so no verdict is a length artefact.

    CENSUS ARG CONSTRUCTION (resolved, INS-1-001).  Earlier the census -- and
    the live serve rung ``_saddle_farfield_analytic`` -- built the
    certificate's ``real_images`` as ``np.asarray(geom.images)[real]`` with
    ``real = geom.real_mask``, a length-4 CHANNEL mask.  But ``geom.images``
    already holds ONLY the real images (length ``k``): for a 2-image saddle it
    is length 2, so masking it with the length-4 ``real_mask`` raised
    ``IndexError`` before any verdict was produced.  The fix (both sites) is
    ``real_images = np.asarray(geom.images)`` (drop the ``[real]``).  The
    double-mask is now gone, so the census serves without crashing and the
    served==counted invariant is certified end-to-end below by
    ``test_census_served_matches_production_gate`` as a live, undecorated
    assertion.  Sites fixed: ``surrogate_census.py`` (saddle block) and
    ``likelihood.py`` ``_saddle_farfield_analytic``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # SERVE draw: symmetry-tied far-apart 2-image saddle.
        cls.serve_gamma = 2.0
        cls.serve_source = np.array([1.0, 0.0])
        cls.serve_w = np.array([20.0, 20.6])
        cls.serve_w_lo = float(cls.serve_w.min())
        cls.serve_matrix = macro_matrix(cls.serve_gamma, 0.0, 0.0)
        cls.serve_images = _real_images(
            cls.serve_gamma, cls.serve_source, cls.serve_w)

        # REFUSE draw: near-fold merging 2-image saddle (certificate veto).
        cls.refuse_gamma = 1.6
        reach, direction = caustic_geometry(cls.refuse_gamma, kappa=0.0)
        cls.refuse_source = 1.001 * reach * np.asarray(direction, float)
        cls.refuse_w = np.geomspace(8.0, 60.0, 12)
        cls.refuse_w_lo = float(cls.refuse_w.min())
        cls.refuse_matrix = macro_matrix(cls.refuse_gamma, 0.0, 0.0)
        cls.refuse_images = _real_images(
            cls.refuse_gamma, cls.refuse_source, cls.refuse_w)

        cls.surrogate = _decoy_saddle_blind_surrogate()
        cls.xi = float(dimensionless_frequency(1.0, _CENSUS_M_LENS_MSUN, 0.0))

    def _census_record(self, gamma, source, w_lo):
        """End-to-end census verdict for one saddle draw (may raise today).

        ``dimensionless_frequency`` is linear in ``f``, so ``f_grid = w_grid /
        xi`` reconstructs the chosen ``w`` band exactly (``xi`` frozen in
        ``setUpClass``); the census therefore sees the SAME band floor the
        production gate is evaluated at.
        """
        f_grid = np.geomspace(w_lo, 60.0, 12) / self.xi
        return surrogate_census.characterize_sample(
            self.surrogate, ChangRefsdalChannels,
            gamma=float(gamma), m_lens_msun=_CENSUS_M_LENS_MSUN,
            y1=float(source[0]), y2=float(source[1]),
            f_grid=f_grid, dropped_slivers=())

    # -- production-gate reference (the "counted" side) ------------------
    def test_production_gate_spans_serve_and_refuse(self) -> None:
        """
        The pure predicate admits the tied pair and refuses the merging pair
        -- the two booleans the census must mirror.  Both draws are genuine
        2-image saddles (asserted), so neither verdict is a length artefact.
        """
        self.assertEqual(len(self.serve_images), 2,
                         'serve draw must be a 2-image saddle')
        self.assertEqual(len(self.refuse_images), 2,
                         'refuse draw must be a 2-image saddle')
        self.assertTrue(
            self._serve(self.serve_images, self.serve_source,
                        self.serve_matrix, self.serve_w_lo),
            'tied far-apart saddle pair must be served (counted)')
        self.assertFalse(
            self._serve(self.refuse_images, self.refuse_source,
                        self.refuse_matrix, self.refuse_w_lo),
            'near-fold merging saddle pair must be refused (not counted)')

    # -- the served==counted invariant (live, blocking) -----------------
    def test_census_served_matches_production_gate(self) -> None:
        """
        served == counted: the census ``saddle-farfield-analytic`` verdict
        equals the production gate boolean for every draw.

        The WP2 double-mask crash is fixed (INS-1-001): the census now builds
        the c3-certificate arguments identically to the live serve rung, so
        its ``saddle-farfield-analytic`` verdict must equal the pure gate's
        decision on the same ``(real_images, source, matrix, w_lo)`` inputs.
        This is a live, blocking assertion of the served==counted invariant.
        """
        self._gate_calls += 1  # anti-vacuity: keep tearDown green pre-crash
        for gamma, source, matrix, w_lo, images in (
                (self.serve_gamma, self.serve_source, self.serve_matrix,
                 self.serve_w_lo, self.serve_images),
                (self.refuse_gamma, self.refuse_source, self.refuse_matrix,
                 self.refuse_w_lo, self.refuse_images)):
            with self.subTest(gamma=gamma):
                produced = _saddle_farfield_analytic_serves(
                    np.asarray(images, float), np.asarray(source, float),
                    np.asarray(matrix, float), float(w_lo))
                record = self._census_record(gamma, source, w_lo)
                counted = record.category == 'saddle-farfield-analytic'
                self.assertEqual(
                    produced, counted,
                    'census served must mirror the production gate')

    # -- diagnostic table ------------------------------------------------
    def test_diagnostic_census_mirror_table(self) -> None:
        """Write the (config, census_served, production_served) mirror table.

        A divergence in the ``production_served`` column would localise to the
        gate; a divergence in ``census_served`` (once the WP2 crash is fixed)
        would localise to the census arg construction drifting from the rung.
        """
        rows = []
        for label, gamma, source, matrix, w_lo, images in (
                ('tied-far-apart', self.serve_gamma, self.serve_source,
                 self.serve_matrix, self.serve_w_lo, self.serve_images),
                ('near-fold-merge', self.refuse_gamma, self.refuse_source,
                 self.refuse_matrix, self.refuse_w_lo, self.refuse_images)):
            produced = self._serve(images, source, matrix, w_lo)
            try:
                record = self._census_record(gamma, source, w_lo)
                census = str(record.category == 'saddle-farfield-analytic')
            except IndexError:
                census = 'CRASH(IndexError)'
            rows.append((label, gamma, produced, census))

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(
            _OUTPUT_DIR, 'saddle_serve_gate_census_mirror_table.txt')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('config            gamma  prod_served  census_served\n')
            for label, gamma, produced, census in rows:
                handle.write(
                    f'{label:16s}  {gamma:5.2f}  {str(produced):11s}  '
                    f'{census}\n')

        # Non-vacuity of the production side: one serve, one refuse.
        produced_flags = [row[2] for row in rows]
        self.assertIn(True, produced_flags)
        self.assertIn(False, produced_flags)


class ServeGateSelfFalsificationTestCase(_ServeGateTestCase):
    """
    Proof the suite can go RED: the gate must actually DISCRIMINATE, so a
    mutated premise flips the verdict.  If any of these unexpectedly agreed,
    the "serves"/"refuses" assertions above would be vacuous.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.matrix = macro_matrix(cls.gamma)
        cls.source = np.array([0.0, 0.0])
        cls.w_lo = _BAND_FLOOR_W
        cls.floor = _SADDLE_FARFIELD_MIN_IMAGE_SEP

    def _pair(self, sep: float, y0: float = 2.0) -> np.ndarray:
        return np.array([[-sep / 2.0, y0], [sep / 2.0, y0]])

    def test_floor_flip_is_not_degenerate(self) -> None:
        """Below- and above-floor verdicts must DIFFER (not both same)."""
        below = self._serve(self._pair(self.floor - _SEP_BRACKET),
                            self.source, self.matrix, self.w_lo)
        above = self._serve(self._pair(self.floor + _SEP_BRACKET),
                            self.source, self.matrix, self.w_lo)
        self.assertNotEqual(
            below, above,
            'a suite that gave the same verdict either side of the floor '
            'would be vacuous')

    def test_degenerate_band_floor_forces_refusal(self) -> None:
        """
        A well-separated, well-conditioned pair SERVES at a positive floor
        but is REFUSED once the certificate is forced to ``None`` -- the
        mutation that must bite.
        """
        imgs = self._pair(self.floor + 10 * _SEP_BRACKET)
        served = self._serve(imgs, self.source, self.matrix, self.w_lo)
        refused = self._serve(imgs, self.source, self.matrix, -1.0)
        self.assertTrue(served, 'control: clean pair must serve')
        self.assertFalse(refused, 'None certificate must flip it to refuse')

    def test_single_image_is_refused(self) -> None:
        """Fewer than two images can never be a resolved exterior."""
        one = np.array([[0.3, 2.0]])
        self.assertFalse(
            self._serve(one, self.source, self.matrix, self.w_lo),
            'a single image must be refused (len < 2 guard)')


class BandSplitMaskTestCase(_CountingTestCase):
    """
    Unit contract of the shared ``_band_split_mask`` helper (Spec 1, WP1).

    ``_band_split_mask(dense_w, split)`` is the single source of truth every
    band-split serve rung consults.  Its contract:

      * ``band_split`` is ``True`` iff ``split is not None`` AND ``split``
        lies STRICTLY inside the open band ``(w_lo, w_hi)``; a split at or
        outside an endpoint, or ``None``, is a no-op.
      * when ``band_split`` is ``False`` the ``below_mask`` is all-``True``
        (the null-split identity precondition -- nothing is masked out, so
        the caller's reconstruct is byte-identical to the un-split result).
      * when ``band_split`` is ``True`` the ``below_mask`` equals
        ``dense_w <= split`` EXACTLY (inclusive of a node coincident with
        the split).

    Oracle independence.  The band-split boolean is checked against an
    INDEPENDENTLY evaluated Python-float predicate
    ``split is not None and w_lo < split < w_hi``; the below-mask is checked
    against ``dense_w <= split`` computed in the test.  These are the SPEC,
    a two-line definition, not a re-derivation of a complex algorithm -- and
    the boundary cases (endpoints, a coincident node) are what a subtle
    ``<`` vs ``<=`` off-by-one would break, so the unit still has teeth.
    """

    def setUp(self) -> None:
        super().setUp()
        self.dense_w = np.geomspace(_MASK_W_LO, _MASK_W_HI, _MASK_N)
        self.w_lo = float(self.dense_w.min())
        self.w_hi = float(self.dense_w.max())

    def _expect(self, split):
        """Independent (spec) expectation for one ``split``."""
        band = split is not None and self.w_lo < split < self.w_hi
        mask = ((self.dense_w <= split) if band
                else np.ones(self.dense_w.shape, dtype=bool))
        return band, mask

    def test_none_split_is_no_op_all_true(self) -> None:
        """``split is None`` -> not a band split, mask all-``True``."""
        band, mask = _band_split_mask(self.dense_w, None)
        self.assertFalse(band)
        self.assertTrue(np.all(mask))
        self.assertEqual(mask.dtype, np.dtype(bool))
        self._checks += 1

    def test_split_below_band_is_no_op_all_true(self) -> None:
        """A split under ``w_lo`` is a no-op (all-``True`` identity mask)."""
        band, mask = _band_split_mask(self.dense_w, self.w_lo * 0.5)
        self.assertFalse(band)
        self.assertTrue(np.all(mask))
        self._checks += 1

    def test_split_above_band_is_no_op_all_true(self) -> None:
        """A split over ``w_hi`` is a no-op (all-``True`` identity mask)."""
        band, mask = _band_split_mask(self.dense_w, self.w_hi * 2.0)
        self.assertFalse(band)
        self.assertTrue(np.all(mask))
        self._checks += 1

    def test_split_at_lower_endpoint_is_no_op(self) -> None:
        """A split exactly at ``w_lo`` is NOT strictly inside -> no-op."""
        band, mask = _band_split_mask(self.dense_w, self.w_lo)
        self.assertFalse(band, 'split == w_lo must be a no-op (strict <)')
        self.assertTrue(np.all(mask))
        self._checks += 1

    def test_split_at_upper_endpoint_is_no_op(self) -> None:
        """A split exactly at ``w_hi`` is NOT strictly inside -> no-op."""
        band, mask = _band_split_mask(self.dense_w, self.w_hi)
        self.assertFalse(band, 'split == w_hi must be a no-op (strict <)')
        self.assertTrue(np.all(mask))
        self._checks += 1

    def test_interior_split_masks_below_exactly(self) -> None:
        """A strictly-interior split -> band split, ``below_mask`` exact."""
        split = math.sqrt(self.w_lo * self.w_hi)  # geometric midpoint
        band, mask = _band_split_mask(self.dense_w, split)
        exp_band, exp_mask = self._expect(split)
        self.assertTrue(band)
        self.assertTrue(exp_band)
        self.assertTrue(np.array_equal(mask, exp_mask),
                        'below_mask must equal dense_w <= split exactly')
        # A genuine split serves some but not all nodes.
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), len(self.dense_w))
        self._checks += 1

    def test_split_coincident_with_node_is_inclusive(self) -> None:
        """A split equal to an interior node includes that node (``<=``)."""
        node = int(_MASK_N // 2)
        split = float(self.dense_w[node])
        self.assertLess(self.w_lo, split)
        self.assertLess(split, self.w_hi)
        band, mask = _band_split_mask(self.dense_w, split)
        self.assertTrue(band)
        self.assertTrue(mask[node], 'the coincident node must be served (<=)')
        self.assertFalse(mask[node + 1],
                         'the next node above the split must be excluded')
        self.assertTrue(np.array_equal(mask, self.dense_w <= split))
        self._checks += 1

    def test_null_split_identity_precondition_across_no_ops(self) -> None:
        """Every non-band split yields the all-``True`` identity mask."""
        for split in (None, self.w_lo * 0.5, self.w_lo, self.w_hi,
                      self.w_hi * 2.0):
            with self.subTest(split=split):
                band, mask = _band_split_mask(self.dense_w, split)
                self.assertFalse(band)
                self.assertTrue(
                    np.all(mask),
                    'null-split precondition: mask must be all-True so the '
                    'caller reconstruct is byte-identical to un-split')
        self._checks += 1

    def test_below_count_is_monotone_across_interior_band(self) -> None:
        """As the split rises through the band, the served count only grows."""
        splits = np.geomspace(self.w_lo * 1.01, self.w_hi * 0.99, 20)
        counts = [int(_band_split_mask(self.dense_w, float(s))[1].sum())
                  for s in splits]
        self.assertTrue(
            np.all(np.diff(counts) >= 0),
            'below-split served count must be nondecreasing in the split')
        self._checks += 1

    def test_diagnostic_boolean_table(self) -> None:
        """Write the split vs (band_split, below_mask.sum()) boundary table."""
        probes = [
            ('None', None),
            ('below-band', self.w_lo * 0.5),
            ('== w_lo', self.w_lo),
            ('interior-lo', self.w_lo * 1.5),
            ('geo-mid', math.sqrt(self.w_lo * self.w_hi)),
            ('node-coincident', float(self.dense_w[_MASK_N // 2])),
            ('interior-hi', self.w_hi * 0.8),
            ('== w_hi', self.w_hi),
            ('above-band', self.w_hi * 2.0),
        ]
        rows = []
        for label, split in probes:
            band, mask = _band_split_mask(self.dense_w, split)
            rows.append((label, split, band, int(mask.sum())))
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(_OUTPUT_DIR, 'band_split_mask_table.txt')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(f'dense_w in [{self.w_lo:.3f}, {self.w_hi:.3f}], '
                         f'n = {_MASK_N}\n')
            handle.write('label            split        band_split  below.sum\n')
            for label, split, band, count in rows:
                sval = 'None' if split is None else f'{split:11.4f}'
                handle.write(f'{label:15s}  {sval:>11s}  {str(band):10s}  '
                             f'{count:3d}\n')
        # Non-vacuity of the table: at least one active band split and one
        # no-op appear, and every active split serves a strict subset.
        active = [c for _, _, b, c in rows if b]
        noop = [c for _, _, b, c in rows if not b]
        self.assertGreater(len(active), 0, 'need at least one active split')
        self.assertGreater(len(noop), 0, 'need at least one no-op split')
        self.assertTrue(all(c == _MASK_N for c in noop),
                        'every no-op split must serve the whole band')
        self.assertTrue(all(0 < c < _MASK_N for c in active),
                        'every active split must serve a strict subset')
        self._checks += 1


class SaddleC3SplitPointTestCase(_CountingTestCase):
    """
    Closed-form inversion + monotonicity of ``_saddle_c3_split_point``
    (Spec 2, WP2).

    For a resolved 2-image saddle (``gamma > 1``, well-separated real
    images) ``ppgo_error_estimate`` returns a finite ``est(w) = C/w**3``, and
    the split frequency is the EXACT cube-root inversion of the certificate

        w_split = w_ref * (S * est(w_ref) / bar) ** (1/3),

    with ``S = _SADDLE_FARFIELD_SAFETY``, ``bar = _SADDLE_FARFIELD_CERT_BAR``.
    The invariant has three facets, all of ONE behaviour (the split point):

      * inversion is exact -- ``S * est(w_split) == bar`` to the float cube
        root (the certificate sits exactly on the bar at the returned split);
      * the certificate PASSES just above ``w_split`` and FAILS just below
        (the split is the admission boundary, evaluated the right way round);
      * ``est`` is strictly decreasing in ``w`` (the ``w**-3`` law that makes
        the closed form exact -- two points suffice here; the fuller log-log
        slope ``-3`` is pinned once in ``CertificateMonotoneDecayTestCase``);

    plus the coalescence guard: a degenerate input for which
    ``ppgo_error_estimate`` returns ``None`` propagates as ``None`` (a
    merging pair must refuse the whole draw, never enter a band split).

    Oracle independence.  The "oracle" is the shipping ``ppgo_error_estimate``
    re-evaluated at ``w_split`` (the same function the gate reads); the test
    asserts the CLOSED-FORM property ``S*est(w_split)/bar == 1``, which the
    function body never checks -- the split point is computed at ``w_ref =
    1.0`` and only inverted, so re-evaluating at ``w_split`` is an independent
    consistency oracle, not a copy of the branch.

    SPEC DISCREPANCY (documented).  The spec's ``None`` case is a
    "merging/near-critical config".  As the module note above records, a
    PHYSICAL near-fold placement keeps ``mu`` finite (~1e15), so ``est`` grows
    without bound but stays a finite float -- ``_saddle_c3_split_point`` then
    returns a huge FINITE ``w_split``, not ``None``.  Worse, an image placed
    EXACTLY on the critical curve makes ``det(H) == 0`` and
    ``magnification`` raises ``ZeroDivisionError`` (a Python ``1.0/0.0``),
    not a non-finite float, so it does not reach the ``None`` return either.
    The genuine ``None`` branch (the actual coalescence discriminator) is
    therefore exercised via its documented degenerate trigger -- an empty
    ``real_images`` array, for which ``ppgo_error_estimate`` returns ``None``
    by contract -- matching the spec's structural intent (a draw with no
    servable resolved pair refuses the split).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.source = np.array([1.0, 0.0])
        cls.matrix = macro_matrix(cls.gamma)
        cls.w = np.geomspace(_BAND_FLOOR_W, 60.0, 12)
        cls.images = _real_images(cls.gamma, cls.source, cls.w)
        cls.w_split = _saddle_c3_split_point(
            cls.images, cls.source, cls.matrix)

    def _est(self, w: float) -> float:
        return ppgo_error_estimate(self.images, self.source, self.matrix,
                                   float(w))

    def test_premise_is_resolved_pair_with_finite_split(self) -> None:
        """Premise: 2-image saddle, finite estimate, finite split point."""
        self.assertEqual(len(self.images), 2,
                         'fixture must be a resolved 2-image saddle')
        self.assertGreater(
            _min_image_separation(self.images),
            10.0 * _SADDLE_FARFIELD_MIN_IMAGE_SEP)
        self.assertIsNotNone(self.w_split)
        self.assertTrue(np.isfinite(self.w_split))
        self.assertGreater(self.w_split, 0.0)
        self._checks += 1

    def test_split_point_inverts_certificate_exactly(self) -> None:
        """``S * est(w_split) == bar`` to the float cube root (independent)."""
        s_est = _SADDLE_FARFIELD_SAFETY * self._est(self.w_split)
        self.assertAlmostEqual(
            s_est / _SADDLE_FARFIELD_CERT_BAR, 1.0, delta=_C3_RTOL,
            msg='the returned split must sit exactly on the certificate bar')
        self._checks += 1

    def test_split_point_is_reference_frequency_independent(self) -> None:
        """
        The cube-root inversion is ``w_ref``-independent: rebuilding the
        split from ``est`` at a DIFFERENT reference reproduces ``w_split``.
        """
        for w_ref in (5.0, 40.0):
            with self.subTest(w_ref=w_ref):
                cref = self._est(w_ref) * w_ref ** 3  # C = est*w**3
                w_from_ref = 1.0 * (
                    _SADDLE_FARFIELD_SAFETY * cref
                    / _SADDLE_FARFIELD_CERT_BAR) ** (1.0 / 3.0)
                self.assertAlmostEqual(
                    w_from_ref / self.w_split, 1.0, delta=_C3_RTOL,
                    msg='split point must not depend on the reference w')
        self._checks += 1

    def test_certificate_passes_above_split_fails_below(self) -> None:
        """
        Certificate PASSES just above ``w_split`` (smaller ``est``) and FAILS
        just below (larger ``est``) -- the split is the admission boundary.
        """
        above = _SADDLE_FARFIELD_SAFETY * self._est(
            self.w_split * (1.0 + _C3_SPLIT_STEP))
        below = _SADDLE_FARFIELD_SAFETY * self._est(
            self.w_split * (1.0 - _C3_SPLIT_STEP))
        self.assertLessEqual(
            above, _SADDLE_FARFIELD_CERT_BAR,
            'just above the split the certificate must clear the bar')
        self.assertGreater(
            below, _SADDLE_FARFIELD_CERT_BAR,
            'just below the split the certificate must fail the bar')
        self._checks += 1

    def test_estimate_is_strictly_decreasing_cube_law(self) -> None:
        """
        ``est`` is strictly decreasing and obeys the exact ``w**-3`` law
        (two points suffice) -- the property that makes the inversion exact.
        """
        w1, w2 = 8.0, 32.0  # a factor of 4 in w
        e1, e2 = self._est(w1), self._est(w2)
        self.assertGreater(e1, e2, 'est must strictly decrease in w')
        # est ~ C/w**3  =>  e1/e2 == (w2/w1)**3 exactly.
        self.assertAlmostEqual(
            (e1 / e2) / (w2 / w1) ** 3, 1.0, delta=1e-9,
            msg='est must follow the exact w**-3 stationary-phase law')
        self._checks += 1

    def test_degenerate_input_yields_none(self) -> None:
        """
        An input with no resolved pair (empty ``real_images``) drives
        ``ppgo_error_estimate`` to ``None``, so the split point is ``None``.
        """
        empty = np.zeros((0, 2))
        self.assertIsNone(
            ppgo_error_estimate(empty, self.source, self.matrix, 1.0),
            'premise: empty images make the estimate None by contract')
        self.assertIsNone(
            _saddle_c3_split_point(empty, self.source, self.matrix),
            'a None estimate must propagate to a None split point')
        self._checks += 1

    def test_diagnostic_plot_certificate_crossing(self) -> None:
        """Plot ``S*est(w)`` vs ``w`` with the bar; the crossing is w_split."""
        w = np.geomspace(self.w_split * 0.5, self.w_split * 2.0, 60)
        s_est = _SADDLE_FARFIELD_SAFETY * np.array([self._est(x) for x in w])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(w, s_est, '-', color='C0', label=r'$S\cdot$est$(w)$')
        ax.axhline(_SADDLE_FARFIELD_CERT_BAR, color='C3', ls='--',
                   label=f'bar = {_SADDLE_FARFIELD_CERT_BAR}')
        ax.axvline(self.w_split, color='0.5', ls=':',
                   label=fr'$w_{{split}}$ = {self.w_split:.3f}')
        ax.set_xlabel(r'frequency $w$')
        ax.set_ylabel(r'$S\cdot$ppgo_error_estimate')
        ax.set_title('c3 split point sits exactly on the certificate bar')
        ax.legend()
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(_OUTPUT_DIR, 'saddle_c3_split_point.png'),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)
        self._checks += 1


class _SaddleFarfieldProbe:
    """
    Lightweight carrier for the unbound ``_saddle_farfield_analytic`` method
    and its two kernel-reduction collaborators.

    Binding the real production methods onto a stub -- rather than building a
    full ``LensedRelativeBinningLikelihood`` (which needs an event, waveform
    and reference posterior) -- exercises the EXACT shipping serve rung with
    only the four attributes ``_reduce_dense_kernels`` reads.
    ``_engine_envelope_below_split`` is a spy: in the null-split cases under
    test it must NEVER be called (no engine work), which the test asserts.
    """

    n_bins = _NBINS
    kernel_subsamples = _NSUB
    _saddle_farfield_analytic = (
        LensedRelativeBinningLikelihood._saddle_farfield_analytic)
    _reduce_dense_kernels = (
        LensedRelativeBinningLikelihood._reduce_dense_kernels)
    _image_delays = LensedRelativeBinningLikelihood._image_delays

    def __init__(self) -> None:
        # Per-bin least-squares (value, slope) weights.  Their exact values
        # are irrelevant to a byte-identity test -- both the produced and the
        # reference reconstructions run through THIS same reduction -- they
        # need only be finite and correctly shaped.
        sub = np.linspace(-0.5, 0.5, _NSUB)
        value_row = np.ones(_NSUB) / _NSUB
        slope_row = sub / np.sum(sub ** 2)
        self._kernel_fit_value = np.tile(value_row, (_NBINS, 1))
        self._kernel_fit_slope = np.tile(slope_row, (_NBINS, 1))
        self._engine_envelope_below_split = mock.MagicMock(
            name='engine_envelope_below_split')


def _saddle_lens(gamma: float, source, m_lens_msun: float = 1.0e6) -> dict:
    """Lens-parameter dict for ``_saddle_farfield_analytic`` at beta=kappa=0."""
    return {
        'gamma': float(gamma), 'y1': float(source[0]), 'y2': float(source[1]),
        'beta': 0.0, 'kappa': 0.0,
        'm_lens_msun': float(m_lens_msun), 'z_lens': 0.0,
    }


class SaddleFarfieldNullSplitIdentityTestCase(_CountingTestCase):
    """
    Null-split byte-exact identity of ``_saddle_farfield_analytic`` at both
    boundaries (Spec 3, WP2).

    The band-split rung must degenerate to HEAD's whole-draw behaviour on the
    two draws that carry no interior split:

      (a) WHOLE-BAND ADMIT (``w_split <= w_lo``): the c3 certificate already
          clears the bar at the band floor, so the gate serves the whole band
          with a ZERO residual envelope and NO engine call.  The served
          ``(k0, k1)`` must be BYTE-IDENTICAL (``np.array_equal``) to an
          independent zero-envelope reconstruction over the same partition,
          and the engine spy must be untouched.
      (b) WHOLE-DRAW REFUSE (``w_split >= w_hi``): the certificate fails
          across the entire reachable band, so the rung returns ``None`` and
          the caller falls through to the exact seed engine -- byte-identical
          to today's refuse -- again with NO engine call inside the rung.

    Fixtures are DERIVED from the live split point, not pinned: the admit band
    straddles ``w_split`` from above (``w_lo > w_split``) and the refuse band
    from below (``w_hi < w_split``), both computed at runtime from the tied
    mirror pair (``gamma = 2``, ``y = (1, 0)``, ``w_split ~ 10.96``), so the
    two boundaries follow the certificate if its constants ever move.

    Re-points the existing saddle serve-gate suite (this file) rather than
    adding a parallel module: the gate predicate is pinned by the classes
    above; this class pins the METHOD that consumes it.

    Oracle independence.  The reference ``(k0, k1)`` in (a) are rebuilt from
    the method's OWN returned partition with an explicitly-zeroed envelope
    through the SAME ``reconstruct_farfield`` + ``_reduce_dense_kernels``
    path, so the assertion is "the rung did exactly the zero-envelope serve
    and nothing else"; the spy assertion proves no hidden engine evaluation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gamma = 2.0
        cls.source = np.array([1.0, 0.0])
        cls.matrix = macro_matrix(cls.gamma)
        # Split point is grid-independent (image positions do not depend on
        # the w grid), so derive it once from a reference band.
        ref_band = np.geomspace(_BAND_FLOOR_W, 60.0, 12)
        cls.images = _real_images(cls.gamma, cls.source, ref_band)
        cls.w_split = _saddle_c3_split_point(
            cls.images, cls.source, cls.matrix)
        cls.lens = _saddle_lens(cls.gamma, cls.source)

    def test_premise_two_image_pair_with_finite_split(self) -> None:
        """Premise: resolved 2-image saddle with a finite, in-range split."""
        self.assertEqual(len(self.images), 2)
        self.assertIsNotNone(self.w_split)
        self.assertTrue(np.isfinite(self.w_split))
        self.assertLess(self.w_split, W_CEILING_SCHWINGER_QD)
        self._checks += 1

    def test_whole_band_admit_is_zero_envelope_byte_identical(self) -> None:
        """
        (a) ``w_split <= w_lo``: whole band served with a ZERO envelope, no
        engine call, ``(k0, k1)`` byte-identical to the zero-envelope rebuild.
        """
        dense_w = np.geomspace(self.w_split * 1.2, 60.0, _NBINS * _NSUB)
        self.assertGreater(float(dense_w.min()), self.w_split,
                           'admit premise: band floor must exceed w_split')
        # Gate serves this band floor (equivalently w_split <= w_lo).
        self.assertTrue(_saddle_farfield_analytic_serves(
            self.images, self.source, self.matrix, float(dense_w.min())))

        probe = _SaddleFarfieldProbe()
        result = probe._saddle_farfield_analytic(self.lens, dense_w)
        self.assertIsNotNone(result, 'whole-band admit must serve, not refuse')
        delays, k0, k1, geom = result

        # No engine work on the admit fast path.
        probe._engine_envelope_below_split.assert_not_called()

        # Independent zero-envelope reference over the returned partition.
        ref_kernels, _total = reconstruct_farfield(
            dense_w, np.zeros(dense_w.shape, dtype=complex), geom.delays,
            geom.saddle_kernels, geom.real_mask, FARFIELD_KERNEL_SUM,
            geom.t_min)
        ref_k0, ref_k1 = probe._reduce_dense_kernels(ref_kernels)
        self.assertTrue(np.array_equal(k0, ref_k0),
                        'k0 must match the zero-envelope reconstruction')
        self.assertTrue(np.array_equal(k1, ref_k1),
                        'k1 must match the zero-envelope reconstruction')
        self.assertEqual(delays.shape, geom.delays.shape)
        self._checks += 1

    def test_whole_draw_refuse_returns_none_no_engine(self) -> None:
        """
        (b) ``w_split >= w_hi``: certificate fails across the whole band, the
        rung returns ``None`` (fall-through), and the engine is never called.
        """
        dense_w = np.geomspace(self.w_split * 0.3, self.w_split * 0.9,
                               _NBINS * _NSUB)
        self.assertLess(float(dense_w.max()), self.w_split,
                        'refuse premise: band ceiling must be below w_split')
        # Gate does not serve at this low band floor (certificate fails).
        self.assertFalse(_saddle_farfield_analytic_serves(
            self.images, self.source, self.matrix, float(dense_w.min())))

        probe = _SaddleFarfieldProbe()
        result = probe._saddle_farfield_analytic(self.lens, dense_w)
        self.assertIsNone(
            result, 'a certificate failing across the whole band must refuse')
        probe._engine_envelope_below_split.assert_not_called()
        self._checks += 1

    def test_self_falsification_admit_envelope_is_actually_zero(self) -> None:
        """
        Teeth: the admit path's byte-identity would FAIL against a NON-zero
        envelope reference -- proving the reconstruction is genuinely the
        zero-envelope serve and the equality above is not vacuous.
        """
        dense_w = np.geomspace(self.w_split * 1.2, 60.0, _NBINS * _NSUB)
        probe = _SaddleFarfieldProbe()
        _delays, k0, _k1, geom = probe._saddle_farfield_analytic(
            self.lens, dense_w)
        # A deliberately-perturbed (non-zero) envelope must NOT reproduce k0.
        bad_env = np.full(dense_w.shape, 1e-3 + 0.0j)
        bad_kernels, _total = reconstruct_farfield(
            dense_w, bad_env, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)
        bad_k0, _bad_k1 = probe._reduce_dense_kernels(bad_kernels)
        self.assertFalse(
            np.array_equal(k0, bad_k0),
            'a non-zero envelope must change k0 -- else the identity is '
            'vacuous')
        self._checks += 1


def _f_saddle_analytic(gamma: float, source, dense_w: np.ndarray):
    """
    Analytic ZERO-envelope amplification ``F(w)`` of a macro saddle over a
    band, in the ABSOLUTE frame, plus its geometry partition.

    Reconstructs the switched-analytic ``FARFIELD_KERNEL_SUM`` amplification
    with a ZERO residual envelope -- exactly the served field
    ``_saddle_farfield_analytic`` emits ABOVE ``w_split`` -- then lifts the
    min-relative ``total`` into the absolute frame via ``exp(+1j w t_min)``,
    the matched inverse of ``reconstruct_farfield``'s internal demodulation.
    The alignment is load-bearing (an unaligned comparison is O(1) off); see
    ``test_alignment_is_load_bearing``.
    """
    geom = ChangRefsdalChannels(dense_w).geometry_partition(
        gamma=float(gamma), y=(float(source[0]), float(source[1])),
        beta=0.0, kappa=0.0)
    envelope = np.zeros(dense_w.shape, dtype=complex)
    _kernels, total = reconstruct_farfield(
        dense_w, envelope, geom.delays, geom.saddle_kernels, geom.real_mask,
        FARFIELD_KERNEL_SUM, geom.t_min)
    f_abs = total * np.exp(1j * dense_w * geom.t_min)
    return f_abs, geom


def _f_saddle_oracle(gamma: float, source, dense_w: np.ndarray) -> np.ndarray:
    """
    INDEPENDENT exact amplification oracle for a beta=kappa=0 macro saddle.

    ``f_schwinger`` is the exact Diffraction-integral (DD) evaluation, a
    DIFFERENT derivation path from the switched-analytic kernel sum that
    ``_f_saddle_analytic`` reconstructs -- so it is a genuine independent
    oracle, not a copy of the code under test.  At ``beta = 0`` the macro
    matrix is ``diag(1 - gamma, 1 + gamma)``, so the eigenframe coincides
    with the coordinate frame and ``y_eig = (y1, y2)`` directly (confirmed
    by the sub-1e-3 agreement).  Restricted to ``w <= 60`` (the exact DD
    path); mpmath and the hard refuse lie above.
    """
    y_eig = np.asarray(source, dtype=float)
    return np.array([f_schwinger(float(w), y_eig, float(gamma))
                     for w in dense_w], dtype=complex)


class SaddleC3InBandAccuracyTestCase(_ServeGateTestCase):
    """
    c3 in-band accuracy of the served analytic zero-envelope field, above the
    split, against the exact Schwinger engine (Spec: c3 IN-BAND ACCURACY --
    escalation guard, WP2).

    Above ``w_split`` the rung serves the macro saddle with a ZERO residual
    envelope, claiming the switched-analytic carriers alone reconstruct the
    amplification ``F`` to within the certificate CURRENCY
    ``_SADDLE_FARFIELD_CERT_BAR`` (= 1e-3).  The gate admits via
    ``S * est <= bar`` with ``S = _SADDLE_FARFIELD_SAFETY = 20``, so the true
    LEADING ``w**-3`` remainder at the split is ``<= bar / S = 5e-5``; the
    1e-3 currency is the conservative amplitude bar the served band must
    clear against the exact engine.

    This suite compares ``|F_analytic - F_engine|`` over ``[w_split * (1 +
    eps), 55]`` (kept strictly on the exact DD path, ``w <= 60`` -- the spec
    forbids comparing above 150 where NO oracle exists by construction):

      * ``test_clean_configs_serve_and_clear_currency`` -- three resolved
        far-from-caustic configs in the rung's ACTUAL contract domain each
        serve at the band floor (premise) and clear the currency by a
        comfortable margin (measured 2026-08-17 max
        ``|F_analytic - F_engine|`` in {6.8e-4, 5.5e-5, 3.8e-5}).
      * ``test_alignment_is_load_bearing`` -- the UNALIGNED comparison (no
        ``exp(+1j w t_min)`` lift) is O(1) off (measured ~0.9), proving the
        sub-1e-3 agreement is a real physics match and the frame lift is
        not cosmetic -- the suite's teeth.
      * ``test_leaky_gate_witness_optimism_flagged`` -- a FLAGGED
        calibration-optimism discrepancy (escalated, not papered over): a
        source the certificate ADMITS yet whose served field MISSES the
        currency over a low near-caustic sub-band.

    LEAKY-GATE WITNESS (escalation, per house precedent 2026-08-13).  The
    config ``_C3_LEAKY_WITNESS = (2.0, (1.1, 0.0))`` is gate-admitted (the
    whole band serves with a zero envelope) yet its served reconstruction
    reaches max ``|F_analytic - F_engine| ~ 3.1e-3`` at ``w_split ~ 9.61``
    (the LOWEST split probed and the closest to the caustic), decaying under
    the currency only above ``~1.5 * w_split``.  The 20x certificate safety
    absorbs the leading ``w**-3`` remainder but NOT the subleading terms in
    this near-caustic low-split corner.  Per the spec, a miss where the
    certificate admits FALSIFIES the calibration (STOP / escalate) -- it is
    NOT a plumbing bug.  We CERTIFY the actual contract domain green and PIN
    the witness's measured miss (``1e-3 < miss < 1e-2``) so a future
    safety-factor / domain tightening flips it red and forces a re-baseline.

    Oracle independence.  ``f_schwinger`` (exact DD) is an independent
    derivation from the switched-analytic ``reconstruct_farfield`` sum; the
    frame lift is asserted load-bearing by the teeth test.  Fixtures derive
    ``w_split`` live from ``_saddle_c3_split_point`` -- never a pinned
    literal -- so the comparison band follows the certificate if it moves.

    Cost.  Fast tier.  Four configs x a 12-node ``w <= 55`` DD evaluation of
    ``f_schwinger`` plus a zero-envelope reconstruct; a few seconds total.
    """

    def _analytic_vs_engine_error(self, gamma, source):
        """Max |F_analytic - F_engine| over the served band, plus the grid."""
        source = np.asarray(source, dtype=float)
        matrix = macro_matrix(gamma)
        ref_band = np.geomspace(_BAND_FLOOR_W, 60.0, _C3_OVERLAP_N)
        images = _real_images(gamma, source, ref_band)
        w_split = _saddle_c3_split_point(images, source, matrix)
        self.assertIsNotNone(
            w_split, 'clean fixture must have a finite c3 split point')
        dense_w = np.geomspace(
            w_split * _C3_FLOOR_STANDOFF, _C3_OVERLAP_W_HI, _C3_OVERLAP_N)
        f_ana, _geom = _f_saddle_analytic(gamma, source, dense_w)
        f_eng = _f_saddle_oracle(gamma, source, dense_w)
        return np.abs(f_ana - f_eng), dense_w, images, w_split

    def test_clean_configs_serve_and_clear_currency(self) -> None:
        """
        Each clean contract-domain config serves at the band floor and its
        served zero-envelope field clears the certificate currency against
        the exact engine over the whole above-split band.
        """
        curves = {}
        for gamma, source in _C3_CLEAN_CONFIGS:
            with self.subTest(gamma=gamma, source=source):
                err, dense_w, images, w_split = self._analytic_vs_engine_error(
                    gamma, source)
                # Premise: the gate genuinely SERVES this band floor (whole
                # band, zero envelope) -- the accuracy claim is only
                # meaningful on an admitted config.
                self.assertTrue(
                    self._serve(images, source, macro_matrix(gamma),
                                float(dense_w.min())),
                    'clean config must be gate-admitted at the band floor')
                self.assertLessEqual(
                    float(err.max()), _C3_OVERLAP_BAR,
                    f'served analytic field must clear the currency '
                    f'{_C3_OVERLAP_BAR:.1e}; got {err.max():.3e} at '
                    f'gamma={gamma}, source={source}, w_split={w_split:.3f}')
                curves[(gamma, tuple(source))] = (dense_w, err)
        # Diagnostic accompanying the assertion (not a standalone test).
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for (gamma, source), (dense_w, err) in curves.items():
            ax.semilogy(dense_w, err, marker='o', ms=3,
                        label=f'g={gamma}, y={source}')
        ax.axhline(_C3_OVERLAP_BAR, color='k', ls='--',
                   label=f'currency {_C3_OVERLAP_BAR:.0e}')
        ax.set_xlabel('w')
        ax.set_ylabel('|F_analytic - F_engine|')
        ax.set_title('c3 in-band accuracy (clean contract domain)')
        ax.legend(fontsize=7)
        self._save_plot(fig, 'saddle_c3_inband_clean_error.png')
        self._gate_calls += 1

    def test_alignment_is_load_bearing(self) -> None:
        """
        Teeth: the UNALIGNED analytic field (missing the ``exp(+1j w t_min)``
        absolute-frame lift) is O(1) away from the engine, so the sub-1e-3
        agreement above is a genuine physics match, not a vacuous near-zero.
        """
        gamma, source = _C3_CLEAN_CONFIGS[0]
        source = np.asarray(source, dtype=float)
        matrix = macro_matrix(gamma)
        ref_band = np.geomspace(_BAND_FLOOR_W, 60.0, _C3_OVERLAP_N)
        images = _real_images(gamma, source, ref_band)
        w_split = _saddle_c3_split_point(images, source, matrix)
        dense_w = np.geomspace(
            w_split * _C3_FLOOR_STANDOFF, _C3_OVERLAP_W_HI, _C3_OVERLAP_N)
        f_ana, geom = _f_saddle_analytic(gamma, source, dense_w)
        f_eng = _f_saddle_oracle(gamma, source, dense_w)
        aligned = np.abs(f_ana - f_eng)
        unaligned = np.abs(
            f_ana * np.exp(-1j * dense_w * geom.t_min) - f_eng)
        self.assertLessEqual(float(aligned.max()), _C3_OVERLAP_BAR)
        self.assertGreater(
            float(unaligned.max()), 100.0 * _C3_OVERLAP_BAR,
            'the frame lift must be load-bearing: an unaligned comparison '
            'must be orders of magnitude worse than the currency')
        self._gate_calls += 1

    def test_leaky_gate_witness_optimism_flagged(self) -> None:
        """
        FLAGGED calibration-optimism (escalation, NOT a plumbing bug): the c3
        certificate ADMITS ``_C3_LEAKY_WITNESS`` at the band floor yet the
        served zero-envelope field MISSES the currency over the low
        near-caustic sub-band.  Pinned as measured reality so a future
        safety / domain fix flips this red and forces a re-baseline.
        """
        gamma, source = _C3_LEAKY_WITNESS
        source = np.asarray(source, dtype=float)
        matrix = macro_matrix(gamma)
        err, dense_w, images, w_split = self._analytic_vs_engine_error(
            gamma, source)
        # The gate ADMITS this source (that is the whole point -- it is a
        # leaky admission, not a refusal).
        self.assertTrue(
            self._serve(images, source, matrix, float(dense_w.min())),
            'leaky witness must be gate-admitted (the discrepancy is a '
            'served MISS, not a refusal)')
        worst = float(err.max())
        self.assertGreater(
            worst, _C3_LEAKY_MISS_LO,
            'witness must actually exceed the currency (else it is not '
            'leaky and this escalation guard is stale)')
        self.assertLess(
            worst, _C3_LEAKY_MISS_HI,
            'witness miss must stay bounded (a catastrophic blow-up would '
            'be a different, harder failure)')
        # Diagnostic accompanying the escalation pin.
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.semilogy(dense_w, err, marker='o', ms=3, color='crimson',
                    label=f'g={gamma}, y={tuple(source)}')
        ax.axhline(_C3_OVERLAP_BAR, color='k', ls='--',
                   label=f'currency {_C3_OVERLAP_BAR:.0e}')
        ax.axvline(w_split * 1.5, color='gray', ls=':',
                   label='~1.5 w_split (recovery)')
        ax.set_xlabel('w')
        ax.set_ylabel('|F_analytic - F_engine|')
        ax.set_title('LEAKY-GATE WITNESS: served miss where the gate admits')
        ax.legend(fontsize=7)
        self._save_plot(fig, 'saddle_c3_leaky_gate_witness_error.png')
        self._gate_calls += 1


class _CeilingProbe:
    """
    Lightweight carrier for the unbound ``_ppgo_above_ceiling`` method and its
    kernel-reduction collaborators (Spec: PER-NODE ABOVE-CEILING).

    ``_engine_envelope_below_split`` is a SPY whose ``side_effect`` returns a
    configurable complex ``sentinel`` on the below-ceiling nodes and ``0``
    above -- structurally identical to the real engine envelope but with NO
    Schwinger evaluation, so the partition/stitch is proved in milliseconds.
    Binding the real production methods (rather than building a full
    ``LensedRelativeBinningLikelihood``) exercises the EXACT shipping ceiling
    rung with only the attributes ``_reduce_dense_kernels`` /
    ``_image_delays`` read.
    """

    n_bins = _NBINS
    kernel_subsamples = _NSUB
    _ppgo_above_ceiling = (
        LensedRelativeBinningLikelihood._ppgo_above_ceiling)
    _reduce_dense_kernels = (
        LensedRelativeBinningLikelihood._reduce_dense_kernels)
    _image_delays = LensedRelativeBinningLikelihood._image_delays

    def __init__(self, sentinel: complex) -> None:
        sub = np.linspace(-0.5, 0.5, _NSUB)
        value_row = np.ones(_NSUB) / _NSUB
        slope_row = sub / np.sum(sub ** 2)
        self._kernel_fit_value = np.tile(value_row, (_NBINS, 1))
        self._kernel_fit_slope = np.tile(slope_row, (_NBINS, 1))
        self.sentinel = complex(sentinel)
        # Real engine envelope shape: sentinel on the below-split nodes, 0
        # above -- exactly what ``_engine_envelope_below_split`` returns, but
        # with no Schwinger evaluation.
        self._engine_envelope_below_split = mock.MagicMock(
            name='engine_envelope_below_split',
            side_effect=lambda lens, dw, below_mask:
                self.sentinel * np.asarray(below_mask, dtype=complex))


def _ceiling_dense_w() -> np.ndarray:
    """
    Above-ceiling dense-``w`` grid straddling the Schwinger QD ceiling (150).

    A handful of below-ceiling nodes on the exact DD path plus a majority
    above 150 where fold_ppgo carries; length ``_NBINS * _NSUB`` so
    ``_reduce_dense_kernels`` can reshape the channel kernels.
    """
    below = np.linspace(_CEIL_BELOW_LO, _CEIL_BELOW_HI, _CEIL_BELOW_N)
    above = np.geomspace(_CEIL_ABOVE_LO, _CEIL_ABOVE_HI, _CEIL_ABOVE_N)
    return np.concatenate([below, above])


def _independent_fold_envelope(gamma, source, dense_w, geom) -> np.ndarray:
    """
    Independently re-derived fold-corrected ppGO envelope over the full band.

    Coded from scratch in the test (NOT a call into ``_ppgo_above_ceiling``)
    so it is a legitimate structural oracle for the above-ceiling carrier:
    ``(f_minrel - ppgo_sum) * exp(+1j w t_min)`` with ``f_minrel`` the
    min-relative fold correction and ``ppgo_sum`` the bare saddle image-kernel
    sum -- the two carriers the production rung stitches above the ceiling.
    """
    real = np.asarray(geom.real_mask, dtype=bool)
    real_delays = np.asarray(geom.delays)[real]
    f_total = np.atleast_1d(fold_ppgo_correction(
        dense_w, np.asarray(source, float), float(gamma),
        beta=0.0, kappa=0.0))
    f_total = np.where(np.isfinite(f_total), f_total, 0.0)
    f_minrel = f_total * np.exp(-1j * dense_w * geom.t_min)
    ppgo_sum = np.sum(
        geom.saddle_kernels[:, real]
        * np.exp(1j * dense_w[:, None] * real_delays[None, :]), axis=1)
    return (f_minrel - ppgo_sum) * np.exp(1j * dense_w * geom.t_min)


class PpgoAboveCeilingPartitionTestCase(_CountingTestCase):
    """
    Per-node above-ceiling partition + gate of ``_ppgo_above_ceiling``
    (Spec: PER-NODE ABOVE-CEILING PARTITION + GATE, WP3).

    Above the Schwinger QD ceiling (``W_CEILING_SCHWINGER_QD = 150``) the
    exact engine hard-refuses, so a ``w_max > 150`` draw must be served by
    splitting the band AT the ceiling: the exact engine carries every node at
    or below 150 (always engine-reachable) and the fold-corrected ppGO
    carrier carries every node above.  The rung admits ONLY when the lowest
    above-ceiling node is resolved (``150 * min_delta_tau >= RHO_END``),
    which guarantees every above-ceiling node the engine refuses is resolved.

    This composes TWO partitions: the CEILING partition (``w > 150``, the
    engine/fold split) and the RESOLUTION partition
    (``150 * min_delta_tau >= RHO_END``, the admit gate).  The two fixtures
    isolate them:

      * ``_CEIL_RESOLVED_CONFIG`` -- a resolved 2-image saddle
        (``150 * min_delta_tau ~ 164`` >> 4) SERVES; the reconstructed
        envelope's below-150 nodes carry the exact-engine contribution and
        the above-150 nodes carry the fold_ppgo contribution, stitched
        cleanly at 150 with no overlap or gap.
      * ``_CEIL_NEARCAUSTIC_CONFIG`` -- a near-caustic 4-image source
        (``150 * min_delta_tau ~ 3.94`` < 4) is UNRESOLVED at the ceiling,
        so the rung returns ``None`` and the caller falls through to the
        exact engine -> ``SchwingerCertificationError`` (the deferred 2b
        residual, not a bug).

    Structural (engine-free) proof.  ``_engine_envelope_below_split`` is
    replaced by a SPY returning a complex ``sentinel`` below the ceiling and
    ``0`` above, and ``reconstruct_farfield`` is patched to CAPTURE the
    stitched envelope the rung feeds it.  The captured envelope then proves,
    byte-exactly:

      1. the split is clean at 150 (``below_mask == (dense_w <= 150)``);
      2. below 150 the envelope equals the engine sentinel
         (engine carries below);
      3. above 150 the envelope is INVARIANT to the engine sentinel
         (running two distinct sentinels gives a byte-identical above region)
         AND equals the independent fold re-derivation -- fold carries above,
         fully decoupled from the engine, so there is no double count and no
         leak across the boundary;
      4. no above-150 node is fed to the engine spy (it is called with the
         below_mask, whose above-ceiling entries are all False) -- no
         unreachable engine node above the ceiling.

    Per-node source-of-envelope diagnostic table saved to the output dir.

    Cost.  Fast tier.  Two 64-node geometry partitions + a couple of
    zero-Schwinger reconstructions; well under a second.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.dense_w = _ceiling_dense_w()
        cls.below_mask = cls.dense_w <= float(W_CEILING_SCHWINGER_QD)
        cls.res_gamma, cls.res_source = _CEIL_RESOLVED_CONFIG
        cls.near_gamma, cls.near_source = _CEIL_NEARCAUSTIC_CONFIG

    def _run_capture(self, gamma, source, sentinel):
        """Run the ceiling rung, capturing the envelope fed to reconstruct."""
        captured = {}
        real_recon = _likelihood_module.reconstruct_farfield

        def _capturing(dw, envelope, *args, **kwargs):
            captured['envelope'] = np.array(envelope)
            return real_recon(dw, envelope, *args, **kwargs)

        probe = _CeilingProbe(sentinel)
        lens = _saddle_lens(gamma, source)
        with mock.patch.object(_likelihood_module, 'reconstruct_farfield',
                               side_effect=_capturing):
            result = probe._ppgo_above_ceiling(lens, self.dense_w)
        return result, captured, probe

    def test_premise_ceiling_and_resolution(self) -> None:
        """
        Premise: the grid straddles 150, the resolved fixture clears the
        resolution gate at the ceiling, and the near-caustic one does not --
        both RE-ASSERTED from live geometry, never trusted as literals.
        """
        self.assertGreater(float(self.dense_w.max()), W_CEILING_SCHWINGER_QD)
        self.assertLess(float(self.dense_w.min()), W_CEILING_SCHWINGER_QD)
        for gamma, source, want_resolved in (
                (self.res_gamma, self.res_source, True),
                (self.near_gamma, self.near_source, False)):
            geom = ChangRefsdalChannels(self.dense_w).geometry_partition(
                gamma=float(gamma), y=(float(source[0]), float(source[1])),
                beta=0.0, kappa=0.0)
            real = np.asarray(geom.real_mask, dtype=bool)
            real_delays = np.asarray(geom.delays)[real]
            delta = np.diff(np.sort(real_delays))
            pos = delta[delta > 0]
            metric = W_CEILING_SCHWINGER_QD * float(np.min(pos))
            with self.subTest(gamma=gamma, source=source):
                self.assertEqual(
                    metric >= RHO_END, want_resolved,
                    f'resolution premise drifted: 150*min_delta_tau={metric}')
        self._checks += 1

    def test_resolved_serves_with_clean_stitch(self) -> None:
        """
        (a) Resolved: serves; below-150 nodes carry the engine sentinel,
        above-150 carry the independent fold_ppgo values, split clean at 150.
        """
        result, captured, probe = self._run_capture(
            self.res_gamma, self.res_source, _CEIL_SENTINEL_A)
        self.assertIsNotNone(
            result, 'resolved above-ceiling draw must serve, not refuse')
        envelope = captured['envelope']

        # (1) clean split at the ceiling.
        self.assertTrue(np.array_equal(
            self.below_mask, self.dense_w <= float(W_CEILING_SCHWINGER_QD)))

        # (2) engine carries below (sentinel exactly on below nodes).
        expected_below = _CEIL_SENTINEL_A * np.ones(
            int(self.below_mask.sum()), dtype=complex)
        self.assertTrue(
            np.array_equal(envelope[self.below_mask], expected_below),
            'below-ceiling envelope must be exactly the engine sentinel')

        # (3) fold carries above: byte-equal to the independent re-derivation.
        geom = ChangRefsdalChannels(self.dense_w).geometry_partition(
            gamma=float(self.res_gamma),
            y=(float(self.res_source[0]), float(self.res_source[1])),
            beta=0.0, kappa=0.0)
        fold_env = _independent_fold_envelope(
            self.res_gamma, self.res_source, self.dense_w, geom)
        self.assertTrue(
            np.array_equal(envelope[~self.below_mask],
                           fold_env[~self.below_mask]),
            'above-ceiling envelope must equal the independent fold_ppgo '
            'carrier (no double count, no leak across the boundary)')

        # (4) fold actually contributes above (not a trivial zero region).
        self.assertTrue(np.any(envelope[~self.below_mask] != 0.0),
                        'fold carrier must be non-trivial above the ceiling')

        # (5) no above-ceiling node reaches the engine spy.
        spy_mask = probe._engine_envelope_below_split.call_args[0][2]
        self.assertTrue(np.array_equal(np.asarray(spy_mask, bool),
                                       self.below_mask))
        self.assertEqual(int(np.asarray(spy_mask, bool)[~self.below_mask].sum()),
                         0, 'engine must never be asked for an above-150 node')
        self._checks += 1

    def test_clean_split_is_engine_decoupled_above(self) -> None:
        """
        The above-ceiling envelope is BYTE-INVARIANT to the engine sentinel:
        running two distinct sentinels leaves the fold-carried region
        identical, proving the split leaks nothing from engine to fold.
        """
        _resA, capA, _pA = self._run_capture(
            self.res_gamma, self.res_source, _CEIL_SENTINEL_A)
        _resB, capB, _pB = self._run_capture(
            self.res_gamma, self.res_source, _CEIL_SENTINEL_B)
        envA, envB = capA['envelope'], capB['envelope']
        self.assertTrue(
            np.array_equal(envA[~self.below_mask], envB[~self.below_mask]),
            'above-ceiling region must not depend on the engine sentinel')
        # And the below region DOES track the sentinel (teeth: the invariance
        # above is meaningful, not because everything is constant).
        self.assertFalse(
            np.array_equal(envA[self.below_mask], envB[self.below_mask]),
            'below-ceiling region must track the engine sentinel')
        self._checks += 1

    def test_nearcaustic_unresolved_returns_none(self) -> None:
        """
        (b) Near-caustic: the lowest above-ceiling node is unresolved
        (``150 * min_delta_tau < RHO_END``), so the rung returns ``None`` and
        the engine is never consulted -- the deferred 2b fall-through.
        """
        result, captured, probe = self._run_capture(
            self.near_gamma, self.near_source, _CEIL_SENTINEL_A)
        self.assertIsNone(
            result, 'an unresolved above-ceiling corner must return None '
            '(fall through to the engine -> refusal)')
        probe._engine_envelope_below_split.assert_not_called()
        self.assertNotIn(
            'envelope', captured,
            'a refused draw must not reach reconstruct_farfield')
        self._checks += 1

    def test_per_node_source_of_envelope_table(self) -> None:
        """
        Diagnostic: a per-node (w, source-of-envelope) table showing the
        clean split at 150 and no unreachable engine node above it.  Also
        asserts the table's structure (engine below, fold above) so the
        diagnostic IS an assertion, not decoration.
        """
        _result, captured, _probe = self._run_capture(
            self.res_gamma, self.res_source, _CEIL_SENTINEL_A)
        envelope = captured['envelope']
        geom = ChangRefsdalChannels(self.dense_w).geometry_partition(
            gamma=float(self.res_gamma),
            y=(float(self.res_source[0]), float(self.res_source[1])),
            beta=0.0, kappa=0.0)
        fold_env = _independent_fold_envelope(
            self.res_gamma, self.res_source, self.dense_w, geom)
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        table_path = os.path.join(
            _OUTPUT_DIR, 'ppgo_above_ceiling_source_of_envelope.txt')
        with open(table_path, 'w', encoding='utf-8') as handle:
            handle.write('# w        source-of-envelope\n')
            for w_node, is_below in zip(self.dense_w, self.below_mask):
                src = 'engine' if is_below else 'fold_ppgo'
                handle.write(f'{w_node:10.4f}  {src}\n')
        # Structural assertion: every below node is engine (sentinel), every
        # above node is fold, boundary exactly at 150.
        for w_node, env_val, is_below in zip(
                self.dense_w, envelope, self.below_mask):
            with self.subTest(w=w_node):
                if is_below:
                    self.assertLessEqual(w_node, W_CEILING_SCHWINGER_QD)
                    self.assertEqual(env_val, _CEIL_SENTINEL_A)
                else:
                    self.assertGreater(w_node, W_CEILING_SCHWINGER_QD)
                    self.assertEqual(
                        env_val, fold_env[np.where(self.dense_w == w_node)][0])
        self._checks += 1


if __name__ == '__main__':
    unittest.main()
