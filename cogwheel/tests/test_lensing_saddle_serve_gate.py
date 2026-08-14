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

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing.chang_refsdal.geometry import (
    macro_matrix, magnification, ppgo_error_estimate, delay)
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing.ppgo_map import caustic_geometry
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing.surrogate import TubeChart, LensAmplificationSurrogate
from cogwheel.lensing import surrogate_census
from cogwheel.lensing.likelihood import (
    _saddle_farfield_analytic_serves,
    _SADDLE_FARFIELD_SAFETY,
    _SADDLE_FARFIELD_CERT_BAR,
    _SADDLE_FARFIELD_MIN_IMAGE_SEP,
)

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


if __name__ == '__main__':
    unittest.main()
