"""
Tests for the interior fold-ppGO certificate layer of the Chang--Refsdal
lens: the exact four-real-image interior predicate (``real_mask.sum() ==
4``), the ported ``_series_coefficients`` third coefficient ``c3``, and
``geometry.ppgo_error_estimate`` (the ``sum_a sqrt|mu_a| |c3_a| /
w_min**3`` worst-case bound a gate reads to admit or refuse raw ppGO on a
band).

WHY THESE ORACLES ARE INDEPENDENT
---------------------------------
Every assertion is judged against something the code under test does NOT
compute for itself:

* the INTERIOR PREDICATE is gated against the closed-form caustic reach
  ``geometry.r_caustic`` -- an image census (``real_mask.sum()``) checked
  against a source-plane boundary the census never touches; a source at
  ``rho = |y|/r_caustic < 1`` must give four real images, one at
  ``rho > 1`` exactly two, and the census must FLIP within ~1e-3 of the
  ``rho = 1`` crossing.  A mislabelled image or a wrong caustic reach is
  the only way this can fail.
* ``c1`` and ``c2`` are cross-checked between the TWO genuinely
  independent shipped derivations -- ``saddle_coefficients`` (closed-form
  ``_c1_polynomial`` / ``_c2_polynomial`` in the radius-aligned frame) and
  ``_series_coefficients`` (a Gaussian moment-table polynomial algebra).
  Neither is derived from the other, so agreement to 1e-12 rules out a
  transcription error in either path.
* ``c3`` being PURELY IMAGINARY is an analytic property of the leading
  omitted stationary-phase term -- no external oracle at all; a real-part
  leak is a transcription bug in the polynomial algebra.
* the ``w**-3`` SCALING is a closed-form identity: the estimate is exactly
  ``(sum_a sqrt|mu_a| |c3_a|) / w_min**3``, so the ratio at two
  frequencies is the pure cube ``(w2/w1)**3``, independent of the
  ``c3``/``mu`` content that is otherwise the hard part.  Differentiating
  the closed form (rather than fitting a slope) is exact.

TOLERANCES
----------
The C1/C2 cross-check bar is the Architect's 1e-12; the two shipped
derivations agree to ~1e-15/1e-14 (three orders of headroom).  The
purely-imaginary bar ``abs(c3.real) < 1e-9 * (abs(c3.imag) + 1)`` is
generous against a measured exact-zero real part.  The ``w**-3`` ratio bar
is 1e-12 against a measured ~1e-16 (pure float64 division roundoff).  The
caustic-flip offset bar is the Architect's 1e-3 against a measured ~1e-13
(the bisection resolves the boundary far tighter than the spec asks).

SELF-FALSIFICATION
------------------
`PpgoCertificateSelfFalsificationTestCase` proves each gate has teeth: a
wrong caustic reach mislabels the interior/exterior census, a real
perturbation of ``c3`` trips the purely-imaginary gate, and a wrong
exponent breaks the ``w**-3`` ratio.  Its ``_expect_checks = False`` opts
it out of the anti-vacuity tally (it asserts that a WRONG oracle would be
caught, so it must not be counted as a real comparison).
"""
from __future__ import annotations

import itertools
import math
import os
import unittest

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import _schwinger
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing.ppgo_map import CERTIFICATION_BAR

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:  # pragma: no cover - plotting is a diagnostic nicety
    _HAVE_MPL = False

#: Directory for diagnostic plots (created lazily by the plotting helpers).
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

#: Interior configs (gamma, theta) whose caustic reach places a 4-image
#: source at ``rho = 0.5`` and a 2-image source at ``rho = 1.5``.  Chosen
#: on distinct rays and shears so a shared bug cannot hide.
_LEG1_CONFIGS: tuple[tuple[float, float], ...] = (
    (0.3, 0.6),
    (0.5, 1.1),
    (0.7, 0.9),
    (0.4, 2.2),
)

#: Interior source rays (gamma, theta) whose ``rho = 0.5`` source has four
#: real images; each exercises the c3/C1/C2 cross-check on four distinct
#: image positions at a fixed matrix.
_C3_SOURCES: tuple[tuple[float, float], ...] = (
    (0.3, 0.6),
    (0.5, 1.1),
    (0.7, 0.9),
)

#: rel-tol for the two independent C1/C2 derivations (Architect's bar).
_C1C2_RTOL = 1e-12
#: rel-tol for the closed-form ``w**-3`` ratio identity (Architect's bar).
_W3_RTOL = 1e-12
#: Absolute bar on the caustic-flip offset (Architect's bar).
_FLIP_OFFSET_BAR = 1e-3

#: Frequency points for the served-ppGO vs f_schwinger consistency specs.
#: All ``<= W_CEILING_SCHWINGER`` (60) so the oracle stays on the exact
#: double-double path -- a few evals, seconds-scale.
_CONSISTENCY_W: tuple[float, ...] = (20.0, 40.0, 60.0)

#: STRICT-interior configs (gamma, theta): the ``rho = 0.5`` source is
#: four-real-image and NOT near the caustic boundary.  ~5 configs x 3 w =
#: ~15 oracle evals.  These certify the certificate is CONSERVATIVE
#: (Fact 3: measured worst ratio ~0.98) against the exact oracle.
_INTERIOR_CONSISTENCY_CONFIGS: tuple[tuple[float, float], ...] = (
    (0.3, 0.6),
    (0.5, 1.1),
    (0.7, 0.9),
    (0.4, 2.2),
    (0.6, 1.8),
)

#: NEAR-caustic configs (gamma, theta, rho): a four-real-image source
#: DELIBERATELY placed at ``rho = |y| / r_caustic`` in ``[0.9, 0.99]`` so
#: two images are merging (small delay-gap ``xi_min``, large ``sqrt|mu|``)
#: and ``c3`` diverges.  ~3 configs x 3 w = ~9 oracle evals.  These certify
#: the certificate SELF-REFUSES before it can go optimistic.
_NEAR_CAUSTIC_CONFIGS: tuple[tuple[float, float, float], ...] = (
    (0.3, 0.6, 0.95),
    (0.5, 1.1, 0.90),
    (0.7, 0.9, 0.95),
)

#: Float slack on the conservativeness ratio ``true_err / cert``.  The
#: invariant is ``true_err <= cert`` (ratio <= 1.0); ``1.02`` absorbs pure
#: float64 roundoff without admitting a genuinely optimistic certificate.
_RATIO_SLACK = 1.02


def _served_and_oracle(gamma: float, theta: float, rho: float,
                       w: float) -> tuple[int, float, float | None, float]:
    """Served raw ppGO vs the exact ``f_schwinger`` oracle at one point.

    For ``kappa = 0``, ``beta = 0`` positive parity the mass-sheet +
    eigenframe reconstruction collapses to ``lam = 1``, ``y_eig = y``,
    ``gamma' = gamma``, ``mass_sheet_phase = 1`` -- so the exact oracle is
    simply ``f_schwinger(w, y, gamma)`` (an INDEPENDENT high-order
    quadrature, not the stationary-phase sum the served ppGO truncates).

    Both the served kernel-sum and the oracle are demodulated by the
    common origin ``exp(-1j w t_min)`` with ``t_min`` the minimum real-image
    Fermat delay, exactly as the fold-ppGO serve does.  Returns
    ``(n_real_images, true_err, cert, xi_min)`` where ``true_err =
    |served - oracle|``, ``cert = ppgo_error_estimate(...)`` and ``xi_min =
    w * min consecutive real-image delay-gap`` (the resolution measure --
    small when two images are merging).
    """
    source = _source_on_ray(gamma, theta, rho)
    matrix = geometry.macro_matrix(gamma)
    images = geometry.find_images(source, matrix)
    delays = sorted(geometry.delay(image, source, matrix) for image in images)
    t_min = delays[0]
    gaps = [hi - lo for lo, hi in zip(delays[:-1], delays[1:])]
    xi_min = w * min(gaps) if gaps else 0.0

    served = sum(
        (geometry.image_kernel(w, image, matrix)
         * np.exp(1j * w * geometry.delay(image, source, matrix)))
        for image in images)
    oracle = _schwinger.f_schwinger(w, source, gamma)  # lam=1 -> F itself
    demod = np.exp(-1j * w * t_min)
    true_err = float(abs(complex(served) * demod - complex(oracle) * demod))
    cert = geometry.ppgo_error_estimate(images, source, matrix, w)
    return len(images), true_err, cert, float(xi_min)


def _source_on_ray(gamma: float, theta: float, rho: float,
                   *, kappa: float = 0.0) -> np.ndarray:
    """Source at fractional caustic reach ``rho`` along ray ``theta``.

    ``rho < 1`` is strictly interior (four real images), ``rho > 1``
    strictly exterior (two).  Derived from the LIVE closed-form caustic
    reach so the fixture follows the boundary if the reach ever moves.
    """
    reach = geometry.r_caustic(gamma, theta, kappa=kappa)
    return rho * reach * np.array([math.cos(theta), math.sin(theta)])


def _real_image_count(gamma: float, y: np.ndarray,
                      *, kappa: float = 0.0) -> int:
    """Number of real images ``real_mask.sum()`` for a source."""
    channels = ChangRefsdalChannels(np.array([10.0, 20.0]))
    channels.reset()  # in-place deterministic labeling; NOT chainable
    partition = channels.geometry_partition(gamma=gamma, y=y, kappa=kappa)
    return int(partition.real_mask.sum())


class CertificateTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison tally.

    Every concrete test increments ``self.n_checks`` once per real
    comparison; ``tearDown`` FAILS if a test that is supposed to compare
    something actually ran zero comparisons -- the guard that stops a
    silently-skipping suite from reading green.  Self-falsification
    classes set ``_expect_checks = False`` to opt out.
    """

    _expect_checks: bool = True

    def setUp(self) -> None:
        self.n_checks = 0

    def tearDown(self) -> None:
        if self._expect_checks:
            self.assertGreater(
                self.n_checks, 0,
                'anti-vacuity: test ran zero real comparisons')


class Leg1InteriorPredicateTestCase(CertificateTestCase):
    """The four-real-image predicate agrees with the closed-form caustic.

    An interior source (``rho = 0.5``) must yield exactly four real images
    (predicate ``real_mask.sum() == 4`` True); an exterior source
    (``rho = 1.5``) exactly two (predicate False).  These are the exact
    configs on which WP2 re-gates the interior fold-ppGO rung.
    """

    def test_interior_sources_give_four_real_images(self) -> None:
        for gamma, theta in _LEG1_CONFIGS:
            with self.subTest(gamma=gamma, theta=theta):
                y = _source_on_ray(gamma, theta, 0.5)
                count = _real_image_count(gamma, y)
                self.assertEqual(
                    count, 4,
                    f'interior rho=0.5 source at (gamma={gamma}, '
                    f'theta={theta}) gave {count} real images, not 4')
                self.n_checks += 1

    def test_exterior_sources_give_two_real_images(self) -> None:
        for gamma, theta in _LEG1_CONFIGS:
            with self.subTest(gamma=gamma, theta=theta):
                y = _source_on_ray(gamma, theta, 1.5)
                count = _real_image_count(gamma, y)
                self.assertEqual(
                    count, 2,
                    f'exterior rho=1.5 source at (gamma={gamma}, '
                    f'theta={theta}) gave {count} real images, not 2')
                self.n_checks += 1

    def test_predicate_flips_at_closed_form_caustic(self) -> None:
        # Bisect rho on the interior->exterior census flip along a ray and
        # confirm it coincides with the closed-form boundary rho = 1 to
        # within the Architect's 1e-3.  The reach itself defines rho = 1,
        # so the measured flip is a pure census/caustic agreement check.
        gamma, theta = 0.3, 0.6
        lo, hi = 0.5, 1.5  # 4 images at lo, 2 images at hi
        self.assertEqual(_real_image_count(gamma, _source_on_ray(
            gamma, theta, lo)), 4)
        self.assertEqual(_real_image_count(gamma, _source_on_ray(
            gamma, theta, hi)), 2)
        for _ in range(60):  # 2^-60 << 1e-3; terminates on machine eps
            mid = 0.5 * (lo + hi)
            if _real_image_count(gamma, _source_on_ray(
                    gamma, theta, mid)) == 4:
                lo = mid
            else:
                hi = mid
        offset = abs(0.5 * (lo + hi) - 1.0)
        self.assertLess(
            offset, _FLIP_OFFSET_BAR,
            f'census flip at rho={0.5 * (lo + hi):.6f} is {offset:.2e} '
            f'from the closed-form caustic rho=1 (bar {_FLIP_OFFSET_BAR})')
        self.n_checks += 1
        self._plot_flip(gamma, theta)

    def _plot_flip(self, gamma: float, theta: float) -> None:
        if not _HAVE_MPL:  # pragma: no cover - diagnostic only
            return
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        rhos = np.linspace(0.3, 1.7, 71)
        counts = [_real_image_count(gamma, _source_on_ray(gamma, theta, r))
                  for r in rhos]
        signed = rhos - 1.0  # signed distance-to-caustic in rho units
        fig, ax = plt.subplots()
        ax.scatter(signed, counts, s=12)
        ax.axvline(0.0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('signed distance to caustic (rho - 1)')
        ax.set_ylabel('real_mask.sum()')
        ax.set_title(f'interior predicate flip (gamma={gamma}, '
                     f'theta={theta})')
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'test_predicate_flips_caustic_scatter.png'),
            dpi=110, bbox_inches='tight')
        plt.close(fig)


def _interior_real_images(gamma: float, theta: float) -> np.ndarray:
    """Four real images of the ``rho = 0.5`` interior source on a ray.

    The source is placed at half the closed-form caustic reach, so the
    fixture is interior by construction and follows the caustic boundary
    if the reach ever moves.
    """
    source = _source_on_ray(gamma, theta, 0.5)
    matrix = geometry.macro_matrix(gamma)
    return geometry.find_images(source, matrix)


class C3CoefficientTestCase(CertificateTestCase):
    """``c3`` reproduces the shipped C1/C2 and is purely imaginary.

    ``_series_coefficients`` returns ``(c1, c2, c3)`` with ``c1 = 1j*C1``
    and ``c2 = C2`` in the shipped convention.  Those first two are
    cross-checked against ``saddle_coefficients`` -- an INDEPENDENT
    closed-form derivation -- to 1e-12.  ``c3``, the leading omitted
    ``w**-3`` term, must be purely imaginary; a real part is a
    transcription bug in the polynomial algebra.
    """

    def test_c1_c2_match_independent_saddle_coefficients(self) -> None:
        for gamma, theta in _C3_SOURCES:
            images = _interior_real_images(gamma, theta)
            self.assertEqual(
                len(images), 4,
                f'ray (gamma={gamma}, theta={theta}) is not interior')
            matrix = geometry.macro_matrix(gamma)
            for idx, image in enumerate(images):
                with self.subTest(gamma=gamma, theta=theta, image=idx):
                    c1, c2, _c3 = geometry._series_coefficients(image, matrix)
                    big_c1, big_c2 = geometry.saddle_coefficients(image, matrix)
                    # c1 = 1j * C1, c2 = C2 (shipped-kernel convention).
                    self.assertTrue(
                        np.isclose(c1, 1j * big_c1, rtol=_C1C2_RTOL, atol=0.0),
                        f'c1={c1} != 1j*C1={1j * big_c1}')
                    self.assertTrue(
                        np.isclose(c2, big_c2, rtol=_C1C2_RTOL, atol=0.0),
                        f'c2={c2} != C2={big_c2}')
                    self.n_checks += 1

    def test_c3_is_purely_imaginary(self) -> None:
        ratios: list[float] = []
        for gamma, theta in _C3_SOURCES:
            images = _interior_real_images(gamma, theta)
            matrix = geometry.macro_matrix(gamma)
            for idx, image in enumerate(images):
                with self.subTest(gamma=gamma, theta=theta, image=idx):
                    _c1, _c2, c3 = geometry._series_coefficients(image, matrix)
                    self.assertLess(
                        abs(c3.real), 1e-9 * (abs(c3.imag) + 1.0),
                        f'c3={c3} has a non-negligible real part')
                    self.n_checks += 1
                    if c3.imag != 0.0:
                        ratios.append(abs(c3.real) / abs(c3.imag))
        self._plot_realimag(ratios)

    def _plot_realimag(self, ratios: list[float]) -> None:
        if not _HAVE_MPL or not ratios:  # pragma: no cover - diagnostic only
            return
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        fig, ax = plt.subplots()
        ax.scatter(range(len(ratios)), ratios, s=14)
        ax.set_xlabel('image index (all interior configs)')
        ax.set_ylabel('abs(c3.real) / abs(c3.imag)')
        ax.set_title('c3 real-part leakage (should be ~0)')
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'test_c3_purely_imaginary_ratio.png'),
            dpi=110, bbox_inches='tight')
        plt.close(fig)


class PpgoErrorEstimateTestCase(CertificateTestCase):
    """``ppgo_error_estimate`` has exact ``w**-3`` scaling and refuses junk.

    The estimate is exactly ``(sum_a sqrt|mu_a| |c3_a|) / w_min**3`` over
    the real images, so the ratio at two frequencies is the pure cube
    ``(w2/w1)**3`` -- independent of the (hard) ``c3``/``mu`` content.  It
    returns ``None`` for a degenerate input: ``w_min <= 0`` or any
    non-finite magnification / ``c3`` (a gate reads ``None`` as "refuse").
    """

    def setUp(self) -> None:
        super().setUp()
        # One interior 4-image config, fixed source/matrix.
        self.gamma, self.theta = 0.3, 0.6
        self.source = _source_on_ray(self.gamma, self.theta, 0.5)
        self.matrix = geometry.macro_matrix(self.gamma)
        self.real_images = geometry.find_images(self.source, self.matrix)
        self.assertEqual(len(self.real_images), 4,
                         'w**-3 fixture must be an interior 4-image source')

    def test_estimate_scales_as_w_minus_three(self) -> None:
        for w1, w2 in ((20.0, 60.0), (15.0, 90.0)):
            with self.subTest(w1=w1, w2=w2):
                est1 = geometry.ppgo_error_estimate(
                    self.real_images, self.source, self.matrix, w1)
                est2 = geometry.ppgo_error_estimate(
                    self.real_images, self.source, self.matrix, w2)
                self.assertIsNotNone(est1)
                self.assertIsNotNone(est2)
                # est(w1)/est(w2) == (w2/w1)**3, exactly (closed form).
                self.assertTrue(
                    np.isclose(est1 / est2, (w2 / w1) ** 3,
                               rtol=_W3_RTOL, atol=0.0),
                    f'ratio {est1 / est2} != cube {(w2 / w1) ** 3}')
                self.n_checks += 1
        self._plot_slope()

    def test_estimate_is_positive_and_finite(self) -> None:
        est = geometry.ppgo_error_estimate(
            self.real_images, self.source, self.matrix, 20.0)
        self.assertIsNotNone(est)
        self.assertTrue(math.isfinite(est) and est > 0.0,
                        f'estimate {est} is not finite-positive')
        self.n_checks += 1

    def test_none_for_nonpositive_w_min(self) -> None:
        for w_min in (0.0, -5.0):
            with self.subTest(w_min=w_min):
                est = geometry.ppgo_error_estimate(
                    self.real_images, self.source, self.matrix, w_min)
                self.assertIsNone(
                    est, f'w_min={w_min} must refuse (None), got {est}')
                self.n_checks += 1

    def test_none_for_nonfinite_magnification(self) -> None:
        # A NaN-coordinate image drives magnification / c3 non-finite;
        # the estimate must refuse rather than propagate NaN.
        bad = np.array([[np.nan, 0.5]])
        est = geometry.ppgo_error_estimate(
            bad, self.source, self.matrix, 20.0)
        self.assertIsNone(est, f'non-finite image must refuse, got {est}')
        self.n_checks += 1

    def _plot_slope(self) -> None:
        if not _HAVE_MPL:  # pragma: no cover - diagnostic only
            return
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        w_grid = np.geomspace(10.0, 200.0, 20)
        ests = [geometry.ppgo_error_estimate(
            self.real_images, self.source, self.matrix, w) for w in w_grid]
        fig, ax = plt.subplots()
        ax.loglog(w_grid, ests, 'o-', ms=4)
        ax.set_xlabel('w_min')
        ax.set_ylabel('ppgo_error_estimate')
        ax.set_title('certificate slope (should be -3)')
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'test_ppgo_error_estimate_slope.png'),
            dpi=110, bbox_inches='tight')
        plt.close(fig)


class InteriorCertificateConservativeTestCase(CertificateTestCase):
    """Certificate is CONSERVATIVE vs ``f_schwinger`` on the true interior.

    On ~5 strict-interior four-image configs (fixed matrices, source at
    ``rho = 0.5`` -- well inside the caustic) at ``w in {20, 40, 60}`` the
    certificate ``ppgo_error_estimate`` must NEVER go optimistic: the
    measured ``true_err = |served - oracle|`` obeys ``true_err <= cert``
    (ratio ``<= 1.0``, ``1.02`` float slack).  The oracle is the exact
    ``f_schwinger`` quadrature -- an independent evaluation of the same
    amplification the served raw ppGO only truncates.  Fact 3: the measured
    worst ratio is ~0.98; a point above 1.0 pinpoints an optimistic
    certificate that a gate would trust wrongly.
    """

    def test_certificate_bounds_true_error_on_interior(self) -> None:
        ratios: list[tuple[float, float]] = []  # (w, ratio) for the plot
        for gamma, theta in _INTERIOR_CONSISTENCY_CONFIGS:
            for w in _CONSISTENCY_W:
                with self.subTest(gamma=gamma, theta=theta, w=w):
                    n_img, true_err, cert, _xi = _served_and_oracle(
                        gamma, theta, 0.5, w)
                    self.assertEqual(
                        n_img, 4,
                        f'(gamma={gamma}, theta={theta}) rho=0.5 is not a '
                        f'four-image interior source ({n_img} images)')
                    # On the true interior the certificate never refuses;
                    # a None here would be a degenerate-input bug.
                    self.assertIsNotNone(
                        cert, 'certificate refused a strict-interior source')
                    ratio = true_err / cert
                    self.assertLessEqual(
                        ratio, _RATIO_SLACK,
                        f'OPTIMISTIC certificate at (gamma={gamma}, '
                        f'theta={theta}, w={w}): true_err={true_err:.3e} > '
                        f'cert={cert:.3e} (ratio {ratio:.3f} > '
                        f'{_RATIO_SLACK})')
                    self.n_checks += 1
                    ratios.append((w, ratio))
        self._plot_ratios(ratios)

    def _plot_ratios(self, ratios: list[tuple[float, float]]) -> None:
        if not _HAVE_MPL or not ratios:  # pragma: no cover - diagnostic only
            return
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        ws, rs = zip(*ratios)
        fig, ax = plt.subplots()
        ax.scatter(ws, rs, s=24)
        ax.axhline(1.0, color='r', lw=0.9, ls='--', label='optimistic wall')
        ax.set_xlabel('w')
        ax.set_ylabel('true_err / cert')
        ax.set_ylim(0.0, max(1.1, max(rs) * 1.1))
        ax.set_title('interior certificate conservativeness (< 1 is safe)')
        ax.legend()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'test_interior_certificate_conservative_ratio.png'),
            dpi=110, bbox_inches='tight')
        plt.close(fig)


class NearCausticCertificateSelfRefusalTestCase(CertificateTestCase):
    """Certificate SELF-REFUSES (or stays conservative) near the caustic.

    On ~3 four-image configs DELIBERATELY placed near the caustic (source
    at ``rho in [0.9, 0.99]`` of the closed-form reach, so two images are
    merging: small ``xi_min``, large ``sqrt|mu|``, diverging ``c3``) at
    ``w in {20, 40, 60}``, EVERY point must satisfy ONE of:

    * the certificate SELF-REFUSES -- ``cert is None`` OR ``cert * 2 >
      CERTIFICATION_BAR`` (the gate would not admit raw ppGO here); OR
    * ``true_err <= cert`` (ratio ``<= 1.02``).

    The forbidden state is ADMIT-AND-OPTIMISTIC: a certificate that both
    passes the gate and under-estimates the true error on a merging
    config.  This is the invariant that makes dropping leg 2 sound -- as an
    image approaches a critical point ``c3`` diverges and the certificate
    must blow up (refuse) before it can go optimistic.
    """

    def test_near_caustic_never_admits_and_optimistic(self) -> None:
        # (xi_min, ratio, admitted) for admitted points only in the plot.
        admitted_points: list[tuple[float, float]] = []
        for gamma, theta, rho in _NEAR_CAUSTIC_CONFIGS:
            for w in _CONSISTENCY_W:
                with self.subTest(gamma=gamma, theta=theta, rho=rho, w=w):
                    n_img, true_err, cert, xi_min = _served_and_oracle(
                        gamma, theta, rho, w)
                    self.assertEqual(
                        n_img, 4,
                        f'near-caustic fixture (gamma={gamma}, theta={theta}, '
                        f'rho={rho}) is not four-image ({n_img})')
                    refuses = (cert is None
                               or cert * 2.0 > CERTIFICATION_BAR)
                    if not refuses:
                        # Admitted: the certificate MUST bound the error.
                        ratio = true_err / cert
                        admitted_points.append((xi_min, ratio))
                        self.assertLessEqual(
                            ratio, _RATIO_SLACK,
                            f'ADMIT-AND-OPTIMISTIC at (gamma={gamma}, '
                            f'theta={theta}, rho={rho}, w={w}): cert={cert:.3e}'
                            f' admitted but true_err={true_err:.3e} exceeds it '
                            f'(ratio {ratio:.3f}); a merging image drove c3 '
                            f'optimistic without refusing')
                    self.n_checks += 1
        self._plot_admitted(admitted_points)

    def _plot_admitted(self, points: list[tuple[float, float]]) -> None:
        if not _HAVE_MPL:  # pragma: no cover - diagnostic only
            return
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        fig, ax = plt.subplots()
        if points:
            xis, rs = zip(*points)
            ax.scatter(xis, rs, s=24, label='admitted')
        else:
            ax.text(0.5, 0.5, 'no admitted points\n(all self-refused)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.axhline(1.0, color='r', lw=0.9, ls='--', label='optimistic wall')
        ax.set_xlabel('xi_min = w * min real-image delay gap')
        ax.set_ylabel('true_err / cert (admitted points only)')
        ax.set_title('near-caustic: admitted points must stay < 1')
        ax.legend()
        fig.savefig(os.path.join(
            _OUTPUT_DIR,
            'test_near_caustic_self_refusal_admitted_ratio.png'),
            dpi=110, bbox_inches='tight')
        plt.close(fig)


class PpgoCertificateSelfFalsificationTestCase(CertificateTestCase):
    """Prove every gate can go red under a deliberately wrong oracle.

    ``_expect_checks = False``: these tests assert that a WRONG oracle
    would be CAUGHT, so they must not count toward the anti-vacuity tally.
    """

    _expect_checks = False

    def test_wrong_caustic_reach_mislabels_census(self) -> None:
        # If the census used HALF the true caustic reach, the rho=1.5
        # source (2 real images) would be mistaken for interior (4). The
        # real census must NOT agree with that inflated boundary.
        gamma, theta = 0.3, 0.6
        exterior = _source_on_ray(gamma, theta, 1.5)  # 2 real images
        # A wrong boundary at 3x reach would call this interior:
        wrong_rho = float(np.linalg.norm(exterior)) / (
            3.0 * geometry.r_caustic(gamma, theta))
        self.assertLess(wrong_rho, 1.0)  # wrong oracle says "interior"
        self.assertEqual(_real_image_count(gamma, exterior), 2)  # truth: not

    def test_real_perturbation_trips_purely_imaginary_gate(self) -> None:
        # A c3 with a real part above the bar must fail the gate the real
        # c3 passes -- proving the purely-imaginary check has teeth.
        _c1, _c2, c3 = geometry._series_coefficients(
            _interior_real_images(0.3, 0.6)[0], geometry.macro_matrix(0.3))
        tampered = complex(c3.imag, c3.imag)  # real part == imag part
        self.assertFalse(
            abs(tampered.real) < 1e-9 * (abs(tampered.imag) + 1.0),
            'a c3 with a large real part must trip the gate')

    def test_wrong_exponent_breaks_w_minus_three_ratio(self) -> None:
        # The ratio identity is specific to the cube: a w**-2 model would
        # predict (w2/w1)**2, which disagrees with the true (w2/w1)**3.
        gamma, theta = 0.3, 0.6
        source = _source_on_ray(gamma, theta, 0.5)
        matrix = geometry.macro_matrix(gamma)
        images = geometry.find_images(source, matrix)
        w1, w2 = 20.0, 60.0
        ratio = (geometry.ppgo_error_estimate(images, source, matrix, w1)
                 / geometry.ppgo_error_estimate(images, source, matrix, w2))
        self.assertFalse(
            np.isclose(ratio, (w2 / w1) ** 2, rtol=_W3_RTOL),
            'a w**-2 exponent must NOT reproduce the measured ratio')

    def test_shrunk_certificate_would_be_caught_on_interior(self) -> None:
        # The conservativeness gate has teeth: an artificially OPTIMISTIC
        # certificate (the true cert shrunk by 1e3) on a real interior
        # point must exceed the 1.02 ratio bar the honest cert passes.
        _n, true_err, cert, _xi = _served_and_oracle(0.3, 0.6, 0.5, 20.0)
        self.assertIsNotNone(cert)
        self.assertLessEqual(true_err / cert, _RATIO_SLACK)  # honest passes
        optimistic = cert * 1e-3
        self.assertGreater(
            true_err / optimistic, _RATIO_SLACK,
            'a certificate 1000x too small must trip the conservativeness bar')

    def test_admitting_near_caustic_would_be_optimistic(self) -> None:
        # The self-refusal is load-bearing: at a merging near-caustic point
        # the true error is orders of magnitude ABOVE CERTIFICATION_BAR, so
        # ANY certificate small enough to pass the gate (cert*2 <= BAR)
        # would be wildly optimistic.  The honest cert must therefore be far
        # too large to admit -- which is exactly the refusal we rely on.
        _n, true_err, cert, _xi = _served_and_oracle(0.3, 0.6, 0.95, 20.0)
        self.assertIsNotNone(cert)
        self.assertGreater(
            true_err, CERTIFICATION_BAR,
            'near-caustic true error must dwarf the admission bar, so any '
            'admitting certificate is optimistic')
        self.assertGreater(
            cert * 2.0, CERTIFICATION_BAR,
            'the honest certificate must refuse this merging config')


if __name__ == '__main__':
    unittest.main()
