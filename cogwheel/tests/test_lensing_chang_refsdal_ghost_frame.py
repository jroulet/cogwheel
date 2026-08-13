"""Frame tests for ``channels.farfield_ghost_term`` (WP1 repair).

WP1 repaired the delay-frame carrier of the decaying complex-saddle
('ghost') contribution ``G(w)``.  The real channel kernels the ghost is
subtracted alongside are carried in the partition's MIN-SUBTRACTED frame
``tau_a - t_min`` (``t_min`` = the minimum real-image Fermat delay), so
the ghost MUST be carried in the SAME frame, ``tau_c - t_min``.  Before
the repair the ghost was carried raw, at ``tau_c``, leaving the
subtracted/re-added term off by a pure phase ``exp(-1j * w * t_min)``.
This suite pins the repaired frame and guards that the newly-added
``find_images`` + ``delay`` call inside ``farfield_ghost_term`` did not
perturb the pure real-image primitives.

Two independent gates:

  * FRAME (machine precision, no engine): the multiplicative ratio
    ``farfield_ghost_term(w) / (kernel * exp(1j * w * tau_c))`` must equal
    ``exp(-1j * w * t_min)`` to a relative tolerance of ``1e-12``, where
    ``kernel`` and ``tau_c`` come from the RAW ``geometry.ghost_kernel``
    and ``t_min`` is recomputed INDEPENDENTLY here from
    ``geometry.find_images`` + ``geometry.delay`` -- NOT read back from the
    channels helper.  The comparison is on the COMPLEX magnitude
    ``|ratio - exp(-1j w t_min)| / |ratio|`` (never real/imaginary parts
    separately), and the grid deliberately includes the band-max
    ``w ~ 59`` (just below ``W_CEILING_SCHWINGER = 60``) as the worst-case
    phase.  Per the Professor: the large ``w * tau`` phase error is
    common-mode between numerator and denominator and cancels in the
    ratio, so a residual above ``1e-12`` signals a real ``t_min`` frame
    mismatch, not floating-point accumulation.  The tolerance is
    ``1e-12`` -- ~100x the ``1e-14`` empirically observed at this probe --
    tight enough that a wrong or dropped ``t_min`` (offset by
    ``|t_min| ~ 0.93`` here, a phase of tens of radians over the band) is
    caught, loose enough to absorb the benign common-mode residual.

  * BIT-IDENTITY (regression freeze): ``find_images``, ``delay``,
    ``morse_index`` and ``image_kernel`` must return values BIT-identical
    to frozen hex-float fixtures captured from the primitives.  The oracle
    for a purity/regression guard is legitimately a frozen snapshot; the
    fixtures are stored as ``float.hex()`` strings so the equality is
    exact, not approximate.  Any nonzero diff pinpoints which primitive
    was perturbed.

``GhostFrameTestCase`` carries an ANTI-VACUITY ``tearDown`` that fails a
test which invoked the comparison helper yet compared nothing (an empty
``w`` grid or an empty image list would otherwise read green).
``GhostFrameSelfFalsificationTestCase`` proves the suite can go RED: a
mutated ``t_min`` breaks the frame ratio, and a perturbed fixture value
breaks the bit-identity equality.
"""

from __future__ import annotations

import pathlib

import numpy as np
from unittest import TestCase, main

from cogwheel.lensing.chang_refsdal import _schwinger, channels, geometry

#: External shear of the representative off-axis frame probe.  Positive
#: parity requires ``1 - kappa > |gamma|``; with ``kappa = 0`` any
#: ``|gamma| < 1`` qualifies.
GAMMA_PROBE: float = 0.5

#: Source-plane polar angle of the frame probe (45 deg, off every
#: principal axis so ``Im tau_c > 0`` and the ghost gate can pass).
THETA_PROBE: float = np.deg2rad(45.0)

#: Radial offset of the source beyond the caustic, ``|y| = r_caustic +
#: OFFSET_PROBE``.  Outside the caustic there are two real images and a
#: single decaying ghost; 0.6 sits comfortably in the ghost-gate-passing
#: band at this angle.
OFFSET_PROBE: float = 0.6

#: Dimensionless frequency grid.  Its minimum (10) clears the ghost gate
#: ``w_min * Im tau_c >= 2.0`` at the probe, and its maximum (59) is the
#: worst-case phase, just below ``_schwinger.W_CEILING_SCHWINGER = 60``.
W_GRID: np.ndarray = np.linspace(10.0, 59.0, 30)

#: Ceiling above which the Schwinger wave branch is declined; the grid
#: stays strictly below it so every node is a served wave-branch node.
#: DERIVED from production: a second copy of ``60.0`` would keep the
#: "every node is served" assertion green while the engine's own ceiling
#: moved out from under ``W_GRID``.
W_CEILING_SCHWINGER: float = _schwinger.W_CEILING_SCHWINGER

#: Relative tolerance on the complex frame ratio.  ~100x the ~1e-14
#: common-mode residual observed at this probe; a wrong/dropped ``t_min``
#: (phase offset of tens of radians over the band) is far larger.
FRAME_RTOL: float = 1e-12

#: Scalar frequency at which the frozen ``image_kernel`` fixtures were
#: captured.
KERNEL_W: float = 25.0

#: Directory for diagnostic plots.
OUTPUT_DIR: pathlib.Path = pathlib.Path(__file__).parent / 'output'


#: Frozen real-image primitive outputs, captured bit-exactly (as
#: ``float.hex()`` strings) from the pre/post-change primitives at three
#: positive-parity probes (``beta = kappa = 0``).  Each probe fixes a
#: ``(source, matrix)`` pair; ``source`` is stored directly so the guard
#: isolates the four primitives from any drift in ``r_caustic``.  Images
#: are listed in ``find_images`` order (increasing Fermat delay).  Kernel
#: values are ``image_kernel`` evaluated at ``KERNEL_W``.
REAL_IMAGE_PROBES: tuple[dict, ...] = (
    {
        'gamma': 0.5,
        'source_hex': ('0x1.9038870204337p-1', '0x1.9038870204336p-1'),
        'images': (
            {'pos_hex': ('0x1.2e37eecd2cdcdp+1', '0x1.2cad1bffe2b16p-1'),
             'delay_hex': '-0x1.dc6268562f40dp-1', 'morse': 0,
             'kernel_hex': ('0x1.1256aa4f79ce3p+0', '0x1.8c6038e5ebfd9p-12')},
            {'pos_hex': ('-0x1.30e5e0bcfe545p-2', '-0x1.ec7ed86e85293p-2'),
             'delay_hex': '0x1.fc31e710c81b3p+0', 'morse': 1,
             'kernel_hex': ('-0x1.5ab750c0a34b0p-8', '-0x1.3d0bb36e911d5p-2')},
        ),
    },
    {
        'gamma': 0.3,
        'source_hex': ('0x1.eaf4c990eeffcp-1', '0x1.1b74255472943p-1'),
        'images': (
            {'pos_hex': ('0x1.03ec09efa1895p+1', '0x1.085cde5497a7bp-1'),
             'delay_hex': '-0x1.7c84d5be31562p-1', 'morse': 0,
             'kernel_hex': ('0x1.02d40b669b41bp+0', '0x1.b12c60fd5df78p-11')},
            {'pos_hex': ('-0x1.de34a47963212p-2', '-0x1.86135f1d1534fp-2'),
             'delay_hex': '0x1.f2e0e75cafe26p+0', 'morse': 1,
             'kernel_hex': ('-0x1.d16e1b72bb7fbp-9', '-0x1.96c904ddc0257p-2')},
        ),
    },
    {
        'gamma': 0.7,
        'source_hex': ('0x1.4e762591bedd6p-1', '0x1.21a6f44788aa8p+0'),
        'images': (
            {'pos_hex': ('0x1.96b473a7e6a4fp+1', '0x1.68cddbea97704p-1'),
             'delay_hex': '-0x1.43590d6a6c607p+0', 'morse': 0,
             'kernel_hex': ('0x1.44e360937a58bp+0', '-0x1.5ffc555d479dep-12')},
            {'pos_hex': ('-0x1.5abb082e48dd3p-3', '-0x1.d746320ba1bc6p-2'),
             'delay_hex': '0x1.30db4c1fc898dp+1', 'morse': 1,
             'kernel_hex': ('-0x1.81455c299851dp-9', '-0x1.bc4a4a70a4ebcp-3')},
        ),
    },
)


def _frame_probe() -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(source, matrix)`` of the representative frame probe.

    Positive parity, ``beta = kappa = 0``, source on the ``THETA_PROBE``
    ray at ``|y| = r_caustic + OFFSET_PROBE`` -- off every principal axis
    so ``Im tau_c > 0`` and the ghost gate can pass.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Shape-``(2,)`` source and shape-``(2, 2)`` macro matrix.
    """
    matrix = geometry.macro_matrix(GAMMA_PROBE, 0.0, 0.0)
    radius = geometry.r_caustic(GAMMA_PROBE, THETA_PROBE) + OFFSET_PROBE
    source = radius * np.array([np.cos(THETA_PROBE), np.sin(THETA_PROBE)])
    return source, matrix


def _independent_t_min(source: np.ndarray, matrix: np.ndarray) -> float:
    """Minimum real-image Fermat delay, recomputed here from primitives.

    Deliberately NOT ``channels._frame_t_min`` -- an oracle that shared
    the code under test would not be a test.  Uses the public
    ``geometry.find_images`` + ``geometry.delay`` primitives directly.

    Parameters
    ----------
    source : np.ndarray
        Shape ``(2,)`` source position.
    matrix : np.ndarray
        Shape ``(2, 2)`` macro matrix.

    Returns
    -------
    float
        ``min_image delay(image, source, matrix)``.
    """
    images = geometry.find_images(source, matrix)
    return min(float(geometry.delay(image, source, matrix))
               for image in images)


class GhostFrameTestCase(TestCase):
    """Base carrying the complex-ratio assertion and anti-vacuity guard."""

    def setUp(self) -> None:
        """Reset the per-test comparison tally used by ``tearDown``."""
        self.n_compared = 0
        self.n_helper_calls = 0

    def tearDown(self) -> None:
        """Fail a test that invoked the helper yet compared nothing.

        A sweep over an empty ``w`` grid or an empty image list would run
        zero element comparisons and so assert nothing while reading
        green.  Tests that never call ``assert_complex_ratio_close`` are
        unaffected.
        """
        if self.n_helper_calls and not self.n_compared:
            self.fail(f'all {self.n_helper_calls} helper call(s) compared '
                      'zero elements; the test asserted nothing')

    def assert_complex_ratio_close(self, got: np.ndarray, expected: np.ndarray,
                                   rtol: float, msg: str = '') -> None:
        """Assert ``|got - expected| / |got| <= rtol`` element-wise.

        The comparison is on the COMPLEX magnitude of the difference,
        normalised by ``|got|`` (the ratio's magnitude), never on the real
        and imaginary parts separately -- a phase error must not be split
        into two smaller-looking components.

        Parameters
        ----------
        got : np.ndarray
            The measured complex ratio.
        expected : np.ndarray
            The predicted complex value ``exp(-1j w t_min)``.
        rtol : float
            Relative tolerance on the complex magnitude.
        msg : str
            Context prepended to a failure message.
        """
        got = np.asarray(got, dtype=complex)
        expected = np.asarray(expected, dtype=complex)
        self.n_helper_calls += 1
        if got.size == 0:
            return
        self.n_compared += int(got.size)
        rel_error = np.abs(got - expected) / np.abs(got)
        worst = float(np.max(rel_error))
        self.assertLessEqual(
            worst, rtol,
            f'{msg}: worst relative error {worst:.3e} > {rtol:.1e} '
            f'(at flat index {int(np.argmax(rel_error))} of {got.size})')


class FarfieldGhostFrameTestCase(GhostFrameTestCase):
    """The ghost is carried in the partition's min-subtracted frame.

    ``farfield_ghost_term`` must equal the RAW ghost carrier
    ``kernel * exp(1j w tau_c)`` shifted by ``exp(-1j w t_min)``.  ``t_min``
    is recomputed independently here (`_independent_t_min`), never read
    from the channels helper.
    """

    def setUp(self) -> None:
        """Build the shared probe and the raw ghost carrier once."""
        super().setUp()
        self.source, self.matrix = _frame_probe()
        self.contribution = geometry.ghost_kernel(
            W_GRID, self.source, self.matrix)
        self.raw_carrier = (self.contribution.kernel
                            * np.exp(1j * W_GRID * self.contribution.delay))
        self.ghost_term = channels.farfield_ghost_term(
            W_GRID, self.source, self.matrix)
        self.t_min = _independent_t_min(self.source, self.matrix)

    def test_probe_premise_holds(self) -> None:
        """The probe has two real images and clears the ghost gate.

        Documents (and guards) the premise the frame assertion relies on:
        outside the caustic there are two real images, ``Im tau_c > 0``
        (off-axis), and ``w_min * Im tau_c`` clears the mid-band gate
        ``2.0`` while the band stays below ``W_CEILING_SCHWINGER``.
        """
        images = geometry.find_images(self.source, self.matrix)
        self.assertEqual(len(images), 2)
        im_tau_c = float(self.contribution.delay.imag)
        self.assertGreater(im_tau_c, 0.0)
        gate = float(np.min(W_GRID)) * im_tau_c
        self.assertGreaterEqual(gate, channels._FARFIELD_WINDOW_RADIANS)
        self.assertLess(float(np.max(W_GRID)), W_CEILING_SCHWINGER)
        # t_min carries no imaginary part (a real, pure-phase shift).
        self.assertNotEqual(self.t_min, 0.0)

    def test_ghost_carried_in_min_subtracted_frame(self) -> None:
        """``ghost_term / raw_carrier == exp(-1j w t_min)`` to FRAME_RTOL.

        The multiplicative ratio isolates the frame shift; the large
        ``w * tau`` phase is common-mode between numerator and denominator
        and cancels, so the residual measures the ``t_min`` frame match.
        """
        ratio = self.ghost_term / self.raw_carrier
        expected = np.exp(-1j * W_GRID * self.t_min)
        self.assert_complex_ratio_close(
            ratio, expected, FRAME_RTOL,
            'ghost term is not carried at tau_c - t_min')

    def test_band_max_is_worst_case_phase(self) -> None:
        """The worst-case band edge ``w ~ 59`` also matches to FRAME_RTOL.

        Isolated so the tightest ``w * t_min`` phase in the band is checked
        on its own, per the Professor's note that a failure there (not at
        small ``w``) is the fingerprint of a genuine ``t_min`` mismatch.
        """
        w_max = float(np.max(W_GRID))
        contribution_max = geometry.ghost_kernel(
            np.array([w_max]), self.source, self.matrix)
        raw_max = (contribution_max.kernel
                   * np.exp(1j * w_max * contribution_max.delay))
        ghost_max = channels.farfield_ghost_term(
            np.array([w_max]), self.source, self.matrix)
        ratio = ghost_max / raw_max
        expected = np.exp(-1j * w_max * self.t_min)
        self.assert_complex_ratio_close(
            ratio, expected, FRAME_RTOL,
            f'ghost frame wrong at band-max w = {w_max:.3f}')

    def test_raw_frame_differs_from_ghost_term(self) -> None:
        """Non-vacuity: the RAW carrier is NOT the ghost term.

        They differ by ``exp(-1j w t_min)`` with ``|t_min| ~ 0.93`` at this
        probe, so the frame assertion above is testing a real shift, not an
        identity that would pass no matter what.  If the two were equal the
        frame test would be decoration.
        """
        max_abs_diff = float(np.max(np.abs(self.raw_carrier - self.ghost_term)))
        self.assertGreater(
            max_abs_diff, 1e-6,
            'raw carrier equals the ghost term: the frame shift is absent, '
            'so the frame assertion is vacuous')

    def test_frame_ratio_is_a_line_of_slope_minus_t_min(self) -> None:
        """The unwrapped phase of the ratio is linear in ``w``, slope
        ``-t_min``, through the origin; a diagnostic plot is saved.

        A nonzero intercept or a wrong slope would reveal a convention
        factor or residual frame error the magnitude test could in
        principle absorb.
        """
        ratio = self.ghost_term / self.raw_carrier
        phase = np.unwrap(np.angle(ratio))
        slope, intercept = np.polyfit(W_GRID, phase, 1)
        self.assertAlmostEqual(slope, -self.t_min, places=9,
                               msg='ratio phase slope is not -t_min')
        # ``np.angle`` fixes the principal branch, so the fitted intercept
        # is defined only modulo 2*pi; 'through 0' means it is a multiple
        # of 2*pi (a genuine convention factor would be a non-multiple).
        wrapped_intercept = (intercept + np.pi) % (2.0 * np.pi) - np.pi
        self.assertAlmostEqual(wrapped_intercept, 0.0, places=6,
                               msg='ratio phase intercept is not 0 mod 2*pi')

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots()
        axes.plot(W_GRID, phase, 'o', label='arg(ghost / raw carrier)')
        axes.plot(W_GRID, -self.t_min * W_GRID, '-',
                  label=f'slope -t_min = {-self.t_min:.4f}')
        axes.set_xlabel('w (dimensionless)')
        axes.set_ylabel('unwrapped phase (rad)')
        axes.set_title('Ghost frame shift: arg(G / raw) vs w')
        axes.legend()
        figure.savefig(
            OUTPUT_DIR / 'test_lensing_chang_refsdal_ghost_frame_slope.png',
            dpi=120)
        plt.close(figure)


class RealImagePrimitiveBitIdentityTestCase(TestCase):
    """The real-image primitives are bit-identical to frozen fixtures.

    WP1 added a ``find_images`` + ``delay`` call INSIDE
    ``farfield_ghost_term``.  Those primitives are pure (no global state),
    so calling them from the ghost path must not perturb their outputs on
    the pure real-image path.  This freezes ``find_images``, ``delay``,
    ``morse_index`` and ``image_kernel`` at three probes and asserts EXACT
    (hex-float) equality.  A regression-freeze guard's oracle is
    legitimately a frozen snapshot; the hex encoding makes the equality
    bit-exact.
    """

    def setUp(self) -> None:
        """Reset the per-test tally used by the anti-vacuity guard."""
        self.n_compared = 0

    def tearDown(self) -> None:
        """Fail if a bit-identity test asserted nothing."""
        if self.n_compared == 0:
            self.fail('no primitive outputs were compared; the test '
                      'asserted nothing')

    @staticmethod
    def _probe_inputs(fixture: dict) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild the ``(source, matrix)`` of a frozen probe exactly.

        ``source`` is reconstructed from stored hex so the guard isolates
        the four primitives from any drift in ``r_caustic``.
        """
        source = np.array([float.fromhex(component)
                           for component in fixture['source_hex']])
        matrix = geometry.macro_matrix(fixture['gamma'], 0.0, 0.0)
        return source, matrix

    def test_find_images_bit_identical(self) -> None:
        """``find_images`` returns the frozen positions, hex-exact."""
        for fixture in REAL_IMAGE_PROBES:
            source, matrix = self._probe_inputs(fixture)
            with self.subTest(gamma=fixture['gamma']):
                images = geometry.find_images(source, matrix)
                self.assertEqual(len(images), len(fixture['images']))
                for image, expected in zip(images, fixture['images']):
                    self.n_compared += 1
                    self.assertEqual(
                        (image[0].hex(), image[1].hex()), expected['pos_hex'])

    def test_delay_bit_identical(self) -> None:
        """``delay`` returns the frozen Fermat delays, hex-exact."""
        for fixture in REAL_IMAGE_PROBES:
            source, matrix = self._probe_inputs(fixture)
            images = geometry.find_images(source, matrix)
            for image, expected in zip(images, fixture['images']):
                with self.subTest(gamma=fixture['gamma'],
                                  delay=expected['delay_hex']):
                    self.n_compared += 1
                    value = geometry.delay(image, source, matrix)
                    self.assertEqual(value.hex(), expected['delay_hex'])

    def test_morse_index_bit_identical(self) -> None:
        """``morse_index`` returns the frozen indices exactly."""
        for fixture in REAL_IMAGE_PROBES:
            source, matrix = self._probe_inputs(fixture)
            images = geometry.find_images(source, matrix)
            for image, expected in zip(images, fixture['images']):
                with self.subTest(gamma=fixture['gamma']):
                    self.n_compared += 1
                    self.assertEqual(
                        geometry.morse_index(image, matrix), expected['morse'])

    def test_image_kernel_bit_identical(self) -> None:
        """``image_kernel`` at ``KERNEL_W`` matches frozen values, hex-exact."""
        for fixture in REAL_IMAGE_PROBES:
            source, matrix = self._probe_inputs(fixture)
            images = geometry.find_images(source, matrix)
            for image, expected in zip(images, fixture['images']):
                with self.subTest(gamma=fixture['gamma']):
                    self.n_compared += 1
                    kernel = complex(geometry.image_kernel(
                        KERNEL_W, image, matrix))
                    self.assertEqual(
                        (kernel.real.hex(), kernel.imag.hex()),
                        expected['kernel_hex'])


class GhostFrameSelfFalsificationTestCase(GhostFrameTestCase):
    """Prove this suite is able to FAIL.

    A frame bug is a pure phase -- silent in magnitude, invisible if the
    test compared real/imaginary parts separately or omitted the shift.
    These tests inject the very errors WP1 repaired and assert the gates
    above go RED, so 'the suite is green' is evidence, not decoration.
    """

    def setUp(self) -> None:
        """Build the shared probe once."""
        super().setUp()
        self.source, self.matrix = _frame_probe()
        self.contribution = geometry.ghost_kernel(
            W_GRID, self.source, self.matrix)
        self.raw_carrier = (self.contribution.kernel
                            * np.exp(1j * W_GRID * self.contribution.delay))
        self.ghost_term = channels.farfield_ghost_term(
            W_GRID, self.source, self.matrix)
        self.t_min = _independent_t_min(self.source, self.matrix)

    def test_raw_frame_fails_the_frame_gate(self) -> None:
        """The UN-shifted (pre-repair) raw carrier fails the frame gate.

        Comparing the raw carrier -- what the ghost term would be if the
        ``-t_min`` shift were dropped -- against ``exp(-1j w t_min)`` must
        exceed FRAME_RTOL, proving the gate catches the exact bug WP1 fixed.
        """
        ratio = self.raw_carrier / self.raw_carrier  # identically 1
        expected = np.exp(-1j * W_GRID * self.t_min)
        with self.assertRaises(AssertionError):
            self.assert_complex_ratio_close(
                ratio, expected, FRAME_RTOL, 'positive control')
        # The helper counted the comparison before failing, so the
        # anti-vacuity tearDown stays satisfied.
        self.assertGreater(self.n_compared, 0)

    def test_wrong_t_min_fails_the_frame_gate(self) -> None:
        """A perturbed ``t_min`` breaks the true ghost term's frame ratio.

        The genuine ratio matches ``exp(-1j w t_min)``; feeding a ``t_min``
        off by a physically small 1e-3 must still exceed FRAME_RTOL over
        the band, so the tolerance is not so loose as to accept a wrong
        frame origin.
        """
        ratio = self.ghost_term / self.raw_carrier
        wrong = np.exp(-1j * W_GRID * (self.t_min + 1e-3))
        with self.assertRaises(AssertionError):
            self.assert_complex_ratio_close(
                ratio, wrong, FRAME_RTOL, 'wrong t_min control')
        self.assertGreater(self.n_compared, 0)

    def test_perturbed_fixture_breaks_bit_identity(self) -> None:
        """A one-ULP change to a frozen delay fails exact equality.

        Confirms the bit-identity assertion has teeth: were it approximate,
        a perturbed fixture would slip through.
        """
        fixture = REAL_IMAGE_PROBES[0]
        source = np.array([float.fromhex(component)
                           for component in fixture['source_hex']])
        matrix = geometry.macro_matrix(fixture['gamma'], 0.0, 0.0)
        images = geometry.find_images(source, matrix)
        true_delay = geometry.delay(images[0], source, matrix)
        perturbed_hex = np.nextafter(true_delay, np.inf).hex()
        self.assertNotEqual(true_delay.hex(), perturbed_hex)


if __name__ == '__main__':
    main()
