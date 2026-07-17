"""
Tests for `lensing.waveform` -- the Chang--Refsdal microlensed waveform
composer.

`LensedWaveformGenerator` wraps an ordinary cogwheel
`waveform.WaveformGenerator` and multiplies every harmonic mode by the
common wave-optics amplification ``F(w(f))`` of a Chang--Refsdal lens,

    h_lensed_lm(f) = F(w(f)) * h_lm(f),   w = 8*pi*G*M_L*(1+z_L)*f / c**3.

This suite pins the four properties that make that claim true and
falsifiable, and it deliberately does NOT re-test the amplification
physics (owned by the `chang_refsdal` suites) -- only that the composer
routes ``F`` through the right grids, converts frequency with the right
constant, and refuses out-of-domain lenses.

WHY A STUB WAVEFORM GENERATOR, NOT A REAL LAL ONE
-------------------------------------------------
The properties under test are structural: the amplification multiplies
whatever the wrapped generator returns, mode by mode, on the waveform's
own frequency grid.  A deterministic `StubWaveformGenerator` (which
reproduces the real `get_hplus_hcross` contract -- the ``phi_ref`` /
``d_luminosity`` scaling, the ``(n_m, 2, n_f)`` vs ``(2, n_f)`` shapes,
and the ``m_arr`` interface -- but with a closed-form, nonzero strain)
is therefore the CORRECT oracle: it makes the unlensed strain an
independently known quantity, so ``h_lensed == F * h_unlensed`` is
checked against a value the production path did not itself derive.  A
real LAL waveform would add nothing testable here and would couple the
suite to approximant physics it does not own.

INDEPENDENT FREQUENCY-CONVENTION ORACLE
---------------------------------------
`WConventionTestCase` recomputes ``w`` from the SEPARATE SI constants
``G``, ``M_sun`` and ``c`` (`XI_PER_MSUN_PER_HZ`), NOT from the module's
own ``_EIGHT_PI_MTSUN_S`` (which is built on ``lal.MTSUN_SI``).  Gating
the module's constant against itself would test nothing; the derivation
here enters through a different constant expression, so a wrong factor
in the module would show up.

ANTI-VACUITY AND SELF-FALSIFICATION
-----------------------------------
Every ``<Thing>TestCase`` derives from `LensedWaveformTestCase`, whose
`tearDown` FAILS a test that made zero comparisons -- so a sweep that
silently iterated an empty grid cannot read green.  `TOL_ROUNDOFF` and
`TOL_CONVENTION` are relative tolerances a few orders of magnitude above
float64 epsilon, justified where used.  `SelfFalsificationTestCase`
proves the suite can go red: it corrupts the amplification routing, the
frequency constant, and the domain gate, and asserts each corruption is
caught.
"""

from __future__ import annotations

import itertools
import pathlib
from unittest import TestCase, main

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (backend must precede import)
import numpy as np  # noqa: E402

from cogwheel import utils  # noqa: E402
from cogwheel.lensing import waveform as lw  # noqa: E402
from cogwheel.lensing.chang_refsdal import geometry  # noqa: E402

lal = utils.import_lal()

#: Float64 machine epsilon; the roundoff unit of every bound here.
EPS = np.finfo(np.float64).eps

#: Relative tolerance for quantities that should agree "to roundoff".
#: ~450x epsilon, leaving headroom for the handful of complex
#: multiplies and broadcasts between the two sides of each comparison
#: while still excluding any O(1) routing error.
TOL_ROUNDOFF = 1e-13

#: Relative tolerance for the frequency-convention check.  The module's
#: ``lal.MTSUN_SI`` and this suite's ``G*M_sun/c**3`` are nominally the
#: same physical constant reached through different expressions; they
#: agree to a few epsilon, so 1e-12 is tight yet not brittle.
TOL_CONVENTION = 1e-12

#: Independent per-solar-mass, per-Hz dimensionless-frequency constant
#: ``xi = 8*pi*G*M_sun/c**3`` [s], built from the SEPARATE SI constants
#: rather than ``lal.MTSUN_SI``.  This is the oracle `WConventionTestCase`
#: gates the module against; see the module docstring.
XI_PER_MSUN_PER_HZ = 8.0 * np.pi * lal.G_SI * lal.MSUN_SI / lal.C_SI ** 3

#: A benign, positive-parity lens far enough from the caustic that the
#: channel engine is well conditioned (matches the committed
#: `test_lensing_channels` "two-image" fixture).
BENIGN_LENS = dict(y=np.array([0.12, 0.035]), gamma=0.2, beta=0.0,
                   kappa=0.0)

#: Reference lens mass / redshift for the convention check.
REF_M_LENS_MSUN = 137.0
REF_Z_LENS = 0.42

#: Modest LIGO-band frequency grid (Hz) used throughout.  Starts above 0
#: so ``w > 0`` and the engine actually runs; a separate DC bin is added
#: where the ``w <= 0`` path is exercised.
FREQ_HZ = np.linspace(20.0, 512.0, 24)

#: Source parameters for the stub generator.  Non-trivial ``phi_ref`` and
#: ``d_luminosity`` ensure the amplification is tested against a fully
#: scaled strain, not the phi_ref=0 / d=1 reference.
STUB_PAR_DIC = {'phi_ref': 0.37, 'd_luminosity': 480.0}

#: Directory for diagnostic plots.
OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'


class StubWaveformGenerator:
    """
    Deterministic stand-in for `waveform.WaveformGenerator`.

    Reproduces the parts of the real contract the composer relies on --
    ``m_arr``, ``harmonic_modes``, and ``get_hplus_hcross(f, par,
    by_m=...)`` with its ``phi_ref`` / ``d_luminosity`` scaling and its
    ``(n_m, 2, n_f)`` (by mode) vs ``(2, n_f)`` (summed) shapes -- but
    with a closed-form, everywhere-nonzero strain so the unlensed side of
    every comparison is independently known.

    Parameters
    ----------
    m_values : Sequence[int]
        Harmonic ``|m|`` numbers to expose, e.g. ``[2, 3, 4]``.
    """

    def __init__(self, m_values) -> None:
        self._m_arr = np.asarray(m_values, dtype=int)

    @property
    def m_arr(self) -> np.ndarray:
        """Int array of ``|m|`` harmonic mode numbers."""
        return self._m_arr.copy()

    @property
    def harmonic_modes(self) -> list:
        """``(l, m)`` modes, one representative ``l`` per ``|m|``."""
        return [(int(m), int(m)) for m in self._m_arr]

    def _reference_strain(self, f_hz: np.ndarray) -> np.ndarray:
        """
        Return the ``phi_ref=0``, ``d_luminosity=1`` per-mode strain,
        shape ``(n_m, 2, n_f)``, nonzero for every entry.
        """
        f_hz = np.asarray(f_hz, dtype=float)
        strain = np.empty((self._m_arr.size, 2, f_hz.size), dtype=complex)
        for index, m in enumerate(self._m_arr):
            amplitude = (1.0 + 0.1 * m) * np.exp(-f_hz / 3000.0)
            phase = 2.0 * np.pi * f_hz * 0.011 * (index + 1)
            # hplus and hcross deliberately distinct so a bug that
            # collapses polarizations would be visible.
            strain[index, 0] = amplitude * np.exp(1j * phase)
            strain[index, 1] = 0.6 * amplitude * np.exp(
                1j * (phase + 0.4 * m)) * 1j
        return strain

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False
                         ) -> np.ndarray:
        """Mirror `WaveformGenerator.get_hplus_hcross`."""
        reference = self._reference_strain(f)
        phi_ref = float(waveform_par_dic['phi_ref'])
        d_luminosity = float(waveform_par_dic['d_luminosity'])
        m_arr = self._m_arr.reshape(-1, 1, 1)
        strain = (np.exp(1j * phi_ref * m_arr) / d_luminosity) * reference
        if by_m:
            return strain
        return np.sum(strain, axis=0)


def _relative_error(got: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise ``|got - reference| / |reference|``."""
    got = np.asarray(got)
    reference = np.asarray(reference)
    return np.abs(got - reference) / np.abs(reference)


def _make_generator(m_values=(2,), *, m_lens_msun=REF_M_LENS_MSUN,
                    z_lens=REF_Z_LENS, **lens_overrides
                    ) -> lw.LensedWaveformGenerator:
    """Build a `LensedWaveformGenerator` over a stub, benign by default."""
    lens = dict(BENIGN_LENS)
    lens.update(lens_overrides)
    return lw.LensedWaveformGenerator(
        StubWaveformGenerator(m_values), m_lens_msun=m_lens_msun,
        z_lens=z_lens, **lens)


class LensedWaveformTestCase(TestCase):
    """
    Base class carrying the anti-vacuity guard.

    A test that iterates a sweep must actually run at least one
    comparison; `note_comparison` records that it did, and `tearDown`
    fails a test that recorded none.  This is what stops an empty grid or
    an accidentally-skipped mode list from reading green.
    """

    def setUp(self) -> None:
        """Reset the per-test comparison tally."""
        self.comparisons = 0

    def tearDown(self) -> None:
        """Fail a test that asserted nothing."""
        if self.comparisons == 0:
            self.fail('anti-vacuity: the test made zero comparisons')

    def note_comparison(self) -> None:
        """Record that one comparison actually ran."""
        self.comparisons += 1

    def assert_relative(self, got, reference, tol, msg) -> None:
        """Assert max elementwise relative error is within `tol`."""
        self.note_comparison()
        error = float(np.max(_relative_error(got, reference)))
        self.assertLessEqual(
            error, tol, f'{msg}: relative error {error:.3e} > {tol:.0e}')


class UnlensedLimitTestCase(LensedWaveformTestCase):
    """
    ``F -> 1`` and ``h_lensed -> h_unlensed`` in the unlensed limit.

    The exact endpoint (``M_L = 0``, so ``w = 0`` identically) is checked
    BIT-EXACTLY -- the strongest possible "to roundoff" statement -- and
    the approach as ``M_L`` shrinks is checked to fall onto the
    float64 floor.
    """

    #: Geometric sequence of tiny lens masses (solar masses); ``w`` over
    #: the band shrinks with ``M_L`` and both ``|F|-1`` and ``|F-1|``
    #: follow it down to the roundoff floor.
    SMALL_MASSES = (1e-3, 1e-6, 1e-9, 1e-12)

    def test_zero_lens_mass_is_bit_exact_unlensed(self):
        """
        ``M_L = 0`` gives ``w = 0`` for every bin, so the composer's
        ``w <= 0`` path must return ``F = 1`` to the bit and the lensed
        strain must equal the unlensed strain exactly.
        """
        generator = _make_generator((2, 3, 4), m_lens_msun=0.0)
        amplification = generator.amplification(FREQ_HZ)
        self.note_comparison()
        self.assertTrue(
            np.array_equal(amplification, np.ones_like(amplification)),
            'F is not identically 1 at zero lens mass')

        for by_m in (False, True):
            lensed = generator.get_hplus_hcross(
                FREQ_HZ, dict(STUB_PAR_DIC), by_m=by_m)
            unlensed = generator.waveform_generator.get_hplus_hcross(
                FREQ_HZ, dict(STUB_PAR_DIC), by_m=by_m)
            self.note_comparison()
            self.assertTrue(
                np.array_equal(lensed, unlensed),
                f'lensed != unlensed bit-exactly at M_L=0 (by_m={by_m})')

    def test_dc_bin_is_exactly_unity(self):
        """
        The ``f = 0`` bin maps to ``w = 0`` and must get ``F = 1``
        exactly even for a massive lens -- the documented ``w <= 0``
        convention.
        """
        generator = _make_generator((2,))
        freqs = np.concatenate(([0.0], FREQ_HZ))
        amplification = generator.amplification(freqs)
        self.note_comparison()
        self.assertEqual(amplification[0], 1.0 + 0.0j,
                         'F at the f=0 (w=0) bin is not exactly 1')

    def test_small_mass_approaches_unlensed_on_the_floor(self):
        """
        As ``M_L -> 0`` the amplification collapses to unity: both
        ``max|F|-1`` and ``max|F-1|`` decrease monotonically with
        ``M_L`` and the smallest mass sits at the roundoff floor.

        ``|F| - 1`` is second order in ``w`` (the leading lensing
        correction is a pure phase), while ``|F - 1|`` -- the strain
        relative error -- is first order; both are driven below
        `TOL_ROUNDOFF` at the smallest mass.
        """
        modulus_dev = []
        strain_dev = []
        for m_lens in self.SMALL_MASSES:
            generator = _make_generator((2,), m_lens_msun=m_lens)
            amplification = generator.amplification(FREQ_HZ)
            modulus_dev.append(float(np.max(np.abs(
                np.abs(amplification) - 1.0))))
            strain_dev.append(float(np.max(np.abs(amplification - 1.0))))
            self.note_comparison()

        modulus_dev = np.array(modulus_dev)
        strain_dev = np.array(strain_dev)
        self.assertTrue(
            np.all(np.diff(modulus_dev) < 0.0),
            f'|F|-1 not monotonically decreasing: {modulus_dev}')
        self.assertTrue(
            np.all(np.diff(strain_dev) < 0.0),
            f'|F-1| not monotonically decreasing: {strain_dev}')
        self.assertLessEqual(
            strain_dev[-1], TOL_ROUNDOFF,
            f'strain deviation {strain_dev[-1]:.3e} above the floor at '
            f'M_L={self.SMALL_MASSES[-1]}')

        self._plot(modulus_dev, strain_dev)

    def _plot(self, modulus_dev, strain_dev) -> None:
        """Diagnostic: deviation vs lens mass, log-log."""
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.loglog(self.SMALL_MASSES, strain_dev, 'o-', label=r'$|F-1|$')
        axis.loglog(self.SMALL_MASSES, modulus_dev, 's-',
                    label=r'$||F|-1|$')
        axis.axhline(TOL_ROUNDOFF, color='k', ls=':', label='TOL_ROUNDOFF')
        axis.set_xlabel(r'$M_L\ [M_\odot]$')
        axis.set_ylabel('max deviation over band')
        axis.set_title(r'Unlensed limit: $F\to1$ as $M_L\to0$')
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'test_lensing_waveform_unlensed_limit.png',
                    dpi=90)
        plt.close(fig)


class WConventionTestCase(LensedWaveformTestCase):
    """
    ``w(f) = 8*pi*G*M_L*(1+z_L)*f/c**3`` -- linear in ``f`` and equal to
    the independently recomputed value.
    """

    def test_w_is_exactly_linear_in_frequency(self):
        """``w`` is linear: ``w(0) = 0`` and ``w(2f) = 2 w(f)``."""
        generator = _make_generator((2,))
        self.note_comparison()
        self.assertEqual(generator.dimensionless_frequency(0.0), 0.0,
                         'w(0) is not exactly 0')

        w_single = generator.dimensionless_frequency(FREQ_HZ)
        w_double = generator.dimensionless_frequency(2.0 * FREQ_HZ)
        self.assert_relative(w_double, 2.0 * w_single, TOL_ROUNDOFF,
                             'w(2f) != 2 w(f)')

    def test_w_matches_independent_constant(self):
        """
        ``w`` equals ``xi * f`` with ``xi`` built from the separate SI
        constants ``G``, ``M_sun``, ``c`` -- not the module's own
        constant.  The ratio ``w / f`` is a flat constant equal to that
        ``xi``.
        """
        expected_xi = (XI_PER_MSUN_PER_HZ * REF_M_LENS_MSUN
                       * (1.0 + REF_Z_LENS))
        generator = _make_generator((2,), m_lens_msun=REF_M_LENS_MSUN,
                                    z_lens=REF_Z_LENS)
        w = generator.dimensionless_frequency(FREQ_HZ)

        self.assert_relative(w, expected_xi * FREQ_HZ, TOL_CONVENTION,
                             'w(f) != xi * f for the independent xi')

        ratio = w / FREQ_HZ
        self.note_comparison()
        self.assertLessEqual(
            float(np.max(np.abs(ratio / expected_xi - 1.0))), TOL_CONVENTION,
            'w/f is not a flat constant equal to the independent xi')

        # Module-level helper and instance accessor must agree exactly.
        self.assert_relative(
            lw.dimensionless_frequency(FREQ_HZ, REF_M_LENS_MSUN, REF_Z_LENS),
            w, 0.0, 'module helper disagrees with instance accessor')

        self._plot(FREQ_HZ, w, expected_xi)

    def _plot(self, freqs, w, expected_xi) -> None:
        """Diagnostic: ``w`` vs ``f`` with the analytic line overlaid."""
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(freqs, w, 'o', label='module w(f)')
        axis.plot(freqs, expected_xi * freqs, '-',
                  label=r'independent $\xi f$')
        axis.set_xlabel('f [Hz]')
        axis.set_ylabel('w')
        axis.set_title('w(f) convention: linear, slope = independent xi')
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'test_lensing_waveform_w_convention.png',
                    dpi=90)
        plt.close(fig)


class PerModeConsistencyTestCase(LensedWaveformTestCase):
    """
    ``F(w(f))`` is applied to EACH mode's frequency array.

    With higher-mode content (22/33/44), every ``|m|`` mode's lensed
    strain must equal ``F(w(f)) * h_unlensed_m(f)`` elementwise, and the
    per-mode ratio must collapse onto the single ``F(w)`` curve -- not be
    applied only to the 22 grid or to the mode-summed strain.
    """

    def test_amplification_applied_per_mode_and_polarization(self):
        """Each ``(m, polarization, f)`` entry carries the same ``F``."""
        generator = _make_generator((2, 3, 4))
        amplification = generator.amplification(FREQ_HZ)
        lensed = generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=True)
        unlensed = generator.waveform_generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=True)

        expected = unlensed * amplification  # (n_m, 2, n_f) * (n_f,)
        self.assert_relative(lensed, expected, TOL_ROUNDOFF,
                             'lensed strain != F * unlensed per mode')

        ratios = []
        m_arr = generator.m_arr
        for mode_index in range(m_arr.size):
            for pol in range(2):
                ratio = (lensed[mode_index, pol]
                         / unlensed[mode_index, pol])
                ratios.append(ratio)
                self.assert_relative(
                    ratio, amplification, TOL_ROUNDOFF,
                    f'mode |m|={m_arr[mode_index]} pol={pol} ratio != F')

        # All per-mode ratios must agree with each other (collapse onto
        # one F(w) curve), independently of F's numerical value.
        reference_ratio = ratios[0]
        for ratio in ratios[1:]:
            self.assert_relative(
                ratio, reference_ratio, TOL_ROUNDOFF,
                'per-mode ratios do not collapse onto a single curve')

        self._plot(amplification, ratios, m_arr)

    def test_summed_strain_is_sum_of_lensed_modes(self):
        """
        The mode-summed lensed strain equals the sum over lensed modes --
        i.e. lensing and mode-summation commute, as they must for a
        per-mode multiplicative factor.
        """
        generator = _make_generator((2, 3, 4))
        by_mode = generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=True)
        summed = generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=False)
        self.assert_relative(summed, np.sum(by_mode, axis=0), TOL_ROUNDOFF,
                             'summed lensed strain != sum of lensed modes')

    def test_decompose_reconstruction_matches_amplification(self):
        """
        The per-image decomposition reconstructs the same ``F(w)`` the
        composer applies, and carries the unlensed per-mode strain for
        the same grid.
        """
        generator = _make_generator((2, 3, 4))
        decomposition = generator.decompose(FREQ_HZ, dict(STUB_PAR_DIC))
        self.assert_relative(
            decomposition.reconstructed_amplification,
            generator.amplification(FREQ_HZ), TOL_ROUNDOFF,
            'decompose reconstruction != amplification')

        unlensed = generator.waveform_generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=True)
        self.assert_relative(
            decomposition.unlensed_hplus_hcross, unlensed, 0.0,
            'decompose unlensed strain != wrapped generator strain')

    def _plot(self, amplification, ratios, m_arr) -> None:
        """Diagnostic: per-mode ratio vs F(w) -- should collapse."""
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(np.real(amplification), np.imag(amplification), 'k-',
                  lw=2, label='F(w)')
        markers = itertools.cycle('os^vD')
        for ratio, marker in zip(ratios, markers):
            axis.plot(np.real(ratio), np.imag(ratio), marker,
                      ms=4, alpha=0.6)
        axis.set_xlabel('Re')
        axis.set_ylabel('Im')
        axis.set_title('Per-mode ratio collapses onto F(w)')
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'test_lensing_waveform_per_mode.png',
                    dpi=90)
        plt.close(fig)


class MacroSaddleRejectionTestCase(LensedWaveformTestCase):
    """
    Macro-saddle (non-positive-parity) lenses are refused at BOTH the
    constructor and the strain-evaluation path, as a propagated
    `geometry.LensDomainError` -- never a warning, ``nan``, or finite
    output.
    """

    #: Non-positive-parity configs ``1 - kappa <= |gamma|``.  The first
    #: two are float64-exact boundary points (``1 - kappa == |gamma|``
    #: bit-for-bit, per FINDINGS F004); the third is a strict interior
    #: violation.
    BAD_CONFIGS = (
        ('boundary 0.5/0.5', dict(kappa=0.5, gamma=0.5)),
        ('boundary 0.75/0.25', dict(kappa=0.75, gamma=0.25)),
        ('interior 0.5/0.6', dict(kappa=0.5, gamma=0.6)),
    )

    #: Positive-parity controls ``1 - kappa > |gamma|`` that must NOT
    #: raise -- proving the gate discriminates rather than always firing.
    GOOD_CONFIGS = (
        ('control 0.0/0.2', dict(kappa=0.0, gamma=0.2)),
        ('control 0.5/0.25', dict(kappa=0.5, gamma=0.25)),
    )

    def _bypass_construct(self, **lens):
        """
        Build a generator with the domain gate bypassed, so its stored
        config can be a macro saddle and the STRAIN path's own guard can
        be exercised in isolation.
        """
        generator = lw.LensedWaveformGenerator.__new__(
            lw.LensedWaveformGenerator)
        generator.waveform_generator = StubWaveformGenerator((2,))
        generator.m_lens_msun = REF_M_LENS_MSUN
        generator.z_lens = REF_Z_LENS
        generator.y = np.array(BENIGN_LENS['y'], dtype=float)
        generator.gamma = float(lens['gamma'])
        generator.beta = 0.0
        generator.kappa = float(lens['kappa'])
        return generator

    def test_constructor_rejects_macro_saddles(self):
        """Construction raises `LensDomainError` for every bad config."""
        for label, lens in self.BAD_CONFIGS:
            with self.subTest(config=label):
                self.note_comparison()
                with self.assertRaises(geometry.LensDomainError):
                    _make_generator((2,), **lens)

    def test_strain_path_rejects_macro_saddles(self):
        """
        The strain path guards independently: even with the constructor
        bypassed, `get_hplus_hcross` and `amplification` on a bad config
        raise `LensDomainError` (propagated, not swallowed) rather than
        returning ``nan``/finite output.
        """
        for label, lens in self.BAD_CONFIGS:
            with self.subTest(config=label):
                generator = self._bypass_construct(**lens)
                self.note_comparison()
                with self.assertRaises(geometry.LensDomainError):
                    generator.amplification(FREQ_HZ)
                with self.assertRaises(geometry.LensDomainError):
                    generator.get_hplus_hcross(FREQ_HZ, dict(STUB_PAR_DIC))
                with self.assertRaises(geometry.LensDomainError):
                    generator.decompose(FREQ_HZ, dict(STUB_PAR_DIC))

    def test_positive_parity_controls_do_not_raise(self):
        """
        Control configs construct and evaluate to finite output -- the
        gate is discriminating, not a blanket refusal.
        """
        truth_table = {}
        for label, lens in self.GOOD_CONFIGS:
            with self.subTest(config=label):
                generator = _make_generator((2,), **lens)
                amplification = generator.amplification(FREQ_HZ)
                strain = generator.get_hplus_hcross(
                    FREQ_HZ, dict(STUB_PAR_DIC))
                self.note_comparison()
                self.assertTrue(np.all(np.isfinite(amplification)),
                                f'{label}: non-finite F')
                self.assertTrue(np.all(np.isfinite(strain)),
                                f'{label}: non-finite strain')
                truth_table[label] = 'returned'
        self.assertEqual(len(truth_table), len(self.GOOD_CONFIGS))


class SelfFalsificationTestCase(TestCase):
    """
    Prove this suite can FAIL.

    A structural routing bug is quiet -- it does not raise -- so a green
    suite is worth only as much as its ability to go red.  These tests
    corrupt each property the suite guards and assert the corruption is
    caught.
    """

    def test_unapplied_amplification_is_detected(self):
        """
        A generator that forgets to apply ``F`` (returns the unlensed
        strain) must fail the per-mode consistency assertion.
        """
        generator = _make_generator((2, 3, 4))
        amplification = generator.amplification(FREQ_HZ)
        unlensed = generator.waveform_generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=True)
        # The BUGGY "lensed" strain is just the unlensed strain.
        error = float(np.max(_relative_error(
            unlensed, unlensed * amplification)))
        self.assertGreater(
            error, TOL_ROUNDOFF,
            'un-amplified strain slipped under the roundoff tolerance; '
            'the per-mode consistency check could not detect a missing F')

    def test_amplification_on_summed_grid_only_is_detected(self):
        """
        Applying ``F`` to the mode-SUMMED strain and redistributing it
        does not reproduce per-mode ``F * h_m`` unless ``F`` is genuinely
        per-mode; a wrong routing must be visible in at least one mode.
        """
        generator = _make_generator((2, 3, 4))
        amplification = generator.amplification(FREQ_HZ)
        unlensed = generator.waveform_generator.get_hplus_hcross(
            FREQ_HZ, dict(STUB_PAR_DIC), by_m=True)
        # BUG: apply F only to mode index 0, leave others unlensed.
        buggy = unlensed.copy()
        buggy[0] = unlensed[0] * amplification
        correct = unlensed * amplification
        error = float(np.max(_relative_error(buggy, correct)))
        self.assertGreater(
            error, TOL_ROUNDOFF,
            'partial-mode amplification slipped under tolerance')

    def test_wrong_frequency_constant_is_detected(self):
        """
        A ``w`` built with a corrupted constant (extra factor of 2) must
        violate the convention tolerance -- the check is discriminating.
        """
        generator = _make_generator((2,), m_lens_msun=REF_M_LENS_MSUN,
                                    z_lens=REF_Z_LENS)
        w = generator.dimensionless_frequency(FREQ_HZ)
        wrong = 2.0 * XI_PER_MSUN_PER_HZ * REF_M_LENS_MSUN \
            * (1.0 + REF_Z_LENS) * FREQ_HZ
        error = float(np.max(_relative_error(w, wrong)))
        self.assertGreater(
            error, TOL_CONVENTION,
            'a doubled frequency constant slipped under TOL_CONVENTION')

    def test_domain_control_genuinely_does_not_raise(self):
        """
        A positive-parity control must NOT raise -- otherwise the
        rejection truth table would be a blanket "always raises" and
        prove nothing.  Assert that expecting it to raise fails.
        """
        with self.assertRaises(self.failureException):
            with self.assertRaises(geometry.LensDomainError):
                _make_generator((2,), kappa=0.0, gamma=0.2)


if __name__ == '__main__':
    main()
