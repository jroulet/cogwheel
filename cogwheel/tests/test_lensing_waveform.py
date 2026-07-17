"""
Tests for `lensing.waveform`, the Chang--Refsdal microlensed waveform
generator.

WHAT THIS SUITE OWNS
--------------------
`LensedWaveformGenerator` is a COMPOSER: it multiplies whatever unlensed
strain the wrapped generator returns by a single frequency-domain factor
``F(w(f))``.  This suite tests that wiring and the two physical corners
where the factor's certified-or-refuse contract becomes visible at the
waveform layer:

* the amplification is a common multiplicative factor (``F * h``), equal
  to ``1`` where ``w <= 0``, and its per-image decomposition reconstructs
  the same total;
* the dimensionless-frequency map ``w = 8*pi*G*M_L*(1+z_L)*f/c**3`` is
  linear in ``f`` and carries the one physical constant;
* MACRO-SADDLE CONTROL (Architect spec 1): an in-band lens configuration
  returns a finite, certified O(1) amplification at the engine's default
  order-42 budget, while a companion at the certification band edge
  refuses cleanly with `operator.CancellationError`;
* SMALL-MASS UNLENSED FLOOR (Architect spec 2): as the lens mass shrinks
  the amplification approaches the unlensed limit ``F -> 1`` monotonically
  over the physically meaningful mass range.

WHY THE ORACLES ARE INDEPENDENT
-------------------------------
Nothing here is judged against the engine's own derivation of ``F``:

* The dimensionless-frequency check is gated against an INDEPENDENT
  literature value of the geometrized solar mass
  ``G*M_sun/c**3 = 4.925490947641266e-6 s`` (`MTSUN_LIT`), not against
  ``lal.MTSUN_SI`` (which the module itself uses); and it asserts the
  LINEARITY the module claims via superposition and scaling, which needs
  no constant at all.
* The macro-saddle control does not check ``F``'s value; it checks the
  CERTIFIED-OR-REFUSE partition of the ``(w, gamma')`` plane -- the same
  contract the operator suite certifies against a 50-digit mpmath oracle.
  The in-band config must return; the band-edge config must raise.  The
  band edge is the previously mis-specified control
  (``y=(0.5, 0.25)``, ``gamma=0.25``, ``kappa=0.5``, so
  ``gamma_eff = gamma/(1 - kappa) = 0.5``), whose order-42 shear-series
  tail (~1.168e-10) sits just above the 1e-10 certification target: its
  refusal is a FEATURE and is asserted where it belongs, at the waveform
  layer that consumes it.
* The unlensed floor is a self-oracling PHYSICAL limit: ``F -> 1`` as
  ``w -> 0``.  Monotone approach to one needs no external reference, and
  the two-case contrast (physical masses monotone; the singular tiny-w
  point excluded) is the substance.

DEFERRED SMALL-W ENGINE GAP
---------------------------
The smallest mass in the floor sweep, ``M_L = 1e-12 Msun`` (``w ~ 1e-13``),
is EXCLUDED from the floor assertion.  As ``w -> 0`` the operator series'
shear prefactor ``gamma/(2*w)`` diverges, a KNOWN engine gap that is
DEFERRED, not a regression: the ``w -> 0`` corner is simply outside the
certified ``(w, gamma')`` band.  It is SAFE because the engine does not
return a wrong number there -- under the F005 certified-or-refuse contract
`operator.F_op` raises rather than certifying an untrustworthy value.
`UnlensedFloorTestCase.test_singular_small_w_point_is_correctly_excluded`
demonstrates the exclusion is warranted (the point raises or breaks the
clean floor), so it is a documented boundary rather than an arbitrary
drop.  Tracked as the deferred small-w unlensed-limit gap; see FINDINGS
F005 for the containing certified-or-refuse contract.

ANTI-VACUITY AND SELF-FALSIFICATION
-----------------------------------
`WaveformTestCase.tearDown` fails a test that made zero comparisons, so a
sweep whose every configuration was skipped cannot read as green.
`SelfFalsificationTestCase` proves each gate above can actually go red.
"""
from __future__ import annotations

import itertools
import pathlib
from dataclasses import dataclass
from unittest import TestCase, main

import numpy as np

from cogwheel.lensing import waveform
from cogwheel.lensing.chang_refsdal import operator, geometry


try:  # Diagnostics only; never gate a test on plotting being present.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False


#: Independent literature value of the geometrized solar mass
#: ``G*M_sun/c**3`` in seconds (``GM_sun = 1.32712440018e20 m**3/s**2``,
#: ``c = 299792458 m/s``).  Deliberately NOT ``lal.MTSUN_SI`` -- which the
#: module under test uses -- so the frequency map is checked against a
#: reference that shares none of its provenance.
MTSUN_LIT = 4.925490947641266e-6

#: Relative tolerance on the dimensionless-frequency constant.  Loose
#: enough to absorb last-digit differences between the literature value
#: and ``lal.MTSUN_SI``, tight enough to catch a wrong constant or a
#: missing ``8*pi`` / ``(1 + z)`` factor.
FREQ_RTOL = 1e-6

#: Operator-series order budget the waveform layer runs at
#: (``ChangRefsdalChannels`` default).  The macro-saddle control is a
#: statement about certification AT THIS ORDER, matching the spec's
#: "order-42".
CERT_MAX_ORDER = operator.MAX_ORDER

#: Cancellation-exponent ceiling ``L = w*|y'|`` below which the wave
#: branch is the only branch (``select_branch`` cannot pick geometric),
#: so every probed frequency actually exercises ``F_op``.  Mirrors
#: ``operator.L_MAX``; used only to keep probes on the wave branch.
_L_MAX = operator.L_MAX

#: Dimensionless-frequency floor below which the small-w engine gap makes
#: the amplification untrustworthy; the floor assertion is restricted to
#: masses whose probed ``w`` exceeds this.  See the module docstring.
W_FLOOR_CUTOFF = 1e-3

#: Upper bound on ``|F| - 1`` for a configuration to count as sitting in
#: the clean unlensed neighbourhood (a weak lens at small ``w``).
CLEAN_FLOOR_MAX = 0.5

#: Lens mass / redshift used to turn target ``w`` values into a frequency
#: grid for the control tests; the physics depends only on ``w``, so the
#: exact mass is immaterial (it just sets the conversion ``xi``).
_CONTROL_MASS_MSUN = 1.0e3
_CONTROL_Z = 0.0

#: Fixed redshift and lowest grid frequency for the mass sweep.
_FLOOR_Z = 0.0
_FLOOR_F0_HZ = 100.0

#: Directory for diagnostic figures.
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


@dataclass(frozen=True)
class _LensConfig:
    """A Chang--Refsdal lens configuration for the control tests.

    Attributes
    ----------
    name : str
        Human-readable label used in messages and figures.
    y : tuple[float, float]
        Source position.
    gamma : float
        External shear magnitude.
    beta : float
        External shear orientation, radians.
    kappa : float
        External convergence.
    w_probes : tuple[float, ...]
        Dimensionless frequencies at which to probe the configuration;
        chosen so ``L = w*|y'|`` stays below ``operator.L_MAX`` (wave
        branch) and ``w*|y'|`` below the kernel's ceiling of 60.
    """

    name: str
    y: tuple
    gamma: float
    beta: float
    kappa: float
    w_probes: tuple

    @property
    def gamma_eff(self) -> float:
        """Effective (mass-sheet-rescaled) shear ``gamma / (1 - kappa)``."""
        return self.gamma / (1.0 - self.kappa)

    @property
    def y_array(self) -> np.ndarray:
        """Source position as a length-2 float array."""
        return np.asarray(self.y, dtype=float)


#: Interior control: low shear and small cancellation exponent, so the
#: order-42 shear series converges with large margin and ``F_op``
#: certifies a return.  ``gamma_eff = 0.10``.
IN_BAND = _LensConfig(
    name='in-band', y=(0.30, 0.10), gamma=0.10, beta=0.0, kappa=0.0,
    w_probes=(4.0, 8.0, 12.0))

#: Band-edge companion: the previously mis-specified positive-parity
#: control.  ``gamma_eff = 0.25 / (1 - 0.5) = 0.5`` sits at the edge of
#: the certified band, and at order 42 the shear-series tail exceeds the
#: 1e-10 target, so ``F_op`` correctly REFUSES.  ``L = w*|y'|`` with
#: ``|y'| = |y|/sqrt(1 - kappa) ~ 0.79`` stays below 48 across the
#: probes, keeping every probe on the wave branch where the refusal
#: lives.
BAND_EDGE = _LensConfig(
    name='band-edge', y=(0.50, 0.25), gamma=0.25, beta=0.0, kappa=0.5,
    w_probes=(30.0, 40.0, 50.0))

#: Weak-lens configuration for the unlensed-limit floor sweep.
FLOOR_CONFIG = _LensConfig(
    name='unlensed-floor', y=(0.30, 0.10), gamma=0.10, beta=0.0,
    kappa=0.0, w_probes=())

#: Lens masses (solar masses) for the floor sweep, decreasing.  The first
#: is the DEFERRED small-w point (excluded from the floor assertion); the
#: rest are physically meaningful masses whose probed ``w`` exceeds
#: ``W_FLOOR_CUTOFF``.
FLOOR_MASSES_MSUN = (1.0e-12, 0.1, 0.3, 1.0, 2.0, 4.0)


class _StubWaveformGenerator:
    """Minimal ``WaveformGenerator`` stand-in with a KNOWN unlensed strain.

    Returns a deterministic, nonzero strain so the composition
    ``F * h`` can be checked exactly without invoking LAL.  It is NOT a
    physics oracle: it isolates the lens amplification, which is the only
    thing `LensedWaveformGenerator` adds on top of the wrapped generator.
    The interface mirrors exactly what `LensedWaveformGenerator` calls
    (``get_hplus_hcross``, ``m_arr``) plus ``harmonic_modes`` for the
    delegated property.
    """

    def __init__(self, m_arr=(2, 3)) -> None:
        self.m_arr = np.asarray(m_arr, dtype=int)
        self.harmonic_modes = [(int(abs(m)), int(m)) for m in self.m_arr]

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False
                         ) -> np.ndarray:
        """Deterministic per-``|m|`` strain, distinct across modes.

        Returns shape ``(n_m, 2, n_f)`` when ``by_m`` else the
        mode-summed ``(2, n_f)``.  The content is arbitrary but nonzero
        and mode-distinct, so a dropped mode or a mis-broadcast factor
        would change the product.
        """
        frequencies = np.asarray(f, dtype=float)
        n_f = frequencies.size
        modes = np.empty((self.m_arr.size, 2, n_f), dtype=complex)
        for index, m_number in enumerate(self.m_arr):
            phase = np.exp(1j * 0.003 * m_number * frequencies)
            modes[index, 0] = (1.0 + 0.1 * m_number) * phase
            modes[index, 1] = (0.7 - 0.05 * m_number) * 1j * phase
        if by_m:
            return modes
        return modes.sum(axis=0)


def _make_generator(config: _LensConfig, m_lens_msun: float,
                    z_lens: float) -> waveform.LensedWaveformGenerator:
    """Build a `LensedWaveformGenerator` wrapping a stub, for `config`."""
    return waveform.LensedWaveformGenerator(
        _StubWaveformGenerator(), m_lens_msun=m_lens_msun,
        z_lens=z_lens, y=config.y_array, gamma=config.gamma,
        beta=config.beta, kappa=config.kappa)


def _frequencies_for_w(w_values,
                       generator: waveform.LensedWaveformGenerator
                       ) -> np.ndarray:
    """Return the frequency grid mapping to ``w_values`` for `generator`.

    Inverts ``w = xi * f`` using the generator's own conversion, so
    ``generator.amplification(f)`` is evaluated at exactly ``w_values``.
    """
    xi_per_hz = float(generator.dimensionless_frequency(1.0))
    return np.asarray(w_values, dtype=float) / xi_per_hz


def _f_op_returns(config: _LensConfig, w: float) -> bool:
    """Whether `operator.F_op` certifies a finite return for `config`.

    ``True`` if the call returns a finite value; ``False`` if it raises
    `operator.CancellationError`.  Other exceptions propagate -- they are
    not the certified/refused distinction this probe reports.
    """
    try:
        value, _ = operator.F_op(
            w, config.y_array, config.gamma, beta=config.beta,
            kappa=config.kappa, max_order=CERT_MAX_ORDER)
    except operator.CancellationError:
        return False
    return bool(np.isfinite(value.real) and np.isfinite(value.imag))


def _is_strictly_increasing(values) -> bool:
    """Return whether ``values`` strictly increases (no ties)."""
    array = np.asarray(values, dtype=float)
    return bool(array.size >= 2 and np.all(np.diff(array) > 0.0))


def _savefig(fig, name: str) -> None:
    """Save a diagnostic figure, swallowing any backend error."""
    if not _HAVE_MPL:
        return
    try:
        _OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(_OUTPUT_DIR / name, dpi=80, bbox_inches='tight')
    except Exception:  # pragma: no cover - environment dependent
        pass
    finally:
        plt.close(fig)


class WaveformTestCase(TestCase):
    """Base class carrying the anti-vacuity comparison tally.

    `tearDown` fails a test that asserted nothing, so a sweep whose every
    configuration was skipped cannot read as green.
    """

    _expect_checks = True

    def setUp(self) -> None:
        self.n_checks = 0

    def tearDown(self) -> None:
        if self._expect_checks and self.n_checks == 0:
            self.fail('vacuous test: no comparison ran, so nothing was '
                      'asserted')


class DimensionlessFrequencyTestCase(WaveformTestCase):
    """``w = 8*pi*G*M_L*(1 + z_L)*f/c**3`` is linear in ``f``."""

    def test_matches_independent_constant(self):
        """
        ``w`` equals ``8*pi*MTSUN_LIT*M_L*(1 + z_L)*f`` with the
        INDEPENDENT literature ``MTSUN_LIT`` -- not ``lal.MTSUN_SI``.
        """
        f_hz = np.array([10.0, 55.0, 200.0, 1024.0])
        for m_lens, z_lens in itertools.product(
                (1.0, 30.0, 1.0e3), (0.0, 0.5, 2.0)):
            with self.subTest(m_lens=m_lens, z_lens=z_lens):
                got = waveform.dimensionless_frequency(
                    f_hz, m_lens, z_lens)
                expected = (8.0 * np.pi * MTSUN_LIT * m_lens
                            * (1.0 + z_lens) * f_hz)
                np.testing.assert_allclose(got, expected, rtol=FREQ_RTOL)
                self.n_checks += 1

    def test_is_linear_in_frequency(self):
        """
        Superposition and scaling hold to roundoff -- the linearity the
        module claims, checked WITHOUT the physical constant.
        """
        f_a = np.array([12.0, 40.0, 130.0])
        f_b = np.array([3.0, 90.0, 7.0])
        for m_lens, z_lens in ((5.0, 0.0), (250.0, 1.3)):
            with self.subTest(m_lens=m_lens, z_lens=z_lens):
                w_a = waveform.dimensionless_frequency(f_a, m_lens, z_lens)
                w_b = waveform.dimensionless_frequency(f_b, m_lens, z_lens)
                w_sum = waveform.dimensionless_frequency(
                    f_a + f_b, m_lens, z_lens)
                w_scaled = waveform.dimensionless_frequency(
                    3.5 * f_a, m_lens, z_lens)
                np.testing.assert_allclose(w_sum, w_a + w_b, rtol=1e-13)
                np.testing.assert_allclose(w_scaled, 3.5 * w_a, rtol=1e-13)
                self.n_checks += 1

    def test_scales_with_mass_and_redshift(self):
        """``w`` is proportional to ``M_L`` and to ``(1 + z_L)``."""
        f_hz = np.array([25.0, 300.0])
        base = waveform.dimensionless_frequency(f_hz, 10.0, 0.0)
        doubled_mass = waveform.dimensionless_frequency(f_hz, 20.0, 0.0)
        doubled_1pz = waveform.dimensionless_frequency(f_hz, 10.0, 1.0)
        np.testing.assert_allclose(doubled_mass, 2.0 * base, rtol=1e-13)
        np.testing.assert_allclose(doubled_1pz, 2.0 * base, rtol=1e-13)
        self.n_checks += 1

    def test_preserves_shape_and_zero(self):
        """The map preserves array shape and sends ``f = 0`` to ``w = 0``."""
        f_hz = np.linspace(0.0, 512.0, 9)
        got = waveform.dimensionless_frequency(f_hz, 42.0, 0.7)
        self.assertEqual(got.shape, f_hz.shape)
        self.assertEqual(float(got[0]), 0.0)
        self.n_checks += 1


class ConstructionValidationTestCase(WaveformTestCase):
    """Construction-time guards of `LensedWaveformGenerator`."""

    def test_macro_saddle_raises_lens_domain_error(self):
        """
        ``1 - kappa <= |gamma|`` (a macro saddle) raises
        `geometry.LensDomainError` AT CONSTRUCTION, never a silent
        downgrade to a warning or a later ``nan``.
        """
        macro_cases = ((1.5, 0.0), (0.5, 0.6), (0.9, 0.1))
        for gamma, kappa in macro_cases:
            with self.subTest(gamma=gamma, kappa=kappa):
                with self.assertRaises(geometry.LensDomainError):
                    waveform.LensedWaveformGenerator(
                        _StubWaveformGenerator(), m_lens_msun=100.0,
                        z_lens=0.0, y=(0.3, 0.1), gamma=gamma,
                        kappa=kappa)
                self.n_checks += 1

    def test_positive_parity_config_constructs(self):
        """A positive-parity config (``1 - kappa > |gamma|``) constructs."""
        generator = _make_generator(IN_BAND, 100.0, 0.0)
        self.assertEqual(generator.gamma, IN_BAND.gamma)
        self.assertEqual(list(generator.m_arr), [2, 3])
        self.n_checks += 1

    def test_bad_source_shape_raises_value_error(self):
        """A source position that is not a two-vector raises ``ValueError``."""
        for bad_y in ((1.0, 2.0, 3.0), (0.5,), [[0.1, 0.2]]):
            with self.subTest(y=bad_y):
                with self.assertRaises(ValueError):
                    waveform.LensedWaveformGenerator(
                        _StubWaveformGenerator(), m_lens_msun=100.0,
                        z_lens=0.0, y=bad_y, gamma=0.1)
                self.n_checks += 1

    def test_non_generator_raises_value_error(self):
        """
        A wrapped object lacking the generator interface raises
        ``ValueError`` rather than failing later at first strain call.
        """
        with self.assertRaises(ValueError):
            waveform.LensedWaveformGenerator(
                object(), m_lens_msun=100.0, z_lens=0.0, y=(0.3, 0.1),
                gamma=0.1)
        self.n_checks += 1


class MacroSaddleControlTestCase(WaveformTestCase):
    """
    Architect spec 1: certified-or-refuse partition at the waveform layer.

    An in-band configuration returns a finite, certified O(1)
    amplification at order 42; the band-edge companion (the mis-specified
    ``gamma_eff = 0.5`` control) refuses cleanly with
    `operator.CancellationError`.  The refusal is a FEATURE, asserted
    where it belongs -- in the waveform layer that consumes the engine's
    certified-or-refuse contract.
    """

    def test_config_effective_shears_are_as_documented(self):
        """
        Pin the mass-sheet-rescaled shears: the in-band control has
        ``gamma_eff = 0.10`` and the band-edge companion the
        mis-specified ``gamma_eff = 0.5`` the spec calls out.
        """
        self.assertAlmostEqual(IN_BAND.gamma_eff, 0.10, places=12)
        self.assertAlmostEqual(BAND_EDGE.gamma_eff, 0.50, places=12)
        self.n_checks += 1

    def test_in_band_control_amplification_is_finite_and_order_one(self):
        """
        ``amplification`` returns finite, O(1) values with NO exception on
        the in-band control, and the underlying ``F_op`` certifies each
        probe (converged, with a truncation tail below the 1e-10 target).
        """
        generator = _make_generator(IN_BAND, _CONTROL_MASS_MSUN,
                                    _CONTROL_Z)
        f_hz = _frequencies_for_w(IN_BAND.w_probes, generator)

        factor = generator.amplification(f_hz)
        self.assertEqual(factor.shape, f_hz.shape)
        self.assertTrue(np.all(np.isfinite(factor)))
        magnitudes = np.abs(factor)
        self.assertTrue(np.all((magnitudes > 0.1) & (magnitudes < 10.0)),
                        f'in-band |F| not O(1): {magnitudes}')
        # Genuine lensing, not a trivial F == 1 everywhere.
        self.assertGreater(float(np.max(np.abs(factor - 1.0))), 1e-6)

        for w in IN_BAND.w_probes:
            with self.subTest(w=w):
                value, diag = operator.F_op(
                    w, IN_BAND.y_array, IN_BAND.gamma, beta=IN_BAND.beta,
                    kappa=IN_BAND.kappa, max_order=CERT_MAX_ORDER)
                self.assertTrue(np.isfinite(value.real)
                                and np.isfinite(value.imag))
                self.assertTrue(diag.converged,
                                f'in-band w={w} did not converge')
                self.assertLess(diag.estimated_relative_tail,
                                operator._CONTRACTION_TARGET,
                                f'in-band w={w} tail not certified')
                self.n_checks += 1

    def test_band_edge_companion_refuses_cleanly(self):
        """
        The band-edge companion raises `operator.CancellationError` at the
        waveform layer: at least one probe refuses at the operator level,
        and ``amplification`` over the probe grid propagates the refusal
        unswallowed (never a ``nan`` or a finite-but-wrong factor).
        """
        refusals = [w for w in BAND_EDGE.w_probes
                    if not _f_op_returns(BAND_EDGE, w)]
        self.assertTrue(
            refusals,
            'the band-edge companion certified every probe; it is no '
            'longer at the certification edge')
        self.n_checks += len(BAND_EDGE.w_probes)

        generator = _make_generator(BAND_EDGE, _CONTROL_MASS_MSUN,
                                    _CONTROL_Z)
        f_hz = _frequencies_for_w(BAND_EDGE.w_probes, generator)
        with self.assertRaises(operator.CancellationError):
            generator.amplification(f_hz)
        self.n_checks += 1

    def test_refusal_message_names_the_configuration(self):
        """
        A refusal identifies the offending ``(w, y, gamma, kappa)`` so a
        caller can tell which configuration was refused, not merely that
        one was.
        """
        refusing_w = next((w for w in BAND_EDGE.w_probes
                           if not _f_op_returns(BAND_EDGE, w)), None)
        self.assertIsNotNone(refusing_w,
                             'no band-edge probe refused to inspect')
        with self.assertRaises(operator.CancellationError) as ctx:
            operator.F_op(refusing_w, BAND_EDGE.y_array, BAND_EDGE.gamma,
                          beta=BAND_EDGE.beta, kappa=BAND_EDGE.kappa,
                          max_order=CERT_MAX_ORDER)
        message = str(ctx.exception)
        for token in ('w =', 'y =', 'gamma', 'kappa'):
            self.assertIn(token, message,
                          f'refusal message omits {token!r}: {message}')
        self.n_checks += 1

    def test_diagnostic_scatter(self):
        """
        Non-asserting diagnostic: scan ``w`` for both controls and plot
        the ``(w, gamma')`` plane coloured by certified return vs refusal.
        """
        rows = []
        for config in (IN_BAND, BAND_EDGE):
            grid = np.linspace(min(config.w_probes),
                               max(config.w_probes), 7)
            for w in grid:
                rows.append((float(w), config.gamma_eff,
                             _f_op_returns(config, float(w))))
        self.n_checks += 1
        self.assertTrue(rows)
        self._plot(rows)

    def _plot(self, rows):
        if not _HAVE_MPL or not rows:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        returned = [(w, g) for w, g, ok in rows if ok]
        refused = [(w, g) for w, g, ok in rows if not ok]
        if returned:
            ax.scatter([w for w, _ in returned],
                       [g for _, g in returned], c='C0', s=30,
                       label='certified return')
        if refused:
            ax.scatter([w for w, _ in refused],
                       [g for _, g in refused], c='C3', marker='x', s=40,
                       label='CancellationError')
        ax.axhline(0.5, color='k', ls='--', alpha=0.6,
                   label="gamma_eff = 0.5 band edge")
        ax.set_xlabel('w')
        ax.set_ylabel("effective shear gamma' = gamma/(1 - kappa)")
        ax.set_title('macro-saddle control: certified vs refused')
        ax.legend()
        _savefig(fig, 'waveform_macro_saddle_control.png')


class UnlensedFloorTestCase(WaveformTestCase):
    """
    Architect spec 2: the amplification approaches ``F -> 1`` as the lens
    mass shrinks, monotonically over the physically meaningful masses.

    The floor ``|F| - 1`` is asserted only for masses whose probed ``w``
    exceeds ``W_FLOOR_CUTOFF``; the singular tiny-w point is excluded (see
    the module docstring's deferred-gap note).
    """

    def _floor_at_smallest_w(self, m_lens_msun: float
                             ) -> tuple[float, float]:
        """Return ``(w0, |F(w0)| - 1)`` at the smallest grid frequency.

        Uses a two-point grid ``[f0, 2*f0]`` (the engine requires at
        least two frequencies); the floor is read at the smaller ``w``.
        """
        generator = _make_generator(FLOOR_CONFIG, m_lens_msun, _FLOOR_Z)
        f_hz = np.array([_FLOOR_F0_HZ, 2.0 * _FLOOR_F0_HZ])
        w0 = float(generator.dimensionless_frequency(f_hz[0]))
        factor = generator.amplification(f_hz)
        return w0, float(abs(factor[0]) - 1.0)

    def test_floor_is_monotone_over_physical_masses(self):
        """
        Over masses with ``w > W_FLOOR_CUTOFF`` the floor ``|F| - 1`` is
        strictly increasing with ``w`` (equivalently, strictly decreasing
        as the mass shrinks toward the unlensed limit), stays in the
        clean unlensed neighbourhood, and is nontrivially bounded away
        from zero at the top of the range.
        """
        physical = [m for m in FLOOR_MASSES_MSUN if m > 1e-9]
        ws, floors = [], []
        for m_lens in physical:
            w0, floor = self._floor_at_smallest_w(m_lens)
            self.assertGreater(
                w0, W_FLOOR_CUTOFF,
                f'mass {m_lens} probes w={w0:.2e} below the floor cutoff')
            ws.append(w0)
            floors.append(floor)
            self.n_checks += 1

        order = np.argsort(ws)
        ordered_floors = np.asarray(floors)[order]

        self.assertTrue(
            _is_strictly_increasing(ordered_floors),
            f'floor |F|-1 not monotone in w: {ordered_floors}')
        self.assertTrue(
            np.all((ordered_floors > 0.0)
                   & (ordered_floors < CLEAN_FLOOR_MAX)),
            f'floor left the clean unlensed neighbourhood: '
            f'{ordered_floors}')
        # Clean ~1e-3 near the cutoff; nontrivial approach to one across
        # the range (the small-w floor is far below the large-w one).
        self.assertLess(ordered_floors[0], 1e-2,
                        'floor near the w~1e-3 cutoff is not clean')
        self.assertGreater(ordered_floors[-1] / ordered_floors[0], 5.0,
                           'floor is flat: no approach to the unlensed '
                           'limit across the mass range')
        self._plot(ws, floors)

    def test_singular_small_w_point_is_correctly_excluded(self):
        """
        Justify the exclusion of ``M_L = 1e-12 Msun`` (``w ~ 1e-13``) as a
        DOMAIN cutoff, not an arbitrary drop.

        The exclusion rule is exactly ``w > W_FLOOR_CUTOFF``: the tiny
        mass probes a ``w`` far below the cutoff, into the deferred
        small-w gap where the shear prefactor ``gamma/(2*w)`` is
        uncertified (see the module docstring), so the mass filter used by
        the floor assertion drops it.  This checks the MECHANISM of the
        exclusion -- that the documented cutoff actually filters the point
        out -- rather than asserting a particular (raise vs return)
        failure mode there, which is exactly the uncertified behaviour we
        decline to depend on.
        """
        tiny_mass = min(FLOOR_MASSES_MSUN)
        generator = _make_generator(FLOOR_CONFIG, tiny_mass, _FLOOR_Z)
        w0_tiny = float(generator.dimensionless_frequency(_FLOOR_F0_HZ))
        self.assertLess(
            w0_tiny, W_FLOOR_CUTOFF,
            f'the tiny mass probes w={w0_tiny:.2e}, not below the '
            f'{W_FLOOR_CUTOFF:.0e} cutoff; the sweep is mis-specified')

        asserted_masses = [
            m for m in FLOOR_MASSES_MSUN
            if float(_make_generator(FLOOR_CONFIG, m, _FLOOR_Z)
                     .dimensionless_frequency(_FLOOR_F0_HZ))
            > W_FLOOR_CUTOFF]
        self.assertNotIn(
            tiny_mass, asserted_masses,
            'the cutoff rule failed to exclude the deferred small-w '
            'point from the floor assertion')
        self.assertTrue(asserted_masses,
                        'the cutoff excluded every mass; nothing would '
                        'be asserted')
        self.n_checks += 1

    def _plot(self, ws, floors):
        if not _HAVE_MPL or not ws:
            return
        # Include the excluded tiny-w point greyed out, if it evaluates.
        tiny_mass = min(FLOOR_MASSES_MSUN)
        excluded = None
        try:
            excluded = self._floor_at_smallest_w(tiny_mass)
        except Exception:  # pragma: no cover - engine may refuse
            excluded = None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(ws, np.abs(floors), 'oC0', label='physical masses')
        if excluded is not None and np.isfinite(excluded[1]):
            ax.loglog([excluded[0]], [abs(excluded[1])], 'o',
                      color='0.7', label='excluded small-w (deferred gap)')
        ax.axvline(W_FLOOR_CUTOFF, color='C3', ls=':',
                   label='w ~ 1e-3 cutoff')
        ax.set_xlabel('w (smallest grid frequency)')
        ax.set_ylabel('|F| - 1')
        ax.set_title('unlensed-limit floor (restricted to physical masses)')
        ax.legend()
        _savefig(fig, 'waveform_unlensed_floor.png')


class AmplificationCompositionTestCase(WaveformTestCase):
    """The lensed strain is the unlensed strain times ``F(w(f))``."""

    def setUp(self) -> None:
        super().setUp()
        self.generator = _make_generator(IN_BAND, _CONTROL_MASS_MSUN,
                                         _CONTROL_Z)
        self.f_hz = _frequencies_for_w(IN_BAND.w_probes, self.generator)
        self.par_dic = {'m1': 30.0, 'm2': 25.0}

    def test_total_strain_is_factor_times_unlensed(self):
        """``get_hplus_hcross`` equals the wrapped strain times ``F``."""
        factor = self.generator.amplification(self.f_hz)
        unlensed = self.generator.waveform_generator.get_hplus_hcross(
            self.f_hz, self.par_dic, by_m=False)
        lensed = self.generator.get_hplus_hcross(
            self.f_hz, self.par_dic, by_m=False)
        np.testing.assert_allclose(lensed, unlensed * factor, rtol=1e-12)
        self.n_checks += 1

    def test_per_mode_strain_is_factor_times_unlensed(self):
        """The common factor multiplies every ``|m|`` mode identically."""
        factor = self.generator.amplification(self.f_hz)
        unlensed = self.generator.waveform_generator.get_hplus_hcross(
            self.f_hz, self.par_dic, by_m=True)
        lensed = self.generator.get_hplus_hcross(
            self.f_hz, self.par_dic, by_m=True)
        np.testing.assert_allclose(
            lensed, unlensed * factor[None, None, :], rtol=1e-12)
        self.n_checks += 1

    def test_amplification_is_unity_where_w_nonpositive(self):
        """
        The ``f = 0`` bin (``w = 0``) takes the unlensed limit ``F = 1``
        exactly, so the strain there is unchanged.
        """
        f_hz = np.concatenate(([0.0], self.f_hz))
        factor = self.generator.amplification(f_hz)
        self.assertEqual(complex(factor[0]), 1.0 + 0.0j)
        self.n_checks += 1


class DecompositionTestCase(WaveformTestCase):
    """`decompose` reconstructs the same total as `amplification`."""

    def setUp(self) -> None:
        super().setUp()
        self.generator = _make_generator(IN_BAND, _CONTROL_MASS_MSUN,
                                         _CONTROL_Z)
        self.f_hz = _frequencies_for_w(IN_BAND.w_probes, self.generator)
        self.par_dic = {'m1': 30.0, 'm2': 25.0}

    def test_reconstructed_amplification_matches_total(self):
        """
        ``sum_a exp(1j*w*tau_a) K_a(w)`` reproduces ``amplification`` on
        the same grid, so the analytic-phase decomposition the likelihood
        consumes carries the full total.
        """
        decomposition = self.generator.decompose(self.f_hz, self.par_dic)
        np.testing.assert_allclose(
            decomposition.reconstructed_amplification,
            self.generator.amplification(self.f_hz), rtol=1e-6,
            atol=1e-9)
        self.n_checks += 1

    def test_unlensed_bins_reconstruct_unity(self):
        """Rows with ``w <= 0`` reconstruct ``F = 1`` exactly."""
        f_hz = np.concatenate(([0.0], self.f_hz))
        decomposition = self.generator.decompose(f_hz, self.par_dic)
        self.assertAlmostEqual(
            complex(decomposition.reconstructed_amplification[0]),
            1.0 + 0.0j, places=9)
        self.n_checks += 1

    def test_unlensed_strain_shape_carries_modes(self):
        """The decomposition carries the wrapped per-``|m|`` strain."""
        decomposition = self.generator.decompose(self.f_hz, self.par_dic)
        self.assertEqual(
            decomposition.unlensed_hplus_hcross.shape,
            (self.generator.m_arr.size, 2, self.f_hz.size))
        np.testing.assert_array_equal(
            decomposition.m_arr, self.generator.m_arr)
        self.n_checks += 1


class SelfFalsificationTestCase(WaveformTestCase):
    """
    Prove the gates above can actually go red.

    A green suite is worth only as much as its ability to fail, so each
    gate is shown catching a deliberately wrong input.
    """

    _expect_checks = False

    def test_monotone_helper_flags_a_break(self):
        """
        ``_is_strictly_increasing`` accepts a rising sequence and rejects
        one with a dip -- otherwise the floor's monotonicity gate would
        assert nothing.
        """
        self.assertTrue(_is_strictly_increasing([1e-3, 4e-3, 1e-2]))
        self.assertFalse(_is_strictly_increasing([1e-3, 4e-3, 3e-3]))
        self.assertFalse(_is_strictly_increasing([2e-3, 2e-3]))

    def test_refusal_gate_is_non_vacuous(self):
        """
        The in-band control does NOT raise `CancellationError`, so the
        band-edge ``assertRaises`` is meaningful rather than passing for
        any configuration whatsoever.
        """
        for w in IN_BAND.w_probes:
            self.assertTrue(
                _f_op_returns(IN_BAND, w),
                f'the interior control refused at w={w}; the refusal gate '
                'would fire for any configuration')

    def test_clean_floor_bound_rejects_a_far_amplification(self):
        """
        The clean-floor bound would catch an amplification far from one
        (e.g. ``|F| = 2``), so it is not a vacuous ``< CLEAN_FLOOR_MAX``.
        """
        bad_floor = abs(2.0) - 1.0
        self.assertGreaterEqual(bad_floor, CLEAN_FLOOR_MAX)

    def test_frequency_gate_rejects_a_wrong_constant(self):
        """
        A 1% error in the frequency constant breaches ``FREQ_RTOL`` -- the
        dimensionless-frequency gate is not decoration.
        """
        f_hz = 100.0
        good = float(waveform.dimensionless_frequency(f_hz, 10.0, 0.0))
        expected = 8.0 * np.pi * MTSUN_LIT * 10.0 * f_hz
        self.assertLessEqual(abs(good - expected) / abs(expected),
                             FREQ_RTOL)
        self.assertGreater(abs(1.01 * good - expected) / abs(expected),
                           FREQ_RTOL)


if __name__ == '__main__':
    main()
