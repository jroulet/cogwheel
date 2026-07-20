"""
Chang--Refsdal microlensed waveform generation for cogwheel.

WHAT
----
`LensedWaveformGenerator` wraps an ordinary cogwheel
`waveform.WaveformGenerator` and applies the frequency-domain wave-optics
amplification of a Chang--Refsdal lens to every harmonic mode,

    h_lensed_lm(f) = F(w(f)) * h_lm(f),

where the amplification ``F(w) = sum_a exp(1j * w * tau_a) * K_a(w)`` is
produced by `chang_refsdal.ChangRefsdalChannels`.  The dimensionless
frequency

    w = 8 * pi * G * M_L * (1 + z_L) * f / c**3

is *linear* in the observed frequency ``f`` and depends only on ``f`` (not
on the mode), so ``F(w(f))`` is a single multiplicative factor shared
across all modes and both polarizations.

WHY A COMPOSER, NOT A SUBCLASS
------------------------------
The lens amplification is orthogonal to how the unlensed strain is built:
it multiplies whatever the wrapped generator returns.  Holding a
`WaveformGenerator` (rather than subclassing and mutating it) keeps the
LAL waveform machinery untouched and lets the lensed generator serialize
its wrapped generator recursively through `utils.JSONMixin`.

WHY IT EXPOSES A DECOMPOSITION, NOT ONLY A TOTAL
------------------------------------------------
The microlensed relative-binning likelihood (Build 3) keeps the
image-delay phases ``exp(1j * w * tau_a)`` analytic and interpolates only
the smooth kernels ``K_a(w)``.  `decompose` therefore returns the
per-image ``(tau_a, K_a)`` decomposition alongside the unlensed per-mode
strain, while `get_hplus_hcross` returns the collapsed total used for the
``F -> 1`` sanity limit and the brute-force reference.

SCOPE
-----
Both macro-image parities are supported: positive parity
(``1 - kappa > abs(gamma)``) and the macro saddle
(``0 < 1 - kappa < abs(gamma)``), whose wave branch routes through the
Schwinger evaluator inside `operator.F_op`.  The macro-geometry domain
refusals -- over-critical / Type III (``1 - kappa <= 0``) and the exact
``det A = 0`` parity boundary (``abs(gamma) == 1 - kappa``) -- raise
`geometry.LensDomainError`, propagated unswallowed from both the
constructor and every strain/decomposition call -- never downgraded to a
warning or a ``nan``.  Beyond the certified Schwinger ceiling the operator
raises `operator.CancellationError` / `SchwingerCertificationError` by
name, likewise never swallowed.

Conventions
-----------
Frequencies in Hz; lens mass ``m_lens_msun`` in solar masses; ``z_lens``
dimensionless.  The Chang--Refsdal source position ``y``, shear magnitude
``gamma`` and orientation ``beta`` (radians), and convergence ``kappa``
follow `chang_refsdal.geometry` (angles in Einstein-radius units).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cogwheel import utils
from cogwheel.lensing.chang_refsdal import channels, geometry

lal = utils.import_lal()

__all__ = ['LensedWaveformGenerator', 'LensedDecomposition',
           'dimensionless_frequency']

#: Seconds of dimensionless-frequency conversion per solar mass of lens,
#: i.e. ``8 * pi * G * M_sun / c**3``.  ``lal.MTSUN_SI`` is the
#: geometrized solar mass ``G * M_sun / c**3`` in seconds, so multiplying
#: by ``m_lens_msun``, ``(1 + z_lens)`` and ``f_hz`` yields the
#: dimensionless ``w``.  This is the single authoritative conversion
#: constant; see `dimensionless_frequency`.
_EIGHT_PI_MTSUN_S = 8.0 * np.pi * lal.MTSUN_SI

#: Fixed number of topology-stable channels in the Chang--Refsdal
#: partition (see `chang_refsdal.channels`); used only to size the
#: unlensed-limit decomposition, where no partition is evaluated.
_N_CHANNELS = 4


def dimensionless_frequency(f_hz, m_lens_msun: float,
                            z_lens: float) -> np.ndarray:
    """
    Dimensionless lensing frequency ``w = xi * f_hz``, linear in ``f_hz``.

    ``w = 8 * pi * G * M_L * (1 + z_L) * f / c**3`` with
    ``xi = 8 * pi * G * M_L * (1 + z_L) / c**3`` a constant with units of
    seconds.  This is the single place the physical constant enters.

    Parameters
    ----------
    f_hz : float or np.ndarray
        Observed frequency (or frequencies) in Hz.
    m_lens_msun : float
        Redshifted-frame lens mass in solar masses.
    z_lens : float
        Lens redshift (dimensionless).

    Returns
    -------
    np.ndarray
        Dimensionless frequency ``w``, same shape as ``f_hz``.
    """
    xi_s = _EIGHT_PI_MTSUN_S * float(m_lens_msun) * (1.0 + float(z_lens))
    return xi_s * np.asarray(f_hz, dtype=float)


@dataclass(frozen=True)
class LensedDecomposition:
    """
    Per-image / per-mode microlensing decomposition at one lens point.

    The amplification reconstructs as
    ``F(w) = sum_a exp(1j * w * tau_a) * K_a(w)`` (see
    `reconstructed_amplification`); the image-delay phases are carried
    analytically so the likelihood interpolates only the smooth kernels.

    Attributes
    ----------
    w : np.ndarray
        Shape ``(n_f,)`` dimensionless frequency on the waveform grid.
    delays : np.ndarray
        Shape ``(n_channels,)`` channel delays ``tau_a`` (dimensionless
        Fermat delays relative to the minimum image).
    kernels : np.ndarray
        Shape ``(n_f, n_channels)`` channel kernels ``K_a(w)``.  Rows
        where ``w <= 0`` (the unlensed limit) reconstruct ``F = 1``.
    real_mask : np.ndarray
        Shape ``(n_channels,)`` boolean, ``True`` where the channel
        holds a real image rather than a parked virtual label.
    unlensed_hplus_hcross : np.ndarray
        Shape ``(n_m, 2, n_f)`` unlensed per-``|m|`` strain
        ``(hplus, hcross)`` from the wrapped generator.
    m_arr : np.ndarray
        Shape ``(n_m,)`` harmonic ``m`` numbers labeling the first axis
        of ``unlensed_hplus_hcross``.
    """

    w: np.ndarray
    delays: np.ndarray
    kernels: np.ndarray
    real_mask: np.ndarray
    unlensed_hplus_hcross: np.ndarray
    m_arr: np.ndarray

    @property
    def reconstructed_amplification(self) -> np.ndarray:
        """
        Coherent channel sum ``sum_a exp(1j * w * tau_a) * K_a(w)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_f,)`` amplification ``F(w)``, matching
            `LensedWaveformGenerator.amplification` on the same grid.
        """
        phases = np.exp(1j * self.w[:, None] * self.delays[None, :])
        return np.sum(phases * self.kernels, axis=1)


class LensedWaveformGenerator(utils.JSONMixin):
    """
    Apply a Chang--Refsdal amplification to a wrapped `WaveformGenerator`.

    The wrapped generator supplies the unlensed per-mode strain; this
    class multiplies each mode by the common factor ``F(w(f))``, which is
    parity-blind: both positive-parity (``1 - kappa > abs(gamma)``) and
    macro-saddle (``0 < 1 - kappa < abs(gamma)``) hosts are served, the
    saddle wave branch routing through the Schwinger evaluator inside
    `operator.F_op`.  Lens parameters are fixed at construction, so the
    macro-geometry domain refusals (over-critical ``1 - kappa <= 0`` and
    the exact ``det A = 0`` parity boundary ``abs(gamma) == 1 - kappa``)
    raise `geometry.LensDomainError` immediately rather than at first use.

    Parameters
    ----------
    waveform_generator : waveform.WaveformGenerator
        Generator producing the unlensed strain via
        ``get_hplus_hcross(f, waveform_par_dic, by_m=...)``.
    m_lens_msun : float
        Redshifted-frame lens mass in solar masses.
    z_lens : float
        Lens redshift.
    y : Sequence[float]
        Shape ``(2,)`` Chang--Refsdal source position.
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.

    Raises
    ------
    geometry.LensDomainError
        If the macro geometry is over-critical (``1 - kappa <= 0``,
        Type III) or sits exactly on the ``det A = 0`` parity boundary
        (``abs(gamma) == 1 - kappa``); raised here at construction by
        `geometry.macro_matrix` and never swallowed.  Both parity
        interiors return normally.
    ValueError
        If ``y`` is not a two-vector or the wrapped object lacks the
        waveform-generator interface.
    """

    def __init__(self, waveform_generator, m_lens_msun: float,
                 z_lens: float, y: Sequence[float], gamma: float,
                 beta: float = 0.0, kappa: float = 0.0) -> None:
        if not (hasattr(waveform_generator, 'get_hplus_hcross')
                and hasattr(waveform_generator, 'm_arr')):
            raise ValueError(
                'waveform_generator must expose the WaveformGenerator '
                'interface (get_hplus_hcross, m_arr); got '
                f'{type(waveform_generator).__name__}.')
        source = np.asarray(y, dtype=float)
        if source.shape != (2,):
            raise ValueError(
                f'The source position y must be a two-vector, got shape '
                f'{source.shape}.')

        # Macro-geometry domain gate at CONSTRUCTION. Both parities are
        # served (positive parity 1 - kappa > |gamma| and the macro saddle
        # 0 < 1 - kappa < |gamma|); macro_matrix raises geometry.LensDomainError
        # by name only for the over-critical / Type III case (1 - kappa <= 0)
        # and the exact det-A = 0 parity boundary (|gamma| == 1 - kappa).
        geometry.macro_matrix(gamma, beta, kappa)

        self.waveform_generator = waveform_generator
        self.m_lens_msun = float(m_lens_msun)
        self.z_lens = float(z_lens)
        self.y = source
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.kappa = float(kappa)

    @property
    def m_arr(self) -> np.ndarray:
        """Harmonic ``m`` numbers of the wrapped generator."""
        return self.waveform_generator.m_arr

    @property
    def harmonic_modes(self) -> list:
        """``(l, m)`` harmonic modes of the wrapped generator."""
        return self.waveform_generator.harmonic_modes

    def dimensionless_frequency(self, f_hz) -> np.ndarray:
        """
        Dimensionless ``w(f)`` for this lens, linear in ``f_hz``.

        Thin instance-level accessor for the module-level
        `dimensionless_frequency`, bound to this lens's mass and redshift.
        """
        return dimensionless_frequency(f_hz, self.m_lens_msun, self.z_lens)

    def _evaluate_channels(self, f_hz):
        """
        Evaluate the channel partition on the positive-``w`` subgrid.

        Returns ``(w, positive_mask, partition)``.  ``partition`` is
        ``None`` when no frequency maps to positive ``w`` (the fully
        unlensed limit).  Named engine refusals -- `geometry.LensDomainError`
        (Type III / parity boundary), `operator.CancellationError` and
        `SchwingerCertificationError` (uncertifiable / above-ceiling
        contraction) -- propagate unswallowed.
        """
        w = self.dimensionless_frequency(f_hz)
        positive = w > 0.0
        if not positive.any():
            return w, positive, None
        engine = channels.ChangRefsdalChannels(w[positive])
        partition = engine.evaluate(gamma=self.gamma, y=self.y,
                                    beta=self.beta, kappa=self.kappa)
        return w, positive, partition

    def amplification(self, f) -> np.ndarray:
        """
        Amplification factor ``F(w(f))`` on the waveform grid.

        Parameters
        ----------
        f : np.ndarray
            Frequencies in Hz.

        Returns
        -------
        np.ndarray
            Shape ``(n_f,)`` complex ``F(w(f))``.  Entries with
            ``w <= 0`` (e.g. the ``f = 0`` bin) are set to the unlensed
            limit ``F = 1``.
        """
        w, positive, partition = self._evaluate_channels(f)
        factor = np.ones(w.shape, dtype=complex)
        if partition is not None:
            factor[positive] = partition.exact_total
        return factor

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False
                         ) -> np.ndarray:
        """
        Lensed strain ``F(w(f)) * h(f)``, mirroring `WaveformGenerator`.

        The amplification is a common multiplicative factor across modes
        and polarizations, applied on the same frequency array ``f`` used
        for the waveform.

        Parameters
        ----------
        f : np.ndarray
            Frequencies in Hz.
        waveform_par_dic : dict
            Source parameters for the wrapped generator.
        by_m : bool
            If ``True`` return per-``|m|`` modes, shape ``(n_m, 2, n_f)``;
            otherwise the mode- and image-summed total, shape
            ``(2, n_f)``.

        Returns
        -------
        np.ndarray
            The lensed ``(hplus, hcross)`` strain.
        """
        unlensed = self.waveform_generator.get_hplus_hcross(
            f, waveform_par_dic, by_m=by_m)
        return unlensed * self.amplification(f)

    def decompose(self, f, waveform_par_dic) -> LensedDecomposition:
        """
        Per-image ``(tau_a, K_a)`` decomposition plus unlensed strain.

        This is the accessor the microlensed likelihood consumes: it
        keeps the image-delay phases analytic (via ``delays``) and returns
        the smooth kernels alongside the unlensed per-mode strain, rather
        than collapsing to the amplified total.

        Parameters
        ----------
        f : np.ndarray
            Frequencies in Hz.
        waveform_par_dic : dict
            Source parameters for the wrapped generator.

        Returns
        -------
        LensedDecomposition
            The channel delays, kernels, real/virtual mask, and the
            unlensed per-``|m|`` strain on the same grid.
        """
        w, positive, partition = self._evaluate_channels(f)
        unlensed = self.waveform_generator.get_hplus_hcross(
            f, waveform_par_dic, by_m=True)

        if partition is None:
            delays = np.zeros(_N_CHANNELS, dtype=float)
            real_mask = np.zeros(_N_CHANNELS, dtype=bool)
            kernels = np.zeros((w.size, _N_CHANNELS), dtype=complex)
            kernels[:, 0] = 1.0
        else:
            delays = partition.delays
            real_mask = partition.real_mask
            kernels = np.zeros((w.size, delays.size), dtype=complex)
            kernels[positive] = partition.kernels
            # Unlensed bins (w <= 0) reconstruct F = 1 exactly through a
            # single channel: exp(1j*w*tau_0) * exp(-1j*w*tau_0) = 1.
            kernels[~positive, 0] = np.exp(
                -1j * w[~positive] * delays[0])

        return LensedDecomposition(
            w=w, delays=delays, kernels=kernels, real_mask=real_mask,
            unlensed_hplus_hcross=unlensed,
            m_arr=self.waveform_generator.m_arr)
