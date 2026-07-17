"""
Multi-component relative-binning likelihood for microlensed CBC signals.

WHAT
----
`LensedRelativeBinningLikelihood` evaluates the Gaussian log-likelihood of
a Chang--Refsdal microlensed compact-binary signal with the relative
binning (heterodyning) speed-up generalized to the multi-component lensed
waveform.  The engine (`chang_refsdal.ChangRefsdalChannels`) decomposes
the amplification into four topology-stable channels,

    F(w) = sum_a exp(1j * w * tau_a) * K_a(w),

so the candidate lensed detector strain is the ``a``-sum of time-shifted,
kernel-weighted copies of the unlensed strain,

    h_L(f) = sum_a exp(2*pi*1j*f*dt_a) * K_a(w(f)) * h(f),

with ``dt_a = xi * tau_a / (2*pi)`` a frequency-independent relative image
delay (``w = xi*f`` is linear in frequency; `lensing.waveform`).  See
`.claude/spec/lensing_paper` Eqs. (fiducial-component)--(slow-component-
ratio) for the derivation this module implements.

HOW (the three design decisions, locked)
-----------------------------------------
1. Delay-continuous summaries, never a product of summaries.  The rapid
   image-delay phase ``exp(2*pi*1j*f*dt_a)`` multiplies the rapid fiducial
   waveform *inside* the frequency sum.  We keep it analytic: within each
   coarse bin the delay phase is evaluated exactly at the bin center and
   expanded to first order across the bin.  Concretely the summaries are
   the delay-free frequency moments

       A^(p)_{m,d,b} = 4 df sum_{f in b} d(f) conj(h0_{m,d}(f)) (f-f_b)^p
       B^(p)_{mm',d,b} = 4 df sum_{f in b}
                         h0_{m,d}(f) conj(h0_{m',d}(f)) wht_d(f)^2 (f-f_b)^p

   (data moments ``p <= 2``; norm moments ``p <= 3`` -- the ``(h|h)`` term
   costs one extra moment because two component ratios multiply).  The
   candidate delay enters analytically as ``exp(-2*pi*1j*f_b*dt_a)`` per
   bin plus a linear in-bin correction folded into the higher moments.

2. Sequential mode-then-image contraction, additive ``M^2 + n_img^2``.
   Because the amplification is a common factor across harmonic modes and
   the unlensed ratio is common across images, the contraction factorizes:
   the mode structure is reduced first (``M`` or ``M^2`` mode pairs), then
   the image structure (``n_img`` or ``n_img^2`` channel pairs) is
   contracted against the mode-reduced tensors.  The cost is additive, not
   multiplicative, and there are no FFTs and no Python loops over frequency
   on the hot path.

3. Lens-aware bins with a guard.  Bins must be fine enough that the linear
   in-bin delay expansion holds for the largest candidate relative image
   delay ``delta_t_max``: ``pi * Delta_f_bin * delta_t_max < bin_delay_tol``.
   `LensedBinningError` is raised (a named error, never a bare ``assert``
   that ``-O`` would strip) when the criterion is violated at construction
   or when a candidate presents delays beyond ``delta_t_max``.

The overall/common time shift reuses `BaseLinearFree`'s
``_get_linearfree_hplus_hcross_dt`` idiom (a single time alignment of the
candidate to the fiducial waveform); the resulting shift is applied as a
common delay in the data term and cancels in the norm term.

SCOPE / ERRORS THAT PROPAGATE
-----------------------------
Positive-parity macro images only: a macro-saddle configuration
(``1 - kappa <= abs(gamma)``) makes the engine raise
`geometry.LensDomainError`, propagated unswallowed.  Likewise a
`operator.CancellationError` -- the engine's named refusal when the
wave-branch contraction cannot certify its accuracy -- is *not* caught:
the certified-or-refuse contract of the engine is preserved end to end.

Conventions
-----------
Frequencies in Hz, times in GPS seconds, delays in seconds; lens mass
``m_lens_msun`` in solar masses.  Inner products follow `CBCLikelihood`
(ASD-drift applied at evaluation, not baked into the summaries).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse

from cogwheel import utils
from cogwheel.likelihood.relative_binning import BaseLinearFree
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.waveform import (LensedWaveformGenerator,
                                       dimensionless_frequency)

__all__ = ['LensedRelativeBinningLikelihood', 'LensedBinningError']

#: Highest frequency moment retained for the data term ``(d|h)``.  The
#: candidate component ratio is expanded to first order across a bin and
#: the delay phase to first order, whose product is quadratic.
_DATA_MAX_MOMENT = 2

#: Highest frequency moment retained for the norm term ``(h|h)``.  Two
#: component ratios and one delay phase multiply, giving a cubic in-bin
#: polynomial -- one moment more than the data term.
_NORM_MAX_MOMENT = 3

#: Default tolerance [rad] for the lens-aware bin criterion
#: ``pi * Delta_f_bin * delta_t_max < bin_delay_tol``.
_DEFAULT_BIN_DELAY_TOL = 0.5

#: Default post-Newtonian phase tolerance [rad] for bin selection when
#: neither ``fbin`` nor ``pn_phase_tol`` is given.  Kept small so the
#: per-bin linear component-ratio expansion is accurate.
_DEFAULT_PN_PHASE_TOL = 0.02

#: Default number of frequency sub-samples per coarse bin used to reduce
#: the candidate channel kernels to their per-bin (value, slope)
#: coefficients.  With bounded channel kernels the plain bin-edge secant
#: (the value 2) is accurate for the gated configurations, so 2 is the
#: default.  The earlier large-amplitude kernel "blow-up" was not an
#: edge-secant aliasing failure but the channel-switch neighbourhood bug
#: in the engine (fixed in the Chang--Refsdal channels; see FINDINGS on
#: the switch-neighbourhood fix); once the kernels are O(1) the secant no
#: longer aliases.
#: A larger value fits a least-squares line over interior sub-samples and
#: is retained as a robustness margin against pure geometric phase
#: oscillation near a caustic, not as a correctness requirement.  Must be
#: >= 2; the value 2 reproduces the plain edge secant.
_DEFAULT_KERNEL_SUBSAMPLES = 2

#: Lens parameters expected in ``par_dic`` (in addition to the waveform
#: parameters) to evaluate the amplification decomposition.
_LENS_PARAMS = ('m_lens_msun', 'z_lens', 'y1', 'y2', 'gamma', 'beta',
                'kappa')

_TWO_PI_I = 2j * np.pi


class LensedBinningError(ValueError):
    """
    Relative-binning bins are too coarse for the requested image delays.

    Raised when ``pi * Delta_f_bin * delta_t_max >= bin_delay_tol`` (the
    lens-aware bin criterion is violated), either at construction for the
    declared ``delta_t_max`` or at evaluation for a candidate whose image
    delays exceed ``delta_t_max``.
    """


def _edge_linear_coefficients(values_at_edges, fbin):
    """
    Per-bin linear (value-at-center, slope) coefficients from edge values.

    A smooth function sampled at the bin edges ``fbin`` is approximated
    within bin ``b`` by ``c0_b + c1_b * (f - f_b)`` with ``f_b`` the bin
    center.  ``c0_b`` is the midpoint value of the linear interpolant and
    ``c1_b`` its slope.

    Parameters
    ----------
    values_at_edges : np.ndarray
        Shape ``(..., n_edges)``; values at ``fbin`` (last axis).
    fbin : np.ndarray
        Shape ``(n_edges,)`` bin-edge frequencies [Hz], increasing.

    Returns
    -------
    c0, c1 : np.ndarray
        Shape ``(..., n_bins)`` center values and slopes [x], [x/Hz].
    """
    lower = values_at_edges[..., :-1]
    upper = values_at_edges[..., 1:]
    widths = np.diff(fbin)
    c0 = 0.5 * (lower + upper)
    c1 = (upper - lower) / widths
    return c0, c1


def _data_term(a_moments, rho0, rho1, kbar0, kbar1, tau, f_center):
    """
    Contract the data term ``(d|h_L)`` per detector, mode-then-image.

    Implements the ``p <= 2`` moment expansion of
    ``sum_{a,m,b} A^(p)_{m,d,b} * [conj(r_m K_a) exp(-2*pi*1j*f*tau_a)]^(p)``
    (paper Eq. data-term-rb generalized to harmonic modes and split into a
    mode reduction followed by an image reduction).

    Parameters
    ----------
    a_moments : sequence of np.ndarray
        ``[A^(0), A^(1), A^(2)]``, each shape ``(n_m, n_det, n_bins)``.
    rho0, rho1 : np.ndarray
        Shape ``(n_m, n_det, n_bins)`` center value and slope of the
        conjugated unlensed component ratio ``conj(r_m)``.
    kbar0, kbar1 : np.ndarray
        Shape ``(n_img, n_bins)`` center value and slope of the
        conjugated candidate kernel ``conj(K_a)``.
    tau : np.ndarray
        Shape ``(n_img,)`` data-frame relative image delays [s]
        (image delay minus the common linear-free time shift).
    f_center : np.ndarray
        Shape ``(n_bins,)`` bin-center frequencies [Hz].

    Returns
    -------
    np.ndarray
        Shape ``(n_det,)`` complex ``(d|h_L)`` per detector (ASD-drift
        not yet applied).
    """
    a0, a1, a2 = a_moments

    # Mode reduction: G^(p,q)_{d,b} = sum_m A^(p)_{m,d,b} rho_q_{m,d,b}
    g00 = np.einsum('mdb,mdb->db', a0, rho0)
    g10 = np.einsum('mdb,mdb->db', a1, rho0)
    g11 = np.einsum('mdb,mdb->db', a1, rho1)
    g20 = np.einsum('mdb,mdb->db', a2, rho0)
    g21 = np.einsum('mdb,mdb->db', a2, rho1)

    tau_a = tau[:, np.newaxis, np.newaxis]  # (n_img, 1, 1)
    # In-bin polynomial of conj(K_a) exp(-2 pi i (f-f_b) tau_a) times the
    # mode-reduced tensors, grouped by the kernel coefficients kbar0/kbar1.
    coeff_k0 = (g00 + g11)[np.newaxis] - _TWO_PI_I * tau_a * (g10 + g21
                                                             )[np.newaxis]
    coeff_k1 = (g10 + g21)[np.newaxis] - _TWO_PI_I * tau_a * g20[np.newaxis]

    phase = np.exp(-_TWO_PI_I * f_center[np.newaxis, :] * tau[:, np.newaxis])
    term = phase[:, np.newaxis, :] * (
        coeff_k0 * kbar0[:, np.newaxis, :]
        + coeff_k1 * kbar1[:, np.newaxis, :])
    return term.sum(axis=(0, 2))


def _norm_term(b_moments, r0, r1, rho0, rho1, k0, k1, kbar0, kbar1,
               delays, f_center):
    """
    Contract the norm term ``(h_L|h_L)`` per detector, mode-then-image.

    Implements the ``p <= 3`` moment expansion of the double component
    sum (paper Eq. norm-term-rb generalized to harmonic modes), factored
    into a mode-pair reduction ``N^(p,q)`` followed by an image-pair
    reduction.  All ``M^2`` ordered mode pairs and ``n_img^2`` ordered
    image pairs are summed explicitly (the result is real).

    Parameters
    ----------
    b_moments : sequence of np.ndarray
        ``[B^(0..3)]``, each shape ``(n_m, n_m, n_det, n_bins)`` over
        ordered mode pairs ``(m, m')``.
    r0, r1, rho0, rho1 : np.ndarray
        Shape ``(n_m, n_det, n_bins)`` center/slope of the component ratio
        ``r_m`` and its conjugate ``rho = conj(r_m)``.
    k0, k1, kbar0, kbar1 : np.ndarray
        Shape ``(n_img, n_bins)`` center/slope of the candidate kernel
        ``K_a`` and its conjugate ``kbar = conj(K_a)``.
    delays : np.ndarray
        Shape ``(n_img,)`` relative image delays [s] (the common time
        shift cancels in the norm term).
    f_center : np.ndarray
        Shape ``(n_bins,)`` bin-center frequencies [Hz].

    Returns
    -------
    np.ndarray
        Shape ``(n_det,)`` real ``(h_L|h_L)`` per detector (ASD-drift not
        yet applied).
    """
    b0, b1, b2, b3 = b_moments

    def reduce_pairs(bp, x_m, y_mprime):
        """Contract ``sum_{m,m'} B^(p) x_m y_m'`` -> ``(n_det, n_bins)``."""
        return np.einsum('mMdb,mdb,Mdb->db', bp, x_m, y_mprime)

    # Mode reduction: N^(p,q) with q the order of the mode-pair ratio
    # mu_q, q in {0,1,2}: mu0 = r0 rho0', mu1 = r1 rho0' + r0 rho1',
    # mu2 = r1 rho1'.
    n00 = reduce_pairs(b0, r0, rho0)
    n11 = reduce_pairs(b1, r1, rho0) + reduce_pairs(b1, r0, rho1)
    n22 = reduce_pairs(b2, r1, rho1)
    n10 = reduce_pairs(b1, r0, rho0)
    n21 = reduce_pairs(b2, r1, rho0) + reduce_pairs(b2, r0, rho1)
    n32 = reduce_pairs(b3, r1, rho1)
    n20 = reduce_pairs(b2, r0, rho0)
    n31 = reduce_pairs(b3, r1, rho0) + reduce_pairs(b3, r0, rho1)
    n30 = reduce_pairs(b3, r0, rho0)

    # Collect by image-side in-bin order s (coefficient of nu_s):
    m0 = n00 + n11 + n22  # (n_det, n_bins)
    m1 = n10 + n21 + n32
    m2 = n20 + n31
    m3 = n30

    # Image reduction over ordered channel pairs (a, c).
    kpair0 = k0[:, np.newaxis, :] * kbar0[np.newaxis, :, :]
    kpair1 = (k1[:, np.newaxis, :] * kbar0[np.newaxis, :, :]
              + k0[:, np.newaxis, :] * kbar1[np.newaxis, :, :])
    kpair2 = k1[:, np.newaxis, :] * kbar1[np.newaxis, :, :]

    delta = delays[:, np.newaxis] - delays[np.newaxis, :]  # (n_img, n_img)
    shift = _TWO_PI_I * delta[:, :, np.newaxis]            # (n_img, n_img, 1)
    nu0 = kpair0
    nu1 = kpair1 + shift * kpair0
    nu2 = kpair2 + shift * kpair1
    nu3 = shift * kpair2

    phase = np.exp(_TWO_PI_I * delta[:, :, np.newaxis]
                   * f_center[np.newaxis, np.newaxis, :])

    # (n_img, n_img, n_det, n_bins): mode-reduced tensors times image nu_s.
    contrib = phase[:, :, np.newaxis, :] * (
        m0[np.newaxis, np.newaxis] * nu0[:, :, np.newaxis, :]
        + m1[np.newaxis, np.newaxis] * nu1[:, :, np.newaxis, :]
        + m2[np.newaxis, np.newaxis] * nu2[:, :, np.newaxis, :]
        + m3[np.newaxis, np.newaxis] * nu3[:, :, np.newaxis, :])
    return contrib.sum(axis=(0, 1, 3)).real


class LensedRelativeBinningLikelihood(BaseLinearFree):
    """
    Fast relative-binning likelihood for Chang--Refsdal microlensed CBCs.

    Subclasses `BaseLinearFree` to reuse the frequency-bin machinery, the
    ASD-drift correction, and the linear-free common-time-shift idiom.  The
    reference waveform ``par_dic_0`` is *unlensed*; the lensing enters only
    at evaluation through the per-candidate channel decomposition, whose
    smooth kernels are interpolated and whose image-delay phases are kept
    analytic (see the module docstring).

    Parameters
    ----------
    event_data : data.EventData
        Conditioned strain and whitening filter.
    waveform_generator : waveform.WaveformGenerator or \
            lensing.waveform.LensedWaveformGenerator
        Unlensed waveform generator (a `LensedWaveformGenerator` is
        accepted and unwrapped to its embedded unlensed generator; the
        lens is supplied per candidate through ``par_dic``).
    par_dic_0 : dict
        Reference (unlensed) waveform parameters, close to the maximum
        likelihood; keys per ``waveform_generator.params``.
    delta_t_max : float
        Largest relative image delay [s] the bins must support.  Sets the
        lens-aware bin criterion and is checked against each candidate.
    fbin : array_like or None
        Bin-edge frequencies [Hz].  Pass this or ``pn_phase_tol``.
    pn_phase_tol : float or None
        Post-Newtonian phase tolerance [rad] for automatic bin selection.
        Defaults to a small value if neither ``fbin`` nor ``pn_phase_tol``
        is given.
    spline_degree : int
        Spline degree used by the base class for the linear-free fit
        weights (the lensed summaries use per-bin moments, not splines).
    bin_delay_tol : float
        Tolerance [rad] in the lens-aware bin criterion.
    kernel_subsamples : int
        Number of frequency sub-samples per coarse bin used to reduce the
        candidate channel kernels to their per-bin (value, slope)
        coefficients by least squares.  Must be ``>= 2``; the default 2
        is the plain bin-edge secant, which is accurate once the channel
        kernels are bounded.  Larger values fit a line over interior
        sub-samples and are a robustness margin against pure geometric
        phase oscillation near a caustic (at the cost of more
        amplification evaluations), not a correctness requirement.  The
        reference summaries and the mode-then-image contraction are
        unaffected.

    Raises
    ------
    LensedBinningError
        If ``pi * max(Delta_f_bin) * delta_t_max >= bin_delay_tol``.
    ValueError
        If ``kernel_subsamples < 2``.
    """

    def __init__(self, event_data, waveform_generator, par_dic_0,
                 delta_t_max, *, fbin=None, pn_phase_tol=None,
                 spline_degree=3, bin_delay_tol=_DEFAULT_BIN_DELAY_TOL,
                 kernel_subsamples=_DEFAULT_KERNEL_SUBSAMPLES):
        if isinstance(waveform_generator, LensedWaveformGenerator):
            base_generator = waveform_generator.waveform_generator
        else:
            base_generator = waveform_generator

        if delta_t_max <= 0:
            raise ValueError('`delta_t_max` must be positive.')
        if kernel_subsamples < 2:
            raise ValueError(
                '`kernel_subsamples` must be >= 2 (2 reproduces the plain '
                f'bin-edge secant); got {kernel_subsamples}.')

        self.delta_t_max = float(delta_t_max)
        self.bin_delay_tol = float(bin_delay_tol)
        self.kernel_subsamples = int(kernel_subsamples)

        # Populated by ``_set_summary`` (triggered by the ``fbin`` setter
        # inside ``super().__init__``).
        self._moment_ops = None
        self._f_center = None
        self.n_bins = None
        self._h0_edges = None
        self._a_moments = None
        self._b_moments = None
        # Per-bin kernel sub-sample grid and least-squares reduction
        # weights (populated by ``_build_kernel_subsampling``).
        self._kernel_dense_f = None
        self._kernel_fit_value = None
        self._kernel_fit_slope = None

        if fbin is None and pn_phase_tol is None:
            pn_phase_tol = _DEFAULT_PN_PHASE_TOL

        super().__init__(event_data, base_generator, par_dic_0,
                         fbin=fbin, pn_phase_tol=pn_phase_tol,
                         spline_degree=spline_degree)

    # -- Parameters ------------------------------------------------------

    @property
    def params(self):
        """Sorted waveform parameters plus the lens parameters."""
        return sorted(set(self.waveform_generator.params) | set(_LENS_PARAMS))

    # -- Summary setup (FFTs / heavy precompute allowed here) ------------

    def _set_summary(self):
        """
        Compute the delay-free frequency-moment summaries.

        Builds ``A^(p)`` (data) and ``B^(p)`` (norm) for the *unlensed*
        fiducial waveform at all detectors, plus the fiducial strain at
        bin edges used to form candidate component ratios.  ASD-drift and
        the linear-free fit weights come from the base class.  Also
        validates the lens-aware bin criterion.
        """
        super()._set_summary()  # asd_drift + linear-free helper weights
        self._build_moment_operators()
        self._build_kernel_subsampling()
        self._validate_bin_delay_criterion()

        # Fiducial unlensed detector strain per |m|, at full resolution and
        # at bin edges.  Use all spin components regardless of
        # ``disable_precession`` so the reference is stable.
        with utils.temporarily_change_attributes(self.waveform_generator,
                                                 disable_precession=False):
            h0_f = self._get_h_f(self.par_dic_0, by_m=True)
            h0_edges = self.waveform_generator.get_strain_at_detectors(
                self.fbin, self.par_dic_0, by_m=True)

        # Undo the big coarse-time shift so the reference is smooth and
        # nonzero at high frequency, stall the ringdown, then reapply it
        # (mirrors ``RelativeBinningLikelihood._set_summary``).
        shift_f = np.exp(-2j * np.pi * self.event_data.frequencies
                         * self.event_data.tcoarse)
        shift_edges = np.exp(-2j * np.pi * self.fbin
                             * self.event_data.tcoarse)
        h0_f = h0_f * shift_f.conj()
        h0_edges = h0_edges * shift_edges.conj()
        self._stall_ringdown(h0_f, h0_edges)
        h0_f = h0_f * shift_f
        h0_edges = h0_edges * shift_edges
        self._h0_edges = h0_edges

        # Data moments A^(p)_{m,d,b} = 4 df sum_{f in b}
        #   blued(f) conj(h0_{m,d}(f)) (f - f_b)^p.
        integrand_dh = self.event_data.blued_strain[np.newaxis] * h0_f.conj()
        self._a_moments = self._bin_moments(integrand_dh, _DATA_MAX_MOMENT)

        # Norm moments B^(p)_{mm',d,b} over all ordered mode pairs.
        wht2 = self.event_data.wht_filter ** 2
        integrand_hh = (h0_f[:, np.newaxis] * h0_f[np.newaxis].conj() * wht2)
        self._b_moments = self._bin_moments(integrand_hh, _NORM_MAX_MOMENT)

    def _build_moment_operators(self):
        """
        Build sparse ``(n_bins, n_rfft)`` frequency-moment operators.

        ``moment_ops[p] @ g`` yields ``4 df sum_{f in b} g(f) (f-f_b)^p``
        with hard bin membership ``f in [fbin[b], fbin[b+1])`` and ``f_b``
        the bin center.
        """
        frequencies = self.event_data.frequencies
        fbin = self.fbin
        n_bins = len(fbin) - 1
        f_center = 0.5 * (fbin[:-1] + fbin[1:])

        in_band = (frequencies >= fbin[0]) & (frequencies <= fbin[-1])
        cols = np.nonzero(in_band)[0]
        bin_of = np.clip(
            np.searchsorted(fbin, frequencies[cols], side='right') - 1,
            0, n_bins - 1)
        offsets = frequencies[cols] - f_center[bin_of]
        prefactor = 4.0 * self.event_data.df

        self._moment_ops = []
        for power in range(_NORM_MAX_MOMENT + 1):
            data = prefactor * offsets ** power
            self._moment_ops.append(scipy.sparse.csr_matrix(
                (data, (bin_of, cols)), shape=(n_bins, len(frequencies))))
        self._f_center = f_center
        self.n_bins = n_bins

    def _build_kernel_subsampling(self):
        """
        Precompute the per-bin sub-sample grid and least-squares weights.

        The candidate channel kernels ``K_a(w)`` are reduced to their
        per-bin ``(value, slope)`` coefficients by fitting a line over
        ``kernel_subsamples`` frequencies inside each bin.  With the
        default ``kernel_subsamples == 2`` this is the plain bin-edge
        secant; larger values add interior sub-samples.  Once the channel
        kernels are bounded (the near-caustic ``K_a`` "blow-up" was the
        engine's channel-switch neighbourhood bug, since fixed in the
        Chang--Refsdal channels, not an edge-secant aliasing failure) the
        secant is accurate for the gated configurations, so the extra
        interior sub-samples are a robustness margin against pure
        geometric phase oscillation near a caustic, not a correctness
        requirement.

        Only the candidate kernel coefficients are affected: the
        reference frequency-moment summaries and the mode-then-image
        contraction are unchanged, so the ``F -> 1`` normalization and
        the additive ``M^2 + n_img^2`` cost are preserved.

        The sub-sample offsets are symmetric about each bin center, so
        ``sum_j offset_j = 0`` and the fit decouples into a value weight
        ``1 / n_sub`` (the mean) and a slope weight
        ``offset_j / sum_k offset_k**2``.  Both are candidate independent
        and depend only on ``fbin``.
        """
        n_sub = self.kernel_subsamples
        lower = self.fbin[:-1][:, np.newaxis]      # (n_bins, 1)
        widths = np.diff(self.fbin)[:, np.newaxis]  # (n_bins, 1)
        # Midpoints of ``n_sub`` equal sub-intervals: strictly interior,
        # strictly increasing across bins, symmetric about the center.
        fractions = (np.arange(n_sub) + 0.5) / n_sub
        dense_f = lower + widths * fractions[np.newaxis, :]  # (n_bins, n_sub)
        offsets = dense_f - self._f_center[:, np.newaxis]    # (n_bins, n_sub)

        self._kernel_dense_f = dense_f.reshape(-1)
        self._kernel_fit_value = np.full((self.n_bins, n_sub), 1.0 / n_sub)
        self._kernel_fit_slope = (
            offsets / np.sum(offsets ** 2, axis=1, keepdims=True))

    def _bin_moments(self, integrand, max_moment):
        """
        Project an integrand onto the frequency-moment summaries.

        Parameters
        ----------
        integrand : np.ndarray
            Shape ``(..., n_rfft)`` complex; oscillatory part of the
            integral (last axis is the FFT grid).
        max_moment : int
            Highest moment order to compute.

        Returns
        -------
        list of np.ndarray
            ``[moment_0, ..., moment_max]``, each shape ``(..., n_bins)``.
        """
        *pre_shape, n_rfft = integrand.shape
        flat = integrand.reshape(-1, n_rfft)
        moments = []
        for power in range(max_moment + 1):
            projected = (self._moment_ops[power] @ flat.T).T
            moments.append(np.asarray(projected).reshape(*pre_shape,
                                                         self.n_bins))
        return moments

    def _validate_bin_delay_criterion(self):
        """
        Raise `LensedBinningError` if the bins cannot resolve the delays.

        The linear in-bin expansion of the image-delay phase requires
        ``pi * Delta_f_bin * delta_t_max < bin_delay_tol`` for the widest
        bin.
        """
        widest_bin = float(np.max(np.diff(self.fbin)))
        criterion = np.pi * widest_bin * self.delta_t_max
        if criterion >= self.bin_delay_tol:
            raise LensedBinningError(
                'Relative-binning bins are too coarse for the requested '
                f'delta_t_max={self.delta_t_max:.6g} s: '
                f'pi*Delta_f_bin*delta_t_max = {criterion:.4g} '
                f'>= bin_delay_tol = {self.bin_delay_tol:.4g}. '
                'Use finer bins (smaller pn_phase_tol) or a smaller '
                'delta_t_max.')

    # -- Amplification decomposition (coarse-node engine call) -----------

    def _lens_params(self, par_dic):
        """Return the lens sub-dictionary, validating required keys."""
        missing = [key for key in _LENS_PARAMS if key not in par_dic]
        if missing:
            raise KeyError(
                f'`par_dic` is missing lens parameters {missing}; expected '
                f'{list(_LENS_PARAMS)}.')
        return {key: par_dic[key] for key in _LENS_PARAMS}

    def _amplification_coefficients(self, par_dic):
        """
        Densely-sampled candidate kernels reduced to per-bin coefficients.

        Evaluates the Chang--Refsdal channel decomposition on the per-bin
        sub-sample grid (``kernel_subsamples`` frequencies per bin) and
        reduces each channel kernel to its best-fit per-bin center value
        and slope by least squares, together with the frequency-
        independent relative image delays.  With the default
        ``kernel_subsamples == 2`` this is the plain bin-edge secant,
        which is accurate now that the channel kernels are bounded;
        interior sub-samples are a robustness margin against pure
        geometric phase oscillation near a caustic, not a correctness
        requirement.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        delays : np.ndarray
            Shape ``(n_channels,)`` relative image delays [s].
        k0, k1 : np.ndarray
            Shape ``(n_channels, n_bins)`` per-bin center value and slope
            [1/Hz] of the candidate kernel ``K_a``.
        partition : ChangRefsdalPartition
            The full engine output at the sub-sample grid.

        Notes
        -----
        `geometry.LensDomainError` (macro-saddle) and
        `operator.CancellationError` (uncertifiable contraction) raised by
        the engine propagate unswallowed, exactly as in
        ``lnlike_bruteforce``, so the two paths refuse symmetrically.
        """
        lens = self._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            self._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        if not np.all(dense_w > 0):
            raise LensedBinningError(
                'All kernel sub-sample frequencies must map to positive '
                'dimensionless frequency w = xi*f; got a non-positive value.')

        engine = ChangRefsdalChannels(dense_w)
        partition = engine.evaluate(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])

        xi = float(dimensionless_frequency(
            1.0, lens['m_lens_msun'], lens['z_lens']))
        delays = xi * partition.delays / (2.0 * np.pi)

        n_channels = partition.kernels.shape[1]
        kernels = partition.kernels.reshape(
            self.n_bins, self.kernel_subsamples, n_channels)
        # Least-squares line per bin: value (mean) and slope, mapped to
        # ``(n_channels, n_bins)`` as the contraction expects.
        k0 = np.einsum('bj,bja->ab', self._kernel_fit_value, kernels)
        k1 = np.einsum('bj,bja->ab', self._kernel_fit_slope, kernels)
        return delays, k0, k1, partition

    def _check_candidate_delays(self, delays):
        """Raise if a candidate's relative delays exceed ``delta_t_max``."""
        largest = float(np.max(np.abs(delays)))
        if largest > self.delta_t_max:
            raise LensedBinningError(
                f'Candidate relative image delay {largest:.6g} s exceeds the '
                f'certified delta_t_max = {self.delta_t_max:.6g} s; the bins '
                'do not resolve it. Rebuild with a larger delta_t_max and '
                'finer bins.')

    # -- Hot path (no FFTs, vectorized contractions) ---------------------

    def _candidate_bin_ratios(self, par_dic):
        """
        Per-bin linear coefficients of the unlensed component ratio.

        Returns ``(r0, r1, dt_linearfree)`` where ``r_m = r0 + r1*(f-f_b)``
        is the candidate/fiducial detector-strain ratio in the linear-free
        (time-aligned) frame, and ``dt_linearfree`` is the common time
        shift back to the approximant convention.
        """
        h_edges = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)
        # Pass a copy: get_hplus_hcross may update ZERO_INPLANE_SPINS in
        # place when precession is disabled.
        _, dt_linearfree = self._get_linearfree_hplus_hcross_dt(
            dict(par_dic), by_m=True)

        # Align the raw candidate strain to the fiducial to keep the ratio
        # smooth across bins; the shift is undone as a common delay later.
        align = np.exp(2j * np.pi * self.fbin * dt_linearfree)
        ratio_edges = h_edges * align / self._h0_edges
        r0, r1 = _edge_linear_coefficients(ratio_edges, self.fbin)
        return r0, r1, dt_linearfree

    def _get_dh_hh_no_asd_drift(self, par_dic):
        """
        Return ``(d|h_L)`` (complex) and ``(h_L|h_L)`` (real) per detector.

        No ASD-drift correction is applied (it multiplies both at the
        evaluation stage, as in `CBCLikelihood`).
        """
        r0, r1, dt_linearfree = self._candidate_bin_ratios(par_dic)
        rho0, rho1 = r0.conj(), r1.conj()

        # Reduce the candidate kernels inside each bin to (value, slope).
        # At the default kernel_subsamples == 2 this is the plain bin-edge
        # secant, which is accurate now that the channel kernels are
        # bounded; interior sub-samples are only a robustness margin.
        delays, k0, k1, _ = self._amplification_coefficients(par_dic)
        self._check_candidate_delays(delays)
        kbar0, kbar1 = k0.conj(), k1.conj()

        # Data frame carries the common linear-free shift; the norm term
        # is invariant to it (it cancels in delta = dt_a - dt_c).
        tau = delays - dt_linearfree

        d_h = _data_term(self._a_moments, rho0, rho1, kbar0, kbar1, tau,
                         self._f_center)
        h_h = _norm_term(self._b_moments, r0, r1, rho0, rho1, k0, k1,
                         kbar0, kbar1, delays, self._f_center)
        return d_h, h_h

    def lnlike_and_metadata(self, par_dic):
        """
        Return ``(lnl, metadata)`` using lensed relative binning.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        lnl : float
            Log likelihood ``sum_d [(d|h)_d.real - (h|h)_d/2] / drift_d^2``.
        metadata : dict
            ``{'lnl': lnl}`` for postprocessing.
        """
        d_h, h_h = self._get_dh_hh_no_asd_drift(par_dic)
        lnl = float((d_h.real - h_h / 2) @ self.asd_drift ** -2)
        return lnl, {'lnl': lnl}

    def get_blob(self, metadata):
        """Return the metadata dict as posterior-sample columns."""
        return metadata

    # -- Brute-force reference (no relative binning) ---------------------

    def lnlike_bruteforce(self, par_dic):
        """
        Exact log likelihood on the full FFT grid, no relative binning.

        Builds the lensed strain ``F(w(f)) * h(f)`` through a
        `LensedWaveformGenerator` (consuming the exact amplification
        ``exact_total``) and evaluates the inner products directly.  This
        is the same-generator reference the relative-binning ``lnlike``
        must reproduce; a macro-saddle `geometry.LensDomainError` or an
        `operator.CancellationError` propagates unswallowed.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        float
            Log likelihood.
        """
        lens = self._lens_params(par_dic)
        lensed_generator = LensedWaveformGenerator(
            self.waveform_generator,
            m_lens_msun=lens['m_lens_msun'], z_lens=lens['z_lens'],
            y=(lens['y1'], lens['y2']), gamma=lens['gamma'],
            beta=lens['beta'], kappa=lens['kappa'])

        event_data = self.event_data
        fslice = event_data.fslice
        frequencies = event_data.frequencies

        strain = self.waveform_generator.get_strain_at_detectors(
            frequencies[fslice], par_dic, by_m=False)  # (n_det, n_band)
        amplification = lensed_generator.amplification(frequencies[fslice])

        lensed_strain = np.zeros(
            (len(event_data.detector_names), len(frequencies)),
            dtype=np.complex128)
        lensed_strain[:, fslice] = amplification * strain

        d_h = self._compute_d_h(lensed_strain)
        h_h = self._compute_h_h(lensed_strain)
        return float(np.sum(d_h.real) - np.sum(h_h) / 2)
