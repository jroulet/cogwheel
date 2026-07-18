"""
Marginalized-extrinsic likelihood for microlensed gravitational waves.

`LensedMarginalizedExtrinsicLikelihood` marginalizes the extrinsic
parameters (sky location, arrival time, polarization, distance and orbital
phase) of a *microlensed* compact-binary signal semi-analytically via the
coherent score, exactly as `MarginalizedExtrinsicLikelihood` does for the
unlensed case.  Lensing lives entirely in the intrinsic sector: given the
lens parameters, ``h_lensed(f) = F(f) * h(f)`` is just a waveform, so the
coherent-score machinery (`CoherentScoreHM`, sky dictionary, distance
lookup table) is reused UNMODIFIED; only the per-mode matched-filter
timeseries ``dh_mptd`` and the norm ``hh_mppd`` that feed it are rebuilt to
carry the amplification ``F``.

The amplification is supplied by an internal `LensedRelativeBinningLikelihood`
engine composed on the SAME frequency bins as the coherent-score summaries.
The engine returns, per proposal, the analytic per-image delays and the
smooth per-bin amplification kernels; from these the total amplification
``F(f_b) = sum_a K_a(f_b) * exp(2j*pi*dt_a*f_b)`` is reconstructed at the bin
EDGES and folded into the data and norm terms:

* DATA: the lensed template is ``F * h`` at the edges.  The image-delay
  linear phase carried by ``F``, mapped through the summary weights'
  ``exp(2j*pi*f*t)`` time kernel, becomes a pure per-image time shift of the
  matched-filter timeseries -- so summing images before the (linear)
  contraction is identical to summing after, and the shifted image
  timeseries is exact.
* NORM: ``|F|^2`` is a real, mode-independent per-edge scaling of the
  UNLENSED single-bin ``(h|h)`` (the delay phases do not shift time in the
  norm), exact under the bin guard because ``|F|`` is smooth across a bin.

Both are exact only under the lens-aware bin guard
``pi * Delta_f_bin * delta_t_max < bin_delay_tol``, which the internal engine
enforces at construction (`LensedBinningError`).

Distance convention (IMPORTANT, deferred to post-analysis)
----------------------------------------------------------
The coherent score marginalizes and (in postprocessing) draws
``d_luminosity``, but for a lensed signal the quantity that enters the strain
is the APPARENT luminosity distance ``d_app = d_luminosity / sqrt(mu_macro)``,
NOT the physical luminosity distance.  The macro magnification is
``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)`` (F009).  Consequently the
'd_luminosity' column produced by `get_blob` / `postprocess_samples` is the
apparent distance ``d_app``; the physical distance is
``d_L = d_app * sqrt(mu_macro)``.  The column is deliberately NOT relabelled
here (relabelling breaks ``lnl_aux`` bookkeeping and downstream plotting).
Both the ``d_app -> d_L`` conversion and the apparent-vs-physical
distance-prior reweighting are DEFERRED to post-analysis.
"""
from __future__ import annotations

import numpy as np

from cogwheel.likelihood.marginalized_extrinsic import (
    MarginalizedExtrinsicLikelihood)
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood,
    _DEFAULT_BIN_DELAY_TOL,
    _DEFAULT_KERNEL_SUBSAMPLES,
    _LENS_PARAMS,
)

__all__ = ['LensedMarginalizedExtrinsicLikelihood']

_TWO_PI_I = 2j * np.pi


class LensedMarginalizedExtrinsicLikelihood(MarginalizedExtrinsicLikelihood):
    """
    Extrinsic-marginalized likelihood for a microlensed CBC signal.

    Sampled parameters are the intrinsic CBC parameters (including
    ``iota``) plus the seven standard lens parameters
    ``('m_lens_msun', 'z_lens', 'y1', 'y2', 'gamma', 'beta', 'kappa')``;
    distance, orbital phase, sky location, arrival time and polarization are
    marginalized by the coherent score and resampled from the conditional
    posterior in postprocessing.

    Notes
    -----
    Constructor-signature coupling (Simplifier watch flag): the extra
    keyword-only arguments ``delta_t_max``, ``bin_delay_tol`` and
    ``kernel_subsamples`` are forwarded verbatim to the internally
    constructed `LensedRelativeBinningLikelihood` engine.  The engine is
    NOT accepted as a constructor argument -- `JSONMixin` round-trips this
    class from primitive ``__init__`` arguments via ``get_init_dict``, so
    the engine must be rebuilt inside the constructor (in `_set_summary`) on
    ``self.fbin``.  If the engine's constructor grows a parameter that
    should be user-tunable, it must be threaded through here too.
    """

    def __init__(self, event_data, waveform_generator, par_dic_0,
                 delta_t_max, *, fbin=None, pn_phase_tol=None,
                 spline_degree=3, t_range=(-.07, .07), coherent_score=None,
                 dlnl_marginalized_threshold=30.,
                 bin_delay_tol=_DEFAULT_BIN_DELAY_TOL,
                 kernel_subsamples=_DEFAULT_KERNEL_SUBSAMPLES):
        """
        Parameters
        ----------
        event_data : data.EventData

        waveform_generator : waveform.WaveformGenerator
            An ordinary (unlensed) generator; lensing enters through the
            lens parameters in ``par_dic``/``par_dic_0``.

        par_dic_0 : dict
            Parameters of the reference waveform, close to the maximum
            likelihood.  Keys must include ``waveform_generator.params`` AND
            the seven lens parameters ``_LENS_PARAMS`` (the base constructor
            evaluates the lens engine on ``par_dic_0`` for beta-temperature
            optimization).

        delta_t_max : float
            Largest relative image delay [s] the frequency bins must
            resolve; sets the lens-aware bin guard
            ``pi * Delta_f_bin * delta_t_max < bin_delay_tol``.  ``t_range``
            must span the image delays, i.e. cover ``[-delta_t_max,
            delta_t_max]`` around the reference arrival time, so no image's
            matched-filter peak is clipped.

        fbin : 1-d array or None
            Bin edges [Hz].  Alternatively pass ``pn_phase_tol``.

        pn_phase_tol : float or None
            Post-Newtonian phase tolerance [rad] for choosing bins.
            Alternatively pass ``fbin``.

        spline_degree : int
            Degree of the relative-binning ratio spline.

        t_range : 2-tuple of floats
            Bounds of the matched-filtering time range [s], relative to
            ``event_data.tgps + par_dic_0['t_geocenter']``.

        coherent_score : CoherentScoreHM or dict, optional
            Coherent score instance or kwargs to build one (default:
            built with default settings).

        dlnl_marginalized_threshold : float
            Extrinsic marginalization refinement threshold (see base class).

        bin_delay_tol : float
            Tolerance [rad] of the lens-aware bin guard, forwarded to the
            engine.

        kernel_subsamples : int
            Per-bin amplification-kernel sub-samples, forwarded to the
            engine.
        """
        # Fail fast: the reference (and every sampled point) must carry the
        # seven lens parameters, because the base constructor's terminal
        # ``lnlike(par_dic_0)`` evaluates the lens engine on ``par_dic_0``.
        missing = set(_LENS_PARAMS) - set(par_dic_0)
        if missing:
            raise ValueError(
                'par_dic_0 must include the lens parameters '
                f'{sorted(missing)} (needs all of {list(_LENS_PARAMS)}); '
                f'got keys {sorted(par_dic_0)}.')

        # Stored BEFORE super().__init__ so the overridden `_set_summary`
        # (invoked from within the base constructor via the `fbin` setter)
        # can build the internal engine with them.  These names must match
        # the constructor signature for `JSONMixin.get_init_dict`.
        self.delta_t_max = delta_t_max
        self.bin_delay_tol = bin_delay_tol
        self.kernel_subsamples = kernel_subsamples
        self._engine = None  # Built by `_set_summary`.

        super().__init__(
            event_data, waveform_generator, par_dic_0, fbin=fbin,
            pn_phase_tol=pn_phase_tol, spline_degree=spline_degree,
            t_range=t_range, coherent_score=coherent_score,
            dlnl_marginalized_threshold=dlnl_marginalized_threshold)

    @property
    def params(self):
        """
        Sampled parameters: intrinsic CBC (``iota`` kept) + the 7 lens
        parameters.  Distance, phase, sky location, time and polarization
        are marginalized by the coherent score.
        """
        # `_MARGINALIZED_EXTRINSIC` is exactly the set of keys that
        # `MarginalizedExtrinsicLikelihood` drops from a waveform generator;
        # derived from its `params` so the two stay in sync.  Computed from
        # `self.waveform_generator` (available from the base constructor's
        # first step) so it never depends on the not-yet-built engine.
        marginalized_extrinsic = (
            set(self.waveform_generator.params)
            - set(MarginalizedExtrinsicLikelihood.params))
        engine_params = (set(self.waveform_generator.params)
                         | set(_LENS_PARAMS))
        return sorted(engine_params - marginalized_extrinsic)

    def _set_summary(self):
        """
        Build the UNLENSED coherent-score summaries, then the internal
        lensed relative-binning engine on the SAME bins.

        Constructing the engine on ``self.fbin`` re-runs the lens-aware bin
        guard (`LensedRelativeBinningLikelihood._validate_bin_delay_criterion`),
        so a construction on bins too coarse for ``delta_t_max`` raises
        `LensedBinningError` here, at construction time.
        """
        super()._set_summary()
        self._engine = LensedRelativeBinningLikelihood(
            self.event_data, self.waveform_generator, self.par_dic_0,
            self.delta_t_max, fbin=self.fbin,
            spline_degree=self._spline_degree,
            bin_delay_tol=self.bin_delay_tol,
            kernel_subsamples=self.kernel_subsamples)

    def _edge_amplification(self, delays, k0, k1):
        """
        Total lensed amplification ``F`` at the bin EDGES ``self.fbin``.

        The engine returns each image's smooth kernel ``K_a`` as a per-bin
        linear model (center value ``k0``, slope ``k1``) on the ``n_bins``
        bin CENTERS, but the coherent-score summary weights are indexed on
        the ``n_bins + 1`` bin EDGES.  Reconstruct ``K_a`` at the edges from
        that certified linear model -- slope-corrected to each edge and
        averaged at shared interior edges (adjacent-bin estimates agree to
        within the bin guard) -- then sum the images with the analytic
        image-delay phase evaluated EXACTLY at the edges::

            F(f_b) = sum_a K_a(f_b) * exp(2j*pi*dt_a*f_b),

        with ``w_b * tau_a = 2*pi*dt_a*f_b`` the dimensionless-frequency
        delay phase (F001: linear in ``f``).

        Parameters
        ----------
        delays : np.ndarray
            Shape ``(n_channels,)`` relative image delays ``dt_a`` [s].
        k0, k1 : np.ndarray
            Shape ``(n_channels, n_bins)`` per-bin center value and slope
            [1/Hz] of each image kernel ``K_a``.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins + 1,)`` complex total amplification at the edges.
        """
        fbin = self.fbin
        f_center = 0.5 * (fbin[:-1] + fbin[1:])          # (n_bins,)
        left = k0 + k1 * (fbin[:-1] - f_center)          # K_a at bin left edges
        right = k0 + k1 * (fbin[1:] - f_center)          # K_a at bin right edges

        kernel_edges = np.empty((k0.shape[0], fbin.size), dtype=complex)
        kernel_edges[:, 0] = left[:, 0]
        kernel_edges[:, -1] = right[:, -1]
        kernel_edges[:, 1:-1] = 0.5 * (right[:, :-1] + left[:, 1:])

        phase = np.exp(_TWO_PI_I * np.outer(delays, fbin))  # (n_channels, b)
        return np.sum(kernel_edges * phase, axis=0)         # (b,)

    def _get_dh_hh_timeshift(self, par_dic):
        """
        Lensed per-mode matched-filter timeseries and norm for the coherent
        score.

        Mirrors `MarginalizedExtrinsicLikelihood._get_dh_hh_timeshift`, with
        the total amplification ``F`` (evaluated at the bin edges, image
        delays analytic) folded in: ``F * h`` in the data term and ``|F|^2``
        scaling the norm term.  The coherent-score summary weights
        ``_d_h_weights`` / ``_h_h_weights`` (built against the UNLENSED
        fiducial) are reused unchanged.
        """
        # Lens engine, evaluated ONCE.  Candidate-side refusals
        # (`geometry.LensDomainError`, `operator.CancellationError`) and the
        # bin-resolution guard (`LensedBinningError`) propagate UNSWALLOWED
        # to the posterior boundary -- matching `lnlike_bruteforce` and the
        # engine's own hot path.  `delays` is `_image_delays(...)` [s].
        delays, k0, k1, _partition = \
            self._engine._amplification_coefficients(par_dic)
        self._engine._check_candidate_delays(delays)
        amplification = self._edge_amplification(delays, k0, k1)  # (b,) complex

        # Unlensed linear-free bin templates, exactly as the base class.
        h_mpb, timeshift = self._get_linearfree_hplus_hcross_dt(
            dict(par_dic) | self._ref_dic, by_m=True)
        h_mpb = h_mpb.astype(np.complex64)  # mpb

        # DATA: lensed template F * h at the edges.  Summing images before
        # the linear contraction equals summing after; the analytic delay
        # phase in F maps, through the weights' exp(2j*pi*f*t) time kernel,
        # to a pure per-image time shift of the matched-filter timeseries.
        h_lensed = (amplification[np.newaxis, np.newaxis, :]
                    * h_mpb).astype(np.complex64)  # mpb
        dh_mptd = (self._d_h_weights[:, np.newaxis]
                   @ h_lensed.conj()[:, :, np.newaxis, :, np.newaxis])[..., 0]

        # NORM: |F|^2 scales the UNLENSED single-bin norm.  F is a
        # mode-independent scalar, so |F|^2 carries all lensing and the base
        # mode-pair einsum is reused with the unlensed h_mpb; exact under the
        # bin guard because |F| is smooth across a bin.
        norm_weight = (
            self._h_h_weights
            * (np.abs(amplification) ** 2)[np.newaxis, np.newaxis, :])  # mdb
        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        hh_mppd = np.einsum('mdb,mpb,mPb->mpPd',
                            norm_weight,
                            h_mpb[m_inds],
                            h_mpb.conj()[mprime_inds]).astype(np.complex64)

        psd_drift_correction = self.asd_drift.astype(np.float32) ** -2  # d
        dh_mptd *= psd_drift_correction
        hh_mppd *= psd_drift_correction
        return dh_mptd, hh_mppd, timeshift
