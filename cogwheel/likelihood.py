"""Compute likelihood of GW events."""

import inspect
import itertools
from functools import wraps
import numpy as np
from scipy import special, stats
from scipy.optimize import differential_evolution, minimize_scalar
import scipy.interpolate
import matplotlib.pyplot as plt

from . import gw_utils
from . import utils
from . gw_prior import UniformTimePrior
from . skyloc_angles import SkyLocAngles
from . waveform import out_of_bounds, APPROXIMANTS


class LikelihoodError(Exception):
    """Base class for all Exceptions in this module."""


def hole_edges(mask):
    """
    Return nholes x 2 array with edges of holes (holes extend from
    [left_edge : right_edge]).
    Appends ones at left and right to catch end holes if present.
    """
    edges = np.diff(np.r_[1, mask, 1])
    left_edges = np.where(edges == -1)[0].astype(np.uint32)
    right_edges = np.where(edges == 1)[0].astype(np.uint32)
    return np.c_[left_edges, right_edges]


def std_from_median(arr):
    """
    Estimator of the standard deviation of an array based on
    the median absolute deviation for robustness to outliers.
    """
    mad = np.median(np.abs(arr - np.median(arr)))
    return mad / (np.sqrt(2) * special.erfinv(.5))


def safe_std(arr, max_contiguous_low=100, expected_high=1.,
             reject_nearby=10):
    """
    Compute the standard deviation of a real array rejecting outliers.
    Outliers may be:
      * Values too high to be likely to come from white Gaussian noise.
      * A contiguous array of values too low to come from white Gaussian
        noise (likely from a hole).
    Once outliers are identified, an extra amount of nearby samples
    is rejected for safety.
    max_contiguous_low: How many contiguous samples below 1 sigma to
                        allow.
    expected_high: Number of times we expect to trigger in white
                   Gaussian noise (used to set the clipping threshold).
    reject_nearby: By how many samples to expand holes for safety.
    """
    good = np.ones(len(arr), dtype=bool)

    # Reject long stretches of low values
    above_one_sigma = np.abs(arr) > 1
    low_edges = hole_edges(above_one_sigma)
    too_long = np.diff(low_edges)[:, 0] > max_contiguous_low
    for left, right in low_edges[too_long]:
        good[left : right] = False

    # Reject high values
    std_est = std_from_median(arr[good])
    thresh = std_est * stats.chi.isf(expected_high / np.count_nonzero(good), 1)
    good[np.abs(arr) > thresh] = False

    # Reject extra nearby samples
    bad_edges = hole_edges(good)
    for left, right in bad_edges:
        good[max(0, left-reject_nearby) : right+reject_nearby] = False

    return np.std(arr[good])


def _check_bounds(lnlike_func):
    """
    Decorator that adds parameter bound checks to a lnlike function.
    `lnlike_func` needs to accept a parameter called `par_dic`.
    Check parameter ranges, return -inf if they are out of bounds.
    """
    parameters = list(inspect.signature(lnlike_func).parameters)
    par_dic_place = parameters.index('par_dic')

    @wraps(lnlike_func)
    def new_lnlike_func(*args, **kwargs):
        try:
            par_dic = args[par_dic_place]
        except IndexError:
            par_dic = kwargs['par_dic']

        if out_of_bounds(par_dic):
            return -np.inf

        return lnlike_func(*args, **kwargs)

    return new_lnlike_func


class CBCLikelihood(utils.JSONMixin):
    """
    Class that accesses the event data and waveform generator; provides
    methods for computing likelihood as a function of compact binary
    coalescence parameters without using relative binning, and also the
    asd-drift correction.
    Subclassed by `ReferenceWaveformFinder` and
    `RelativeBinningLikelihood`.
    """
    def __init__(self, event_data, waveform_generator):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`.
        waveform_generator: Instance of `waveform.WaveformGenerator`.
        """
        # Check consistency between `event_data` and `waveform_generator`:
        for attr in ['detector_names', 'tgps', 'tcoarse']:
            if getattr(event_data, attr) != getattr(waveform_generator, attr):
                raise ValueError('`event_data` and `waveform_generator` have '
                                 f'inconsistent values of {attr}.')

        self.event_data = event_data
        self.waveform_generator = waveform_generator

        self.asd_drift = None

    @property
    def asd_drift(self):
        """
        Array of len ndetectors with ASD drift-correction.
        Values > 1 mean local noise variance is higher than average.
        """
        return self._asd_drift

    @asd_drift.setter
    def asd_drift(self, value):
        """Ensure asd_drift is a numpy array of the correct length."""
        if value is None:
            value = np.ones(len(self.event_data.detector_names))
        elif len(value) != len(self.event_data.detector_names):
            raise ValueError('ASD-drift must match number of detectors.')

        self._asd_drift = np.asarray(value, dtype=np.float_)

    def compute_asd_drift(self, par_dic, tol=.02, **kwargs):
        """
        Estimate local standard deviation of the matched-filter output
        at the time of the event for each detector.

        Note: all harmonic_modes of the approximant are used even if
        `waveform_generator.harmonic_modes` is set differently. This is
        so that `asd_drift` does not change and one can make apples-to-
        apples comparisons of the likelihood toggling harmonic modes on
        and off.

        Parameters
        ----------
        par_dic: dictionary of waveform parameters.
        tol: stochastic error tolerance in the measurement, used to
             decide the number of samples.
        kwargs: passed to `safe_std`, keys include:
                `max_contiguous_low`, `expected_high`, `reject_nearby`.
        """
        # Use all available modes to get a waveform, then reset
        harmonic_modes = self.waveform_generator.harmonic_modes
        self.waveform_generator.harmonic_modes = None
        normalized_h_f = self._get_h_f(par_dic, normalize=True)
        self.waveform_generator.harmonic_modes = harmonic_modes

        z_cos, z_sin = self._matched_filter_timeseries(normalized_h_f)
        whitened_h_f = (np.sqrt(2 * self.event_data.nfft * self.event_data.df)
                        * self.event_data.wht_filter * normalized_h_f)
        # Undo previous asd_drift so nsamples is independent of it
        nsamples = np.ceil(4 * self.asd_drift**-4
                           * np.sum(np.abs(whitened_h_f)**4, axis=-1)
                           / (tol**2 * self.event_data.nfft)).astype(int)

        nsamples = np.minimum(nsamples, z_cos.shape[1])

        asd_drift = self.asd_drift.copy()
        for i_det, n_s in enumerate(nsamples):
            places = (i_det, np.arange(-n_s//2, n_s//2))
            asd_drift[i_det] *= safe_std(np.r_[z_cos[places], z_sin[places]],
                                         **kwargs)
        return asd_drift

    def get_average_frequency(self, par_dic, ref_det_name=None, moment=1.):
        """
        Return average frequency in Hz, defined as
        ``(avg(f^moment))^(1/moment)``
        where ``avg`` is the frequency-domain average with weight
        ~ |h(f)|^2 / PSD(f).

        The answer is rounded to nearest Hz to ease reporting.

        Parameters
        ----------
        par_dic: dictionary of waveform parameters.
        ref_det_name: name of the detector from which to get the PSD,
                      e.g. 'H' for Hanford, or `None` (default) to
                      combine the PSDs of all detectors.
        moment: nonzero float, controls the frequency weights in the
                average.
        """
        det_ind = ...
        if ref_det_name:
            det_ind = self.event_data.detector_names.index(ref_det_name)

        weight = (np.abs(self._get_h_f(par_dic) * self.event_data.wht_filter)**2
                 )[det_ind, self.event_data.fslice]
        weight /= weight.sum()

        frequencies = self.event_data.frequencies[self.event_data.fslice]

        return np.round((weight * frequencies**moment).sum() ** (1/moment))

    @_check_bounds
    def lnlike_fft(self, par_dic):
        """
        Return log likelihood computed on the FFT grid, without using
        relative binning.
        """
        h_f = self._get_h_f(par_dic)
        h_h = self._compute_h_h(h_f)
        d_h = self._compute_d_h(h_f)
        return np.sum(d_h.real) - np.sum(h_h) / 2

    def _get_h_f(self, par_dic, *, normalize=False, by_m=False):
        """
        Return ndet x nfreq array with waveform strain at detectors,
        evaluated on the FFT frequency grid and zeroized outside
        `(fmin, fmax)`.
        Parameters
        ----------
        par_dic: dictionary of waveform parameters.
        normalize: bool, whether to normalize the waveform by sqrt(h|h)
                   at each detector.

        Return
        ------
        n_detectors x n_frequencies array with strain at detector.
        If by_m, output is (n_m x n_detectors x n_frequencies)
        """

        shape = ((len(self.waveform_generator._harmonic_modes_by_m),) if by_m
                 else ()) + self.event_data.strain.shape
        h_f = np.zeros(shape, dtype=np.complex_)
        h_f[..., self.event_data.fslice] \
            = self.waveform_generator.get_strain_at_detectors(
                self.event_data.frequencies[self.event_data.fslice], par_dic,
                by_m)
        if normalize:
            h_f /= np.sqrt(self._compute_h_h(h_f))[..., np.newaxis]
        return h_f

    def _compute_h_h(self, h_f):
        """
        Return array of len ndetectors with inner product (h|h).
        ASD drift correction is applied. Relative binning is not used.
        """
        return (4 * self.event_data.df * self.asd_drift**-2
                * np.linalg.norm(h_f * self.event_data.wht_filter, axis=-1)**2)

    def _compute_d_h(self, h_f):
        """
        Return array of len ndetectors with complex inner product (d|h).
        ASD drift correction is applied. Relative binning is not used.
        """
        return (4 * self.event_data.df * self.asd_drift**-2
                * np.sum(self.event_data.blued_strain * np.conj(h_f), axis=-1))

    def _matched_filter_timeseries(self, normalized_h_f):
        """
        Return (z_cos, z_sin), the matched filter output of a normalized
        template and its Hilbert transform.
        A constant asd drift correction per `self.asd_drift` is applied.

        Parameters
        ----------
        h_f: ndet x nfft array, normalized frequency domain waveform.

        Return
        ------
        z_cos, z_sin: each is a ndet x nfft time series.
        """
        factor = (2 * self.event_data.nfft * self.event_data.df
                  * self.asd_drift[:, np.newaxis]**-2)
        z_cos = factor * np.fft.irfft(self.event_data.blued_strain
                                      * np.conj(normalized_h_f))
        z_sin = factor * np.fft.irfft(self.event_data.blued_strain
                                      * np.conj(1j * normalized_h_f))
        return z_cos, z_sin

    def plot_whitened_wf(self, par_dic, trng=(-.7, .1), plot_data=True,
                         fig=None, figsize=None, by_m=False, **wf_plot_kwargs):
        """
        Plot the whitened strain and waveform model in the time domain
        in all detectors.

        Parameters:
        -----------
        par_dic: Waveform parameters to use, as per `self.params`.
        trng: Range of time to plot relative to `self.tgps`.
        plot_data: Flag to include detector data in plot.
        fig: `plt.Figure` object. `None` (default) creates a new figure.
        figsize: Figure (width, height) in inches, used if `fig=None`.
        **wf_plot_kwargs: Keyword arguments passed to pyplot.plot()
                          for waveform (data plot arguments are fixed).
                          Data plot arguments are set to
                          c=`C0`, lw=.2, label=`Data` and can be changed
                          or expanded using the keyword data_plot_kwargs.
                          That is, the data's plot arguments will be
                          updated after the defaults are set, using
                          wf_plot_kwargs.pop(`data_plot_kwargs`)

        Return:
        -------
        fig, ax: Figure and axes array with plots.
        """
        if fig is None:
            fig = self._setup_data_figure(figsize)
        axes = fig.get_axes()

        time = self.event_data.t - self.event_data.tcoarse
        data_t_wht = self._get_whitened_td(self.event_data.strain)
        wf_t_wht = self._get_whitened_td(self._get_h_f(par_dic, by_m=by_m))
        if by_m:
            wf_t_wht = np.array([wf_t_wht[:, j, :]
                                 for j in range(len(data_t_wht))])
            lab0 = wf_plot_kwargs.pop('label', '')
            if lab0:
                lab0 = f'{lab0}: '
        # Plot
        data_plot_kwargs = wf_plot_kwargs.pop('data_plot_kwargs', {})
        for ax, data_det, wf_det in zip(axes, data_t_wht, wf_t_wht):
            if plot_data:
                ax.plot(time, data_det, 'C0', lw=.2, label='Data',
                        **data_plot_kwargs)
            if by_m:
                for j, lmlist in enumerate(
                        self.waveform_generator._harmonic_modes_by_m.values()):
                    lab_lm = lab0 + ', '.join([str(lm) for lm in lmlist])
                    ax.plot(time, wf_det[j], label=lab_lm, **wf_plot_kwargs)
            else:
                ax.plot(time, wf_det, **wf_plot_kwargs)

        plt.xlim(trng)
        return fig

    def _get_whitened_td(self, strain_f):
        """
        Take a frequency-domain strain defined on the FFT grid
        `self.event_data.frequencies` and return a whitened time domain
        strain defined on `self.event_data.t`.
        """
        return (np.sqrt(2 * self.event_data.nfft * self.event_data.df)
                * np.fft.irfft(strain_f * self.event_data.wht_filter))

    def _setup_data_figure(self, figsize=None):
        """Return a new `Figure` with subplots for each detector."""
        fig, axes = plt.subplots(len(self.event_data.detector_names),
                                 sharex=True, sharey=True, figsize=figsize,
                                 tight_layout=True)
        axes = np.atleast_1d(axes)
        fig.text(.0, .54, 'Whitened Strain', rotation=90, ha='left',
                 va='center', fontsize='large')

        for ax, det in zip(axes, self.event_data.detector_names):
            ax.text(.02, .95, det, ha='left', va='top', transform=ax.transAxes)
            ax.tick_params(which='both', direction='in', right=True, top=True)

        plt.xlabel('Time (s)', size=12)
        return fig

    def __repr__(self):
        return f'{self.__class__.__name__}({self.event_data.eventname})'


class ReferenceWaveformFinder(CBCLikelihood):
    """
    Find a high-likelihood solution without using relative binning.
    Identify best reference detector and detector pair based on SNR.
    Identify best reference frequency based on best-fit waveform.

    Some simplfying restrictions are placed, for speed:
        * (2, 2) mode
        * Aligned, equal spins
        * Inclination = 1 radian
        * Polarization = 0
    """
    def __init__(self, event_data, waveform_generator):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`.
        waveform_generator: Instance of `waveform.WaveformGenerator`.
        """
        super().__init__(event_data, waveform_generator)
        waveform_generator.harmonic_modes = [(2, 2)]

    def find_bestfit_pars(self, tc_rng=(-.1, .1), seed=0,
                          f_ref_moment=1.):
        """
        Find a good fit solution with restricted parameters
        (face-on, equal aligned spins).
        Does not use relative binning so it can be used to find
        a fiducial relative binning waveform.

        First maximize likelihood incoherently w.r.t. intrinsic
        parameters using mchirp, eta, chieff.
        Then maximize likelihood w.r.t. extrinsic parameters, leaving
        the intrinsic fixed.

        Parameters
        ----------
        tc_rng: 2-tuple with minimum and maximum times to look for
                triggers.
        seed: To initialize the random state of stochastic maximizers.
        f_ref_moment: nonzero float, controls the choice of `f_ref`.
                      See `CBCLikelihood.get_average_frequency`.
        """
        # Set initial parameter values, will improve them in stages.
        par_dic = {par: 0. for par in self.waveform_generator.params}
        par_dic['d_luminosity'] = 1.
        par_dic['iota'] = 1.  # So we have higher modes
        par_dic['f_ref'] = 20.

        # Optimize intrinsic parameters
        self._optimize_m1m2s1zs2z_incoherently(par_dic, tc_rng, seed)

        # Use waveform to define asd_drift, reference detector, detector
        # pair and reference frequency:
        self.asd_drift = self.compute_asd_drift(par_dic)
        lnl_by_detectors = self.lnlike_max_amp_phase_time(
            par_dic, tc_rng, return_by_detectors=True)
        sorted_dets = [det for _, det in sorted(zip(
            lnl_by_detectors, self.waveform_generator.detector_names))][::-1]
        ref_det_name = sorted_dets[0]
        detector_pair = ''.join(
            dict.fromkeys(sorted_dets + ['H', 'L']))[:2]  # 2 dets guaranteed
        par_dic['f_ref'] = self.get_average_frequency(
            par_dic, ref_det_name, f_ref_moment)

        # Optimize time, sky location, orbital phase and distance
        self._optimize_t_refdet(par_dic, ref_det_name, tc_rng)
        t0_refdet = self._optimize_skyloc(par_dic, ref_det_name, detector_pair,
                                          seed)
        self._optimize_phase_and_distance(par_dic)

        return {'par_dic': par_dic,
                'f_ref': par_dic['f_ref'],
                'ref_det_name': ref_det_name,
                'detector_pair': detector_pair,
                't0_refdet': t0_refdet}

    @_check_bounds
    def lnlike_max_amp_phase_time(self, par_dic, tc_rng,
                                  return_by_detectors=False):
        """
        Return log likelihood maximized over amplitude, phase and time
        incoherently across detectors.
        """
        normalized_h_f = self._get_h_f(par_dic, normalize=True)
        z_cos, z_sin = self._matched_filter_timeseries(normalized_h_f)
        inds = np.arange(int(tc_rng[0] / self.event_data.dt),
                         int(tc_rng[1] / self.event_data.dt))
        lnl = np.max(z_cos[:, inds]**2 + z_sin[:, inds]**2, axis=1) / 2
        if return_by_detectors:
            return lnl
        return np.sum(lnl)

    @_check_bounds
    def lnlike_max_amp_phase(self, par_dic, ret_amp_phase_bf=False,
                             det_inds=...):
        """
        Return log likelihood maximized over amplitude and phase
        coherently across detectors (the same amplitude rescaling
        and phase is applied to all detectors).
        """
        h_f = self._get_h_f(par_dic)
        h_h = np.sum(self._compute_h_h(h_f)[det_inds])
        d_h = np.sum(self._compute_d_h(h_f)[det_inds])
        lnl = np.abs(d_h)**2 / h_h / 2

        if not ret_amp_phase_bf:
            return lnl

        phase_bf = np.angle(d_h)
        amp_bf = np.abs(d_h) / h_h
        return lnl, amp_bf, phase_bf

    def _optimize_m1m2s1zs2z_incoherently(self, par_dic, tc_rng, seed):
        """
        Optimize mchirp, eta and chieff by likelihood maximized over
        amplitude, phase and time incoherently across detectors.
        Modify inplace the entries of `par_dic` correspondig to
        `m1, m2, s1z, s2z` with the new solution.
        """
        # eta_max < .25 to avoid q = 1 solutions that lack harmonics:
        eta_rng = gw_utils.q_to_eta(self.event_data.q_min), .24
        chieff_rng = (-.999, .999)

        def lnlike_incoherent(mchirp, eta, chieff):
            m1, m2 = gw_utils.mchirpeta_to_m1m2(mchirp, eta)
            intrinsic = dict(m1=m1, m2=m2, s1z=chieff, s2z=chieff)
            return self.lnlike_max_amp_phase_time(par_dic | intrinsic, tc_rng)

        print(f'Searching incoherent solution for {self.event_data.eventname}')
        result = differential_evolution(
            lambda mchirp_eta_chieff: -lnlike_incoherent(*mchirp_eta_chieff),
            bounds=[self.event_data.mchirp_range, eta_rng, chieff_rng],
            seed=seed)
        print(f'Set intrinsic parameters, lnL = {-result.fun}')

        mchirp, eta, chieff = result.x
        par_dic['m1'], par_dic['m2'] = gw_utils.mchirpeta_to_m1m2(mchirp, eta)
        par_dic['s1z'] = par_dic['s2z'] = chieff

    #### EVDAT QUESTION: should this be talking more with event_data?
    def _optimize_t_refdet(self, par_dic, ref_det_name, tc_rng):
        """
        Find coalescence time that optimizes SNR maximized over
        amplitude and phase at reference detector.
        Update the 't_geocenter' entry of `par_dic` inplace.
        Note that `t_geocenter` can and should be recomputed if `ra`,
        `dec` change.
        """
        i_refdet = self.event_data.detector_names.index(ref_det_name)

        def lnlike_refdet(t_geocenter):
            return self.lnlike_max_amp_phase(
                {**par_dic, 't_geocenter': t_geocenter}, det_inds=i_refdet)

        tc_arr = np.arange(*tc_rng, 2**-10)
        ind = np.argmax([lnlike_refdet(tgeo) for tgeo in tc_arr])
        result = minimize_scalar(lambda tgeo: -lnlike_refdet(tgeo),
                                 bracket=tc_arr[ind-1 : ind+2], bounds=tc_rng)
        par_dic['t_geocenter'] = result.x
        print(f'Set time, lnL({ref_det_name}) = {-result.fun}')

    def _optimize_skyloc(self, par_dic, ref_det_name, detector_pair, seed):
        """
        Find right ascension and declination that optimize likelihood
        maximized over amplitude and phase.
        t_geocenter is readjusted so as to leave t_refdet unchanged.
        Update 't_geocenter', ra', 'dec' entries of `par_dic` inplace.
        Return `t0_refdet`, best fit time [s] relative to `tgps` at the
        reference detector.
        """
        skyloc = SkyLocAngles(detector_pair, self.event_data.tgps)
        time_transformer = UniformTimePrior(tgps=self.event_data.tgps,
                                            ref_det_name=ref_det_name,
                                            t0_refdet=np.nan, dt0=np.nan)
        t0_refdet = time_transformer.inverse_transform(
            **{key: par_dic[key] for key in ['t_geocenter', 'ra', 'dec']}
            )['t_refdet']  # Will hold t_refdet fixed


        def lnlike_skyloc(thetanet, phinet):
            ra, dec = skyloc.thetaphinet_to_radec(thetanet, phinet)
            t_geocenter_dic = time_transformer.transform(t_refdet=t0_refdet,
                                                         ra=ra, dec=dec)
            return self.lnlike_max_amp_phase(
                {**par_dic, 'ra': ra, 'dec': dec, **t_geocenter_dic})

        result = differential_evolution(
            lambda thetaphinet: -lnlike_skyloc(*thetaphinet),
            bounds=[(0, np.pi), (0, 2*np.pi)], seed=seed)
        thetanet, phinet = result.x
        par_dic['ra'], par_dic['dec'] = skyloc.thetaphinet_to_radec(thetanet,
                                                                    phinet)
        par_dic['t_geocenter'] = time_transformer.transform(
            t_refdet=t0_refdet, ra=par_dic['ra'], dec=par_dic['dec']
            )['t_geocenter']

        print(f'Set sky location, lnL = {-result.fun}')
        return t0_refdet

    def _optimize_phase_and_distance(self, par_dic):
        """
        Find phase and distance that optimize coherent likelihood.
        Update 'vphi', 'd_luminosity' entries of `par_dic` inplace.
        Return log likelihood.
        """
        max_lnl, amp_bf, phase_bf = self.lnlike_max_amp_phase(
            par_dic, ret_amp_phase_bf=True)
        par_dic['vphi'] += phase_bf / 2
        par_dic['d_luminosity'] /= amp_bf
        lnl = self.lnlike_fft(par_dic)
        np.testing.assert_allclose(max_lnl, lnl)

        print(f'Set polarization and distance, lnL = {lnl}')


TOLERANCE_PARAMS = {'ref_wf_lnl_difference': np.inf,
                    'raise_ref_wf_outperformed': False,
                    'check_relative_binning_every': np.inf,
                    'relative_binning_dlnl_tol': .1,
                    'lnl_drop_from_peak': 20.}


class ReferenceWaveformOutperformedError(LikelihoodError):
    """
    The relative-binning reference waveform's log likelihood was
    exceeded by more than `tolerance_params['ref_wf_lnl_difference']` at
    some other parameter values and
    `tolerance_params['raise_ref_wf_outperformed']` was `True`.
    """


class RelativeBinningError(LikelihoodError):
    """
    The log likelihood computed with relative-binning was different than
    that on the FFT grid by more than
    `tolerance_params['relative_binning_dlnl_tol']`.
    """


class RelativeBinningLikelihood(CBCLikelihood):
    """
    Generalization of `CBCLikelihood` that implements computation of
    likelihood with the relative binning method.
    """
    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None, tolerance_params=None,
                 spline_degree=3):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`
        waveform_generator: Instance of `waveform.WaveformGenerator`.
        par_dic_0: dictionary with parameters of the reference waveform,
                   should be close to the maximum likelihood waveform.
        fbin: Array with edges of the frequency bins used for relative
              binning [Hz]. Alternatively, pass `pn_phase_tol`.
        pn_phase_tol: Tolerance in the post-Newtonian phase [rad] used
                      for defining frequency bins. Alternatively, pass
                      `fbin`.
        tolerance_params: dictionary with relative-binning tolerance
                          parameters. Keys may include a subset of:
          * `ref_wf_lnl_difference`: float, tolerable improvement in
            value of log likelihood with respect to reference waveform.
          * `raise_ref_wf_outperformed`: bool. If `True`, raise a
            `RelativeBinningError` if the reference waveform is
            outperformed by more than `ref_wf_lnl_difference`.
            If `False`, silently update the reference waveform.
          * check_relative_binning_every: int, every how many
            evaluations to test accuracy of log likelihood computation
            with relative binning against the expensive full FFT grid.
          * relative_binning_dlnl_tol: float, absolute tolerance in
            log likelihood for the accuracy of relative binning.
            Whenever the tolerance is exceeded, `RelativeBinningError`
            is raised.
          * lnl_drop_from_peak: float, disregard accuracy tests if the
            tested waveform achieves a poor log likelihood compared to
            the reference waveform.
        """
        if (fbin is None) == (pn_phase_tol is None):
            raise ValueError('Pass exactly one of `fbin` or `pn_phase_tol`.')

        super().__init__(event_data, waveform_generator)

        self._spline_degree = spline_degree
        self._par_dic_0 = par_dic_0

        if pn_phase_tol:
            self.pn_phase_tol = pn_phase_tol
        else:
            self.fbin = fbin

        self.tolerance_params = TOLERANCE_PARAMS | (tolerance_params or {})

        self._lnlike_evaluations = 0

    @_check_bounds
    def lnlike(self, par_dic, bypass_tests=False):
        """
        Return log likelihood using relative binning. Apply relative-
        binning robustness tests per `self.tolerance_params`.
        """
        lnl = self.lnlike_detectors_no_asd_drift(par_dic) @ self.asd_drift**-2

        if bypass_tests:
            return lnl

        # Relative-binning robustness tests
        self._lnlike_evaluations += 1
        if lnl - self._lnl_0 > self.tolerance_params['ref_wf_lnl_difference']:
            self.test_relative_binning_accuracy(par_dic)
            if self.tolerance_params['raise_ref_wf_outperformed']:
                raise ReferenceWaveformOutperformedError(
                    f'lnl = {lnl}, lnl_0 = {self._lnl_0}, par_dic = {par_dic}')
            self.par_dic_0 = par_dic  # Update relative binning solution
            lnl = self._lnl_0

        if (self._lnlike_evaluations
                % self.tolerance_params['check_relative_binning_every'] == 0):
            self.test_relative_binning_accuracy(par_dic)

        return lnl

    def lnlike_detectors_no_asd_drift(self, par_dic):
        """
        Return an array of length n_detectors with the values of
        `(d|h) - (h|h)/2`, no ASD-drift correction applied, using
        relative binning.

        Parameters
        ----------
        par_dic: dictionary per `self.waveform_generator.params`.
        """
        h_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)

        # Sum over m and f axes, leave det ax unsummed.
        d_h = (self._d_h_weights * h_fbin.conj()).real.sum(axis=(0, -1))

        m_inds, mprime_inds = self._get_m_mprime_inds()
        h_h = ((self._h_h_weights * h_fbin[m_inds] * h_fbin[mprime_inds].conj()
               ).real.sum(axis=(0, -1)))
        return d_h - h_h/2

    @property
    def pn_phase_tol(self):
        """
        Tolerance in the post-Newtonian phase [rad] used for defining
        frequency bins.
        """
        return self._pn_phase_tol

    @pn_phase_tol.setter
    def pn_phase_tol(self, pn_phase_tol):
        """
        Compute frequency bins such that across each frequency bin the
        change in the post-Newtonian waveform phase with respect to the
        fiducial waveform is bounded by `pn_phase_tol` [rad].
        """
        pn_exponents = [-5/3, -2/3, 1]
        if APPROXIMANTS[self.waveform_generator.approximant].tides:
            pn_exponents += [5/3]
        pn_exponents = np.array(pn_exponents)

        pn_coeff_rng = 2*np.pi / np.abs(self.event_data.fmin**pn_exponents
                                        - self.event_data.fmax**pn_exponents)

        f_arr = np.linspace(self.event_data.fmin, self.event_data.fmax, 10000)

        diff_phase = np.sum([np.sign(exp) * rng * f_arr**exp
                             for rng, exp in zip(pn_coeff_rng, pn_exponents)],
                            axis=0)
        diff_phase -= diff_phase[0]  # Worst case scenario differential phase

        # Construct frequency bins
        nbin = np.ceil(diff_phase[-1] / pn_phase_tol).astype(int)
        diff_phase_arr = np.linspace(0, diff_phase[-1], nbin + 1)
        self.fbin = np.interp(diff_phase_arr, diff_phase, f_arr)
        self._pn_phase_tol = pn_phase_tol

    @property
    def fbin(self):
        """Edges of the frequency bins for relative binning [Hz]."""
        return self._fbin

    @fbin.setter
    def fbin(self, fbin):
        """
        Set frequency bin edges, round them to fall in the FFT array.
        Compute auxiliary quantities related to frequency bins.
        Set `_pn_phase_tol` to `None` to keep logs clean.
        """
        fbin_ind = np.unique(np.searchsorted(self.event_data.frequencies,
                                             fbin - self.event_data.df/2))
        self._fbin = self.event_data.frequencies[fbin_ind]  # Bin edges

        self._set_splines()
        self._set_summary()
        self._pn_phase_tol = None  # Erase potentially outdated information

    @property
    def spline_degree(self):
        """
        Integer between 1 and 5, degree of the spline used to
        interpolate waveform ratios.
        """
        return self._spline_degree

    @spline_degree.setter
    def spline_degree(self, spline_degree):
        self._spline_degree = spline_degree
        self._set_splines()
        self._set_summary()

    @property
    def par_dic_0(self):
        """Dictionary with reference waveform parameters."""
        return self._par_dic_0

    @par_dic_0.setter
    def par_dic_0(self, par_dic_0):
        self._par_dic_0 = par_dic_0
        self._set_summary()

    def _set_splines(self):
        self._splines = scipy.interpolate.interp1d(
            self.fbin, np.eye(len(self.fbin)), kind=self._spline_degree,
            bounds_error=False, fill_value=0.)(self.event_data.frequencies).T

    def _set_summary(self):
        """
        Compute summary data for the fiducial waveform at all detectors.
        `asd_drift` is not applied to the summary data, to not have to
        keep track of it.
        Update `asd_drift` using the reference waveform.
        The summary data `self._d_h_weights` and `self._d_h_weights` are
        such that:
            (d|h) ~= sum(_d_h_weights * conj(h_fbin)) / asd_drift^2
            (h|h) ~= sum(_h_h_weights * abs(h_fbin)^2) / asd_drift^2

        Note: all spin components in `self.par_dic_0` are used, even if
        `self.waveform_generator.disable_precession` is set to `True`.
        This is so that the reference waveform remains the same when
        toggling `disable_precession`.

        # TODO: maybe enforce a minimum amplitude for h_0?
        """
        # Don't zero the in-plane spins for the reference waveform
        disable_precession = self.waveform_generator.disable_precession
        self.waveform_generator.disable_precession = False

        self._h0_f = self._get_h_f(self.par_dic_0, by_m=True)
        self._h0_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, self.par_dic_0, by_m=True)  # n_m x ndet x len(fbin)

        d_h0 = self.event_data.blued_strain * np.conj(self._h0_f)
        self._d_h_weights = (self._get_summary_weights(d_h0)
                             / np.conj(self._h0_fbin))

        m_inds, mprime_inds = self._get_m_mprime_inds()
        h0m_h0mprime = (self._h0_f[m_inds] * self._h0_f[mprime_inds].conj()
                        * self.event_data.wht_filter ** 2)
        self._h_h_weights = (self._get_summary_weights(h0m_h0mprime)
                             / (self._h0_fbin[m_inds]
                                * self._h0_fbin[mprime_inds].conj()))
        # Count off-diagonal terms twice:
        self._h_h_weights[~np.equal(m_inds, mprime_inds)] *= 2

        self.asd_drift = self.compute_asd_drift(self.par_dic_0)
        self._lnl_0 = self.lnlike(self.par_dic_0, bypass_tests=True)

        # Reset
        self.waveform_generator.disable_precession = disable_precession

    def _get_m_mprime_inds(self):
        """
        Return two lists of integers, these zipped are pairs (i, j) of
        indices with j >= i that run through the number of m modes.
        """
        return map(list, zip(*itertools.combinations_with_replacement(
            range(len(self.waveform_generator._harmonic_modes_by_m)), 2)))

    def _get_summary_weights(self, integrand):
        """
        Return summary data to compute efficiently integrals of the form
            4 integral g(f) r(f) df,
        where r(f) is a smooth function.
        The above integral is approximated by
            summary_weights * r(fbin)
        which is the exact result of replacing `r(f)` by a spline that
        interpolates it at `fbin`.
        This implementation is simple but slow and memory-intensive.
        Could be faster by using a sparse basis for the splines.
        Could use less memory with a `for` loop instead of `np.dot`.

        Parameters
        ----------
        integrand: g(f) in the above notation (the oscillatory part of
                   the integrand), array whose last axis corresponds to
                   the fft frequency grid.

        Return
        ------
        summary_weights: array shaped like `integrand` except the last
                         axis now correponds to the frequency bins.
        """
        return 4 * self.event_data.df * np.dot(integrand, self._splines)

    def _get_h_f_interpolated(self, par_dic):
        """
        Fast approximation to `_get_h_f`.
        Return ndet x nfreq array with waveform strain at detectors
        evaluated on the FFT frequency grid and zeroized outside
        `(fmin, fmax)`, computed using relative binning from a low
        frequency resolution waveform.
        """
        h_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)

        ratio = scipy.interpolate.interp1d(
            self.fbin, h_fbin / self._h0_fbin, assume_sorted=True,
            kind=self.spline_degree, bounds_error=False, fill_value=0.
            )(self.event_data.frequencies)

        # Sum over harmonic modes
        return np.sum(ratio * self._h0_f, axis=0)

    def test_relative_binning_accuracy(self, par_dic):
        """
        Return 2-tuple with log likelihood evaluated with and without
        the relative binning method.
        Raise `RelativeBinningError` if the difference is bigger than
        `self.tolerance_params['relative_binning_dlnl_tol']`.
        """
        lnl_rb = self.lnlike(par_dic, bypass_tests=True)
        lnl_fft = self.lnlike_fft(par_dic)

        good_wf = (self._lnl_0 - max(lnl_rb, lnl_fft)
                   < self.tolerance_params['lnl_drop_from_peak'])

        tol_exceeded = (np.abs(lnl_rb - lnl_fft)
                        > self.tolerance_params['relative_binning_dlnl_tol'])

        if good_wf and tol_exceeded:
            raise RelativeBinningError(
                'Relative-binning tolerance exceeded:\n'
                f'lnl_rb = {lnl_rb}\nlnl_fft = {lnl_fft}\npar_dic = {par_dic}')
        return lnl_rb, lnl_fft

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return super().get_init_dict() | ({'fbin': None} if self.pn_phase_tol
                                          else {})
