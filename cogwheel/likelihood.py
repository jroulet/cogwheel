"""Compute likelihood of GW events."""

import inspect
import itertools
from functools import wraps
import numpy as np
from scipy import special, stats
from scipy.optimize import differential_evolution, minimize_scalar
import scipy.interpolate
import matplotlib.pyplot as plt

from cogwheel import data
from cogwheel import gw_utils
from cogwheel import utils
from cogwheel import waveform
from cogwheel.gw_prior import UniformTimePrior
from cogwheel.skyloc_angles import SkyLocAngles


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

        if not waveform.within_bounds(par_dic):
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
                  * self.asd_drift**-2)[:, np.newaxis]
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


class RelativeBinningLikelihood(CBCLikelihood):
    """
    Generalization of `CBCLikelihood` that implements computation of
    likelihood with the relative binning method.
    """
    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None, spline_degree=3):
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
        spline_degree: int, degree of the spline used to interpolate the
                       ratio between waveform and reference waveform for
                       relative binning.
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

    @_check_bounds
    def lnlike(self, par_dic):
        """Return log likelihood using relative binning."""
        return self.lnlike_detectors_no_asd_drift(par_dic) @ self.asd_drift**-2

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

        # Sum over m and f axes, leave detector axis unsummed.
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
        if waveform.APPROXIMANTS[self.waveform_generator.approximant].tides:
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

    def _get_h_f_interpolated(self, par_dic, *, normalize=False, by_m=False):
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

        h_f = ratio * self._h0_f

        if normalize:
            h_f /= np.sqrt(self._compute_h_h(h_f))[..., np.newaxis]

        if by_m:
            return h_f
        return np.sum(h_f, axis=0)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return super().get_init_dict() | ({'fbin': None} if self.pn_phase_tol
                                          else {})

    @classmethod
    def from_reference_waveform_finder(
            cls, reference_waveform_finder, approximant,
            fbin=None, pn_phase_tol=.05, spline_degree=3):
        """
        Provide
        """
        waveform_generator = reference_waveform_finder.waveform_generator \
            .reinstantiate(approximant=approximant, harmonic_modes=None)

        return cls(reference_waveform_finder.event_data, waveform_generator,
                   reference_waveform_finder.par_dic_0, fbin,
                   pn_phase_tol, spline_degree)


class ReferenceWaveformFinder(RelativeBinningLikelihood):
    """
    Find a high-likelihood solution.

    Some simplfying restrictions are placed, for speed:
        * (2, 2) mode
        * Aligned, equal spins
        * Inclination = 1 radian
        * Polarization = 0
    """
    @classmethod
    def from_event(cls, event, mchirp_guess, approximant='IMRPhenomXAS',
                   pn_phase_tol=.02, spline_degree=3,
                   **maximization_kwargs):
        """
        Constructor that finds a reference waveform solution
        automatically by maximizing the likelihood.

        Parameters
        ----------
        event_data: Instance of `data.EventData`, or string with event
                    name (must correspond to a file in `data.DATADIR`),
                    or path to ``npz`` file with `EventData` instance.
        mchirp_guess: float, estimate of the detector-frame chirp mass
                      of the signal.
        approximant: str, approximant name.
        pn_phase_tol: float
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins.
        spline_degree: int, degree of the spline used to interpolate the
                       ratio between waveform and reference waveform for
                       relative binning.
        **maximization_kwargs: passed to `self.find_bestfit_pars()`.
        """
        if isinstance(event, data.EventData):
            event_data = event
        else:
            try:
                event_data = data.EventData.from_npz(event)
            except FileNotFoundError:
                event_data = data.EventData.from_npz(filename=event)

        waveform_generator = waveform.WaveformGenerator.from_event_data(
            event_data, approximant, harmonic_modes=[(2, 2)])

        # Set initial parameter dictionary. Will get improved by
        # `find_bestfit_pars()`. Serves dual purpose as maximum
        # likelihood result and relative binning reference.
        par_dic_0 = dict.fromkeys(waveform_generator.params, 0.)
        par_dic_0['d_luminosity'] = 1.
        par_dic_0['iota'] = 1.  # So waveform has higher modes
        par_dic_0['f_ref'] = 100.
        par_dic_0['m1'], par_dic_0['m2'] = gw_utils.mchirpeta_to_m1m2(
            mchirp_guess, eta=.2)

        ref_wf_finder = cls(event_data, waveform_generator, par_dic_0,
                            pn_phase_tol=pn_phase_tol,
                            spline_degree=spline_degree)
        ref_wf_finder.find_bestfit_pars(**maximization_kwargs)
        return ref_wf_finder

    def find_bestfit_pars(self, mchirp_range=None, time_range=(-.1, .1),
                          seed=0):
        """
        Find a good fit solution with restricted parameters
        (face-on, equal aligned spins).
        Will update `self.par_dic_0` in stages. The relative binning
        summary data (and `asd_drift`) will be updated after maximizing
        over intrinsic parameters and also after setting the sky
        location.

        First maximize likelihood incoherently w.r.t. intrinsic
        parameters using mchirp, eta, chieff.
        Then maximize likelihood w.r.t. extrinsic parameters, leaving
        the intrinsic fixed.
        Speed is prioritized over quality of the maximization. Use
        `Posterior.refine_reference_waveform()` to refine the solution
        if needed.

        Parameters
        ----------
        mchirp_range: 2-tuple with minimum and maximum detector-frame
                      chirp mass (Msun), optional. If not given, will
                      guess based on `self.par_dic_0`.
        time_range: 2-tuple with minimum and maximum times to look for
                    triggers.
        seed: To initialize the random state of stochastic maximizers.
        """
        # Optimize intrinsic parameters, update relative binning summary:
        mchirp_range = mchirp_range or gw_utils.estimate_mchirp_range(
            gw_utils.mchirp(self.par_dic_0['m1'], self.par_dic_0['m2']))
        self._optimize_m1m2s1zs2z_incoherently(mchirp_range, time_range, seed)

        # Use waveform to define reference detector, detector pair
        # and reference frequency:
        kwargs = self.get_coordinate_system_kwargs(time_range)
        self.par_dic_0['f_ref'] = kwargs['f_avg']

        # Optimize time, sky location, orbital phase and distance
        self._optimize_t_refdet(kwargs['ref_det_name'], time_range)
        self._optimize_skyloc(kwargs['detector_pair'], seed)
        self._optimize_phase_and_distance()

    @_check_bounds
    def lnlike_max_amp_phase_time(self, par_dic, time_range,
                                  return_by_detectors=False):
        """
        Return log likelihood maximized over amplitude, phase and time
        incoherently across detectors.
        """
        normalized_h_f = self._get_h_f_interpolated(par_dic, normalize=True)
        z_cos, z_sin = self._matched_filter_timeseries(normalized_h_f)
        inds = np.arange(int(time_range[0] / self.event_data.dt),
                         int(time_range[1] / self.event_data.dt))
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
        h_f = self._get_h_f_interpolated(par_dic)
        h_h = np.sum(self._compute_h_h(h_f)[det_inds])
        d_h = np.sum(self._compute_d_h(h_f)[det_inds])
        lnl = np.abs(d_h)**2 / h_h / 2

        if not ret_amp_phase_bf:
            return lnl

        phase_bf = np.angle(d_h)
        amp_bf = np.abs(d_h) / h_h
        return lnl, amp_bf, phase_bf

    def _optimize_m1m2s1zs2z_incoherently(self, mchirp_range, time_range,
                                          seed):
        """
        Optimize mchirp, eta and chieff by likelihood maximized over
        amplitude, phase and time incoherently across detectors.
        Modify in-place the entries of `self.par_dic_0` correspondig to
        `m1, m2, s1z, s2z` with the new solution.
        """
        # eta_max < .25 to avoid q = 1 solutions that lack harmonics:
        eta_range = (.05, .24)
        chieff_range = (-.999, .999)

        def get_updated_par_dic(mchirp, eta, chieff):
            """Return `self.par_dic_0` with updated m1, m2, s1z, s2z."""
            m1, m2 = gw_utils.mchirpeta_to_m1m2(mchirp, eta)
            intrinsic = dict(m1=m1, m2=m2, s1z=chieff, s2z=chieff)
            return self.par_dic_0 | intrinsic

        def lnlike_incoherent(mchirp, eta, chieff):
            """
            Log likelihood maximized over amplitude, phase and time
            incoherently across detectors.
            """
            par_dic = get_updated_par_dic(mchirp, eta, chieff)
            return self.lnlike_max_amp_phase_time(par_dic, time_range)

        print(f'Searching incoherent solution for {self.event_data.eventname}')

        result = differential_evolution(
            lambda mchirp_eta_chieff: -lnlike_incoherent(*mchirp_eta_chieff),
            bounds=[mchirp_range, eta_range, chieff_range], seed=seed)

        self.par_dic_0 = get_updated_par_dic(*result.x)
        print(f'Set intrinsic parameters, lnL = {-result.fun}')

    def _optimize_t_refdet(self, ref_det_name, time_range):
        """
        Find coalescence time that optimizes SNR maximized over
        amplitude and phase at reference detector.
        Update the 't_geocenter' entry of `par_dic` in-place.
        Note that `t_geocenter` can and should be recomputed if `ra`,
        `dec` change.
        """
        i_refdet = self.event_data.detector_names.index(ref_det_name)

        def lnlike_refdet(t_geocenter):
            return self.lnlike_max_amp_phase(
                self.par_dic_0 | {'t_geocenter': t_geocenter}, det_inds=i_refdet)

        tc_arr = np.arange(*time_range, 2**-10)
        ind = np.argmax([lnlike_refdet(tgeo) for tgeo in tc_arr])
        result = minimize_scalar(lambda tgeo: -lnlike_refdet(tgeo),
                                 bracket=tc_arr[ind-1 : ind+2],
                                 bounds=time_range)
        self.par_dic_0['t_geocenter'] = result.x
        print(f'Set time, lnL({ref_det_name}) = {-result.fun}')

    def _optimize_skyloc(self, detector_pair, seed):
        """
        Find right ascension and declination that optimize likelihood
        maximized over amplitude and phase.
        t_geocenter is readjusted so as to leave t_refdet unchanged.
        Update 't_geocenter', ra', 'dec' entries of `self.par_dic_0`
        in-place.
        """
        skyloc = SkyLocAngles(detector_pair, self.event_data.tgps)
        time_transformer = UniformTimePrior(tgps=self.event_data.tgps,
                                            ref_det_name=detector_pair[0],
                                            t0_refdet=np.nan, dt0=np.nan)
        t0_refdet = time_transformer.inverse_transform(
            **{key: self.par_dic_0[key] for key in ['t_geocenter', 'ra', 'dec']}
            )['t_refdet']  # Will hold t_refdet fixed

        def get_updated_par_dic(thetanet, phinet):
            """Return `self.par_dic_0` with updated ra, dec, t_geocenter."""
            ra, dec = skyloc.thetaphinet_to_radec(thetanet, phinet)
            t_geocenter_dic = time_transformer.transform(
                t_refdet=t0_refdet, ra=ra, dec=dec)
            return self.par_dic_0 | {'ra': ra, 'dec': dec} | t_geocenter_dic

        def lnlike_skyloc(thetanet, phinet):
            par_dic = get_updated_par_dic(thetanet, phinet)
            return self.lnlike_max_amp_phase(par_dic)

        result = differential_evolution(
            lambda thetaphinet: -lnlike_skyloc(*thetaphinet),
            bounds=[(0, np.pi), (0, 2*np.pi)], seed=seed)

        self.par_dic_0 = get_updated_par_dic(*result.x)
        print(f'Set sky location, lnL = {-result.fun}')

    def _optimize_phase_and_distance(self):
        """
        Find phase and distance that optimize coherent likelihood.
        Update 'vphi', 'd_luminosity' entries of `self.par_dic_0`
        in-place.
        Return log likelihood.
        """
        max_lnl, amp_bf, phase_bf = self.lnlike_max_amp_phase(
            self.par_dic_0, ret_amp_phase_bf=True)
        self.par_dic_0['vphi'] += phase_bf / 2
        self.par_dic_0['d_luminosity'] /= amp_bf

        print(f'Set polarization and distance, lnL = {max_lnl}')

    def get_coordinate_system_kwargs(self, time_range=None):
        """
        Return dictionary with parameters commonly required to set up
        coordinate system for sampling.
        Can be used to instantiate some classes defined in `gw_prior`.

        Parameters
        ----------
        time_range: 2-tuple or None
            Used to set criterion for sorting detectors by decreasing
            likelihood. If passed, likelihood will be incoherent,
            maximized over amplitude, phase, and time in (tmin, tmax)
            relative to `self.event_data.tgps`.
            Otherwise, likelihood is coherent (default).

        Return
        ------
        dictionary with entries for:
            * tgps
            * par_dic
            * f_avg
            * f_ref
            * ref_det_name
            * detector_pair
            * t0_refdet
            * mchirp_range
        """
        if time_range:
            lnl_by_detectors = self.lnlike_max_amp_phase_time(
                self.par_dic_0, time_range, return_by_detectors=True)
        else:
            lnl_by_detectors = self.lnlike_detectors_no_asd_drift(
                self.par_dic_0) * self.asd_drift**-2

        sorted_dets = [det for _, det in sorted(zip(
            lnl_by_detectors, self.waveform_generator.detector_names))][::-1]
        ref_det_name = sorted_dets[0]
        detector_pair = ''.join(dict.fromkeys(sorted_dets + ['H', 'L']))[:2]

        f_avg = self.get_average_frequency(self.par_dic_0, ref_det_name)

        delay = gw_utils.time_delay_from_geocenter(
            ref_det_name, self.par_dic_0['ra'], self.par_dic_0['dec'],
            self.event_data.tgps)[0]
        t0_refdet = self.par_dic_0['t_geocenter'] + delay
        mchirp_range = gw_utils.estimate_mchirp_range(
            gw_utils.mchirp(self.par_dic_0['m1'], self.par_dic_0['m2']),
            snr=np.sqrt(2*lnl_by_detectors.sum()))

        return {'tgps': self.event_data.tgps,
                'par_dic': self.par_dic_0,
                'f_avg': f_avg,
                'f_ref': self.par_dic_0['f_ref'],
                'ref_det_name': ref_det_name,
                'detector_pair': detector_pair,
                't0_refdet': t0_refdet,
                'mchirp_range': mchirp_range}
