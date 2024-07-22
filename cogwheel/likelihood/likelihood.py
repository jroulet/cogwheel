"""
Compute likelihood of GW events.

A class ``CBCLikelihood`` is defined. This can be used to compute
likelihoods without resorting to the relative binning algorithm (slow)
and is subclassed by ``BaseRelativeBinning``.
"""
import inspect
from functools import wraps
import numpy as np
from scipy import special, stats
import matplotlib.pyplot as plt

from cogwheel import utils
from cogwheel import waveform


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
    Estimator of the standard deviation of an array based on the median
    absolute deviation for robustness to outliers.
    """
    mad = np.median(np.abs(arr - np.median(arr)))
    return mad / (np.sqrt(2) * special.erfinv(.5))


def check_bounds(lnlike_func):
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
    Subclassed by ``RelativeBinningLikelihood``.
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
    def params(self):
        """Parameters expected in `par_dic` for likelihood evaluation."""
        return self.waveform_generator.params

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

        self._asd_drift = np.asarray(value, dtype=np.float64)

    def compute_asd_drift(self, par_dic, tol=.02,
                          max_tcorr_contiguous_low=16., **kwargs):
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
        par_dic: dict
            Waveform parameters, keys should match
            ``self.waveform_generator.params``.

        tol: float
            Stochastic measurement error tolerance, used to decide the
            number of samples.

        max_tcorr_contiguous_low: float
            Maximum number of contiguous correlation times with values
            below the average noise level to allow (these are classified
            as a hole and disregarded in the average).

        **kwargs:
            Passed to `safe_std`, keys include:
                `expected_high`, `reject_nearby`.
        """
        # Use all available modes and spin components to get a waveform
        with utils.temporarily_change_attributes(self.waveform_generator,
                                                 disable_precession=False,
                                                 harmonic_modes=None):
            # Undo previous asd_drift so result is independent of it
            normalized_h_f = (self._get_h_f(par_dic, normalize=True)
                              / self.asd_drift[:, np.newaxis])

        z_cos, z_sin = self._matched_filter_timeseries(normalized_h_f)
        whitened_h_f = (np.sqrt(2 * self.event_data.nfft * self.event_data.df)
                        * self.event_data.wht_filter * normalized_h_f)

        correlation_length = (4 * np.sum(np.abs(whitened_h_f)**4, axis=-1)
                              / self.event_data.nfft)

        asd_drift = np.ones_like(self.asd_drift)
        for i_det, ncorr in enumerate(correlation_length):
            nsamples = min(np.ceil(ncorr / tol**2).astype(int),
                           z_cos.shape[1])
            max_contiguous_low = np.ceil(max_tcorr_contiguous_low * ncorr
                                        ).astype(int)

            places = (i_det, np.arange(-nsamples//2, nsamples//2))
            asd_drift[i_det] = self._safe_std(
                np.r_[z_cos[places], z_sin[places]], max_contiguous_low,
                **kwargs)

        return asd_drift

    def _safe_std(self, arr, max_contiguous_low=np.inf,
                  expected_high=1., reject_nearby=.5):
        """
        Compute the standard deviation of a real array rejecting
        outliers.
        Outliers may be:
          * Values too high to likely come from white Gaussian noise.
          * A contiguous array of values too low to come from white
            Gaussian noise (likely from a hole).
        Once outliers are identified, an extra amount of nearby samples
        is rejected for safety.

        Parameters
        ----------
        max_contiguous_low: int
            How many contiguous samples below 1 sigma to allow.

        expected_high: float
            Number of times we expect to trigger in white Gaussian noise
            (used to set the clipping threshold).

        reject_nearby: float
            By how many seconds to expand holes for safety.
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
        thresh = std_est * stats.chi.isf(
            expected_high / np.count_nonzero(good), 1)
        good[np.abs(arr) > thresh] = False

        # Reject extra nearby samples
        bad_edges = hole_edges(good)
        for left, right in bad_edges:
            reject_inds = int(reject_nearby*2*self.event_data.frequencies[-1])
            good[max(0, left-reject_inds) : right+reject_inds] = False

        return np.std(arr[good])

    def get_average_frequency(self, par_dic, ref_det_name=None,
                              moment=1.):
        """
        Return average frequency in Hz, defined as
        ``(avg(f^moment))^(1/moment)``
        where ``avg`` is the frequency-domain average with weight
        ~ |h(f)|^2 / PSD(f).

        The answer is rounded to nearest Hz to ease reporting.

        Parameters
        ----------
        par_dic: dict
            Waveform parameters, keys should match
            ``self.waveform_generator.params``.

        ref_det_name: str or None
            Name of the detector from which to get the PSD, e.g. 'H' for
            Hanford, or `None` (default) to combine the PSDs of all
            detectors.

        moment: nonzero float
            Controls the frequency weights in the average.
        """
        det_ind = ...
        if ref_det_name:
            det_ind = self.event_data.detector_names.index(ref_det_name)

        weight = np.abs(self._get_h_f(par_dic) * self.event_data.wht_filter
                       )[det_ind, self.event_data.fslice] ** 2
        weight /= weight.sum()

        frequencies = self.event_data.frequencies[self.event_data.fslice]

        return np.round((weight * frequencies**moment).sum() ** (1/moment))

    @check_bounds
    def lnlike_fft(self, par_dic):
        """
        Return log likelihood computed on the FFT grid, without using
        relative binning.

        Parameters
        ----------
        par_dic: dict
            Waveform parameters, keys should match
            ``self.waveform_generator.params``.
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
        par_dic: dict
            Waveform parameters, keys should match
            ``self.waveform_generator.params``.

        normalize: bool
            Whether to normalize the waveform by sqrt(h|h) at each
            detector.

        Return
        ------
        Array of shape (n_m?, n_detectors, n_frequencies) with strain at
        detector. `n_m` is there only if `by_m=True`.
        """
        shape = (self.waveform_generator.m_arr.shape if by_m else ()
                ) + self.event_data.strain.shape
        h_f = np.zeros(shape, dtype=np.complex128)
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
        No ASD drift correction is applied.

        Parameters
        ----------
        h_f: (ndet, nrfft) array
            Normalized frequency domain waveform.

        Return
        ------
        z_cos, z_sin: each is a (ndet, nfft) time series.
        """
        factor = 2 * self.event_data.nfft * self.event_data.df
        z_cos = factor * np.fft.irfft(self.event_data.blued_strain
                                      * np.conj(normalized_h_f))
        z_sin = factor * np.fft.irfft(self.event_data.blued_strain
                                      * np.conj(1j * normalized_h_f))
        return z_cos, z_sin

    def plot_whitened_wf(self, par_dic, trng=(-.7, .1), plot_data=True,
                         fig=None, figsize=None, by_m=False,
                         **wf_plot_kwargs):
        """
        Plot the whitened strain and waveform model in the time domain
        in all detectors.

        Parameters:
        -----------
        par_dic: dict
            Waveform parameters, keys should match
            ``self.waveform_generator.params``.

        trng: (float, float)
            Range of time to plot relative to `self.tgps` (s).

        plot_data: bool
            Whether to include detector data in plot.

        fig: `plt.Figure`, optional
            `None` (default) creates a new figure.

        figsize: (float, float)
            Figure width and height in inches, used if `fig=None`.

        **wf_plot_kwargs:
            Keyword arguments passed to ``plt.plot()`` for waveform.
            Additionally, keyword arguments for the data plot can be
            passed as a dict ``data_plot_kwargs``.

        Return:
        -------
        fig: Figure with plots.
        """
        if fig is None:
            fig = self._setup_data_figure(figsize)
        axes = fig.get_axes()

        time = self.event_data.times - self.event_data.tcoarse
        data_t_wht = self._get_whitened_td(self.event_data.strain)
        wf_t_wht = self._get_whitened_td(self._get_h_f(par_dic, by_m=by_m))

        # Plot
        data_plot_kwargs = ({'c': 'C0', 'lw': .2, 'label': 'Data'}
                            | wf_plot_kwargs.pop('data_plot_kwargs', {}))
        for ax, data_det, wf_det in zip(axes, data_t_wht, wf_t_wht):
            if plot_data:
                ax.plot(time, data_det, **data_plot_kwargs)
            ax.plot(time, wf_det, **wf_plot_kwargs)

        plt.xlim(trng)
        return fig

    def _get_whitened_td(self, strain_f):
        """
        Take a frequency-domain strain defined on the FFT grid
        ``self.event_data.frequencies`` and return a whitened time
        domain strain defined on `self.event_data.times`.
        """
        return (np.sqrt(2 * self.event_data.nfft * self.event_data.df)
                * np.fft.irfft(strain_f * self.event_data.wht_filter))

    def _setup_data_figure(self, figsize=None):
        """Return a new ``Figure`` with subplots for each detector."""
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

    def postprocess_samples(self, samples):
        """
        Placeholder method that will be called after sampling and may be
        overriden by subclasses. (E.g. marginalized likelihoods
        demarginalize the distribution in postprocessing.)

        Parameters
        ----------
        samples: pandas.DataFrame
            Rows are samples, columns must contain `.params`.
        """
        del samples

    def __repr__(self):
        return f'{self.__class__.__name__}({self.event_data.eventname})'
