"""Download, process and store data about GW events."""

import json
import logging
import pathlib
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gwpy.timeseries
import gwosc

from cogwheel import gw_utils
from cogwheel import utils
from cogwheel import waveform

# gwpy meddles with matplotlib, undo:
plt.rcdefaults()
gwpy.plot.axes.register_projection(gwpy.plot.axes._Axes)


DATADIR = pathlib.Path(__file__).parent/'data'
GWOSC_FILES_DIR = DATADIR/'gwosc_files'

ASD_DIR = DATADIR/'example_asds'
ASDS = {path.name.removesuffix('.npy'): path
        for path in sorted(ASD_DIR.glob('*.npy'))}

EVENTS_METADATA = pd.read_csv(DATADIR/'events_metadata.csv', index_col=0)


def make_asd_func(frequencies, asd):
    """
    Interpolate amplitude spectral density as a function of frequency.

    Frequencies outside the given range are assigned a large but finite
    ASD.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequency values for the ASD.

    asd : numpy.ndarray
        Array of amplitude spectral density values corresponding to the
        frequencies.

    Returns
    -------
    function
        A function that interpolates the ASD based on input frequency.
    """
    return interpolate.interp1d(frequencies, asd, bounds_error=False,
                                fill_value=1)


class DataError(Exception):
    """Base class for exceptions in this module."""


class EventData(utils.JSONMixin):
    """
    Class to save an event's frequency-domain strain data and whitening
    filter for multiple detectors.
    """

    @classmethod
    def from_timeseries(
            cls, filenames, eventname, detector_names, tgps,
            t_before=16., t_after=16., wht_filter_duration=32., fmin=15.,
            df_taper=1., fmax=1024., **kwargs):
        """
        Parameters
        ----------
        filenames : list of paths
            Paths pointing to ``gwpy.timeseries.Timeseries`` objects,
            containing data for each detector.

        eventname : str
            Name of the event, e.g. ``'GW150914'``.

        detector_names : sequence of str
            E.g. ``'HLV'`` or ``('H', 'L', 'V')`` for
            Hanford-Livingston-Virgo.

        tgps : float
            GPS time of the event (s).

        t_before, t_after : float
            Number of seconds of valid data (i.e. without edge effects)
            to keep before/after `tgps`. The total segment of data will
            have extra duration of ``wht_filter_duration / 2`` seconds
            to either side.

        wht_filter_duration : float
            Desired impulse response length of the whitening filter (s),
            will be ``wht_filter_duration / 2`` seconds to either side.
            Note: the whitening filter will only be approximately FIR.
            This is also the duration of each chunk in which the
            individual PSDs are measured for Welch, and the extent of
            the tapering in time-domain (s).

        fmin : float or sequence of floats
            Minimum frequency at which the whitening filter will have
            support (Hz). Multiple values can be passed, one for each
            detector.

        df_taper : float
            Whitening filter is highpassed. See ``highpass_filter``.

        fmax : float
            Desired Nyquist frequency (Hz), half the sampling frequency.

        **kwargs
            Keyword arguments to ``gwpy.timeseries.TimeSeries.read``.

        Returns
        -------
            ``EventData`` instance.

        See Also
        --------
        download_timeseries
        """
        if len(filenames) != len(detector_names):
            raise ValueError(
                'Length of `filenames` and `detector_names` are mismatched.')

        f_strain_whtfilter_tcoarses = []
        for filename, fmin_ in zip(*np.broadcast_arrays(filenames, fmin)):
            timeseries = cls._read_timeseries(filename, tgps, **kwargs)
            f_strain_whtfilter_tcoarses.append(
                cls._get_f_strain_whtfilter_from_timeseries(
                    timeseries, tgps, t_before, t_after, wht_filter_duration,
                    fmin_, df_taper, fmax))
        (frequencies, *f_copies), strain, wht_filter, (tcoarse, *t_copies) = (
            np.array(arr) for arr in zip(*f_strain_whtfilter_tcoarses))

        for f_copy in f_copies:
            np.testing.assert_array_equal(frequencies, f_copy)

        for t_copy in t_copies:
            np.testing.assert_equal(tcoarse, t_copy)

        return cls(eventname, frequencies, strain, wht_filter,
                   detector_names, tgps, tcoarse)

    @classmethod
    def gaussian_noise(
            cls, eventname, duration, detector_names, asd_funcs, tgps,
            tcoarse=None, fmin=15., df_taper=1., fmax=1024., seed=None):
        """
        Constructor that generates stationary colored Gaussian noise.

        Note: the data will be periodic.

        Parameters
        ----------
        eventname : str
            Name of event.

        duration : float
            Number of seconds of data.

        detector_names : sequence of str
            E.g. ``'HLV'`` or ``('H', 'L', 'V')`` for
            Hanford-Livingston-Virgo.

        asd_funcs : sequence of callables or strings
            Functions that return the noise amplitude spectral density
            (1/Hz), of the same length as `detector_names`. See
            ``make_asd_func`` to construct an interpolator.
            Alternatively, a string that is a key in ``ASDS`` can be
            passed to use a predefined ASD (e.g. 'asd_H_O3').

        tgps : float
            GPS time of event.

        tcoarse : float
            Time of event relative to beginning of data. Defaults to
            ``duration / 2``, the center of the segment.

        fmin : float
            Minimum frequency at which the whitening filter will have
            support (Hz). It is important for performance.

        df_taper : float
            Whitening filter is highpassed. See ``highpass_filter``.

        fmax : float
            Desired Nyquist frequency (Hz), half the sampling frequency.

        seed : int, optional
            Use some fixed value for reproducibility.

        Returns
        -------
        Instance of ``EventData``.

        See Also
        --------
        make_asd_func
        EventData.inject_signal
        """
        if len(detector_names) != len(asd_funcs):
            raise ValueError(
                'Lengths of `detector_names` and `asd_funcs` should match.')

        asd_funcs = list(asd_funcs)  # Ensure it is mutable
        for i, asd_func in enumerate(asd_funcs):
            if isinstance(asd_func, str):
                if not asd_func in ASDS:
                    raise ValueError(f'Unknown asd_func {asd_func!r}. '
                                     f'Allowed values are {list(ASDS)}')
                asd_funcs[i] = make_asd_func(*np.load(ASDS[asd_func]))

        tcoarse = duration / 2 if tcoarse is None else tcoarse
        dt = 1 / (2*fmax)
        frequencies = np.fft.rfftfreq(n=int(duration / dt), d=dt)
        asd = np.array([asd_func(frequencies) for asd_func in asd_funcs])

        real, imag = np.random.default_rng(seed).normal(
            scale=np.sqrt(duration) / 2 * asd, size=(2,) + asd.shape)
        strain = real + 1j * imag
        strain[:, [0, -1]] = strain[:, [0, -1]].real  # Real at f = 0 & Nyquist

        wht_filter = highpass_filter(frequencies, fmin, df_taper) / asd
        return cls(eventname, frequencies, strain, wht_filter, detector_names,
                   tgps, tcoarse)

    def __init__(self, eventname, frequencies, strain, wht_filter,
                 detector_names, tgps, tcoarse, injection=None):
        """
        Parameters
        ----------
        eventname : str
            Event name, e.g. ``'GW150914'``.

        frequencies : 1-d array
            Frequencies (Hz), as in ``numpy.fft.rfftfreq()``, uniform
            and starting at 0.

        strain : array of shape (ndet, nfreq)
            Frequency-domain strain data (1/Hz).

        wht_filter : array of shape (ndet, nfreq)
            Frequency-domain whitening filter. It is important that
            entries are 0 below some minimum frequency so waveforms
            don't need to be queried at arbitrarily low frequency.

        detector_names : sequence of str
            E.g. ``'HLV'`` or ``('H', 'L', 'V')`` for
            Hanford-Livingston-Virgo.

        tgps : float
            GPS time of the event (s).

        tcoarse : float
            Time of event relative to beginning of data.

        injection : dict, optional
            For bookkeeping, not actually used.
        """
        super().__init__()
        assert strain.shape[0] == len(detector_names)
        assert wht_filter.shape[0] == len(detector_names)
        assert strain.shape[1] == len(frequencies)
        assert wht_filter.shape[1] == len(frequencies)

        self.frequencies = frequencies
        np.testing.assert_allclose(self.df, np.diff(frequencies),
                                   err_msg='Irregular frequency grid.')
        np.testing.assert_equal(frequencies[0], 0,
                                err_msg='Frequency grid must start at 0')

        self.eventname = eventname
        self.detector_names = tuple(detector_names)
        self.tgps = tgps
        self.tcoarse = tcoarse
        self.wht_filter = wht_filter

        self.blued_strain = None  # Set by ``._set_strain()``
        self._set_strain(strain)

        nonzero = np.nonzero(np.sum(self.wht_filter, axis=0))[0]
        self.fslice = slice(nonzero[0], nonzero[-1] + 1)
        self.fbounds = self.frequencies[nonzero[[0, -1]]]

        self._check_wht_filter_validity()

        if injection:
            for key in {'d_h', 'h_h'} & injection.keys():
                injection[key] = np.asarray(injection[key])
        self.injection = injection

    def _set_strain(self, strain):
        self.strain = strain
        self.blued_strain = self.wht_filter**2 * self.strain

    @property
    def df(self):
        """Frequency resolution (Hz)."""
        return self.frequencies[1] - self.frequencies[0]

    @property
    def nfft(self):
        """Number of time-domain samples."""
        return 2 * (len(self.frequencies) - 1)

    @property
    def times(self):
        """
        Times of the data, starting at 0 (s) (event is at ``tcoarse``).
        """
        return np.linspace(0, 1/self.df, self.nfft, endpoint=False)

    @staticmethod
    def _read_timeseries(filename, tgps, **kwargs):
        """
        Return a ``gwpy.timeseries.TimeSeries``, cropped around
        the event to exclude any nans.
        """
        try:
            timeseries = gwpy.timeseries.TimeSeries.read(filename, **kwargs)
        except ValueError:
            kwargs['format'] = 'hdf5.gwosc'
            timeseries = gwpy.timeseries.TimeSeries.read(filename, **kwargs)

        i_event = np.searchsorted(timeseries.times.value, tgps)
        i_nan = np.where(np.isnan(timeseries.value))[0]

        if i_event in i_nan:
            raise DataError(f'{filename} has no data at event time.')

        i_start = 0
        if np.any(before := i_nan < i_event):
            i_start = np.max(i_nan[before]) + 1

        i_end = len(timeseries)
        if np.any(after := i_nan > i_event):
            i_end = np.min(i_nan[after]) - 1

        t_start = timeseries.times[i_start]
        t_end = timeseries.times[i_end - 1] + timeseries.dt

        if len(i_nan) > 0:
            print(f'Warning: keeping only {(t_end - t_start)} of valid data '
                  'near event.')

        timeseries = timeseries.crop(t_start, t_end)
        assert not any(np.isnan(timeseries))
        return timeseries

    @staticmethod
    def _get_f_strain_whtfilter_from_timeseries(
            timeseries: gwpy.timeseries.TimeSeries, tgps: float,
            t_before=16., t_after=16., wht_filter_duration=32.,
            fmin=15., df_taper=1., fmax=1024.):
        """
        Parameters
        ----------
        timeseries : `gwpy.timeseries.TimeSeries`
            Object containing single-detector data. Needs to be long
            enough to measure the PSD with the Welch method.

        tgps : float
            GPS time of event.

        t_before, t_after : float
            Number of seconds of valid data (i.e. without edge effects)
            to keep before/after `tgps`. The total segment of data will
            have extra duration of ``wht_filter_duration / 2`` seconds
            to either side.

        wht_filter_duration : float
            Desired impulse response length of the whitening filter (s).
            Note: the whitening filter will only be approximately FIR.
            This is also the duration of each chunk in which the
            individual PSDs are measured for Welch, and the extent of
            the tapering in time-domain (s).

        fmin : float
            Minimum frequency at which the whitening filter will have
            support (Hz). It is important for performance.

        df_taper : float
            Whitening filter is highpassed. See ``highpass_filter``.

        fmax : float
            Desired Nyquist frequency (Hz), half the sampling frequency.
        """
        if (wht_filter_duration / timeseries.dt.value) % 2 != 0:
            raise NotImplementedError(
                'Make `wht_filter_duration` an even multiple of `dt`.')

        timeseries = timeseries.detrend()

        segment_duration = wht_filter_duration + t_before + t_after
        tcoarse = wht_filter_duration / 2 + t_before
        t_start = tgps - tcoarse
        segment = timeseries.pad(int(segment_duration / timeseries.dt.value)
                                ).crop(t_start, t_start + segment_duration)

        rfftfreq = np.fft.rfftfreq(len(segment), timeseries.dt.value)
        i_max = np.searchsorted(rfftfreq, fmax) + 1
        rfftfreq_down = rfftfreq[:i_max]

        # Construct whitening filter
        measured_asd = timeseries.asd(wht_filter_duration,
                                      overlap=wht_filter_duration/2,
                                      method='median', window='hann')
        asd = measured_asd.interpolate(1 / segment_duration).value[:i_max]

        highpass = highpass_filter(rfftfreq_down, fmin, df_taper)
        raw_wht_filter_td = np.fft.irfft(highpass / asd)

        window_fir = signal.windows.tukey(
            int(wht_filter_duration * 2 * fmax), .1)
        window_fir_padded_shifted = np.fft.fftshift(
            np.pad(window_fir, (len(raw_wht_filter_td)-len(window_fir)) // 2))

        # Whitening filter will be exactly 0 below fmin,
        # approximately FIR, exactly zero phase:
        wht_filter = highpass * np.fft.rfft(
            window_fir_padded_shifted * raw_wht_filter_td).real

        # Taper and downsample data
        ntaper = int(wht_filter_duration / 2 / segment.dt.value)
        taper = np.sin(np.linspace(0, np.pi/2, ntaper))**2
        first_valid_ind, last_valid_ind = np.where(segment.value)[0][[0, -1]]
        i_event = np.searchsorted(segment.times.value, tgps)
        if (first_valid_ind + ntaper > i_event
                or last_valid_ind - ntaper < i_event):
            raise DataError('Event too close to the edge of valid data. '
                            'Consider reducing `wht_filter_duration`')

        segment[first_valid_ind : first_valid_ind+ntaper] *= taper
        segment[last_valid_ind-ntaper+1 : last_valid_ind+1] *= taper[::-1]

        data_fd = np.fft.rfft(segment.value)

        data_fd_down = data_fd[:i_max] * rfftfreq_down[-1] / rfftfreq[-1]
        data_fd_down[-1] = data_fd_down[-1].real

        # Multiply by dt because the rest of the code uses the
        # convention of the continuous Fourier transform:
        data_fd_down /= 2 * fmax

        return rfftfreq_down, data_fd_down, wht_filter, tcoarse

    def inject_signal(self, par_dic, approximant):
        """
        Add a signal to the data.

        Injection parameters will be stored as a dictionary in the
        ``injection`` attribute. The inner product ⟨h|h⟩ at each
        detector (ignoring ASD-drift correction) is also stored.
        The signal is computed only at frequencies where the whitening
        filter has support.

        Parameters
        ----------
        par_dic : dict
            Parameter values, keys should match
            ``waveform.WaveformGenerator.params``

        approximant : str
            Name of approximant.
        """
        if self.injection:
            logging.warning(
                'Injecting another signal in addition to a previous injection.'
                ' Injection metadata will be overwritten. '
                'Strain data will contain all the injected events.')

        waveform_generator = waveform.WaveformGenerator.from_event_data(
            self, approximant)
        h_f = np.zeros_like(self.strain)
        h_f[:, self.fslice] = waveform_generator.get_strain_at_detectors(
            self.frequencies[self.fslice], par_dic)
        self._set_strain(self.strain + h_f)

        h_h = 4 * self.df * np.linalg.norm(h_f * self.wht_filter, axis=-1)**2
        d_h = 4 * self.df * (self.blued_strain * h_f.conj()).real.sum(axis=-1)
        self.injection = {'par_dic': par_dic,
                          'approximant': approximant,
                          'd_h': d_h,
                          'h_h': h_h}

    def specgram(self, xlim=None, nfft=64, noverlap=None, vmax=25.):
        """
        Plot a spectrogram of the whitened data.

        The colorbar is in units of the expected power from Gaussian
        noise.

        Parameters
        ----------
        xlim : (float, float)
            Optional, time range to plot relative to time of the event.

        nfft : int
            Number of samples to use in each spectrum computation. Sets
            the frequency resolution.

        noverlap : int
            How many samples to overlap between adjacent spectra.
            Defaults to ``nfft / 2``.

        vmax : float
            Upper limit for the color scale.
        """
        noverlap = noverlap or nfft / 2

        f_sampling = 2 * self.fbounds[1]

        norm = plt.Normalize(0, vmax)
        fig, axes = plt.subplots(len(self.detector_names),
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': .1},
                                 squeeze=False)
        axes = axes[:, 0]
        for i, ax in enumerate(axes):
            wht_data_td = (np.fft.irfft(self.strain[i] * self.wht_filter[i])
                           * np.sqrt(2 * f_sampling))
            ax.specgram(wht_data_td * np.sqrt(f_sampling),
                        NFFT=nfft, noverlap=48, Fs=f_sampling,
                        xextent=(self.times[[0, -1]] - self.tcoarse),
                        scale='linear', norm=norm)
            ax.grid()
            ax.text(.02, .95, self.detector_names[i], ha='left', va='top',
                    transform=ax.transAxes, c='w')

        axes[0].set_title(self.eventname)
        minus_tgps = f' - {self.tgps}' if self.tgps != 0 else ''
        axes[-1].set_xlabel(rf'$t_{{\rm GPS}}{minus_tgps}$ (s)')
        axes[-1].set_xlim(xlim)

        plt.figtext(0., .5, 'Frequency (Hz)', rotation=90,
                    ha='left', va='center', fontsize='large')

        fig.colorbar(plt.cm.ScalarMappable(norm=norm), pad=.03,
                     ax=axes.tolist(), label=r'Power ($\sigma^2$)')

    def to_npz(self, *, filename=None, overwrite=False,
               permissions=0o644):
        """Save class as ``.npz`` file (by default in `DATADIR`)."""
        filename = pathlib.Path(filename or self.get_filename(self.eventname))
        if not overwrite and filename.exists():
            raise FileExistsError(f'{filename} already exists. '
                                  'Pass `overwrite=True` to overwrite.')

        dic = self.get_init_dict()
        dic['injection'] = json.dumps(dic['injection'], cls=utils.NumpyEncoder)
        np.savez(filename, **dic)
        filename.chmod(permissions)

    @classmethod
    def from_npz(cls, eventname=None, *, filename=None):
        """Load a `.npz` file previously saved with `to_npz()`."""
        if eventname:
            if filename:
                raise ValueError('Pass exactly one of `eventname`, `filename`')
            filename = cls.get_filename(eventname)
        dic = {key: val[()] for key, val in np.load(filename).items()}

        if 'injection' in dic:
            dic['injection'] = json.loads(dic['injection'])

        # Backward compatibility:
        if 'psd' in dic:
            assert 'wht_filter' not in dic
            dic['wht_filter'] = dic.pop('fd_filter') / np.sqrt(dic.pop('psd'))
        for deprecated_key in {'mchirp_range', 'q_min'} & dic.keys():
            del dic[deprecated_key]

        return cls(**dic)

    @staticmethod
    def get_filename(eventname=None):
        """Return npz filename to save/load class instance."""
        return DATADIR/f'{eventname}.npz'

    def _check_wht_filter_validity(self):
        """
        GWOSC data at 4096 Hz sampling rate has a downsampling artifact
        that manifests as an overly low PSD, i.e. high whitening filter,
        above 1620 Hz (see https://gwosc.org/yellow_box/).
        This method checks if the whitening filter looks suspicious, and
        issues a warning if yes.
        """
        if self.fbounds[1] < 1620:
            return

        ind_split = np.searchsorted(self.frequencies, 1620)
        wht_filter_pre, wht_filter_post = np.split(self.wht_filter,
                                                   [ind_split], axis=1)

        if np.linalg.norm(wht_filter_pre) < np.linalg.norm(wht_filter_post):
            logging.warning(
                'The whitening filter has most power at high frequencies. '
                'This is likely a downsampling artifact in the GWOSC data. '
                'Make sure the whitening filter is correct! '
                'If it is not, try downloading their 16384 Hz data, or using '
                'fmax <= 1620 Hz. See https://gwosc.org/yellow_box/ .'
                'To plot the whitening filter do:\n\t'
                'plt.plot(event_data.frequencies, event_data.wht_filter.T)\n'
                'To download 16384 Hz data do:\n\t'
                'cogwheel.data.download_timeseries('
                f'{self.eventname!r}, overwrite=True, sample_rate=2**14)')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.eventname})'


def highpass_filter(frequencies, fmin=15., df_taper=1.):
    """
    High-pass frequency domain filter with a sin^2 tapering.

    Parameters
    ----------
    frequencies : float 1d array
        Sorted, regularly spaced frequencies.

    fmin : float
        Minimum frequency, filter will be 0 below this.

    df_taper : float
        Width of the frequency band over which the filter goes to 1.

    Returns
    -------
    float array
        Same shape as `frequencies`, is 0 below `fmin` and 1 above
        `fmin + df_taper`.
    """
    fd_filter = np.ones(len(frequencies))
    i_1 = np.searchsorted(frequencies, fmin) - 1
    i_2 = np.searchsorted(frequencies, fmin + df_taper)
    fd_filter[:i_1] = 0.
    fd_filter[i_1 : i_2] = np.sin(np.linspace(0, np.pi/2, i_2-i_1))**2
    return fd_filter


def download_timeseries(eventname, outdir=None, tgps=None,
                        interval=(-2048, 2048), overwrite=False,
                        **kwargs):
    """
    Download data from GWOSC, save as hdf5 format that can be read by
    `gwpy.timeseries.Timeseries.read()`.

    Files are saved as ``'{det}_{eventname}.hdf5'``, e.g.
    ``'H_GW150914.hdf5'``.
    Return the paths to the files, the detector names and the GPS time;
    these are useful inputs to ``EventData.from_timeseries``.

    Parameters
    ----------
    eventname : str
        Name of event.

    outdir : path
        Directory into which to save the files. Defaults to
        ``GWOSC_FILES_DIR/eventname``.

    tgps : float, optional
        GPS time of event. Can be ``None`` if `eventname` is known to
        GWOSC.

    interval : (float, float)
        Start and end time relative to tgps (s).

    overwrite : bool
        If ``False``, will skip the download when a file already exists.

    **kwargs
        Passed to ``gwpy.timeseries.TimeSeries.fetch_open_data``.

    Returns
    -------
    filenames : list of pathlib.Path
        Paths to the hdf5 files with the downloaded data.

    detector_names : tuple of str
        E.g. ``('H', 'L', 'V')``.

    tgps : float
        GPS time of the event.
    """
    tgps = tgps or gwosc.datasets.event_gps(eventname)
    outdir = pathlib.Path(outdir or GWOSC_FILES_DIR/eventname)

    utils.mkdirs(outdir)

    filenames = []
    for detector_name in gw_utils.DETECTORS:
        path = outdir/f'{detector_name}_{eventname}.hdf5'
        if path.exists() and not overwrite:
            print(f'Skipping existing file {path.resolve()}')
            filenames.append(path)
            continue

        try:
            timeseries = _fetch_open_data(detector_name, tgps, interval,
                                          **kwargs)
        except ValueError:  # That detector has no data
            pass
        else:
            if not np.isnan(timeseries[np.searchsorted(timeseries.times.value,
                                                       tgps)]):
                if path.exists():
                    path.unlink()
                timeseries.write(path)
                filenames.append(path)

    detector_names = tuple(path.name[0] for path in filenames)
    return filenames, detector_names, tgps


def _fetch_open_data(detector_name, tgps, interval, **kwargs):
    """
    Similar to ``gwpy.timeseries.TimeSeries.fetch_open_data`` except
    that if it fails to find files for the whole requested time range,
    it will make an attempt to download a reduced range by restricting
    to the file containing the event. If that also fails, ValueError is
    raised.

    Parameters
    ----------
    detector_name : str
        Detector initial, e.g. 'H' for Hanford.

    tgps : float
        GPS time of event.

    interval : (float, float)
        Start and end time relative to tgps (s).

    **kwargs
        Passed to ``gwpy.timeseries.TimeSeries.fetch_open_data``.

    Returns
    -------
        gwpy.timeseries.TimeSeries
    """
    start, end = np.add(tgps, interval).astype(int)
    ifo = detector_name.removesuffix('1') + '1'
    try:
        timeseries = gwpy.timeseries.TimeSeries.fetch_open_data(
            ifo, start, end, **kwargs)
    except ValueError:
        # This detector has no data for the whole range.
        # Attempt downloading only the file that contains ``tgps``.
        # Assume file boundaries ocurr at multiples of 4096 s GPS time.
        t_lower = int(tgps) // 4096 * 4096
        t_upper = t_lower + 4096
        start, end = np.clip((start, end), t_lower, t_upper)
        timeseries = gwpy.timeseries.TimeSeries.fetch_open_data(
            ifo, start, end, **kwargs)

    return timeseries
