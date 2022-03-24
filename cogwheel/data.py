"""Download, process and store data about GW events."""

import pathlib
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import gwpy.timeseries
import gwosc

from cogwheel import utils
from cogwheel import gw_utils

DATADIR = pathlib.Path(__file__).parent/'data'
GWOSC_FILES_DIR = DATADIR/'gwosc_files'


class DataError(Exception):
    """Base class for exceptions in this module."""


class EventData(utils.JSONMixin):
    """
    Class to save an event's frequency-domain strain data and whitening
    filter for multiple detectors.
    """
    def __init__(self, eventname, frequencies, strain, wht_filter,
                 detector_names, tgps, tcoarse):
        """
        Parameters
        ----------
        eventname: str
            Event name, e.g. ``'GW151914'``.

        frequencies: 1-d array
            Frequencies (Hz), as in np.fft.rfftfreq(), uniform and
            starting at 0.

        strain: array of shape (ndet, nfreq)
            Frequency-domain strain data (1/Hz).

        wht_filter: array of shape (ndet, nfreq)
            Frequency-domain whitening filter. It is important that
            entries are 0 below some minimum frequency so waveforms
            don't need to be queried at arbitrarily low frequency.

        detector_names: string
            Detectors' initials, e.g. ``'HLV'`` for Hanford-Livingston-
            Virgo.

        tgps: float
            GPS time of the event (s).

        tcoarse: float
            Time of event relative to beginning of data.
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
        self.detector_names = detector_names
        self.tgps = tgps
        self.tcoarse = tcoarse
        self.strain = strain
        self.wht_filter = wht_filter
        self.blued_strain = self.wht_filter**2 * self.strain

        nonzero = np.nonzero(np.sum(self.wht_filter, axis=0))[0]
        self.fslice = slice(nonzero[0], nonzero[-1] + 1)
        self.fbounds = self.frequencies[nonzero[[0, -1]]]

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
        """Times of the data, starting at 0 (s) (event is at ``tcoarse``)."""
        return np.linspace(0, 1/self.df, self.nfft, endpoint=False)

    @classmethod
    def from_timeseries(
            cls, filenames, eventname, detector_names, tgps,
            signal_duration=16., wht_filter_duration=32., fmin=15.,
            df_taper=1., fmax=1024.):
        """
        Parameters
        ----------
        filenames: list of paths
            Paths pointing to ``gwpy.timeseries.Timeseries`` objects,
            containing data for each detector.

        eventname: str
            Name of the event, e.g. ``'GW150914'``.

        detector_names: string
            Detectors' initials, e.g. ``'HLV'`` for Hanford-Livingston-
            Virgo.

        tgps: float
            GPS time of the event (s).

        signal_duration: float
            Estimated lower bound for the duration of the signal (s).

        wht_filter_duration: float
            Desired impulse response length of the whitening filter (s).
            Note: the whitening filter will only be approximately FIR.
            This is also the duration of each chunk in which the
            individual PSDs are measured for Welch, and the extent of
            the tapering in time-domain (s).

        fmin: float or sequence of floats
            Minimum frequency at which the whitening filter will have
            support (Hz). It is important for performance. Multiple
            values can be passed, one for each detector.

        df_taper: float
            Whitening filter is highpassed. See ``highpass_filter``.

        fmax: float
            Desired Nyquist frequency (Hz), half the sampling frequency.

        Return
        ------
            ``EventData`` instance.
        """
        if len(filenames) != len(detector_names):
            raise ValueError(
                'Length of `filenames` and `detector_names` are mismatched.')

        f_strain_whtfilter_tcoarses = []
        for filename, fmin_ in np.transpose(np.broadcast_arrays(filenames,
                                                                fmin)):
            timeseries = cls._read_timeseries(filename, tgps)
            f_strain_whtfilter_tcoarses.append(
                cls._get_f_strain_whtfilter_from_timeseries(
                    timeseries, tgps, signal_duration, wht_filter_duration,
                    fmin_, df_taper, fmax))
        (frequencies, *f_copies), strain, wht_filter, (tcoarse, *t_copies) = (
            np.array(arr) for arr in zip(*f_strain_whtfilter_tcoarses))

        for f_copy in f_copies:
            np.testing.assert_array_equal(frequencies, f_copy)

        for t_copy in t_copies:
            np.testing.assert_equal(tcoarse, t_copy)

        return cls(eventname, frequencies, strain, wht_filter,
                   detector_names, tgps, tcoarse)

    @staticmethod
    def _read_timeseries(filename, tgps):
        timeseries = gwpy.timeseries.TimeSeries.read(filename)

        i_event = np.searchsorted(timeseries.times.value, tgps)
        i_nan = np.where(np.isnan(timeseries.value))[0]

        i_start = 0
        if np.any(before := (i_nan < i_event)):
            i_start = np.max(i_nan[before]) + 1

        i_end = len(timeseries)
        if np.any(after := (i_nan > i_event)):
            i_end = np.min(i_nan[after]) - 1

        t_start = timeseries.times[i_start]
        t_end = timeseries.times[i_end - 1] + timeseries.dt

        if len(i_nan) > 0:
            print(f'Warning: keeping only {(t_end - t_start)} of valid data '
                  'near event.')

        timeseries = timeseries.crop(t_start, t_end)
        assert not any(np.isnan(timeseries))
        return timeseries

    @classmethod
    def _get_f_strain_whtfilter_from_timeseries(
            cls, timeseries: gwpy.timeseries.TimeSeries, tgps: float,
            signal_duration=15., wht_filter_duration=16.,
            fmin=15., df_taper=1., fmax=1024.):
        """
        Parameters
        ----------
        timeseries: `gwpy.timeseries.TimeSeries`
            Object containing single-detector data. Needs to be long
            enough to measure the PSD with the Welch method.

        tgps: float, GPS time of event.

        signal_duration: float
            Estimated lower bound for the duration of the signal (s).

        wht_filter_duration: float
            Desired impulse response length of the whitening filter (s).
            Note: the whitening filter will only be approximately FIR.
            This is also the duration of each chunk in which the
            individual PSDs are measured for Welch, and the extent of
            the tapering in time-domain (s).

        fmin: float
            Minimum frequency at which the whitening filter will have
            support (Hz). It is important for performance.

        df_taper: float
            Whitening filter is highpassed. See ``highpass_filter``.

        fmax: float
            Desired Nyquist frequency (Hz), half the sampling frequency.
        """
        if (wht_filter_duration / timeseries.dt.value) % 2 != 0:
            raise NotImplementedError(
                'Make `wht_filter_duration` an even multiple of `dt`.')

        timeseries = timeseries.detrend()

        segment_duration = utils.next_pow_2(signal_duration
                                            + 2 * wht_filter_duration)
        tcoarse = (segment_duration + signal_duration) / 2
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

        window_fir = signal.tukey(int(wht_filter_duration * 2 * fmax), .1)
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

        segment[first_valid_ind : first_valid_ind + ntaper] *= taper
        segment[last_valid_ind - ntaper + 1 : last_valid_ind + 1] *= taper[::-1]

        data_fd = np.fft.rfft(segment.value)

        data_fd_down = data_fd[:i_max] * rfftfreq_down[-1] / rfftfreq[-1]
#         data_fd_down[-1] = data_fd_down[-1].real

        # Multiply by dt because the rest of the code uses the
        # convention of the continuous Fourier transform:
        data_fd_down /= 2 * fmax

        return rfftfreq_down, data_fd_down, wht_filter, tcoarse

    def specgram(self, xlim=None, nfft=64, noverlap=None, vmax=25.):
        """
        Plot a spectrogram of the whitened data in units of the expected
        power from Gaussian noise.

        Parameters
        ----------
        xlim: (float, float)
            Optional, time range to plot relative to time of the event.

        nfft: int
            Number of samples to use in each spectrum computation. Sets
            the frequency resolution.

        noverlap: int
            How many samples to overlap between adjacent spectra.
            Defaults to ``nfft / 2``.

        vmax: float
            Upper limit for the color scale.
        """
        noverlap = noverlap or nfft / 2

        f_sampling = 2 * self.fbounds[1]

        norm = plt.Normalize(0, vmax)
        fig, axes = plt.subplots(len(self.detector_names),
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': .1, 'wspace': .1})
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
        axes[-1].set_xlabel(rf'$t_{{\rm GPS}} - {self.tgps}$ (s)')
        axes[-1].set_xlim(xlim)

        plt.figtext(0., .5, 'Frequency (Hz)', rotation=90,
                    ha='left', va='center', fontsize='large')

        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm), pad=.03,
                                ax=axes.tolist(), label=r'Power ($\sigma^2$)')
        colorbar.ax.grid()

    def to_npz(self, *, filename=None, overwrite=False,
               permissions=0o644):
        """Save class as ``.npz`` file (by default in `DATADIR`)."""
        filename = pathlib.Path(filename or self.get_filename(self.eventname))
        if not overwrite and filename.exists():
            raise FileExistsError(f'{filename} already exists. '
                                  'Pass `overwrite=True` to overwrite.')
        np.savez(filename, **self.get_init_dict())
        filename.chmod(permissions)

    @classmethod
    def from_npz(cls, eventname=None, *, filename=None):
        """Load a `.npz` file previously saved with `to_npz()`."""
        if eventname:
            if filename:
                raise ValueError('Pass exactly one of `eventname`, `filename`')
            filename = cls.get_filename(eventname)
        dic = {key: val[()] for key, val in np.load(filename).items()}

        return cls(**dic)

    @staticmethod
    def get_filename(eventname=None):
        """Return npz filename to save/load class instance."""
        return DATADIR/f'{eventname}.npz'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.eventname})'


def highpass_filter(frequencies, fmin=15., df_taper=1.):
    """
    High-pass frequency domain filter with a sin^2 tapering between
    `fmin` and `fmin + df_taper` [Hz].
    """
    fd_filter = np.ones(len(frequencies))
    i_1 = np.searchsorted(frequencies, fmin) - 1
    i_2 = np.searchsorted(frequencies, fmin + df_taper)
    fd_filter[:i_1] = 0.
    fd_filter[i_1 : i_2] = np.sin(np.linspace(0, np.pi/2, i_2-i_1))**2
    return fd_filter


def download_timeseries(eventname, outdir=None, tgps=None,
                        interval=(-2048, 2048), overwrite=False):
    """
    Download data from gwosc, save as hdf5 format that can be read by
    `gwpy.timeseries.Timeseries.read()`.
    Files are saved as ``'{det}_{eventname}.hdf5'``, e.g.
    ``'H_GW150914.hdf5'``.

    Parameters
    ----------
    eventname: str
        Name of event.

    outdir: path
        Directory into which to save the files. Defaults to
        GWOSC_FILES_DIR/eventname

    tgps: float
        GPS time of event.

    interval: (float, float)
        Start and end time relative to tgps (s).

    overwrite: bool
        If ``False``, will skip the download when a file already exists.
    """
    tgps = tgps or gwosc.datasets.event_gps(eventname)
    outdir = pathlib.Path(outdir or GWOSC_FILES_DIR/eventname)

    utils.mkdirs(outdir)

    for det in gw_utils.DETECTORS:
        path = outdir/f'{det}_{eventname}.hdf5'
        if path.exists() and not overwrite:
            print(f'Skipping existing file {path.resolve()}')
            continue

        try:
            timeseries = gwpy.timeseries.TimeSeries.fetch_open_data(
                f'{det}1', *np.add(tgps, interval).astype(int))
        except ValueError:  # That detector has no data
            pass
        else:
            timeseries.write(path)
