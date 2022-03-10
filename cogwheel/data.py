"""Store data about GW events."""

import pathlib
import numpy as np

from . import utils

DATADIR = pathlib.Path(__file__).parent/'data'


class EventData(utils.JSONMixin):
    """
    Class to save an event's frequency-domain strain data and psd for
    multiple detectors, as well as some metadata.
    """
    def __init__(self, eventname, frequencies, strain, psd,
                 detector_names, tgps, tcoarse, mchirp_range, q_min,
                 fd_filter=None):
        """
        Parameters
        ----------
        eventname: string, e.g. `'GW151914'`.
        frequencies: array of frequencies in Hz.
                     As in np.fft.rfftfreq(), uniform and starting at zero
        strain: ndet x nfreq array, frequency-domain strain data.
        psd: ndet x nfreq array, frequency-domain PSD.
        detector_names: string, e.g. `'HLV'`.
        tgps: float, GPS time of the event.
        tcoarse: float, time of event relative to beginning of data.
        mchirp_range: array with detector-frame chirp mass bounds.
        q_min: float, minimum mass ratio 0 < q_min <= 1.
        fd_filter: ndet x nfreq (or just nfreq) array, frequency-domain
                   filter to apply multiplicatively to the whitened data
                   and template. Defaults to `default_filter()` in all
                   detectors.
        """
        super().__init__()
        assert strain.shape[0] == len(detector_names)
        assert psd.shape[0] == len(detector_names)
        assert strain.shape[1] == len(frequencies)
        assert psd.shape[1] == len(frequencies)

        self.df = frequencies[1] - frequencies[0]
        np.testing.assert_allclose(self.df, np.diff(frequencies),
                                   err_msg='Irregular frequency grid.')
        np.testing.assert_equal(frequencies[0], 0,
                                err_msg=f'Frequency grid must start at 0')
        self.eventname = eventname
        self.frequencies = frequencies
        self.strain = strain
        self.psd = psd
        self.detector_names = detector_names
        self.tgps = tgps
        self.tcoarse = tcoarse
        #### EVDAT QUESTION: do we really need these in the instance?
        self.mchirp_range = mchirp_range
        #### EVDAT QUESTION: if so, should we take tc_range too?
        ## --> maybe mchirp is needed but I think qmin less than tc_range
        self.q_min = q_min

        self.nfft = 2 * (len(self.frequencies) - 1)
        self.t = np.linspace(0, 1/self.df, self.nfft, endpoint=False)
        self.dt = self.t[1] - self.t[0]

        self.fslice = None  # Set by fd_filter.setter
        self.blued_strain = None  # Set by fd_filter.setter
        self.wht_filter = None  # Set by fd_filter.setter
        self.fmin = None  # Set by fd_filter.setter
        self.fmax = None  # Set by fd_filter.setter
        self.fd_filter = fd_filter

    @property
    def fd_filter(self):
        """Frequency domain filter."""
        return self._fd_filter

    @fd_filter.setter
    def fd_filter(self, fd_filter):
        """
        Set a frequency domain filter to apply in addition to the
        whitening filter, e.g. to restrict frequency range.
        `fd_filter` is a ndet x nfrequencies array. If a 1d array is
        passed, it reshaped to 2d.
        The auxiliary attributes `fslice`, `blued_strain`, `wht_filter`
        are computed using this filter.
        """
        if fd_filter is None:
            fd_filter = default_filter(self.frequencies)
        fd_filter = np.atleast_2d(fd_filter)
        np.testing.assert_equal(fd_filter.shape[1], len(self.frequencies))

        self._fd_filter = fd_filter

        nonzero = np.nonzero(np.sum(self.fd_filter / self.psd, axis=0))[0]
        self.fslice = slice(nonzero[0], nonzero[-1] + 1)
        self.fmin = self.frequencies[nonzero[0]]
        self.fmax = self.frequencies[nonzero[-1]]

        self.wht_filter = self.fd_filter / np.sqrt(self.psd)
        self.blued_strain = self.wht_filter**2 * self.strain

    def to_npz(self, *, filename=None, overwrite=False,
               permissions=0o644):
        """Save class as `.npz` file (by default in `DATADIR`)."""
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
        dic = {key: val[()] for key, val in np.load(filename).iteritems()}
        return cls(**dic)

    @staticmethod
    def get_filename(eventname=None):
        """Return npz filename to save/load class instance."""
        return DATADIR/f'{eventname}.npz'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.eventname})'


def default_filter(frequencies, fmin=15., df_taper=1.):
    """
    High-pass frequency domain filter with a sin^2 tapering between
    `fmin` and `fmin + df_taper` [Hz].
    """
    fd_filter = np.ones(len(frequencies))
    i1 = np.searchsorted(frequencies, fmin) - 1
    i2 = np.searchsorted(frequencies, fmin + df_taper)
    fd_filter[:i1] = 0.
    fd_filter[i1 : i2] = np.sin(np.linspace(0, np.pi/2, i2-i1))**2
    return fd_filter
