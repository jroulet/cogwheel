"""
Implement class ``SkyDictionary``, useful for marginalizing over sky
location.
"""
import collections
import itertools
import numpy as np
import scipy.signal
from scipy.stats import qmc

from cogwheel import gw_utils
from cogwheel import utils


class SkyDictionary(utils.JSONMixin):
    """
    Given a network of detectors, this class generates a set of
    samples covering the sky location isotropically in Earth-fixed
    coordinates (lat, lon).
    The samples are assigned to bins based on the arrival-time delays
    between detectors. This information is accessible as dictionaries
    ``delays2inds_map``, ``delays2genind_map``.
    Antenna coefficients F+, Fx (psi=0) and detector time delays from
    geocenter are computed and stored for all samples.

    Public attributes
    -----------------
    detector_names: tuple
        E.g. ``('H', 'L', 'V')``.

    nsky: int
        Number of sky-location samples.

    f_sampling: float
        Inverse of the time bin size at each detector (Hz).

    seed:
        Sets the initial random state.

    sky_samples: dict
        Contains entries for 'lat' and 'lon' (rad) of the sky samples.

    fplus_fcross_0: (nsky, n_det, 2) float array
        Antenna coefficients at the sky samples for psi=0.

    geocenter_delay_first_det: (nsky,) float array
        Time delay (s) from geocenter to the first detector (per
        ``.detector_names``) for each sky sample.

    delays: (n_det - 1, nsky) float array
        Time delay (s) from the first detector (per ``.detector_names``)
        to the remaining detectors, for each sky sample.

    delays2inds_map: dict
        Each key is a tuple of (n_det-1) ints, corresponding to
        discretized delays from the first detector to the remaining
        detectors.
        Each value is a list of ints, corresponding to indices of
        sky-samples that have the correct delay (to the resolution given
        by ``f_sampling``).

    Public methods
    --------------
    resample_timeseries
    get_sky_inds_and_prior
    choose_f_sampling
    """
    DEFAULT_F_SAMPLING = 2**13

    def __init__(self, detector_names, *,
                 f_sampling: int = DEFAULT_F_SAMPLING,
                 nsky: int = 10**6, seed=0):
        self.detector_names = tuple(detector_names)
        self.nsky = nsky
        self.f_sampling = f_sampling
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.sky_samples = self._create_sky_samples()
        self.fplus_fcross_0 = gw_utils.get_fplus_fcross_0(
            self.detector_names, **self.sky_samples).astype(np.float32)
        geocenter_delays = gw_utils.get_geocenter_delays(
            self.detector_names, **self.sky_samples)
        self.geocenter_delay_first_det = geocenter_delays[0]
        self.delays = geocenter_delays[1:] - geocenter_delays[0]

        self.delays2inds_map = self._create_delays2inds_map()

        discrete_delays = np.array(list(self.delays2inds_map))

        if len(self.detector_names) == 1:
            self._min_delay = np.array([])
            self._max_delay = np.array([])
        else:
            discrete_delays = np.array(list(self.delays2inds_map))
            self._min_delay = np.min(discrete_delays, axis=0)
            self._max_delay = np.max(discrete_delays, axis=0)

        self._delays_bounds = {}
        self._delays_prior = {}
        weights = np.linalg.norm(self.fplus_fcross_0, axis=(1, 2)) ** 3
        for n_detectors in range(2, len(detector_names) + 1):
            for order in itertools.permutations(range(len(detector_names)),
                                                n_detectors):
                pairwise_delays = self._discretize(
                    geocenter_delays[order[1:],] - geocenter_delays[order[0]])

                bins = [np.arange(*bounds) + .5
                        for bounds in zip(pairwise_delays.min(axis=1) - 1,
                                          pairwise_delays.max(axis=1) + 1)]
                self._delays_prior[order] = np.histogramdd(
                    pairwise_delays.T, weights=weights, bins=bins)[0].copy('F')

                if n_detectors == 2:
                    self._delays_bounds[order] = (pairwise_delays.min(),
                                                  pairwise_delays.max())

        # (n_det-1)-dimensional array of generators yielding sky-indices
        self._ind_generators = np.full(self._max_delay - self._min_delay + 1,
                                       iter(()))
        self.set_generators()

        # (n_det-1)-dimensional float array with dt * d(Omega / 4pi)
        self._sky_prior = np.zeros(self._max_delay - self._min_delay + 1)
        for key, inds in self.delays2inds_map.items():
            self._sky_prior[key] = len(inds) / self.nsky / self.f_sampling

    def set_generators(self):
        """
        Set attribute ``self._ind_generators``, this method can be
        called after instantiation to make the method
        ``.get_sky_inds_and_prior`` have reproducible output.
        """
        for key, inds in self.delays2inds_map.items():
            self._ind_generators[key] = itertools.cycle(inds)

    def resample_timeseries(self, timeseries, times, axis=-1):
        """
        Resample a timeseries to match the SkyDict's sampling frequency.
        The sampling frequencies of the SkyDict and ``timeseries`` must
        be multiples (or ``ValueError`` is raised).
        The data is smoothed at the edges to make it periodic.

        Parameters
        ----------
        timeseries: array_like
            The data to resample.

        times: array_like
             Equally-spaced sample positions associated with the signal
             data in `timeseries`.

        axis: int
            The axis of timeseries that is resampled. Default is -1.

        Return
        ------
        resampled_timeseries, resampled_times
            A tuple containing the resampled array and the corresponding
            resampled positions.
        """
        shape = [1 for _ in timeseries.shape]
        shape[axis] = timeseries.shape[axis]
        window = np.sqrt(
            scipy.signal.windows.tukey(shape[axis], .1)).reshape(shape)
        endpoint = np.mean(timeseries.take([0, -1], axis))
        timeseries = (timeseries - endpoint) * window + endpoint

        fs_ratio = np.round(self.f_sampling * (times[1] - times[0]),
                            decimals=10)  # Prevent machine precision problems
        if fs_ratio != 1:
            timeseries, times = scipy.signal.resample(
                timeseries, int(len(times) * fs_ratio), times, axis=axis)
            if not np.isclose(1 / self.f_sampling, times[1] - times[0]):
                raise ValueError(
                    '`times` is incommensurate with `f_sampling`.')

        return timeseries, times

    def get_sky_inds_and_prior(self, delays):
        """
        Parameters
        ----------
        delays: int array of shape (n_det-1, n_samples)
            Time-of-arrival delays in units of 1 / self.f_sampling

        Return
        ------
        sky_inds: tuple of ints of length n_physical
            Indices of self.sky_samples with the correct time delays.

        sky_prior: float array of length n_physical
            Prior probability density for the time-delays (s).

        physical_mask: boolean array of length n_samples
            Some choices of time of arrival at detectors may not
            correspond to any physical sky location, these are flagged
            ``False`` in this array. Unphysical samples are discarded.
        """
        assert len(delays) == len(self.detector_names) - 1

        if len(self.detector_names) == 1:  # Single-detector case
            n_samples = delays.shape[1]
            physical_mask = np.full(n_samples, True)
            sky_inds = np.fromiter(
                itertools.islice(self._ind_generators[()], n_samples),
                int, n_samples)
            sky_prior = np.full(n_samples, self._sky_prior)
            return sky_inds, sky_prior, physical_mask

        # First mask: are individual delays plausible? This is necessary
        # in order to interpret the delays as indices to self._sky_prior
        physical_mask = np.all((delays.T >= self._min_delay)
                               & (delays.T <= self._max_delay), axis=1)

        # Submask: for the delays that survive the first mask, are there
        # any sky samples with the correct delays at all detector pairs?
        sky_prior = self._sky_prior[tuple(delays[:, physical_mask])]
        submask = sky_prior > 0

        physical_mask[physical_mask] *= submask
        sky_prior = sky_prior[submask]

        # Generate sky samples for the physical delays
        generators = self._ind_generators[tuple(delays[:, physical_mask])]
        sky_inds = np.fromiter(map(next, generators), int, len(generators))
        return sky_inds, sky_prior, physical_mask

    @classmethod
    def choose_f_sampling(cls, f_nyquist: int) -> int:
        """
        Return closest frequency to the default f_sampling that still
        makes it an integer multiple of the sampling frequency
        corresponding to `f_nyquist` (Hz). The output of this function
        can be used as ``f_sampling`` parameter in ``__init__``.
        """
        fmax = 2 * f_nyquist
        return int(fmax * np.round(cls.DEFAULT_F_SAMPLING / fmax))

    def _create_sky_samples(self):
        """
        Return a dictionary of samples in terms of 'lat' and 'lon' drawn
        isotropically by means of a Quasi Monte Carlo (Halton) sequence.
        """
        u_lat, u_lon = qmc.Halton(2, seed=self._rng).random(self.nsky).T

        return {'lat': np.arcsin(2*u_lat - 1),
                'lon': 2 * np.pi * u_lon}

    def _create_delays2inds_map(self):
        """
        Return a dictionary mapping arrival time delays to sky-sample
        indices.
        Its keys are tuples of ints of length (n_det - 1), with time
        delays to the first detector in units of 1/self.f_sampling.
        Its values are list of indices to ``self.sky_samples`` of
        samples that have the corresponding (discretized) time delays.
        """
        if len(self.detector_names) == 1:
            return {(): list(range(self.nsky))}

        # (ndet-1, nsky)
        delays_keys = zip(*self._discretize(self.delays))

        delays2inds_map = collections.defaultdict(list)
        for i_sample, delays_key in enumerate(delays_keys):
            delays2inds_map[delays_key].append(i_sample)

        return delays2inds_map

    def _discretize(self, delays: float) -> int:
        """Return discretized delay in units of 1/self.f_sampling."""
        return np.rint(delays * self.f_sampling).astype(int)

    def apply_tdet_prior(self, t_arrival_lnprob):
        """
        Change `t_arrival_lnprob` inplace to account for the
        astrophysical prior for time delays.

        Parameters
        ----------
        t_arrival_lnprob: (n_det, n_times) float array
            Incoherent proposal for log probability of arrival times at
            each detector.
        """
        # Sort detectors by SNR
        det_order = tuple(np.argsort(t_arrival_lnprob.max(axis=1)))[::-1]

        if len(det_order) > 1:
            # Apply prior at 2nd detector
            prob_t0 = utils.exp_normalize(t_arrival_lnprob[det_order[0]])
            t_arrival_lnprob[det_order[1]] += self._second_detector_lnprior(
                det_order, prob_t0)

        if len(det_order) > 2:
            # Apply prior at 3rd detector
            prob_t1 = utils.exp_normalize(t_arrival_lnprob[det_order[1]])
            t_arrival_lnprob[det_order[2]] += self._third_detector_lnprior(
                det_order, prob_t0, prob_t1)

        # Note: update code when there are > 3 detectors, as it is now
        # it will work but may be inefficient if the 4th detector has
        # low SNR.

    def _second_detector_lnprior(self, det_order, prob_t0):
        """
        Log of prior probability of time of arrival at the second
        detector (by `det_order`) given the probability of arrival at
        the first detector.
        """
        prior_t1 = scipy.signal.convolve(
            prob_t0, self._delays_prior[det_order[:2]], 'same')
        return np.log(prior_t1 + 1e-10)

    def _third_detector_lnprior(self, det_order, prob_t0, prob_t1):
        """
        Log of prior probability of time of arrival at the third
        detector (by `det_order`) given the probability of arrival at
        the first two detectors.
        """
        prob_dt_01 = scipy.signal.correlate(prob_t1, prob_t0, 'same')
        dt_01 = scipy.signal.correlation_lags(len(prob_t1), len(prob_t0),
                                              'same')
        min_delay, max_delay = self._delays_bounds[det_order[:2]]
        mask = (dt_01 >= min_delay) & (dt_01 <= max_delay)
        prior_dt02 = (prob_dt_01[mask]
                      @ self._delays_prior[det_order[:3]])
        prior_t2 = scipy.signal.convolve(prob_t0, prior_dt02, 'same')
        return np.log(prior_t2 + 1e-10)
