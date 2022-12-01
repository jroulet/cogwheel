"""
Implement class ``SkyDictionary``, useful for marginalizing over sky
location.
"""

import collections
import itertools
import numpy as np
from scipy.stats import qmc

import lal

from cogwheel import gw_utils
from cogwheel import skyloc_angles
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

    """
    def __init__(self, detector_names, *, f_sampling: int = 2**13,
                 nsky: int = 10**6, seed=0):
        self.detector_names = tuple(detector_names)
        self.nsky = nsky
        self.f_sampling = f_sampling
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        
        self.sky_samples = self._create_sky_samples()
        self.fplus_fcross_0 = gw_utils.get_fplus_fcross_0(self.detector_names,
                                                          **self.sky_samples)
        geocenter_delays = gw_utils.get_geocenter_delays(
            self.detector_names, **self.sky_samples)
        self.geocenter_delay_first_det = geocenter_delays[0]
        self.delays = geocenter_delays[1:] - geocenter_delays[0]

        self.delays2inds_map = self._create_delays2inds_map()
        self.delays2genind_map = {
            delays_key: self._create_index_generator(inds)
            for delays_key, inds in self.delays2inds_map.items()}

    def _create_sky_samples(self):
        samples = {}
        u_lat, u_lon = qmc.Halton(2, seed=self._rng).random(self.nsky).T

        samples['lat'] = np.arcsin(2*u_lat - 1)
        samples['lon'] = 2 * np.pi * u_lon
        return samples

    def _create_delays2inds_map(self):
        # (ndet-1, nsky)
        delays_keys = zip(*np.rint(self.delays * self.f_sampling).astype(int))

        delays2inds_map = collections.defaultdict(list)
        for i_sample, delays_key in enumerate(delays_keys):
            delays2inds_map[delays_key].append(i_sample)

        return delays2inds_map

    def _create_index_generator(self, inds):
        delays_key_prior = (self.f_sampling**(len(self.detector_names) - 1)
                            * len(inds) / self.nsky)
        while True:
            for i_sample in inds:
                yield i_sample, delays_key_prior
