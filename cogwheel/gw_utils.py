"""Utility functions specific to gravitational waves."""
import scipy.interpolate
import numpy as np
import lal

from cogwheel import utils


# ----------------------------------------------------------------------
# Detector locations and responses:

DETECTORS = {'H': lal.CachedDetectors[lal.LHO_4K_DETECTOR],
             'L': lal.CachedDetectors[lal.LLO_4K_DETECTOR],
             'V': lal.CachedDetectors[lal.VIRGO_DETECTOR]}

DETECTOR_ARMS = {
    'H': (np.array([lal.LHO_4K_ARM_X_DIRECTION_X,
                    lal.LHO_4K_ARM_X_DIRECTION_Y,
                    lal.LHO_4K_ARM_X_DIRECTION_Z]),
          np.array([lal.LHO_4K_ARM_Y_DIRECTION_X,
                    lal.LHO_4K_ARM_Y_DIRECTION_Y,
                    lal.LHO_4K_ARM_Y_DIRECTION_Z])),
    'L': (np.array([lal.LLO_4K_ARM_X_DIRECTION_X,
                    lal.LLO_4K_ARM_X_DIRECTION_Y,
                    lal.LLO_4K_ARM_X_DIRECTION_Z]),
          np.array([lal.LLO_4K_ARM_Y_DIRECTION_X,
                    lal.LLO_4K_ARM_Y_DIRECTION_Y,
                    lal.LLO_4K_ARM_Y_DIRECTION_Z])),
    'V': (np.array([lal.VIRGO_ARM_X_DIRECTION_X,
                    lal.VIRGO_ARM_X_DIRECTION_Y,
                    lal.VIRGO_ARM_X_DIRECTION_Z]),
          np.array([lal.VIRGO_ARM_Y_DIRECTION_X,
                    lal.VIRGO_ARM_Y_DIRECTION_Y,
                    lal.VIRGO_ARM_Y_DIRECTION_Z]))}


EARTH_CROSSING_TIME = 2 * 0.02128  # 2 R_Earth / c (seconds)


@utils.lru_cache()
def fplus_fcross(detector_names, ra, dec, psi, tgps):
    """
    Return a (2 x n_detectors) array with F+, Fx antenna coefficients.
    Note: For caching, ``detector_names`` has to be hashable, e.g. a
    string like ``'HLV'`` or a tuple ``('H', 'L', 'V')`` but not a list.
    """
    gmst = lal.GreenwichMeanSiderealTime(tgps)  # [radians]
    return np.transpose([
        lal.ComputeDetAMResponse(DETECTORS[det].response, ra, dec, psi, gmst)
        for det in detector_names])


@utils.lru_cache()
def time_delay_from_geocenter(detector_names, ra, dec, tgps):
    """
    Return an array with delay times from Earth center [seconds].
    Note: For caching, ``detector_names`` has to be hashable, e.g. a
    string like ``'HLV'`` or a tuple ``('H', 'L', 'V')`` but not a list.
    """
    return np.array([
        lal.TimeDelayFromEarthCenter(DETECTORS[det].location, ra, dec, tgps)
        for det in detector_names])


@utils.lru_cache()
def detector_travel_times(detector_name_1, detector_name_2):
    """Return light travel time between two detectors [seconds]."""
    return np.linalg.norm(DETECTORS[detector_name_1].location
                          - DETECTORS[detector_name_2].location) / lal.C_SI


# ----------------------------------------------------------------------
# Similar to the above, but in Earth-fixed coordinates and vectorized:
def get_geocenter_delays(detector_names, lat, lon):
    """
    Return array of shape (n_detectors, ...) time delays from geocenter
    [s]. Vectorized over lat, lon.
    """
    locations = np.array([DETECTORS[detector_name].location
                          for detector_name in detector_names])  # (ndet, 3)
    #JM 08/11/22 prevert cyclic reference of gw_utils.py and skyloc_angles.py
    # direction = skyloc_angles.latlon_to_cart3d(lat, lon) #

    direction = np.array([np.cos(lon) * np.cos(lat),
                     np.sin(lon) * np.cos(lat),
                     np.sin(lat)])

    return -np.einsum('di,i...->d...', locations, direction) / lal.C_SI


def get_fplus_fcross_0(detector_names, lat, lon):
    """
    Return array with antenna response functions fplus, fcross with
    polarization psi=0.
    Vectorized over lat, lon. Return shape is (..., n_det, 2)
    where `...` is the shape of broadcasting (lat, lon).
    """
    responses = np.array([DETECTORS[detector_name].response
                          for detector_name in detector_names]
                        )  # (n_det, 3, 3)

    lat, lon = np.broadcast_arrays(lat, lon)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    x = np.array([sinlon, -coslon, np.zeros_like(sinlon)])  # (3, ...)
    dx = np.einsum('dij,j...->di...', responses, x)  # (n_det, 3, ...)

    y = np.array([-coslon * sinlat,
                  -sinlon * sinlat,
                  coslat])  # (3, ...)
    dy = np.einsum('dij,j...->di...', responses, y)

    fplus0 = (np.einsum('i...,di...->d...', x, dx)
              - np.einsum('i...,di...->d...', y, dy))
    fcross0 = (np.einsum('i...,di...->d...', x, dy)
               + np.einsum('i...,di...->d...', y, dx))

    return np.moveaxis([fplus0, fcross0], (0, 1), (-1, -2))


#-----------------------------------------------------------------------
# Coordinate transformations

def eta_to_q(eta):
    """q = m2/m1 as a function of eta = q / (1+q)**2."""
    return (1 - np.sqrt(1 - 4*eta) - 2*eta) / (2*eta)


def q_to_eta(q):
    """eta = q / (1+q)**2 as a function of q = m2/m1."""
    return q / (1+q)**2


def mchirpeta_to_m1m2(mchirp, eta):
    """Return `m1, m2` given `mchirp, eta`."""
    q = eta_to_q(eta)
    m1 = mchirp * (1 + q)**.2 / q**.6
    m2 = q * m1
    return m1, m2


def mchirpeta_to_mtot(mchirp, eta):
    """Return `mtot` given `mchirp, eta`."""
    return mchirp * eta**-.6


def m1m2_to_mchirp(m1, m2):
    """Return chirp mass given component masses."""
    return (m1*m2)**.6 / (m1+m2)**.2


def chieff(m1, m2, s1z, s2z):
    return (m1*s1z + m2*s2z) / (m1+m2)


class _ChirpMassRangeEstimator:
    """
    Class that implements a rough estimation of the chirp mass posterior
    support based on the best fit chirp mass value and the signal to
    noise ratio.
    Intended for setting ranges for sampling or maximization.
    Keep in mind this is very approximate, check your results
    responsibly.
    """
    def __init__(self, mchirp_0=60):
        self.mchirp_0 = mchirp_0
        mchirp_grid = np.geomspace(.1, 1000, 10000)
        x_grid = np.vectorize(self._x_of_mchirp)(mchirp_grid)
        self._mchirp_of_x = scipy.interpolate.interp1d(x_grid, mchirp_grid)

    def _x_of_mchirp(self, mchirp):
        """
        Chirp-mass reparametrization in which the uncertainty
        is approximately homogeneous.
        """
        if mchirp > self.mchirp_0:
            return mchirp
        return (-3/5 * self.mchirp_0**(8/3) * mchirp**(-5/3)
                + 8/5 * self.mchirp_0)

    def __call__(self, mchirp, sigmas=5, snr=8):
        """
        Return an array with the minimum and maximum estimated
        values of mchirp with posterior support.
        Intended for setting ranges for sampling or maximization.
        Keep in mind this is very approximate, check your results
        responsibly.

        Parameters
        ----------
        mchirp: Estimate of the center of the mchirp distribution.
        sigmas: How big (conservative) to make the range compared
                to the expected width of the distribution.
        snr: Signal-to-noise ratio, lower values give bigger ranges.
        """
        x = self._x_of_mchirp(mchirp)
        dx = 200. * sigmas / snr
        mchirp_min = self._mchirp_of_x(x - dx)
        mchirp_max = self._mchirp_of_x(x + dx)
        return np.array([mchirp_min, mchirp_max])

    def expand_range(self, mchirp_0, mchirp_bound, factor=2):
        """
        Auxiliary function to adjust one of the edges of the chirp mass
        range.
        Return a new value for the edge of the mchirp interval that is
        expanded by `factor` (in terms of a reparametrization of chirp
        mass in which the uncertainty is approximately homogeneous).
        For example, if one decides that (say) the lower end of the
        chirp mass range is overestimated, one can use this function to
        get a more conservative (lower) value.

        Parameters
        ----------
        mchirp_0: float
            A central value of the chirp mass range (Msun), typically
            the `mchirp` argument passed to ``__call__``.

        mchirp_bound: float
            Value of the chirp mass bound that we want to adjust (Msun).

        factor: float
            By how much to change `mchirp_bound` away from `mchirp_0`.
            A value larger than 1 expands the range.
        """
        x_0 = self._x_of_mchirp(mchirp_0)
        x_bound = self._x_of_mchirp(mchirp_bound)
        new_x_bound = x_0 + factor * (x_bound - x_0)
        return self._mchirp_of_x(new_x_bound)[()]


estimate_mchirp_range = _ChirpMassRangeEstimator()
