"""Utility functions specific to gravitational waves."""
import scipy.interpolate
import numpy as np

import lal

from cogwheel import grid

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


def fplus_fcross(detector_names, ra, dec, psi, tgps):
    """
    Return a (2 x n_detectors) array with F+, Fx antenna coefficients.
    """
    gmst = lal.GreenwichMeanSiderealTime(tgps)  # [radians]
    return np.transpose([
        lal.ComputeDetAMResponse(DETECTORS[det].response, ra, dec, psi, gmst)
        for det in detector_names])


def fplus_fcross_detector(detector_name, ra, dec, psi, tgps):
    """
    Return a (2 x n_angles) array with F+, Fx for many angles at one detector.
    ra, dec, psi, tgps can be scalars or arrays, and any with length > 1 must
     share the same length.all have length n_angles, tgps can be scalar or array
    """
    if hasattr(tgps, '__len__'):
        gmst = [lal.GreenwichMeanSiderealTime(t) for t in tgps]
    else:
        gmst = lal.GreenwichMeanSiderealTime(tgps)
    return np.transpose([
        lal.ComputeDetAMResponse(DETECTORS[detector_name].response, r, d, p, g)
        for r, d, p, g in np.broadcast(ra, dec, psi, gmst)])


def time_delay_from_geocenter(detector_names, ra, dec, tgps):
    """Return an array with delay times from Earth center [seconds]."""
    return np.array([
        lal.TimeDelayFromEarthCenter(DETECTORS[det].location, ra, dec, tgps)
        for det in detector_names])


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


def mchirp(m1, m2):
    """Return chirp mass given component masses."""
    return (m1*m2)**.6 / (m1+m2)**.2


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
        return self._mchirp_of_x(new_x_bound)


estimate_mchirp_range = _ChirpMassRangeEstimator()


# ----------------------------------------------------------------------
# Latex formatting

_LABELS = {
    # Mass
    'mchirp': r'$\mathcal{M}^{\rm det}$',
    'lnq': r'$\ln q$',
    'q': r'$q$',
    'eta': r'$\eta$',
    'm1': r'$m_1^{\rm det}$',
    'm2': r'$m_2^{\rm det}$',
    'mtot': r'$M_{\rm tot}^{\rm det}$',
    'm1_source': r'$m_1^{\rm src}$',
    'm2_source': r'$m_2^{\rm src}$',
    'mtot_source': r'$M_{\rm tot}^{\rm src}$',
    'mchirp_source': r'$\mathcal{M}^{\rm src}$',
    # Spin
    'chieff': r'$\chi_{\rm eff}$',
    'cumchidiff': r'$\int \pi (\chi_{\rm diff})$',
    's1phi_hat': r'$\hat{\phi}_{s1}$',
    's2phi_hat': r'$\hat{\phi}_{s2}$',
    'cums1r_s1z': r'$C_1^\perp$',
    'cums2r_s2z': r'$C_2^\perp$',
    's1x': r'$s_{1x}$',
    's1y': r'$s_{1y}$',
    's1z': r'$s_{1z}$',
    's2x': r'$s_{2x}$',
    's2y': r'$s_{2y}$',
    's2z': r'$s_{2z}$',
    's1': r'$|s_1|$',
    's2': r'$|s_2|$',
    's1theta': r'$\theta_{s1}$',
    's2theta': r'$\theta_{s2}$',
    's1r': r'$s_1^\perp$',
    's2r': r'$s_2^\perp$',
    's1phi': r'$\phi_{s1}$',
    's2phi': r'$\phi_{s2}$',
    'chip': r'$\chi_p$',
    # Distance
    'd_hat': r'$\hat{D}$',
    'd_luminosity': r'$D_L$',
    'z': r'$z$',
    # Orientation
    'phi_ref': r'$\phi_{\rm ref}$',
    'psi_hat': r'$\hat{\psi}$',
    'psi': r'$\psi$',
    'cosiota': r'$\cos \iota$',
    'iota': r'$\iota$',
    'thetaJN': r'$\theta_{JN}$',
    'costheta_jn': r'$\cos \theta_{JN}$',
    'phi_jl': r'$\phi_{JL}$',
    'phi12': r'$\phi_{12}$',
    # Location
    'costhetanet': r'$\cos \theta_{\rm net}$',
    'phinet_hat': r'$\hat{\phi}_{\rm net}$',
    'ra': r'$\alpha$',
    'dec': r'$\delta$',
    'thetanet': r'$\theta_{\rm net}$',
    'phinet': r'$\phi_{\rm net}$',
    # Time
    't_refdet': r'$t_{\rm ref\,det}$',
    'tc': r'$t_c$',
    't_geocenter': r'$t_{\rm geocenter}$',
    # Likelihood
    'lnl': r'$\ln \mathcal{L}$',
    'lnl_H': r'$\ln \mathcal{L}_H$',
    'lnl_L': r'$\ln \mathcal{L}_L$',
    'lnl_V': r'$\ln \mathcal{L}_V$',
    }

_UNITS = (dict.fromkeys(['mchirp', 'm1', 'm2', 'mtot', 'mtot_source',
                         'm1_source', 'm2_source', 'mchirp_source'],
                        r'M$_\odot$')
          | dict.fromkeys(['t_refdet', 'tc', 't_geocenter'], 's')
          | {'d_hat': r'Mpc M$_\odot^{-1}$',
             'd_luminosity': 'Mpc',
             })

LATEX_LABELS = grid.LatexLabels(_LABELS, _UNITS)
