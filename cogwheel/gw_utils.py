"""Utility functions specific to gravitational waves."""
import numpy as np

import lal

from . import grid

DETECTORS = {'H': lal.CachedDetectors[lal.LHO_4K_DETECTOR],
             'L': lal.CachedDetectors[lal.LLO_4K_DETECTOR],
             'V': lal.CachedDetectors[lal.VIRGO_DETECTOR]}


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

def fplus_fcross_analytic(det_polar, det_azimuth, psi):
    """
    Get time-indep. geometric F+, Fx following https://arxiv.org/pdf/1102.5421.pdf
    where det_polar, det_azimuth are the polar and azimuthal angles of the
    line-of-sight in the frame whose x-y plane is defined by the detector arms
    with the z-axis pointing toward the sky
    """
    costheta = np.cos(det_polar)
    part1 = 0.5 * (1 + (costheta ** 2)) * np.cos(2 * det_azimuth)
    part2 = costheta * np.sin(2 * det_azimuth)
    c2ps = np.cos(2 * psi)
    s2ps = np.sin(2 * psi)
    return part1 * c2ps - part2 * s2ps, part1 * s2ps + part2 * c2ps

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
    'cums1r_s1z': r'$\int \pi (s_1^\perp | s_{1z})$',
    'cums2r_s2z': r'$\int \pi (s_2^\perp | s_{2z})$',
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
    'd_hat': r'$D_{\rm eff} / \mathcal{M}$',
    'd_luminosity': r'$D_L$',
    'z': r'$z$',
    # Orientation
    'vphi': r'$\varphi$',
    'psi_hat': r'$\hat{\psi}$',
    'psi': r'$\psi$',
    'cosiota': r'$\cos \iota$',
    'iota': r'$\iota$',
    'thetaJN': r'$\theta_{JN}$',
    'costhetaJN': r'$\cos \theta_{JN}$',
    'phiJL': r'$\phi_{JL}$',
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
