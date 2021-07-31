"""Utility functions specific to gravitational waves."""

import numpy as np

import lal

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

def time_delay_from_geocenter(detector_names, ra, dec, tgps):
    """Return an array with delay times from Earth center [seconds]."""
    return np.array([
        lal.TimeDelayFromEarthCenter(DETECTORS[det].location, ra, dec, tgps)
        for det in detector_names])

#--------------------------------------------
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
