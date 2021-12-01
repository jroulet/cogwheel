"""
Module for functions that are useful in higher level data analysis but
which cannot be compared directly to LVC due to convention differences.
"""
import numpy as np

def fplus_fcross_analytic(det_polar, det_azimuth, psi):
    """
    Get time-indep. geometric F+, Fx following https://arxiv.org/pdf/1102.5421.pdf
    where det_polar, det_azimuth are the polar and azimuthal angles of the
    line-of-sight in the frame whose x-y plane is defined by the detector arms
    with the z-axis pointing toward the sky.
    """
    costheta = np.cos(det_polar)
    part1 = 0.5 * (1 + (costheta ** 2)) * np.cos(2 * det_azimuth)
    part2 = costheta * np.sin(2 * det_azimuth)
    c2ps = np.cos(2 * psi)
    s2ps = np.sin(2 * psi)
    return part1 * c2ps - part2 * s2ps, part1 * s2ps + part2 * c2ps


