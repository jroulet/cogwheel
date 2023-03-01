"""
Provide functions to transform between d_luminosity and redshift
as well as computing ratios of comoving volume-time to Euclidean
universe (i.e., luminosity volume and present-day observer time)
volume-time. See:
    d_luminosity_of_z
    z_of_d_luminosity
    comoving_to_luminosity_diff_vt_ratio
"""

import astropy
from astropy.cosmology import Planck18
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


@np.vectorize
def d_luminosity_of_z(z, cosmology=Planck18):
    """Return luminosity distance (Mpc) as a function of redshift."""
    return (cosmology.luminosity_distance(z) / astropy.units.Mpc).decompose()


def _construct_z_of_d_luminosity(cosmology=Planck18, z_max=10.):
    """Return the function inverse to d_luminosity_of_z."""
    z_arr = np.linspace(0, z_max, 1000)
    d_luminosity_arr = d_luminosity_of_z(z_arr, cosmology)
    interp_z_of_d = InterpolatedUnivariateSpline(d_luminosity_arr, z_arr)

    def _z_of_d_luminosity(d_luminosity):
        """
        Return redshift as a function of luminosity distance (Mpc).
        """
        return interp_z_of_d(d_luminosity)[()]

    dz_dd_luminosity = interp_z_of_d.derivative()
    return _z_of_d_luminosity, dz_dd_luminosity


z_of_d_luminosity, _dz_dd_luminosity = _construct_z_of_d_luminosity()


def comoving_to_luminosity_diff_vt_ratio(d_luminosity):
    """
    Return the ratio of differential comoving volume-time to
    luminosity-volume and observer-time, i.e.
        d(Vcom tcom)/d(Vlum t)
            = (1+z)^-4 * (1 - d_l / (1+z) * dz/dd_l)
    where (Vcom tcom) is the comoving volume-time, Vlum is the
    luminosity volume, t is the present-day observer time, z is the
    cosmological redshift, d_l the luminosity distance and
    dz/dd_l the derivative of redshift with respect to luminosity
    distance.
    """
    z = z_of_d_luminosity(d_luminosity)
    dz_dd = _dz_dd_luminosity(d_luminosity)
    return (1+z)**-4 * (1 - d_luminosity / (1+z) * dz_dd)
