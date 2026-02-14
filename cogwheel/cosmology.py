"""
Cosmological corrections to an Euclidean universe.

Provide functions to transform between d_luminosity and redshift as well
as computing ratios of comoving volume-time to Euclidean universe (i.e.,
luminosity volume and present-day observer time) volume-time.
"""

import astropy
from astropy.cosmology import Planck18
from scipy.interpolate import make_interp_spline
import numpy as np


@np.vectorize
def d_luminosity_of_z(z, cosmology=Planck18):
    """Return luminosity distance (Mpc) as a function of redshift."""
    return (cosmology.luminosity_distance(z) / astropy.units.Mpc).decompose()


def _construct_z_of_d_luminosity(cosmology=Planck18, z_max=10.):
    """Return the function inverse to ``d_luminosity_of_z``."""
    z_arr = np.linspace(0, z_max, 1000)
    d_luminosity_arr = d_luminosity_of_z(z_arr, cosmology)
    interp_z_of_d = make_interp_spline(d_luminosity_arr, z_arr)
    interp_z_of_d.extrapolate = False

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
    Return ratio of differential comoving volume-time to luminosity-
    volume and observer-time.

    Compute the ratio:

    .. math::
        \\frac{d(V_C t_C)}{d(V_L t)}
        = (1+z)^{-4} (1 - \\frac{d_L}{1+z} \\frac{dz}{dd_L})

    where :math:`(V_C t_C)` is the comoving volume-time, :math:`V_L` is
    the luminosity volume, :math:`t` is the present-day observer time,
    :math:`z` is the cosmological redshift, :math:`d_L` the luminosity
    distance and :math:`dz/dd_L` the derivative of redshift with respect
    to luminosity distance.
    """
    z = z_of_d_luminosity(d_luminosity)
    dz_dd = _dz_dd_luminosity(d_luminosity)
    return (1+z)**-4 * (1 - d_luminosity / (1+z) * dz_dd)
