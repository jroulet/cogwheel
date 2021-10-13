import numpy as np
from scipy.interpolate import UnivariateSpline

#     Cosmological Distance
# -----------------------------
from astropy.cosmology import Planck15
from astropy.cosmology import z_at_value as z_at_val
import astropy.units as astro_units

u_Mpc = astro_units.Mpc
u_Gpc = astro_units.Gpc
u_Mpc3 = u_Mpc ** 3
u_Gpc3 = u_Gpc ** 3
DEFAULT_COSMOLOGY = Planck15

c_over_H0 = 4422  # [Mpc]
Omega_m = 0.308
z_max = 10

def _x(z):
    return (1 - Omega_m) / (Omega_m * (1 + z) ** 3)

def phi(x):
    return ((1 + 1.32 * x + 0.4415 * x ** 2 + 0.02656 * x ** 3)
            / (1 + 1.392 * x + .5121 * x ** 2 + 0.03944 * x ** 3))

def dL(z):
    return (2 * c_over_H0 * (1 + z) / np.sqrt(Omega_m)
            * (phi(_x(0)) - phi(_x(z)) / np.sqrt(1 + z)))

zs = np.linspace(0, z_max, 1000)
# z_of_DL = interp1d(dL(zs), zs)
z_of_DL = UnivariateSpline(dL(zs), zs, s=0)
dz_dDL = z_of_DL.derivative()  # Function of DL

# distance and volume from z
def DL_Mpc_of_z(z, cosmo=DEFAULT_COSMOLOGY):
    d_obj = cosmo.luminosity_distance(z)
    if d_obj.unit == u_Mpc:
        return d_obj.value
    else:
        raise RuntimeError(f'astropy.cosmology distance units changed from Mpc to {d_obj.unit}')

def Dcomov_Mpc_of_z(z, cosmo=DEFAULT_COSMOLOGY):
    d_obj = cosmo.comoving_distance(z)
    if d_obj.unit == u_Mpc:
        return d_obj.value
    else:
        raise RuntimeError(f'astropy.cosmology distance units changed from Mpc to {d_obj.unit}')

def Vcomov_Mpc3_of_z(z, cosmo=DEFAULT_COSMOLOGY):
    v_obj = cosmo.comoving_volume(z)
    if v_obj.unit == u_Mpc3:
        return v_obj.value
    else:
        raise RuntimeError(f'astropy.cosmology volume units changed from cubic Mpc to {v_obj.unit}')

# FASTER z from distance (interpolated)
z_of_DL_Mpc = UnivariateSpline(DL_Mpc_of_z(zs, cosmo=DEFAULT_COSMOLOGY), zs, s=0)
z_of_Dcomov_Mpc = UnivariateSpline(Dcomov_Mpc_of_z(zs, cosmo=DEFAULT_COSMOLOGY), zs, s=0)
z_of_Vcomov_Mpc3 = UnivariateSpline(Vcomov_Mpc3_of_z(zs, cosmo=DEFAULT_COSMOLOGY), zs, s=0)

# z from distance or volume
def z_at_value(f, val):
    """INPUT MUST HAVE ASTROPY UNITS"""
    # modification to allow arrays to be passed (error in astropy.cosmo.z_at_value)
    if hasattr(val.value, '__len__'):
        return np.array([z_at_val(f, v) for v in val])
    else:
        return z_at_val(f, val)

#### NOTE THESE ARE SLOW! use interpolated versions for arrays (same name with _Mpc suffix)
def z_of_DL_astropy(DL, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    xx = (np.asarray(DL) if hasattr(DL, '__len__') else DL)
    return z_at_value(cosmo.luminosity_distance, xx * D_units)

def z_of_Dcomov(Dcomov, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    xx = (np.asarray(Dcomov) if hasattr(Dcomov, '__len__') else Dcomov)
    return z_at_value(cosmo.comoving_distance, xx * D_units)

def z_of_Vcomov(Vcomov, V_units=u_Mpc3, cosmo=DEFAULT_COSMOLOGY):
    xx = (np.asarray(Vcomov) if hasattr(Vcomov, '__len__') else Vcomov)
    return z_at_value(cosmo.comoving_volume, xx * V_units)

# distance and volume conversion
def Vcomov_of_Dcomov(Dcomov):
    xx = (np.asarray(Dcomov) if hasattr(Dcomov, '__len__') else Dcomov)
    return 4 * np.pi * (xx ** 3) / 3.

def Dcomov_of_Vcomov(Vcomov):
    xx = (np.asarray(Vcomov) if hasattr(Vcomov, '__len__') else Vcomov)
    return (0.75 * xx / np.pi) ** (1 / 3.)

def Dcomov_of_DL(DL, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    if hasattr(DL, '__len__'):
        xx = np.asarray(DL)
        zz = (z_of_DL_Mpc(xx) if (D_units == u_Mpc) and (cosmo == DEFAULT_COSMOLOGY)
              else z_at_value(cosmo.luminosity_distance, xx * D_units))
        return xx / (1 + zz)
    else:
        return DL / (1 + z_at_value(cosmo.luminosity_distance, DL * D_units))

def DL_of_Dcomov(Dcomov, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    if hasattr(Dcomov, '__len__'):
        xx = np.asarray(Dcomov)
        zz = (z_of_Dcomov_Mpc(xx) if (D_units == u_Mpc) and (cosmo == DEFAULT_COSMOLOGY)
              else z_at_value(cosmo.comoving_distance, xx * D_units))
        return xx * (1 + zz)
    else:
        return Dcomov * (1 + z_at_value(cosmo.comoving_distance, Dcomov * D_units))

def DL_of_Vcomov(Vcomov, V_units=u_Mpc3, cosmo=DEFAULT_COSMOLOGY):
    not_scalar = hasattr(Vcomov, '__len__')
    xx = (np.asarray(Vcomov) if not_scalar else Vcomov)
    zz = (z_of_Vcomov_Mpc3(xx) if not_scalar and (V_units == u_Mpc3) and (cosmo == DEFAULT_COSMOLOGY)
          else z_at_value(cosmo.comoving_volume, xx * V_units))
    return Dcomov_of_Vcomov(xx) * (1 + zz)

def Vcomov_of_DL(DL, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    return Vcomov_of_Dcomov(Dcomov_of_DL(DL, D_units=D_units, cosmo=cosmo))

