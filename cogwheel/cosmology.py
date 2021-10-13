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
z_max = 10
zs = np.linspace(0, z_max, 10000)
z_of_DL_Mpc = UnivariateSpline(DL_Mpc_of_z(zs, cosmo=DEFAULT_COSMOLOGY), zs, s=0)
dz_dDL = z_of_DL_Mpc.derivative()  # Function of DL
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
def z_of_DL_astropy(DL_astropy, cosmo=DEFAULT_COSMOLOGY):
    """INPUT MUST BE AN ASTROPY OBJECT or array of astropy objects (needs units)"""
    return z_at_value(cosmo.luminosity_distance, DL_astropy)

def z_of_Dcomov_astropy(Dcomov_astropy, cosmo=DEFAULT_COSMOLOGY):
    """INPUT MUST BE AN ASTROPY OBJECT or array of astropy objects (needs units)"""
    return z_at_value(cosmo.comoving_distance, Dcomov_astropy)

def z_of_Vcomov_astropy(Vcomov_astropy, cosmo=DEFAULT_COSMOLOGY):
    """INPUT MUST BE ASTROPY OBJECT or array of astropy objects (needs units)"""
    return z_at_value(cosmo.comoving_volume, Vcomov_astropy)

# distance and volume conversion
def DtoV(distance):
    return 4 * np.pi * (distance ** 3) / 3.

def VtoD(volume):
    return (0.75 * volume / np.pi) ** (1 / 3.)

def Vcomov_of_Dcomov(Dcomov):
    return DtoV((np.asarray(Dcomov) if hasattr(Dcomov, '__len__') else Dcomov))

def Dcomov_of_Vcomov(Vcomov):
    return VtoD((np.asarray(Vcomov) if hasattr(Vcomov, '__len__') else Vcomov))

def Dcomov_of_DL(DL, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    if hasattr(DL, '__len__'):
        xx = np.asarray(DL)
        zz = (z_of_DL_Mpc(xx) if (D_units == u_Mpc) and (cosmo == DEFAULT_COSMOLOGY)
              else z_at_value(cosmo.luminosity_distance, xx * D_units))
        return xx / (1 + zz)
    else:
        return DL / (1 + z_at_value(cosmo.luminosity_distance, DL * D_units))

def Vcomov_of_DL(DL, D_units=u_Mpc, cosmo=DEFAULT_COSMOLOGY):
    return DtoV(Dcomov_of_DL(DL, D_units=D_units, cosmo=cosmo))

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


