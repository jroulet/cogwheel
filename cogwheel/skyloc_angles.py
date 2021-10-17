"""
Implementation of SkyLocAngles class to handle coordinate
transformations to and from right ascension and declination.
"""
import numpy as np

import lal

from . import utils
from . gw_utils import DETECTORS

class SkyLocAngles(utils.JSONMixin):
    """
    Class that defines a coordinate system for sky localization.
    Converts back and forth between (ra, dec) and (thetanet,
    phinet). The latter are spherical angles associated to a
    Cartesian system (i, j, k) where k is the line connecting two
    detectors, j is the zenith at the midpoint between the two detectors
    and i = cross(j, k) is the horizon at the midpoint.
    The coordinate system is specified by two detectors and a GPS time.
    """
    def __init__(self, detector_pair: str, tgps: float):
        """
        Parameters
        ----------
        detector_pair: Length-2 string with names of the detectors
                       used to define the coordinate system, e.g. "HL".
        tgps: GPS time used to define the coordinate system.
        """
        assert len(detector_pair) == 2, \
            f'Need 2 detectors from {list(DETECTORS)}, e.g. "HL".'
        self.detector_pair = detector_pair
        self.tgps = tgps
        self._gmst = lal.GreenwichMeanSiderealTime(tgps)  # [radians]

        # 2 detector cartesian locations in meters, fixed to Earth:
        # x = Greenwich & Equator, z = North pole.
        self._det1_location = DETECTORS[self.detector_pair[0]].location
        self._det2_location = DETECTORS[self.detector_pair[1]].location
        midpoint_location = (self._det1_location + self._det2_location) / 2

        # Cartesian axes of the new coordinate system, fixed to Earth:
        # i = horizon, j = zenith, k = line connecting 2 detectors.
        self._k_axis = normalize(self._det1_location - self._det2_location)
        self._i_axis = normalize(np.cross(midpoint_location, self._k_axis))
        self._j_axis = np.cross(self._k_axis, self._i_axis)

        self._rotation_matrix = get_rotation_matrix(
            self._i_axis, self._j_axis, self._k_axis)

    def radec_to_thetaphinet(self, ra, dec):
        """
        Transform sky location angles from (ra, dec) to
        (thetanet, phinet).
        """
        lon = ra_to_lon(ra, self._gmst)
        xyz = latlon_to_cart3d(dec, lon)
        ijk = self._rotation_matrix @ xyz
        thetanet, phinet = cart3d_to_thetaphi(ijk)
        return thetanet, phinet

    def thetaphinet_to_radec(self, thetanet, phinet):
        """
        Transform sky location angles from (thetanet, phinet)
        to (ra, dec).
        """
        ijk = thetaphi_to_cart3d(thetanet, phinet)
        xyz = self._rotation_matrix.T @ ijk
        dec, lon = cart3d_to_latlon(xyz)
        ra = lon_to_ra(lon, self._gmst)
        return ra, dec

    def __repr__(self):
        return f'SkyLocAngles({self.detector_pair!r}, {self.tgps})'


def latlon_to_cart3d(lat, lon):
    """
    Return a unit vector (x, y, z) from a latitude and longitude in
    radians.
    Axes directions are: x = Greenwich & Equator, z = North pole
    """
    return np.array([np.cos(lon) * np.cos(lat),
                     np.sin(lon) * np.cos(lat),
                     np.sin(lat)])


def thetaphi_to_cart3d(theta, phi):
    """
    Return a unit vector (x, y, z) from spherical angles in radians.
    """
    return np.array([np.cos(phi) * np.sin(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(theta)])


def cart3d_to_latlon(unit_vector):
    """
    Return latitude and longitude in radians corresponding to a
    Cartesian 3d vector assumed normalized to 1.
    """
    lat = np.arcsin(unit_vector[2])
    lon = np.arctan2(unit_vector[1], unit_vector[0])
    return lat, lon


def cart3d_to_thetaphi(unit_vector):
    """
    Return spherical angles in radians corresponding to a Cartesian 3d
    vector r, assumed normalized to 1.
    """
    theta = np.arccos(unit_vector[2])
    phi = np.arctan2(unit_vector[1], unit_vector[0]) % (2*np.pi)
    return theta, phi


def normalize(vector):
    """Divide a vector by its norm."""
    return vector / np.linalg.norm(vector)


def ra_to_lon(ra, gmst):
    """
    Convert right ascension to longitude.

    Parameters
    ----------
    ra: right ascension in radians.
    gmst: Greenwich meridian standard time in radians.

    Return
    ------
    Longitude in radians.
    """
    return (ra - gmst) % (2*np.pi)


def lon_to_ra(lon, gmst):
    """
    Convert longitude to right ascension.

    Parameters
    ----------
    lon: longitude in radians.
    gmst: Greenwich meridian standard time in radians.

    Return
    ------
    Right ascension in radians.
    """
    return (lon + gmst) % (2*np.pi)


def x_rotation_matrix(angle):
    """Rotation matrix about x axis in 3d space."""
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, cos, -sin],
                     [0, sin, cos]])


def z_rotation_matrix(angle):
    """Rotation matrix about z axis in 3d space."""
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def get_rotation_matrix(x_3d, y_3d, z_3d):
    """
    Return a rotation matrix R such that
        R @ r = r',
    where r' are the coordinates in the new frame
    (@ is matrix multiplication).
    The new frame's axis are the unit vectors
    x_3d, y_3d, z_3d (each a normalized size 3 numpy array).
    Equivalently,
        R @ x_3d = (1, 0, 0)
        R @ y_3d = (0, 1, 0)
        R @ z_3d = (0, 0, 1).
    """
    alpha = np.arctan2(z_3d[0], -z_3d[1])
    beta = np.arccos(z_3d[2])
    gamma = np.arctan2(x_3d[2], y_3d[2])

    rot_1 = z_rotation_matrix(alpha)
    rot_2 = x_rotation_matrix(beta)
    rot_3 = z_rotation_matrix(gamma)

    return (rot_1 @ rot_2 @ rot_3).T
