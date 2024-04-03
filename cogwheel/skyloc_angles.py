"""
Implementation of SkyLocAngles class to handle coordinate
transformations to and from right ascension and declination.
"""
import numpy as np

import lal

from cogwheel import utils
from cogwheel import gw_utils


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
                       used to define the coordinate system, e.g. "HL",
                       or length-1 string, e.g. "H".
        tgps: GPS time used to define the coordinate system.
        """
        self.detector_pair = detector_pair
        self.tgps = tgps
        self._gmst = lal.GreenwichMeanSiderealTime(tgps)  # [radians]
        self._rotation_matrix = get_rotation_matrix(*self._get_ij_axes())

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

    def _get_ij_axes(self):
        """
        First two cartesian axes of the new coordinate system, fixed to
        Earth (the third axis is their cross product).
        If `self.detector_pair` has 2 detectors the system is:
            i = horizon
            j = zenith
            k = line connecting 2 detectors.
        If it has 1 detector the system is:
            k = x_arm + y_arm
            i = y_arm - x_arm
            j = zenith
        Otherwise a `ValueError` is raised.
        """
        if len(self.detector_pair) == 2:
            det1_location = gw_utils.DETECTORS[self.detector_pair[0]].location
            det2_location = gw_utils.DETECTORS[self.detector_pair[1]].location
            midpoint_location = (det1_location + det2_location) / 2

            k_axis = normalize(det1_location - det2_location)
            i_axis = normalize(np.cross(midpoint_location, k_axis))
            j_axis = np.cross(k_axis, i_axis)
            return i_axis, j_axis

        if len(self.detector_pair) == 1:
            x_arm, y_arm = gw_utils.DETECTOR_ARMS[self.detector_pair[0]]

            # Arms are normalized but not perfectly orthogonal, fix:
            y_arm = np.cross(np.cross(x_arm, y_arm), x_arm)

            k_axis = normalize(x_arm + y_arm)
            i_axis = normalize(y_arm - x_arm)
            j_axis = np.cross(k_axis, i_axis)
            return i_axis, j_axis

        raise ValueError('Need 1 or 2 detectors from '
                         f'{list(gw_utils.DETECTORS)}, e.g. "HL".')

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


def get_rotation_matrix(x_3d, y_3d):
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
    if not np.allclose(np.matmul((x_3d, y_3d), np.transpose((x_3d, y_3d))),
                       np.eye(2)):
        raise ValueError('Pass normalized orthogonal axes.')

    z_3d = np.cross(x_3d, y_3d)
    alpha = np.arctan2(z_3d[0], -z_3d[1])
    beta = np.arccos(z_3d[2])
    gamma = np.arctan2(x_3d[2], y_3d[2])

    rot_1 = z_rotation_matrix(alpha)
    rot_2 = x_rotation_matrix(beta)
    rot_3 = z_rotation_matrix(gamma)

    return (rot_1 @ rot_2 @ rot_3).T
