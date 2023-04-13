"""Tests for the `gw_utils` module."""

from unittest import TestCase, main
import numpy as np

import lal

from cogwheel import gw_utils
from cogwheel import skyloc_angles


class AntennaCoefficientsTestCase(TestCase):
    """Tests for computation of fplus, fcross."""
    @staticmethod
    def _get_random_args():
        ra = np.random.uniform(0, np.pi)
        dec = np.random.uniform(-np.pi/2, np.pi/2)
        tgps = np.random.uniform(1e9)
        detector_names = tuple(gw_utils.DETECTORS)
        return ra, dec, tgps, detector_names

    def test_psi(self):
        """
        Test sign convention for the sign of the polarization.
        We define psi such that
            (F+, Fx) = ((c, s), (-s, c)) @ (F+(psi=0), Fx(psi=0))
        where c = cos(2 psi), s = sin(2 psi).
        We compute antenna coefficients for a random location and
        polarization and verify that it is the same as if we apply the
        polarization manually.
        """
        ra, dec, tgps, detector_names = self._get_random_args()
        psi = np.random.uniform(0, np.pi)

        fp_fc = gw_utils.fplus_fcross(detector_names, ra, dec, psi, tgps)
        fp0_fc0 = gw_utils.fplus_fcross(detector_names, ra, dec, 0, tgps)

        psi_rotation = np.array([[np.cos(2*psi), np.sin(2*psi)],
                                 [-np.sin(2*psi), np.cos(2*psi)]])

        np.testing.assert_allclose(fp_fc, psi_rotation @ fp0_fc0)

    def test_fplus_fcross(self):
        """Test tensor-product computation of (F+, Fx) against LAL."""
        ra, dec, tgps, detector_names = self._get_random_args()
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        lon = skyloc_angles.ra_to_lon(ra, gmst)

        np.testing.assert_allclose(
            gw_utils.fplus_fcross(detector_names, ra, dec, 0, tgps),
            gw_utils.get_fplus_fcross_0(detector_names, dec, lon).T)


if __name__ == '__main__':
    main()
