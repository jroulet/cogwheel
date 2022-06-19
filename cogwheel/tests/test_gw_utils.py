"""Tests for the `gw_utils` module."""

from unittest import TestCase, main
import numpy as np

from cogwheel.gw_utils import fplus_fcross, DETECTORS


class PolarizationTestCase(TestCase):
    """
    Test sign convention for the sign of the polarization.
    We define psi such that
        (F+, Fx) = ((c, s), (-s, c)) @ (F+(psi=0), Fx(psi=0))
    where c = cos(2 psi), s = sin(2 psi).
    """
    @staticmethod
    def test_psi():
        """
        Compute antenna pattern for a random location and polarization
        and verify that it is the same as if we apply the polarization
        manually.
        """
        ra = np.random.uniform(0, np.pi)
        dec = np.random.uniform(-np.pi/2, np.pi/2)
        psi = np.random.uniform(0, np.pi)
        tgps = np.random.uniform(1e9)

        fp_fc = fplus_fcross(tuple(DETECTORS), ra, dec, psi, tgps)
        fp0_fc0 = fplus_fcross(tuple(DETECTORS), ra, dec, 0, tgps)

        psi_rotation = np.array([[np.cos(2*psi), np.sin(2*psi)],
                                 [-np.sin(2*psi), np.cos(2*psi)]])

        np.testing.assert_allclose(fp_fc, psi_rotation @ fp0_fc0)


if __name__ == '__main__':
    main()
