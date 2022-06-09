"""Tests for the `skyloc_angles` module."""

import itertools
from unittest import TestCase, main
import numpy as np

from cogwheel import skyloc_angles
from cogwheel import gw_utils


DETECTOR_PAIRS = [''.join(pair)
                  for pair in itertools.combinations(gw_utils.DETECTORS, 2)]


def instantiate_skyloc(detector_pair=None, tgps=None):
    """
    Return a SkyLocAngles instance.
    If `detector_pair` and/or `tgps` are unspecified, they are chosen
    randomly.
    """
    if detector_pair is None:
        detector_pair = np.random.choice(DETECTOR_PAIRS)
    if tgps is None:
        tgps = np.random.uniform(0, 1e9)
    return skyloc_angles.SkyLocAngles(detector_pair, tgps)


class SkyLocAnglesTestCase(TestCase):
    """Class to test `SkyLocAngles`."""
    def test_inverse_transform(self):
        """
        Test that `SkyLocAngles.radec_to_thetaphinet()` and
        `SkyLocAngles.thetaphinet_to_radec()` are mutual inverses.
        """
        for detector_pair in DETECTOR_PAIRS:
            skyloc = instantiate_skyloc(detector_pair)

            ra = np.random.uniform(0, 2*np.pi)
            dec = np.arcsin(np.random.uniform(-1, 1))
            thetanet, phinet = skyloc.radec_to_thetaphinet(ra, dec)
            ra_, dec_ = skyloc.thetaphinet_to_radec(thetanet, phinet)

            self.assertAlmostEqual(ra, ra_, msg=skyloc)
            self.assertAlmostEqual(dec, dec_, msg=skyloc)

    @staticmethod
    def test_constant_timelag():
        """
        Test that changing phinet at fixed thetanet keeps the time-lag
        between detectors constant.
        """
        for detector_pair in DETECTOR_PAIRS:
            skyloc = instantiate_skyloc(detector_pair)

            thetanet = np.arccos(np.random.uniform(-1, 1))
            phinets = np.linspace(0, 2*np.pi, 20)

            radecs = [skyloc.thetaphinet_to_radec(thetanet, phinet)
                      for phinet in phinets]

            timelags = [np.subtract(*gw_utils.time_delay_from_geocenter(
                            detector_pair, ra, dec, skyloc.tgps))
                        for ra, dec in radecs]
            np.testing.assert_allclose(timelags, timelags[0])


if __name__ == '__main__':
    main()
