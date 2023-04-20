"""Tests for the `waveform` module."""

import itertools
from unittest import TestCase, main
import numpy as np

from cogwheel import waveform, gw_utils

DETECTOR_PAIRS = [''.join(pair)
                  for pair in itertools.combinations(gw_utils.DETECTORS, 2)]


def get_random_init_parameters():
    """
    Return a dictionary with kwargs for `waveform.WaveformGenerator`.
    """
    return {'detector_names': np.random.choice(DETECTOR_PAIRS),
            'tgps': np.random.uniform(0, 1e9),
            'tcoarse': np.random.uniform(0, 128)}


def get_random_par_dic(aligned_spins=False, tides=False):
    """
    Return a dictionary of waveform parameters, per
    `waveform.WaveformGenerator.params`.
    """
    par_dic = {}

    par_dic['m2'], par_dic['m1'] = np.sort(np.random.uniform(1, 100, 2))

    s1tilt, s2tilt = np.arccos(np.random.uniform(-1, 1, 2))
    if aligned_spins:
        s1tilt, s2tilt = 0, 0
    s1phi, s2phi = np.random.uniform(0, 2*np.pi, 2)
    s1, s2 = np.random.uniform(0, 1, 2)

    par_dic['s1x_n'] = s1 * np.sin(s1tilt) * np.cos(s1phi)
    par_dic['s1y_n'] = s1 * np.sin(s1tilt) * np.sin(s1phi)
    par_dic['s1z'] = s1 * np.cos(s1tilt)

    par_dic['s2x_n'] = s2 * np.sin(s2tilt) * np.cos(s2phi)
    par_dic['s2y_n'] = s2 * np.sin(s2tilt) * np.sin(s2phi)
    par_dic['s2z'] = s2 * np.cos(s2tilt)

    par_dic['l1'], par_dic['l2'] = 0, 0
    if tides:
        par_dic['l1'], par_dic['l2'] = np.random.uniform(0, 1000, 2)

    par_dic['iota'] = np.arccos(np.random.uniform(-1, 1))
    par_dic['phi_ref'] = np.random.uniform(0, 2*np.pi)

    par_dic['ra'] = np.random.uniform(0, 2*np.pi)
    par_dic['dec'] = np.arcsin(np.random.uniform(-1, 1))

    par_dic['psi'] = np.random.uniform(0, np.pi)

    par_dic['t_geocenter'] = np.random.uniform(-.1, .1)

    par_dic['d_luminosity'] = np.random.uniform(10, 1e3)

    par_dic['f_ref'] = np.random.uniform(10, 200)

    assert sorted(par_dic) == waveform.WaveformGenerator.params
    return par_dic


class WaveformGeneratorTestCase(TestCase):
    """Class to test `WaveformGenerator`."""
    @staticmethod
    def test_phi_ref_and_distance():
        """
        Test that `WaveformGenerator` correctly implements `phi_ref` and
        `d_luminosity` as fast parameters.
        """
        for approximant, app_metadata in waveform.APPROXIMANTS.items():
            wfg = waveform.WaveformGenerator(
                **get_random_init_parameters(), approximant=approximant)
            for i in range(100):
                par_dic = get_random_par_dic(app_metadata.aligned_spins,
                                             app_metadata.tides)
                waveform_par_dic = {par: par_dic[par]
                                    for par in wfg._waveform_params}
                lal_dic = wfg.create_lal_dict()

                f = np.linspace(0, 1e3, 500)

                # These two ways should give the same answer:
                hplus_hcross = wfg.get_hplus_hcross(f, waveform_par_dic)

                hplus_hcross_ = waveform.compute_hplus_hcross(
                    f, waveform_par_dic, approximant, lal_dic=lal_dic)

                mask = hplus_hcross_ != 0
                h_100hz = np.linalg.norm(
                    hplus_hcross_[:, np.searchsorted(f, 100)])
                assert np.allclose(hplus_hcross, hplus_hcross_,
                                   atol=1e-3*h_100hz, rtol=1e-3), (
                    '`waveform.WaveformGenerator.get_waveform()` gives a '
                    'different answer than `waveform.compute_waveform()` for '
                    f'waveform_par_dic={waveform_par_dic}\n'
                    f'f=\n{f}\napproximant={approximant}\nNumber of waveforms '
                    f'tried before failure={i}.\nThe relative difference is\n'
                    f'{np.abs(1 - hplus_hcross[mask] / hplus_hcross_[mask])}.')


if __name__ == '__main__':
    main()
