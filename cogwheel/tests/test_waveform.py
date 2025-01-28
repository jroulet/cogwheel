"""Tests for the `waveform` module."""

import itertools
from unittest import TestCase, main
import numpy as np

from cogwheel import waveform, gw_utils

# Add `IMRPhenomXODE` to `waveform.APPROXIMANTS`:
from cogwheel.waveform_models import xode as _

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
    rng = np.random.default_rng()
    par_dic = {}

    par_dic['m2'], par_dic['m1'] = np.sort(rng.uniform(5, 100, 2))

    s1tilt, s2tilt = np.arccos(rng.uniform(-1, 1, 2))
    if aligned_spins:
        s1tilt, s2tilt = 0, 0
    s1phi, s2phi = rng.uniform(0, 2*np.pi, 2)
    s1, s2 = rng.uniform(0, 1, 2)

    par_dic['s1x_n'] = s1 * np.sin(s1tilt) * np.cos(s1phi)
    par_dic['s1y_n'] = s1 * np.sin(s1tilt) * np.sin(s1phi)
    par_dic['s1z'] = s1 * np.cos(s1tilt)

    par_dic['s2x_n'] = s2 * np.sin(s2tilt) * np.cos(s2phi)
    par_dic['s2y_n'] = s2 * np.sin(s2tilt) * np.sin(s2phi)
    par_dic['s2z'] = s2 * np.cos(s2tilt)

    par_dic['l1'], par_dic['l2'] = 0, 0
    if tides:
        par_dic['l1'], par_dic['l2'] = rng.uniform(0, 1000, 2)

    par_dic['iota'] = np.arccos(rng.uniform(-1, 1))
    par_dic['phi_ref'] = rng.uniform(0, 2*np.pi)

    par_dic['ra'] = rng.uniform(0, 2*np.pi)
    par_dic['dec'] = np.arcsin(rng.uniform(-1, 1))

    par_dic['psi'] = rng.uniform(0, np.pi)

    par_dic['t_geocenter'] = rng.uniform(-.1, .1)

    par_dic['d_luminosity'] = rng.uniform(10, 1e3)

    par_dic['f_ref'] = rng.uniform(10, 200)

    assert sorted(par_dic) == waveform.WaveformGenerator.params
    return par_dic


class _WaveformGenerator(waveform.WaveformGenerator):
    """
    For testing the property that

        h_m(phi_ref, d_luminosity)
            = h_m(0, 1) e^(i m phi_ref) / d_luminosity.
    """
    def get_hplus_hcross_explicit(self, f, waveform_par_dic):
        """
        Don't assume analytic dependence on d_luminosity and phi_ref.
        """
        lal_dic = self.create_lal_dict()
        hplus_hcross_modes = waveform.APPROXIMANTS[self.approximant] \
            .hplus_hcross_by_mode_func(f,
                                       waveform_par_dic,
                                       self.approximant,
                                       self.harmonic_modes,
                                       lal_dic)

        return sum(hplus_hcross_modes.values())



class WaveformGeneratorTestCase(TestCase):
    """Class to test `WaveformGenerator`."""
    @staticmethod
    def test_phi_ref_and_distance():
        """
        Test that `WaveformGenerator` correctly implements `phi_ref` and
        `d_luminosity` as fast parameters.
        """
        for approximant, app_metadata in waveform.APPROXIMANTS.items():
            wfg = _WaveformGenerator(**get_random_init_parameters(),
                                     approximant=approximant)
            for i in range(100):
                par_dic = get_random_par_dic(app_metadata.aligned_spins,
                                             app_metadata.tides)
                waveform_par_dic = {par: par_dic[par]
                                    for par in wfg._waveform_params}

                f = np.linspace(10, 1e3, 500)

                # These two ways should give the same answer:
                hplus_hcross = wfg.get_hplus_hcross(f, waveform_par_dic)
                hplus_hcross_ = wfg.get_hplus_hcross_explicit(
                    f, waveform_par_dic).astype(np.complex128)

                mask = hplus_hcross_ != 0
                max_relative_difference = np.max(np.abs(
                    1 - hplus_hcross[mask] / hplus_hcross_[mask]))

                m_tot = waveform_par_dic['m1'] + waveform_par_dic['m2']
                ind_isco = np.searchsorted(f, gw_utils.isco_frequency(m_tot))
                h_isco = np.linalg.norm(hplus_hcross_[:, ind_isco])
                np.testing.assert_allclose(
                    hplus_hcross, hplus_hcross_, atol=1e-3*h_isco, rtol=1e-3,
                    err_msg=('`get_hplus_hcross()` gives a different answer '
                             'than `get_hplus_hcross_explicit()` for:\n'
                             f'{waveform_par_dic=}\n\n{f=}\n{approximant=}\n'
                             f'Number of waveforms tried before failure={i}.\n'
                             'The maximum relative difference is '
                             f'{max_relative_difference}.'))


if __name__ == '__main__':
    main()
