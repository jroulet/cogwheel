"""Generate strain waveforms and project them onto detectors."""

import numpy as np
from collections import defaultdict

import lal
import lalsimulation

from . import gw_utils

ZERO_INPLANE_SPINS = {'s1x': 0.,
                      's1y': 0.,
                      's2x': 0.,
                      's2y': 0.}

DEFAULT_PARS = {**ZERO_INPLANE_SPINS,
                's1z': 0.,
                's2z': 0.,
                'l1': 0.,
                'l2': 0.}

HARMONIC_MODES = {'IMRPhenomXPHM': [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)],
                  'IMRPhenomD': [(2, 2)]}

ALIGNED_SPINS = {'IMRPhenomXPHM': False,
                 'IMRPhenomD': True}

TIDES = {'IMRPhenomXPHM': False,
         'IMRPhenomD': False}


def out_of_bounds(par_dic):
    """
    Return whether parameters in `par_dic` are out of physical bounds.
    """
    return (any(par_dic[positive] < 0 for positive in
                ['m1', 'm2', 'd_luminosity', 'l1', 'l2', 'iota'])
            or any(np.linalg.norm(s) > 1 for s in [
                [par_dic['s1x'], par_dic['s1y'], par_dic['s1z']],
                [par_dic['s2x'], par_dic['s2y'], par_dic['s2z']]])
            or par_dic['iota'] > np.pi
            or np.abs(par_dic['dec'] > np.pi/2))


def compute_hplus_hcross(f_ref, f, par_dic, approximant: str,
                         harmonic_modes=None):
    """
    Generate frequency domain waveform using LAL.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    approximant: String with the approximant name.
    f_ref: Reference frequency in Hz
    f: Frequency array in Hz
    par_dic: Dictionary of source parameters. Needs to have these keys:
                 m1, m2, d_luminosity, iota, vphi;
             plus, optionally:
                 s1x, s1y, s1z, s2x, s2y, s2z, l1, l2.
    harmonic_modes: Optional, list of 2-tuples with (l, m) pairs
                  specifying which (co-precessing frame) higher-order
                  modes to include.
    """
    # Parameters ordered for lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_params = [
        'vphi', 'm1_kg', 'm2_kg', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'f_ref', 'd_luminosity_meters', 'iota', 'lal_dic', 'approximant', 'f']

    par_dic = {**DEFAULT_PARS, **par_dic}

    # SI unit conversions
    par_dic['d_luminosity_meters'] = par_dic['d_luminosity'] * 1e6 * lal.PC_SI
    par_dic['m1_kg'] = par_dic['m1'] * lal.MSUN_SI
    par_dic['m2_kg'] = par_dic['m2'] * lal.MSUN_SI

    lal_dic = lal.CreateDict()  # Contains tidal and higher-mode parameters
    # Tidal parameters
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        lal_dic, par_dic['l1'])
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        lal_dic, par_dic['l2'])
    # Higher-mode parameters
    if harmonic_modes is not None:
        mode_array = lalsimulation.SimInspiralCreateModeArray()
        for l, m in harmonic_modes:
            assert (l, abs(m)) in HARMONIC_MODES[approximant], \
                f'Mode ({l}, {m}) not supported by {approximant}.'
            lalsimulation.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsimulation.SimInspiralWaveformParamsInsertModeArray(lal_dic,
                                                               mode_array)
    par_dic['lal_dic'] = lal_dic

    par_dic['approximant'] = lalsimulation.GetApproximantFromString(
        approximant)

    par_dic['f_ref'] = f_ref

    f0_is_0 = int(f[0] == 0)  # In this case we will set h(f=0) = 0
    par_dic['f'] = lal.CreateREAL8Sequence(len(f) - f0_is_0)
    par_dic['f'].data = f[f0_is_0:]

    try:
        hplus, hcross = lalsimulation.SimInspiralChooseFDWaveformSequence(
            *[par_dic[par] for par in lal_params])
    except:
        print('Error when calling LAL at these parameters: ', par_dic)
        raise
    hplus_hcross = np.zeros((2, len(f)), dtype=np.complex_)
    hplus_hcross[0, f0_is_0:] = hplus.data.data
    hplus_hcross[1, f0_is_0:] = hcross.data.data
    return hplus_hcross


class WaveformGenerator:
    """
    Class that provides methods for generating frequency domain
    waveforms.
    "Fast" and "slow" parameters are distinguished: the last waveform
    call is cached and can be computed fast if only fast parameters
    are changed.
    The boolean attribute `disable_precession` can be set to ignore
    inplane spins.
    """
    params = {
        'm1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'l1', 'l2',
        'iota', 'vphi', 'ra', 'dec', 'psi', 't_geocenter', 'd_luminosity'}

    fast_params = {'ra', 'dec', 'psi', 't_geocenter', 'd_luminosity'}
    slow_params = params - fast_params

    _projection_params = {'ra', 'dec', 'psi', 't_geocenter'}
    _waveform_params = params - _projection_params

    def __init__(self, detector_names, tgps, tcoarse, approximant, f_ref,
                 harmonic_modes=None, disable_precession=False):
        super().__init__()

        self.detector_names = detector_names
        self.tgps = tgps
        self.tcoarse = tcoarse
        self._approximant = approximant
        self.harmonic_modes = harmonic_modes
        self.disable_precession = disable_precession
        self.f_ref = f_ref

        self.n_slow_evaluations = 0
        self.n_fast_evaluations = 0
        self.cache = {'slow_par_dic': {},
                      'approximant': None,
                      'f_ref': None,
                      'f': None,
                      'harmonic_modes_by_m': {},
                      'hplus_hcross_0': None}

    @property
    def approximant(self):
        """String with waveform approximant name."""
        return self._approximant

    @approximant.setter
    def approximant(self, app: str):
        """
        Set `approximant` and reset `harmonic_modes` per
        `HARMONIC_MODES[approximant]`; print a warning that this
        was done.
        Raise `ValueError` if `HARMONIC_MODES` does not contain the
        requested approximant.
        """
        if app not in HARMONIC_MODES:
            raise ValueError(f'Add {app} to `waveform.HARMONIC_MODES`.')
        self._approximant = app
        self.harmonic_modes = None
        print(f'`approximant` changed to {app}, setting `harmonic_modes` to '
              f'{self.harmonic_modes}.')

    @property
    def harmonic_modes(self):
        return self._harmonic_modes

    @harmonic_modes.setter
    def harmonic_modes(self, modes):
        """
        Set `self._harmonic_modes` implementing defaults based on the
        approximant, this requires hardcoding which modes are
        implemented by each approximant; it is a necessary evil because
        we need to know `m` in order to make `vphi` a fast parameter.
        Also set `self._harmonic_modes_by_m` with a dictionary whose
        keys are `m` and whose values are a list of `(l, m)` tuples with
        that `m`.
        """
        self._harmonic_modes = modes or HARMONIC_MODES[self.approximant]

        self._harmonic_modes_by_m = defaultdict(list)
        for l, m in self._harmonic_modes:
            self._harmonic_modes_by_m[m].append((l, m))

    def get_strain_at_detectors(self, f, par_dic, by_m=False):
        """
        Get strain measurable at detectors.

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        par_dic: parameter dictionary per `WaveformGenerator.params`.
        by_m: bool, whether to return waveform separated by `m`
              harmonic mode (summed over `l`), or already summed.

        Return
        ------
        n_detectors x n_frequencies array with strain at detector.
        """
        waveform_par_dic = {par: par_dic[par] for par in self._waveform_params}

        # hplus_hcross shape: (n_m x 2 x n_frequencies), n_m optional
        hplus_hcross = self.get_hplus_hcross(f, waveform_par_dic, by_m)

        # fplus_fcross shape: (2 x n_detectors)
        fplus_fcross = np.array(gw_utils.fplus_fcross(
            self.detector_names, par_dic['ra'], par_dic['dec'], par_dic['psi'],
            self.tgps))

        time_delays = gw_utils.time_delay_from_geocenter(
            self.detector_names, par_dic['ra'], par_dic['dec'], self.tgps)

        # shifts shape: (n_detectors x n_frequencies)
        shifts = np.exp(-2j*np.pi * f * (self.tcoarse
                                         + par_dic['t_geocenter']
                                         + time_delays[:, np.newaxis]))

        # Detector strain (n_m x n_detectors x n_frequencies), n_m optional
        return np.sum(fplus_fcross[..., np.newaxis]
                      * hplus_hcross[..., np.newaxis, :], axis=-3) * shifts

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False):
        """
        Return hplus, hcross waveform strain.
        Note: inplane spins will be zeroized if `self.disable_precession`
              is `True`.

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        waveform_par_dic: dictionary per `WaveformGenerator._waveform_params`.

        Return
        ------
        2 x len(f) array with (hplus, hcross).
        """
        if self.disable_precession:
            waveform_par_dic.update(ZERO_INPLANE_SPINS)

        slow_par_dic = {par: waveform_par_dic[par] for par in self.slow_params}
        self._rotate_inplane_spins(slow_par_dic, waveform_par_dic['vphi'])

        # Are cached parameters the same?
        same_cache = (
            all(np.isclose(self.cache['slow_par_dic'].get(par, np.nan),
                           slow_par_dic[par]) for par in self.slow_params)
            and self.cache['approximant'] == self.approximant
            and self.cache['f_ref'] == self.f_ref
            and np.array_equal(self.cache['f'], f)
            and self.cache['harmonic_modes_by_m'] == self._harmonic_modes_by_m)
        if same_cache:
            hplus_hcross_0 = self.cache['hplus_hcross_0']
            self.n_fast_evaluations += 1
        else:
            # Compute the waveform mode by mode and update cache.
            waveform_par_dic_0 = {**slow_par_dic,
                                  'd_luminosity': 1., 'vphi': 0.}
            # hplus_hcross_0 is a (n_m x 2 x n_frequencies) arrays with
            # sum_l (hlm+, hlmx), at vphi=0, d_luminosity=1Mpc.
            hplus_hcross_0 = np.array(
                [compute_hplus_hcross(self.f_ref, f, waveform_par_dic_0,
                                      self.approximant, modes)
                 for modes in self._harmonic_modes_by_m.values()])
            self.cache = {'approximant': self.approximant,
                          'f_ref': self.f_ref,
                          'f': f,
                          'slow_par_dic': slow_par_dic,
                          'harmonic_modes_by_m': self._harmonic_modes_by_m,
                          'hplus_hcross_0': hplus_hcross_0}
            self.n_slow_evaluations += 1

        # hplus_hcross is a (n_m x 2 x n_frequencies) array.
        m_arr = np.array(list(self._harmonic_modes_by_m)).reshape(-1, 1, 1)
        hplus_hcross = (np.exp(1j * m_arr * waveform_par_dic['vphi'])
                        / waveform_par_dic['d_luminosity'] * hplus_hcross_0)
        if by_m:
            return hplus_hcross
        return np.sum(hplus_hcross, axis=0)

    @staticmethod
    def _rotate_inplane_spins(dic, vphi):
        """
        Rotate inplane spins (s1x, s1y) and (s2x, s2y) by an angle `vphi`,
        inplace in `dic`.
        """
        sin_vphi = np.sin(vphi)
        cos_vphi = np.cos(vphi)
        rotation = np.array([[cos_vphi, -sin_vphi],
                             [sin_vphi, cos_vphi]])
        dic['s1x'], dic['s1y'] = rotation @ np.array([dic['s1x'], dic['s1y']])
        dic['s2x'], dic['s2y'] = rotation @ np.array([dic['s2x'], dic['s2y']])
