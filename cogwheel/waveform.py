"""Generate strain waveforms and project them onto detectors."""
import itertools
from collections import defaultdict, namedtuple
import numpy as np

import lal
import lalsimulation

from cogwheel import gw_utils
from cogwheel import utils

ZERO_INPLANE_SPINS = {'s1x_n': 0.,
                      's1y_n': 0.,
                      's2x_n': 0.,
                      's2y_n': 0.}

DEFAULT_PARS = {**ZERO_INPLANE_SPINS,
                's1z': 0.,
                's2z': 0.,
                'l1': 0.,
                'l2': 0.}

FORCE_NNLO_ANGLES = (
    ('SimInspiralWaveformParamsInsertPhenomXPrecVersion', 102),)


def compute_hplus_hcross(f, par_dic, approximant: str,
                         harmonic_modes=None, lal_dic=None):
    """
    Generate frequency domain waveform using LAL.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    f: 1d array of type float
        Frequency array in Hz

    par_dic: dict
        Source parameters. Needs to have these keys:
            * m1, m2: component masses (Msun)
            * d_luminosity: luminosity distance (Mpc)
            * iota: inclination (rad)
            * phi_ref: phase at reference frequency (rad)
            * f_ref: reference frequency (Hz)
        plus, optionally:
            * s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z: dimensionless spins
            * l1, l2: dimensionless tidal deformabilities

    approximant: str
        Approximant name.

    harmonic_modes: list of 2-tuples with (l, m) pairs, optional
        Which (co-precessing frame) higher-order modes to include.

    lal_dic: LALDict, optional
        Contains special approximant settings.
    """

    # Parameters ordered for lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_params = [
        'phi_ref', 'm1_kg', 'm2_kg', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'f_ref', 'd_luminosity_meters', 'iota', 'lal_dic', 'approximant', 'f']

    par_dic = DEFAULT_PARS | par_dic

    # Transform inplane spins to LAL's coordinate system.
    inplane_spins_xy_n_to_xy(par_dic)

    # SI unit conversions
    par_dic['d_luminosity_meters'] = par_dic['d_luminosity'] * 1e6 * lal.PC_SI
    par_dic['m1_kg'] = par_dic['m1'] * lal.MSUN_SI
    par_dic['m2_kg'] = par_dic['m2'] * lal.MSUN_SI

    par_dic['lal_dic'] = lal_dic or lal.CreateDict()
    # Tidal parameters
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        par_dic['lal_dic'], par_dic['l1'])
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        par_dic['lal_dic'], par_dic['l2'])
    # Higher-mode parameters
    if harmonic_modes is not None:
        mode_array = lalsimulation.SimInspiralCreateModeArray()
        for l, m in harmonic_modes:
            lalsimulation.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsimulation.SimInspiralWaveformParamsInsertModeArray(
            par_dic['lal_dic'], mode_array)

    par_dic['approximant'] = lalsimulation.GetApproximantFromString(
        approximant)

    f0_is_0 = f[0] == 0  # In this case we will set h(f=0) = 0
    par_dic['f'] = lal.CreateREAL8Sequence(len(f))
    par_dic['f'].data = f
    if f0_is_0:
        par_dic['f'].data[0] = par_dic['f'].data[1]

    try:
        hplus, hcross = lalsimulation.SimInspiralChooseFDWaveformSequence(
            *[par_dic[par] for par in lal_params])
    except Exception:
        print('Error while calling LAL at these parameters:', par_dic)
        raise
    hplus_hcross = np.stack([hplus.data.data, hcross.data.data])
    if f0_is_0:
        hplus_hcross[:, 0] = 0

    return hplus_hcross


def compute_hplus_hcross_by_mode(f, par_dic, approximant: str,
                                 harmonic_modes, lal_dic=None):
    """
    Return dictionary of the form {(l, m): h_lm} with the contribution
    of each harmonic mode to hplus, hcross.

    Parameters
    ----------
    f: 1d array of type float
        Frequency array in Hz

    par_dic: dict
        Source parameters. Needs to have these keys:
            * m1, m2: component masses (Msun)
            * d_luminosity: luminosity distance (Mpc)
            * iota: inclination (rad)
            * phi_ref: phase at reference frequency (rad)
            * f_ref: reference frequency (Hz)
        plus, optionally:
            * s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z: dimensionless spins
            * l1, l2: dimensionless tidal deformabilities

    approximant: str
        Approximant name.

    harmonic_modes: list of 2-tuples with (l, m) pairs
        Which (co-precessing frame) higher-order modes to include.

    lal_dic: LALDict, optional
        Contains special approximant settings.
    """
    return {mode: compute_hplus_hcross(f, par_dic, approximant,
                                       harmonic_modes=[mode], lal_dic=lal_dic)
            for mode in harmonic_modes}


Approximant = namedtuple(
    'Approximant',
    ('harmonic_modes', 'aligned_spins', 'tides', 'hplus_hcross_by_mode_func'),
    defaults=([(2, 2)], True, False, compute_hplus_hcross_by_mode))

APPROXIMANTS = {
    'IMRPhenomD_NRTidalv2': Approximant(tides=True),
    'IMRPhenomD': Approximant(),
    'IMRPhenomXPHM': Approximant(harmonic_modes=[(2, 2), (2, 1), (3, 3),
                                                 (3, 2), (4, 4)],
                                 aligned_spins=False),
    'IMRPhenomXAS': Approximant(),
    }


def inplane_spins_xy_n_to_xy(par_dic):
    """
    Rotate inplane spins (s1x_n, s1y_n) and (s2x_n, s2y_n) by an angle
    `-phi_ref` to get (s1x, s1y), (s2x, s2y).
    `par_dic` needs to have keys 's1x_n', 's1y_n', 's2x_n', 's2y_n'.
    Entries for 's1x', 's1y', 's2x', 's2y' will be added.

    `x_n`, `y_n` are axes perpendicular to the orbital angular momentum
    `L`, so that the line of sight `N` lies in the y-z plane, i.e.
        N = (0, sin(iota), cos(iota))
    in the (x_n, y_n, z) system.
    `x`, `y` are axes perpendicular to the orbital angular momentum `L`,
    so that the orbital separation is the x direction.
    The two systems coincide when `phi_ref=0`.
    """
    sin_phi_ref = np.sin(par_dic['phi_ref'])
    cos_phi_ref = np.cos(par_dic['phi_ref'])
    rotation = np.array([[cos_phi_ref, sin_phi_ref],
                         [-sin_phi_ref, cos_phi_ref]])

    ((par_dic['s1x'], par_dic['s2x']),
     (par_dic['s1y'], par_dic['s2y'])
        ) = rotation.dot(((par_dic['s1x_n'], par_dic['s2x_n']),
                          (par_dic['s1y_n'], par_dic['s2y_n'])))


def inplane_spins_xy_to_xy_n(par_dic):
    """
    Rotate inplane spins (s1x, s1y) and (s2x, s2y) by an angle
    `phi_ref` to get (s1x_n, s1y_n), (s2x_n, s2y_n).
    `par_dic` needs to have keys 's1x', 's1y', 's2x', 's2y'.
    Entries for 's1x_n', 's1y_n', 's2x_n', 's2y_n' will be added.

    `x_n`, `y_n` are axes perpendicular to the orbital angular momentum
    `L`, so that the line of sight `N` lies in the y-z plane, i.e.
        N = (0, sin(iota), cos(iota))
    in the (x_n, y_n, z) system.
    `x`, `y` are axes perpendicular to the orbital angular momentum `L`,
    so that the orbital separation is the x direction.
    The two systems coincide when `phi_ref=0`.
    """
    sin_phi_ref = np.sin(par_dic['phi_ref'])
    cos_phi_ref = np.cos(par_dic['phi_ref'])
    rotation = np.array([[cos_phi_ref, -sin_phi_ref],
                         [sin_phi_ref, cos_phi_ref]])

    ((par_dic['s1x_n'], par_dic['s2x_n']),
     (par_dic['s1y_n'], par_dic['s2y_n'])
        ) = rotation.dot(((par_dic['s1x'], par_dic['s2x']),
                          (par_dic['s1y'], par_dic['s2y'])))


def within_bounds(par_dic: dict) -> bool:
    """
    Return whether parameters in `par_dic` are within physical bounds.
    """
    return (all(par_dic[positive] >= 0
                for positive in {'m1', 'm2', 'd_luminosity', 'l1', 'l2', 'iota'
                                }.intersection(par_dic))
            and np.all(np.linalg.norm(
                [(par_dic['s1x_n'], par_dic['s1y_n'], par_dic['s1z']),
                 (par_dic['s2x_n'], par_dic['s2y_n'], par_dic['s2z'])],
                axis=1) <= 1)
            and par_dic.get('iota', 0) <= np.pi
            and np.abs(par_dic.get('dec', 0)) <= np.pi/2
           )


class WaveformGenerator(utils.JSONMixin):
    """
    Class that provides methods for generating frequency domain
    waveforms, in terms of `hplus, hcross` or projected onto detectors.
    "Fast" and "slow" parameters are distinguished: the last waveform
    calls are cached and can be computed fast when only fast parameters
    are changed.
    The attribute `n_cached_waveforms` can be used to control how many
    waveform calls to save in the cache.
    The boolean attribute `disable_precession` can be set to ignore
    inplane spins.
    """
    params = sorted(['d_luminosity', 'dec', 'f_ref', 'iota', 'l1', 'l2',
                     'm1', 'm2', 'psi', 'ra', 's1x_n', 's1y_n', 's1z',
                     's2x_n', 's2y_n', 's2z', 't_geocenter', 'phi_ref'])

    fast_params = sorted(['d_luminosity', 'dec', 'psi', 'ra', 't_geocenter',
                          'phi_ref'])
    slow_params = sorted(set(params) - set(fast_params))

    _projection_params = sorted(['dec', 'psi', 'ra', 't_geocenter'])
    _waveform_params = sorted(set(params) - set(_projection_params))
    polarization_params = sorted(set(params) - {'psi'})

    def __init__(self, detector_names, tgps, tcoarse, approximant,
                 harmonic_modes=None, disable_precession=False,
                 n_cached_waveforms=1, lalsimulation_commands=()):
        super().__init__()

        if approximant == 'IMRPhenomXODE':
            from cogwheel.waveform_models import xode as _  # TODO more elegant

        self.detector_names = tuple(detector_names)
        self.tgps = tgps
        self.tcoarse = tcoarse
        self._approximant = approximant
        self.harmonic_modes = harmonic_modes
        self.disable_precession = disable_precession
        self.lalsimulation_commands = lalsimulation_commands
        self.n_cached_waveforms = n_cached_waveforms

        self.n_slow_evaluations = 0
        self.n_fast_evaluations = 0

        self._cached_f = None

    @classmethod
    def from_event_data(cls, event_data, approximant,
                        harmonic_modes=None, disable_precession=False,
                        n_cached_waveforms=1, lalsimulation_commands=()):
        """
        Constructor that takes `detector_names`, `tgps` and `tcoarse`
        from an instance of `data.EventData`.
        """
        return cls(event_data.detector_names, event_data.tgps,
                   event_data.tcoarse, approximant, harmonic_modes,
                   disable_precession, n_cached_waveforms,
                   lalsimulation_commands)

    @property
    def approximant(self):
        """String with waveform approximant name."""
        return self._approximant

    @approximant.setter
    def approximant(self, approximant: str):
        """
        Set `approximant` and reset `harmonic_modes` per
        `APPROXIMANTS[approximant].harmonic_modes`; print a warning that
        this was done.
        Raise `ValueError` if `APPROXIMANTS` does not contain the
        requested approximant.
        """
        if approximant not in APPROXIMANTS:
            raise ValueError(f'Add {approximant} to `waveform.APPROXIMANTS`.')
        self._approximant = approximant

        old_harmonic_modes = self.harmonic_modes
        self.harmonic_modes = None
        if self.harmonic_modes != old_harmonic_modes:
            print(f'`approximant` changed to {approximant!r}, setting'
                  f'`harmonic_modes` to {self.harmonic_modes}.')
        utils.clear_caches()

    @property
    def harmonic_modes(self):
        """List of `(l, m)` pairs."""
        return self._harmonic_modes

    @harmonic_modes.setter
    def harmonic_modes(self, harmonic_modes):
        """
        Set `self._harmonic_modes` implementing defaults based on the
        approximant, this requires hardcoding which modes are
        implemented by each approximant.
        Also set `self._harmonic_modes_by_m` with a dictionary whose
        keys are `m` and whose values are a list of `(l, m)` tuples with
        that `m`.
        """
        if harmonic_modes is None:
            harmonic_modes = APPROXIMANTS[self.approximant].harmonic_modes
        else:
            harmonic_modes = [tuple(mode) for mode in harmonic_modes]
        self._harmonic_modes = harmonic_modes

        self._harmonic_modes_by_m = defaultdict(list)
        for l, m in self._harmonic_modes:
            self._harmonic_modes_by_m[m].append((l, m))
        utils.clear_caches()

    @property
    def m_arr(self):
        """Int array of m harmonic mode numbers."""
        return np.fromiter(self._harmonic_modes_by_m, int)

    @property
    def n_cached_waveforms(self):
        """Nonnegative integer, number of cached waveforms."""
        return self._n_cached_waveforms

    @n_cached_waveforms.setter
    def n_cached_waveforms(self, n_cached_waveforms):
        self.cache = [{'slow_par_vals': np.array(np.nan),
                       'approximant': None,
                       'f': None,
                       'harmonic_modes_by_m': {},
                       'hplus_hcross_0': None,
                       'lalsimulation_commands': ()}
                      for _ in range(n_cached_waveforms)]
        self._n_cached_waveforms = n_cached_waveforms

    @property
    def lalsimulation_commands(self):
        """
        Tuple of `(key, value)` where `key` is the name of a
        `lalsimulation` function and `value` is its second argument,\
        after `lal_dic`.
        """
        return self._lalsimulation_commands

    @lalsimulation_commands.setter
    def lalsimulation_commands(self, lalsimulation_commands):
        self._lalsimulation_commands = lalsimulation_commands
        utils.clear_caches()

    def get_m_mprime_inds(self):
        """
        Return two lists of integers, these zipped are pairs (i, j) of
        indices with j >= i that run through the number of m modes.
        """
        return map(list, zip(*itertools.combinations_with_replacement(
            range(len(self._harmonic_modes_by_m)), 2)))

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
        Array of shape (n_m?, n_detectors, n_frequencies) with strain at
        detector, `n_m` is there only if `by_m=True`.
        """
        # shape: (n_m?, 2, n_detectors, n_frequencies)
        hplus_hcross_at_detectors = self.get_hplus_hcross_at_detectors(
            f, par_dic, by_m)

        # fplus_fcross shape: (2, n_detectors)
        fplus_fcross = gw_utils.fplus_fcross(
            self.detector_names, par_dic['ra'], par_dic['dec'], par_dic['psi'],
            self.tgps)

        # Detector strain (n_m?, n_detectors, n_frequencies)
        return np.einsum('pd, ...pdf -> ...df',
                         fplus_fcross, hplus_hcross_at_detectors)

    def get_hplus_hcross_at_detectors(self, f, par_dic, by_m=False):
        """
        Return plus and cross polarizations with time shifts applied
        (but no fplus, fcross).

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        par_dic: parameter dictionary per `WaveformGenerator.params`.
        by_m: bool, whether to return waveform separated by `m`
              harmonic mode (summed over `l`), or already summed.

        Return
        ------
        Array of shape (n_m?, 2, n_detectors, n_frequencies) with hplus,
        hcross at detector, `n_m` is there only if `by_m=True`.
        """
        waveform_par_dic = {par: par_dic[par] for par in self._waveform_params}

        # hplus_hcross shape: (n_m?, 2, n_frequencies)
        hplus_hcross = self.get_hplus_hcross(f, waveform_par_dic, by_m)

        # shifts shape: (n_detectors, n_frequencies)
        if not np.array_equal(f, self._cached_f):
            self._get_shifts.cache_clear()
            self._cached_f = f
        shifts = self._get_shifts(par_dic['ra'], par_dic['dec'],
                                  par_dic['t_geocenter'])

        # hplus, hcross (n_m?, 2, n_detectors, n_frequencies)
        return np.einsum('...pf, df -> ...pdf', hplus_hcross, shifts)

    @utils.lru_cache(maxsize=16)
    def _get_shifts(self, ra, dec, t_geocenter):
        """Return (n_detectors, n_frequencies) array with e^(-2 i f t_det)."""
        time_delays = gw_utils.time_delay_from_geocenter(
            self.detector_names, ra, dec, self.tgps)
        return np.exp(-2j*np.pi * self._cached_f
                      * (self.tcoarse
                         + t_geocenter
                         + time_delays[:, np.newaxis]))

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False):
        """
        Return hplus, hcross waveform strain.
        Note: inplane spins will be zeroized if `self.disable_precession`
              is `True`.

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        waveform_par_dic: dictionary per
                          `WaveformGenerator._waveform_params`.
        by_m: bool, whether to return harmonic modes separately by m (l
              summed over) or all modes already summed over.

        Return
        ------
        array with (hplus, hcross), of shape `(2, len(f))` if `by_m` is
        `False`, or `(n_m, 2, len(f))` if `by_m` is `True`, where `n_m`
        is the number of harmonic modes with different `m`.
        """
        if self.disable_precession:
            waveform_par_dic.update(ZERO_INPLANE_SPINS)

        slow_par_vals = np.array([waveform_par_dic[par]
                                  for par in self.slow_params])

        # Attempt to use cached waveform for fast evaluation:
        if matching_cache := self._matching_cache(slow_par_vals, f):
            hplus_hcross_0 = matching_cache['hplus_hcross_0']
            self.n_fast_evaluations += 1
        else:
            # Compute the waveform mode by mode and update cache.
            lal_dic = self.create_lal_dict()

            waveform_par_dic_0 = dict(zip(self.slow_params, slow_par_vals),
                                      d_luminosity=1., phi_ref=0.)

            # hplus_hcross_0 is a (n_m x 2 x n_frequencies) array with
            # sum_l (hlm+, hlmx), at phi_ref=0, d_luminosity=1Mpc.
            hplus_hcross_modes \
                = APPROXIMANTS[self.approximant].hplus_hcross_by_mode_func(
                    f,
                    waveform_par_dic_0,
                    self.approximant,
                    self.harmonic_modes,
                    lal_dic)

            hplus_hcross_0 = np.array(
                [np.sum([hplus_hcross_modes[mode] for mode in m_modes], axis=0)
                 for m_modes in self._harmonic_modes_by_m.values()])

            cache_dic = {'approximant': self.approximant,
                         'f': f,
                         'slow_par_vals': slow_par_vals,
                         'harmonic_modes_by_m': self._harmonic_modes_by_m,
                         'hplus_hcross_0': hplus_hcross_0,
                         'lalsimulation_commands': self.lalsimulation_commands}

            # Append new cached waveform and delete oldest
            self.cache.append(cache_dic)
            self.cache.pop(0)

            self.n_slow_evaluations += 1

        # hplus_hcross is a (n_m x 2 x n_frequencies) array.
        m_arr = self.m_arr.reshape(-1, 1, 1)
        hplus_hcross = (np.exp(1j * waveform_par_dic['phi_ref'] * m_arr)
                        / waveform_par_dic['d_luminosity'] * hplus_hcross_0)
        if by_m:
            return hplus_hcross
        return np.sum(hplus_hcross, axis=0)

    def create_lal_dict(self):
        """Return a LAL dict object per ``self.lalsimulation_commands``."""
        lal_dic = lal.CreateDict()
        for function_name, value in self.lalsimulation_commands:
            getattr(lalsimulation, function_name)(lal_dic, value)
        return lal_dic

    def _matching_cache(self, slow_par_vals, f, eps=1e-6):
        """
        Return entry of the cache that matches the requested waveform, or
        `False` if none of the cached waveforms matches that requested.
        """
        for cache_dic in self.cache[ : : -1]:
            if (np.linalg.norm(slow_par_vals-cache_dic['slow_par_vals']) < eps
                    and cache_dic['approximant'] == self.approximant
                    and np.array_equal(cache_dic['f'], f)
                    and cache_dic['harmonic_modes_by_m']
                        == self._harmonic_modes_by_m
                    and cache_dic['lalsimulation_commands']
                        == self.lalsimulation_commands):
                return cache_dic

        return False
