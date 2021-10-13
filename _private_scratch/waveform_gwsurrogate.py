"""Generate strain waveforms with gwsurrogate and project them onto detectors."""
import numpy as np
import scipy.signal as spsig
from copy import deepcopy as dcopy
import os
import sys
from . import harmonic_mode_utils as hm_utils

import gwsurrogate


COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
import gw_utils
import utils
import waveform

ZERO_INPLANE_SPINS = {'s1x': 0.,
                      's1y': 0.,
                      's2x': 0.,
                      's2y': 0.}

DEFAULT_PARS = {**ZERO_INPLANE_SPINS,
                's1z': 0.,
                's2z': 0.,
                'l1': 0.,
                'l2': 0.}

APPROXIMANTS = {}

class Approximant:
    """Bookkeeping of LAL approximants' metadata."""
    def __init__(self, approximant: str, harmonic_modes: list,
                 aligned_spins: bool, tides: bool):
        self.approximant = approximant
        self.harmonic_modes = harmonic_modes
        self.aligned_spins = aligned_spins
        self.tides = tides
        APPROXIMANTS[approximant] = self

# precessing surrogate model
APPROX_PRE = 'NRSur7dq4'
LOAD_SUR_PRE = True
SUR_PRE = (gwsurrogate.LoadSurrogate(APPROX_PRE) if LOAD_SUR_PRE == True else None)
#  =>  remaining precessing kwargs:  f_ref,  dt,  ellMax
ALLMODES_PRE = [(2, 2), (2, -2), (2, 1), (2, -1), (2, 0),
                (3, 3), (3, -3), (3, 2), (3, -2), (3, 1), (3, -1), (3, 0),
                (4, 4), (4, -4), (4, 3), (4, -3), (4, 2), (4, -2), (4, 1), (4, -1), (4, 0)]
Approximant(APPROX_PRE, [lm for lm in ALLMODES_PRE if lm[0] >= 0], False, False)

########################################################################
# non-precessing hybrid surrogate model
APPROX_HYB = 'NRHybSur3dq8'
LOAD_SUR_HYB = True
SUR_HYB = (gwsurrogate.LoadSurrogate(APPROX_HYB) if LOAD_SUR_HYB == True else None)
ALLMODES_HYB = [(2, 2), (2, -2), (2, 1), (2, -1),
                (3, 3), (3, -3), (3, 2), (3, -2), (3, 1), (3, -1),
                (4, 4), (4, -4), (4, 3), (4, -3), (4, 2), (4, -2), (5, 5), (5, -5)]
Approximant(APPROX_HYB, [lm for lm in ALLMODES_HYB if lm[0] > 0], True, False)


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

def argmaxnd(arr):
    return np.unravel_index(np.argmax(arr), arr.shape)

#### WINDOWING and padding
WINDOW_KWS_DEFAULTS = {'winfront':True, 'winback':True,
                       'frontfrac':0.1, 'ampfrac':0.01}

def marr_tail_pts(marr, ampfrac=0.01):
    """get smallest ntail s.t. zav[-ntail] > frac*max(zav), zav = (abs(z) + np.roll(abs(z),1))/2"""
    zamp = np.abs(marr)
    zamp = 0.5*(zamp + np.roll(zamp, 1, axis=-1))
    ampcuts = ampfrac * np.max(zamp, axis=-1)
    minus_ind = 1
    while np.all(zamp[..., -minus_ind] < ampcuts) == True:
        minus_ind += 1
    return minus_ind

def tukwin_front(nfront):
    return spsig.hann(2*nfront, sym=True)[:nfront]

def tukwin_back(nback):
    return spsig.hann(2*nback, sym=True)[-nback:]

def tukwin_npts(ntot, nwin):
    tukwin = np.ones(ntot)
    hannwin = spsig.hann(2*nwin, sym=True)
    tukwin[:nwin] = hannwin[:nwin]
    tukwin[-nwin:] = hannwin[-nwin:]
    return tukwin

def get_double_tukey(wfarr, nfront=None, ampfrac=0.01, frontfrac=0.1):
    if nfront is None:
        nfront = int(np.floor(wfarr.shape[-1] * frontfrac))
    tukout = np.ones_like(wfarr)
    tukout[..., :nfront] *= tukwin_front(nfront)
    nback = marr_tail_pts(wfarr, ampfrac=ampfrac)
    tukout[..., -nback:] *= tukwin_back(nback)
    return tukout

def apply_double_tukey(wfarr, winfront=True, winback=True,
                       ampfrac=0.01, nfront=None, frontfrac=0.1):
    if winfront:
        if nfront is None:
            nfront = int(np.floor(wfarr.shape[-1] * frontfrac))
        wfarr[..., :nfront] *= tukwin_front(nfront)
    if winback:
        nback = marr_tail_pts(wfarr, ampfrac=ampfrac)
        wfarr[..., -nback:] *= tukwin_back(nback)
    return

def tukwin_bandpass(taper_width, f_nyq=None, rfft_len=None, f_rfft=None):
    """
    taper_width is frequency interval length in Hz to be tapered at each end
    f_nyq & rfft_len are the nyquist frequency and length of the rfftfreq array
    """
    if f_rfft is None:
        return spsig.tukey(rfft_len, alpha=(2 * taper_width / f_nyq), sym=True)
    else:
        return spsig.tukey(len(f_rfft), alpha=(2 * taper_width / f_rfft[-1]), sym=True)

def zeropad_end(wfarr, pad_to_N):
    if pad_to_N == wfarr.shape[-1]:
        return wfarr
    elif pad_to_N > wfarr.shape[-1]:
        new_arr = np.zeros((*wfarr.shape[:-1], pad_to_N), dtype=wfarr.dtype)
        new_arr[..., :wfarr.shape[-1]] = wfarr
        return new_arr
    else:
        raise RuntimeError(f'wfarr of shape {wfarr.shape} cannot be padded to {pad_to_N}')


########    WAVEFORM GENERATION TECHNICAL PARAMETERS
# * f_ref * (float, hertz) is the frequency of the 22 mode at the reference epoch (<=> t_ref)
FREF_PE = 36.  # standardized reference frequency (Hz) for precessing PE
FREF_MB = 20.

# * dt * (float, seconds) is the inverse of the sampling rate for generating the waveform,
#  so it sets the true physical nyquist frequency of our fourier analysis
DT_MB = 1. / 1024.
DT_PE = 1. / 1024.

NFFT_MB = 2 ** 14
# * fcut22 * (float, hertz) is 22 frequency at the time when the front-end window reaches unity
# --> so, if no front window is applied, this is the same as the model's f_low variable,
#  but if the front is tukey-windowed, we have f_low < fcut22 so that we don't lose SNR from
#  applying the "smooth to zero" on times when the 22 signal is in a band >= fcut22
FCUT22_MB = 12.

CONST_SUR_KWS = {'units':'mks', 'dist_mpc':1, 'skip_param_checks':True,
                 'df':None, 'freqs':None, 'times':None, 'precessing_opts':None,
                 'tidal_opts':None, 'par_dict':None, 'taper_end_duration':None}

SUR_KWS_DEFAULTS_HYB = {'f_ref':FREF_MB, 'f_low':FCUT22_MB,
                        'inclination':None, 'phi_ref':None, 'ellMax':None}
SUR_KWS_DEFAULTS_PRE = {'f_ref':FREF_PE, 'f_low':0,
                        'inclination':None, 'phi_ref':None, 'mode_list':None}

DEFAULT_MARR_KWS = {APPROX_HYB: {**CONST_SUR_KWS, **SUR_KWS_DEFAULTS_HYB},
                    APPROX_PRE: {**CONST_SUR_KWS, **SUR_KWS_DEFAULTS_PRE}}
MODE_INPUT_KEY = {APPROX_HYB: 'mode_list', APPROX_PRE: 'ellMax'}
FUNDAMENTAL_MODE_INPUT = {APPROX_HYB: [(2, 2)], APPROX_PRE: 2}
HARMONIC_MODE_INPUT_FUNC = {APPROX_HYB: lambda lms: (None if lms is None else
                                                     [tuple(lm) for lm in lms]),
                            APPROX_PRE: lambda lms: (None if lms is None else
                                                     np.max([lm[0] for lm in lms]))}

#### GENERATING waveforms

def gen_wfdic_1mpc(par_dic, approximant, dt, f_ref, harmonic_modes=None,
                   return_t=False, max_inband=False, frontfrac=None,
                   **surrogate_kwargs):
    """
    generate UNWINDOWED mode dictionary from gwsurrogate model
    NOTE: max_inband likely to fail with precessing model, should be good for hybrid
    """
    if approximant == APPROX_PRE:
        use_sur = SUR_PRE
    elif approximant == APPROX_HYB:
        use_sur = SUR_HYB
    else:
        raise RuntimeError(f'{approximant} is not a valid approximant')
    tmarr_kws = {**DEFAULT_MARR_KWS[approximant], **surrogate_kwargs}
    tmarr_kws[MODE_INPUT_KEY[approximant]] = HARMONIC_MODE_INPUT_FUNC[approximant](harmonic_modes)
    tmarr_kws['dt'] = dt
    tmarr_kws['f_ref'] = f_ref
    tmarr_kws['dist_mpc'] = 1
    q1 = par_dic['m1'] / par_dic['m2']
    chi1 = [par_dic[k] for k in ['s1x', 's1y', 's1z']]
    chi2 = [par_dic[k] for k in ['s2x', 's2y', 's2z']]
    mt = par_dic['m1'] + par_dic['m2']

    if max_inband:
        # option for continuing to lower f_low until tukey window does not
        # touch the part of the waveform with f_22 >= original_f_low
        # **WARNING** for the precessing model this is likely to cause error
        kws22 = dcopy(tmarr_kws)
        kws22[MODE_INPUT_KEY[approximant]] = FUNDAMENTAL_MODE_INPUT[approximant]
        t, _, _ = use_sur(q1, chi1, chi2, M=mt, **kws22)
        if frontfrac is None:
            frontfrac = WINDOW_KWS_DEFAULTS['frontfrac']
        ntot = len(t) + int(np.floor(len(t) * frontfrac))
        if ntot % 2 != 0:
            ntot += 1
        f_low = tmarr_kws.pop('f_low') / 2.
        assert f_low > 0, 'max_inband version requires f_low > 0'
        t, wfdic, _ = use_sur(q1, chi1, chi2, M=mt, f_low=f_low, **tmarr_kws)
        while len(t) < ntot:
            f_low /= 2.
            t, wfdic, _ = use_sur(q1, chi1, chi2, M=mt, f_low=f_low, **tmarr_kws)
        for k in wfdic.keys():
            wfdic[k] = wfdic[k][-ntot:]
    else:
        t, wfdic, _ = use_sur(q1, chi1, chi2, M=mt, **tmarr_kws)
    return ((t, wfdic) if return_t else wfdic)


def compute_hplus_hcross(f_ref, f, par_dic, approximant: str,
                         harmonic_modes=None, window_kwargs={},
                         surrogate_kwargs={}):
    """
    Generate frequency domain waveform using gwsurrogate.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    approximant: APPROX_PRE or APPROX_HYB string specifying surrogate
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
    # match f to time grid
    dt = 0.5 / f[-1]
    df = np.min(np.diff(f))
    T = 1 / df
    nfft = int(np.ceil(T / dt))
    # ensure valid approximant
    if approximant == APPROX_PRE:
        use_sur = SUR_PRE
    elif approximant == APPROX_HYB:
        use_sur = SUR_HYB
    else:
        raise RuntimeError(f'{approximant} is not a valid approximant')
    tmarr_kws = {**DEFAULT_MARR_KWS[approximant], **surrogate_kwargs}
    tmarr_kws[MODE_INPUT_KEY[approximant]] = HARMONIC_MODE_INPUT_FUNC[approximant](harmonic_modes)
    tmarr_kws['dt'] = dt
    tmarr_kws['f_ref'] = f_ref
    tmarr_kws['dist_mpc'] = 1

    par_dic = {**DEFAULT_PARS, **par_dic}
    t, hpihc, _ = use_sur(par_dic['m1'] / par_dic['m2'], [par_dic[k] for k in ['s1x', 's1y', 's1z']],
                          [par_dic[k] for k in ['s2x', 's2y', 's2z']], M=par_dic['m1'] + par_dic['m2'],
                          phi_ref=par_dic['vphi'], inclination=par_dic['iota'], **tmarr_kws)
    # windowing -- control with window_kwargs dict, defaults to WINDOW_KWS_DEFAULTS
    apply_double_tukey(hpihc, **{**WINDOW_KWS_DEFAULTS, **window_kwargs})

    hp_hc_dimless = np.fft.rfft(np.roll(zeropad_end(np.array([hpihc.real, -hpihc.imag]), nfft),
                                        -np.searchsorted(t, 0), axis=-1), n=nfft, axis=-1)

    if f[0] == 0:
        # In this case we will set h(f=0) = 0
        hp_hc_dimless[:, 0] = 0
    fmask = np.isin(np.fft.rfftfreq(nfft, d=dt), f)
    return hp_hc_dimless[:, fmask] * dt / par_dic['d_luminosity']


class SurrogateWaveformGenerator(waveform.WaveformGenerator):
    """
    Class that provides methods for generating frequency domain
    waveforms, in terms of `hplus, hcross` or projected onto detectors.
    "Fast" and "slow" parameters are distinguished: the last waveform
    calls are cached and can be computed fast when only fast parameters
    are changed.
    The boolean attribute `disable_precession` can be set to ignore
    inplane spins.
    """
    def __init__(self, detector_names, tgps, tcoarse, approximant, f_ref,
                 harmonic_modes=None, disable_precession=False,
                 n_cached_waveforms=1, window_kwargs={}, surrogate_kwargs={}):
        super().__init__(detector_names, tgps, tcoarse, approximant, f_ref,
                         harmonic_modes=harmonic_modes, disable_precession=disable_precession,
                         n_cached_waveforms=n_cached_waveforms)
        self.window_kwargs = {**WINDOW_KWS_DEFAULTS, **window_kwargs}
        self.surrogate_kwargs = {**DEFAULT_MARR_KWS[approximant], **surrogate_kwargs}
        self.surrogate_kwargs['f_ref'] = f_ref

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False, **surrogate_kwargs):
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
        self._rotate_inplane_spins(slow_par_vals, waveform_par_dic['vphi'])

        # Attempt to use cached waveform for fast evaluation:
        matching_cache = self._matching_cache(slow_par_vals, f)
        if matching_cache:
            hplus_hcross_0 = matching_cache['hplus_hcross_0']
            self.n_fast_evaluations += 1
        else:
            # Compute the waveform mode by mode and update cache.
            waveform_par_dic_0 = dict(zip(self.slow_params, slow_par_vals),
                                      d_luminosity=1., vphi=0.)
            dt = 0.5 / f[-1]
            df = np.min(np.diff(f))
            T = 1 / df
            nfft = int(np.ceil(T / dt))
            #hm_utils.Y_lm()
            # hplus_hcross_0 is a (n_m x 2 x n_frequencies) arrays with
            # sum_l (hlm+, hlmx), at vphi=0, d_luminosity=1Mpc.
            waveform_times, waveform_mode_dic_0 = gen_wfdic_1mpc(waveform_par_dic_0,
                self.approximant, dt, self.f_ref, harmonic_modes=self.harmonic_modes,
                return_t=True, **surrogate_kwargs)
            ##################################################################
            hplus_hcross_0 = np.array(
                [compute_hplus_hcross(self.f_ref, f, waveform_par_dic_0,
                                      self.approximant, modes,
                                      window_kwargs=self.window_kwargs,
                                      surrogate_kwargs=self.surrogate_kwargs)
                 for modes in self._harmonic_modes_by_m.values()])
            cache_dic = {'approximant': self.approximant,
                         'f_ref': self.f_ref,
                         'f': f,
                         'slow_par_vals': slow_par_vals,
                         'harmonic_modes_by_m': self._harmonic_modes_by_m,
                         'hplus_hcross_0': hplus_hcross_0}
            # Append new cached waveform and delete oldest
            self.cache.append(cache_dic)
            self.cache.pop(0)

            self.n_slow_evaluations += 1

        # hplus_hcross is a (n_m x 2 x n_frequencies) array.
        m_arr = np.array(list(self._harmonic_modes_by_m)).reshape((-1, 1, 1))
        hplus_hcross = (np.exp(1j * m_arr * waveform_par_dic['vphi'])
                        / waveform_par_dic['d_luminosity'] * hplus_hcross_0)
        if by_m:
            return hplus_hcross
        return np.sum(hplus_hcross, axis=0)
