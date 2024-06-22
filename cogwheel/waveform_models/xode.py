"""
Interface cogwheel with IMRPhenomXODE.

For this module to run, you need to install IMRPhenomXODE as follows:
1. Clone the repo (https://github.com/hangyu45/IMRPhenomXODE)
   anywhere in your system.
2. Make a symbolic link to the IMRPhenomXODE repository:
```bash
cd <path_to_cogwheel>/cogwheel/waveform_models
ln -s <path_to_IMRPhenomXODE>/src/ IMRPhenomXODE
```
Note, IMRPhenomXODE requires a `lalsimulation` version >= 5.1.0

Example usage
-------------
```python
import pandas as pd

from cogwheel import data
from cogwheel import gw_plotting
from cogwheel import posterior
from cogwheel import sampling
import cogwheel.waveform_models.xode

parentdir = 'example'  # Parameter estimation runs will be saved here
eventname = 'GW151226'
mchirp_guess = data.EVENTS_METADATA['mchirp'][eventname]
post = posterior.Posterior.from_event(eventname,
                                      mchirp_guess,
                                      approximant='IMRPhenomXODE',
                                      prior_class='IntrinsicIASPrior')

pym = sampling.PyMultiNest(post, run_kwargs={'n_live_points': 256})
rundir = pym.get_rundir(parentdir)
pym.run(rundir)  # Will take a bit

samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
gw_plotting.CornerPlot(samples[post.prior.sampled_params]).plot()
```
"""
import numpy as np

from cogwheel import waveform

try:
    from .IMRPhenomXODE.waveLib import get_hp_hc_each_prec_mode_f_sequence
except ImportError as err:
    raise ImportError(
        'It seems that you are trying to use IMRPhenomXODE but did not '
        f'install it. Follow the instructions in {__file__} (reproduced'
        f' below)\n\n{__doc__}') from err

CONFIG = {'use_N4LO_prec': True,
          'SEOB_22_cal': True,
          'SEOB_HM_cal': True,
          'atol': 1e-3,
          'rtol': 1e-3}


def compute_hplus_hcross_by_mode_xode(f, par_dic,
                                      approximant='IMRPhenomXODE',
                                      harmonic_modes=None,
                                      lal_dic=None):
    """
    Generate frequency domain waveform with IMRPhenomXODE.
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
            * h1, h2: component dissipation numbers
            * s1z, s2z: dimensionless spins

    approximant: 'IMRPhenomXODE'
        ``ValueError`` is raised if it is not 'IMRPhenomXODE'.

    harmonic_modes: list of (int, int) pairs
        (l, m) numbers of the harmonic modes to include, must be
        from the ones accepted by IMRPhenomXPHM. Note: l >= m >= 0.

    lal_dic: LALDict, optional
        Contains special approximant settings.
    """
    if approximant != 'IMRPhenomXODE':
        raise ValueError('`approximant` must be "IMRPhenomXODE"`')

    par_dic = waveform.DEFAULT_PARS | par_dic

    # Transform inplane spins to LAL's coordinate system.
    waveform.inplane_spins_xy_n_to_xy(par_dic)
    if harmonic_modes is None:
        harmonic_modes = waveform.APPROXIMANTS['IMRPhenomXODE'].harmonic_modes

    ll_list_neg, mm_list_neg = np.transpose(harmonic_modes)
    mm_list_neg *= -1

    kwargs_xode = {**CONFIG,
                   'freqs': f,
                   'approximant': 'XODE',
                   'll_list_neg': ll_list_neg,
                   'mm_list_neg': mm_list_neg,
                   'mass1': par_dic['m1'],
                   'mass2': par_dic['m2'],
                   'spin1x': par_dic['s1x'],
                   'spin1y': par_dic['s1y'],
                   'spin1z': par_dic['s1z'],
                   'spin2x': par_dic['s2x'],
                   'spin2y': par_dic['s2y'],
                   'spin2z': par_dic['s2z'],
                   'f_ref': par_dic['f_ref'],
                   'phi_ref': par_dic['phi_ref'],
                   'iota': par_dic['iota'],
                   'f_lower': f[0],
                   'distance': par_dic['d_luminosity'],
                   'aux_par': lal_dic,
                   }

    try:
        h_pmf = get_hp_hc_each_prec_mode_f_sequence(**kwargs_xode)
    except Exception:
        print('Error while calling XODE at these parameters:', kwargs_xode)
        raise
    return dict(zip(harmonic_modes, np.moveaxis(h_pmf, 1, 0)))


waveform.APPROXIMANTS['IMRPhenomXODE'] = waveform.Approximant(
    harmonic_modes= [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)],
    aligned_spins=False,
    hplus_hcross_by_mode_func=compute_hplus_hcross_by_mode_xode)
