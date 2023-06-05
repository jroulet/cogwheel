import multiprocessing
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cogwheel import data
from cogwheel import likelihood
from cogwheel import waveform

from . import config
from .injection_prior import InjectionPrior


N_CORES = multiprocessing.cpu_count()  # How many cores to use, edit if desired

event_data = data.EventData.gaussian_noise('', **config.EVENT_DATA_KWARGS)

waveform_generator = waveform.WaveformGenerator.from_event_data(
    event_data, config.APPROXIMANT)


def generate_injections_above_threshold(mchirp_range):
    """
    Return a DataFrame with injections from the prior guaranteeing that
    ⟨ℎ∣ℎ⟩ is above ``config.H_H_MIN``, and that d_luminosity is below
    ``config.D_LUMINOSITY_MAX``.
    The number of injections is ``config.N_INJECTIONS_ABOVE_THRESHOLD``.
    """
    injs_above_threshold = pd.DataFrame()
    batch_size = 10 * config.N_INJECTIONS_ABOVE_THRESHOLD

    while len(injs_above_threshold) < config.N_INJECTIONS_ABOVE_THRESHOLD:
        injs_above_threshold = pd.concat(
            (injs_above_threshold,
             _generate_injections_above_threshold(mchirp_range, batch_size)),
            ignore_index=True)

    return injs_above_threshold[:config.N_INJECTIONS_ABOVE_THRESHOLD]


def _generate_injections_above_threshold(mchirp_range,
                                         n_total_injections):
    """
    Generate a batch of injections that pass the ⟨ℎ∣ℎ⟩ threshold and
    are within the maximum luminosity distance.
    The number of injections that will pass the thresholds is unknown
    in advance.

    Parameters
    ----------
    mchirp_range: (float, float)
        Minimum and maximum detector-frame chirp masses (Msun).

    n_total_injections: int
        Number of injections generated (before applying the ⟨ℎ∣ℎ⟩ cut).
    """
    injection_prior = InjectionPrior(mchirp_range=mchirp_range,
                                     detector_pair='H',
                                     tgps=0.,
                                     f_ref=config.F_REF)
    injections = injection_prior.generate_random_samples(n_total_injections)
    h_h_1mpc = _compute_h_h_1mpc(injections)

    # Choose physical units for distance, so that the loudest signal
    # placed at d_ref barely makes the threshold
    d_ref = np.sqrt(np.max(h_h_1mpc) / config.H_H_MIN)

    injections['d_luminosity'] = injections['dimensionless_distance'] * d_ref
    del injections['dimensionless_volume']
    del injections['dimensionless_distance']

    injections['h_h'] = h_h_1mpc / injections['d_luminosity']**2
    injections_above_threshold = injections[
        (injections['h_h'] > config.H_H_MIN)
        & (injections['d_luminosity'] < config.D_LUMINOSITY_MAX)
        ].reset_index(drop=True)
    return injections_above_threshold


def _compute_h_h_1mpc(injections, rtol=1e-3, n_cores=None):
    """
    Compute ⟨ℎ∣ℎ⟩ at 1 Mpc, ensuring a relative accuracy of `rtol`.

    Parameters
    ----------
    injections: pd.DataFrame
        Samples including all parameters except for time and distance.

    rtol: float
        Relative tolerance in ⟨ℎ∣ℎ⟩ computation.

    Return
    ------
    h_h_1mpc, relative_error: float arrays, same length as `injections`.
    """
    # Attempt relative binning and estimate accuracy
    pn_phase_tol = .01
    h_h_1mpc = _compute_h_h_1mpc_rb(injections, pn_phase_tol, n_cores)
    h_h_1mpc_lowres = _compute_h_h_1mpc_rb(injections, pn_phase_tol*2, n_cores)

    # Recompute inaccurate ones exactly
    recompute = np.abs(h_h_1mpc / h_h_1mpc_lowres - 1) > rtol
    h_h_1mpc[recompute] = _compute_h_h_1mpc_fft(injections[recompute], n_cores)

    return h_h_1mpc


def _compute_h_h_1mpc_rb(injections, pn_phase_tol, n_cores=None):
    """Compute ⟨ℎ∣ℎ⟩ at 1 Mpc using relative binning."""
    n_cores = n_cores or N_CORES

    par_dic_0 = {**injections.median(), 'd_luminosity': 1}

    like = likelihood.RelativeBinningLikelihood(
        event_data, waveform_generator, par_dic_0, pn_phase_tol=pn_phase_tol)

    par_dics = ({**row, 'd_luminosity': 1}
                for _, row in injections.iterrows())

    with multiprocessing.Pool(n_cores) as pool:
        h_h_1mpc = pool.starmap(_get_h_h_rb,
                                ((like, par_dic) for par_dic in par_dics))
    return np.array(h_h_1mpc)


def _get_h_h_rb(like, par_dic):
    return np.sum(like._get_dh_hh_no_asd_drift(par_dic)[1])


def _compute_h_h_1mpc_fft(injections, n_cores=None):
    """Compute ⟨ℎ∣ℎ⟩ at 1 Mpc without using relative binning."""
    n_cores = n_cores or N_CORES

    like = likelihood.CBCLikelihood(event_data, waveform_generator)
    like.asd_drift = None  # Sets it to 1

    par_dics = ({**row, 'd_luminosity': 1}
                for _, row in injections.iterrows())

    with multiprocessing.Pool(n_cores) as pool:
        h_h_1mpc = pool.starmap(_get_h_h_fft,
                                ((like, par_dic) for par_dic in par_dics))
    return np.array(h_h_1mpc)


def _get_h_h_fft(like, par_dic):
    return np.sum(like._compute_h_h(like._get_h_f(par_dic)))


def test_h_h_distribution(injections_above_threshold):
    """
    Plot test that distribution of ⟨ℎ∣ℎ⟩ is as expected.

    Parameters
    ----------
    injections_above_threshold: pandas.DataFrame
        Output of ``generate_injections_above_threshold()``.
    """
    h_h_distribution = stats.pareto(b=1.5, scale=config.H_H_MIN)

    plt.figure()
    x = np.geomspace(config.H_H_MIN,
                     1.2 * injections_above_threshold['h_h'].max(),
                     100)

    # Predicted distribution
    plt.plot(x, h_h_distribution.pdf(x), c='C1', label='Prediction')

    # Monte Carlo of what distribution should look like with these many samples
    for i in range(20):
        plt.hist(h_h_distribution.rvs(size=len(injections_above_threshold)),
                 bins=x, density=True, histtype='step', alpha=.3,
                 label='Monte Carlo' if i == 0 else None)

    # Actual outcome
    plt.hist(injections_above_threshold['h_h'],
             bins=x, density=True, label='Outcome')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(rf'$\langle h \mid h \rangle_{{>{config.H_H_MIN}}}$')
    plt.ylabel('PDF')
    plt.title(rf'$N_{{\rm samples}} = {len(injections_above_threshold)}$')
