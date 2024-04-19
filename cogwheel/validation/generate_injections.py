"""
Generate injection samples drawn from a user-prescribed distribution.

Usage:
    1. Make a config file (see `cogwheel/validation/example/config.py`)
    2. Call ``generate_injections_from_config``
"""

import argparse
import multiprocessing
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cogwheel import data
from cogwheel import waveform
import cogwheel.likelihood
from cogwheel.validation import load_config


def generate_injections_from_config(config,
                                    n_cores=multiprocessing.cpu_count(),
                                    overwrite=False):
    """
    Create a pandas DataFrame with injection samples and save it in
    ``config.INJECTION_DIR/config.INJECTIONS_FILENAME``.

    Parameters
    ----------
    config: module
        Output of ``cogwheel.validation.load_config``, contains
        configuration parameters.

    n_cores: int
        Number of computing cores to use.

    overwrite: bool
        Whether to overwrite the injections file if it already exists.
    """
    injections_path = config.INJECTION_DIR/config.INJECTIONS_FILENAME
    if not overwrite and injections_path.exists():
        raise FileExistsError(f'{injections_path.as_posix()} exists, '
                              'pass `overwrite=True` to overwrite')

    print(f'Generating injections in {injections_path.as_posix()}...')
    injections_above_threshold = _generate_injections_above_threshold(config,
                                                                      n_cores)
    injections_above_threshold.to_feather(injections_path)


def _generate_injections_above_threshold(config, n_cores):
    """
    Return a DataFrame with injections from the prior guaranteeing that
    ⟨ℎ∣ℎ⟩ is above ``config.H_H_MIN``, and that d_luminosity is below
    ``config.D_LUMINOSITY_MAX``.
    The number of injections is ``config.N_INJECTIONS``.
    """
    injs_above_threshold = pd.DataFrame()
    batch_size = 10 * config.N_INJECTIONS

    event_data = data.EventData.gaussian_noise('', **config.EVENT_DATA_KWARGS)
    waveform_generator = waveform.WaveformGenerator.from_event_data(
        event_data, config.APPROXIMANT)
    likelihood = cogwheel.likelihood.CBCLikelihood(event_data,
                                                   waveform_generator)
    likelihood.asd_drift = None  # Sets it to 1

    while len(injs_above_threshold) < config.N_INJECTIONS:
        batch = _batch_of_injections_above_threshold(config, batch_size,
                                                     likelihood, n_cores)
        injs_above_threshold = pd.concat((injs_above_threshold, batch),
                                         ignore_index=True)

    return injs_above_threshold[:config.N_INJECTIONS]


def _batch_of_injections_above_threshold(config, batch_size, likelihood,
                                         n_cores):
    """
    Generate a batch of injections that pass the ⟨ℎ∣ℎ⟩ threshold and are
    within the maximum luminosity distance.
    The number of injections that will pass the thresholds is unknown in
    advance.

    Parameters
    ----------
    config: module
        Contains configuration parameters.

    batch_size: int
        Number of injections generated (before applying the ⟨ℎ∣ℎ⟩ cut).

    likelihood: cogwheel.likelihood.CBCLikelihood
        Use its waveform generator and PSD for computing ⟨ℎ∣ℎ⟩.

    n_cores: int
        Number of computing cores to use.
    """
    injection_prior = config.INJECTION_PRIOR_CLS(**config.PRIOR_KWARGS)
    injections = injection_prior.generate_random_samples(batch_size)
    h_h_1mpc = _compute_h_h_1mpc(injections, likelihood, n_cores)

    # Choose physical units for distance, so that the if the loudest
    # signal was placed at d_ref it would barely make the threshold
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


def _compute_h_h_1mpc(injections, likelihood, n_cores, rtol=1e-3):
    """
    Compute ⟨ℎ∣ℎ⟩ at 1 Mpc, ensuring a relative accuracy of `rtol`.

    Parameters
    ----------
    injections: pd.DataFrame
        Samples including all parameters except for time and distance.

    likelihood: cogwheel.likelihood.CBCLikelihood
        Use its waveform generator and PSD for computing ⟨ℎ∣ℎ⟩.

    n_cores: int
        Number of computing cores to use.

    rtol: float
        Relative tolerance in ⟨ℎ∣ℎ⟩ computation.

    Return
    ------
    h_h_1mpc: float array, same length as `injections`.
    """
    # Attempt relative binning and estimate accuracy
    pn_phase_tol = .01
    h_h_1mpc = _compute_h_h_1mpc_rb(injections, pn_phase_tol, likelihood,
                                    n_cores)
    h_h_1mpc_lowres = _compute_h_h_1mpc_rb(injections, pn_phase_tol*2,
                                           likelihood, n_cores)

    # Recompute inaccurate ones exactly
    recompute = np.abs(h_h_1mpc / h_h_1mpc_lowres - 1) > rtol
    h_h_1mpc[recompute] = _compute_h_h_1mpc_fft(injections[recompute],
                                                likelihood, n_cores)

    return h_h_1mpc


def _compute_h_h_1mpc_rb(injections, pn_phase_tol, likelihood, n_cores):
    """Compute ⟨ℎ∣ℎ⟩ at 1 Mpc using relative binning."""
    par_dic_0 = {**injections.median(), 'd_luminosity': 1}

    likelihood_rb = cogwheel.likelihood.RelativeBinningLikelihood(
        **likelihood.get_init_dict(),
        par_dic_0=par_dic_0, pn_phase_tol=pn_phase_tol)
    likelihood_rb.asd_drift = None  # Sets it to 1

    par_dics = ({**row, 'd_luminosity': 1} for _, row in injections.iterrows())
    args = ((likelihood_rb, par_dic) for par_dic in par_dics)

    h_h_1mpc = _starmap(_get_h_h_rb, args, n_cores)
    return np.array(h_h_1mpc)


def _get_h_h_rb(likelihood, par_dic):
    return np.sum(likelihood._get_dh_hh_no_asd_drift(par_dic)[1])


def _compute_h_h_1mpc_fft(injections, likelihood, n_cores):
    """Compute ⟨ℎ∣ℎ⟩ at 1 Mpc without using relative binning."""
    par_dics = ({**row, 'd_luminosity': 1} for _, row in injections.iterrows())
    args = ((likelihood, par_dic) for par_dic in par_dics)
    h_h_1mpc = _starmap(_get_h_h_fft, args, n_cores)
    return np.array(h_h_1mpc)


def _get_h_h_fft(likelihood, par_dic):
    h_f = likelihood._get_h_f(par_dic)
    h_h = likelihood._compute_h_h(h_f)
    return h_h.sum()


def _starmap(func, iterable, n_cores):
    """
    Similar to ``multiprocessing.Pool.starmap`` but if n_cores == 1 it
    just does a for loop. Useful hack if ``multiprocessing`` misbehaves.
    """
    if n_cores != 1:
        with multiprocessing.Pool(n_cores) as pool:
            return pool.starmap(func, iterable)
    else:
        return [func(*args) for args in iterable]


def test_h_h_distribution(config):
    """
    Plot test that distribution of ⟨ℎ∣ℎ⟩ is as expected.
    It is assumed that injections have already been generated (see
    ``generate_injections_from_config``).

    Parameters
    ----------
    config: module
        Output of ``cogwheel.validation.load_config()``, contains
        configuration parameters.
    """
    injections_above_threshold = pd.read_feather(
        config.INJECTION_DIR/config.INJECTIONS_FILENAME)
    h_h_distribution = stats.pareto(b=1.5, scale=config.H_H_MIN)

    plt.figure()
    x = np.geomspace(config.H_H_MIN,
                     1.2 * injections_above_threshold['h_h'].max(),
                     100)

    # Predicted distribution
    plt.plot(x, h_h_distribution.pdf(x), c='C1', label='Prediction')

    # Monte Carlo of how distribution should look like with these many samples
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


def main(config_filename, n_cores=0, overwrite=False):
    """
    Create a pandas DataFrame with injection samples and save it in
    ``config.INJECTION_DIR/config.INJECTIONS_FILENAME``.
    Also plot tests of the distribution of ⟨ℎ∣ℎ⟩.
    """
    if n_cores <= 0:
        n_cores += multiprocessing.cpu_count()

    config = load_config(config_filename)

    generate_injections_from_config(config, n_cores, overwrite)
    test_h_h_distribution(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Generate injection samples drawn from a user-prescribed distribution.
        """)

    parser.add_argument('config_filename', help='Path to a config file.')
    parser.add_argument('n_cores',
                        help='Number of computing cores for parallelization.',
                        default=0, type=int)

    parser.add_argument('--overwrite', action='store_true',
                        help='pass to overwrite existing injections file')

    main(**vars(parser.parse_args()))
