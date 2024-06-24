"""P-P plots."""

import json
from pstats import Stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cogwheel.validation import inference

from cogwheel import gw_plotting
from cogwheel import gw_utils
from cogwheel import sampling
from cogwheel import utils


def pp_plot(credible_intervals, params=None, ax=None,
            show_xy_labels=True, show_title=True, show_legend=True):
    """
    Make a probability-probability plot.

    credible_intervals: pandas.DataFrame
        Output of ``get_credible_intervals``.

    params: sequence of str, optional
        Subset of parameters to plot.

    ax: matplotlib.axes.Axes, optional
        Where to draw the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    if params is None:
        params = list(credible_intervals)

    clean_credible_intervals = credible_intervals.dropna()
    for par in params:
        sorted_credible_intervals = np.sort(clean_credible_intervals[par])
        ax.plot(sorted_credible_intervals,
                np.linspace(0, 1, len(clean_credible_intervals)),
                label=gw_plotting.CornerPlot.DEFAULT_LATEX_LABELS[par], lw=1)
    ax.plot((0, 1), (0, 1), 'k:')  # Diagonal line

    if show_title:
        ax.set_title(f'$N = {len(clean_credible_intervals)} / '
                     f'{len(credible_intervals)}$',
                     fontsize='medium')

    ax.tick_params(axis='x', direction='in', top=True)
    ax.tick_params(axis='y', direction='in', right=True)
    ax.grid(linestyle=':')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if show_legend:
        ax.legend(fontsize=10, frameon=True, framealpha=.5, labelspacing=0.25,
                  loc='upper left', edgecolor='none', borderpad=0.3)

    if show_xy_labels:
        ax.set_xlabel('Credible interval')
        ax.set_ylabel('Fraction of injections in credible interval')


def get_credible_intervals(config, params=None,
                           min_required_samples=30):
    """
    Compute credible interval at which the injected value is recovered
    for multiple parameters and injections.

    Parameters
    ----------
    config: module
        Output of ``load_config``.

    params: sequence of str, optional
        Parameters for which to compute credible intervals. Defaults to
        all parameters available in both the injection json and samples.

    min_required_samples: int
        Minimum number of samples with ⟨ℎ∣ℎ⟩ > config.H_H_MIN to
        tolerate.

    Return
    ------
    credible_intervals: pandas.DataFrame
        Columns correspond to different source parameters and rows to
        different injections.
    """
    rundirs = sorted(config.INJECTION_DIR.glob('runs/INJ*/run_*/'))
    return pd.DataFrame.from_records(
        _get_credible_intervals(config, rundir, params, min_required_samples)
        for rundir in rundirs)


def _get_credible_intervals(config, rundir, params=None,
                            min_required_samples=30):
    """
    Compute credible interval for multiple parameters for a single
    injection.

    Parameters
    ----------
    config: module
        Output of ``load_config``.

    rundir: os.PathLike
        Path to run directory containing samples and injection json.

    params: sequence of str, optional
        Parameters for which to compute credible intervals. Defaults to
        all parameters available in both the injection json and samples.

    min_required_samples: int
        Minimum number of samples with ⟨ℎ∣ℎ⟩ > config.H_H_MIN to
        tolerate.

    Return
    ------
    dict of the form {par: credible_interval}, or empty dict if
    something fails.
    """
    try:
        samples = _get_samples(config, rundir)
    except FileNotFoundError:
        print(f'Skipping {rundir} with no samples...')
        return {}

    if (n_loud := len(samples)) < min_required_samples:
        print(f'Skipping {rundir} with only {n_loud} samples with '
              f'⟨ℎ∣ℎ⟩ above {config.H_H_MIN}.')
        return {}

    with open(rundir/inference.INJECTION_DICT_FILENAME,
              encoding='utf-8') as file:
        injection = json.load(file)['par_dic']

    if params is None:
        params = [par for par in injection
                  if par in samples and np.ptp(samples[par]) != 0]

    return {par: credible_interval(samples[par],
                                   injection[par],
                                   samples.get(utils.WEIGHTS_NAME))
            for par in params}


def credible_interval(arr, value, weights=None):
    """
    Parameters
    ----------
    arr: float array
        Posterior samples.

    value: float
        Truth.

    weights: float array, optional
        Sample weights, has same shape as `arr`.

    Return
    ------
    credible_interval: float between 0 and 1.
        (Weighted) fraction of `arr` whose values are below `value`.
    """
    if weights is None:
        weights = np.ones_like(arr)

    return weights[arr < value].sum() / weights.sum()


def _get_samples(config, rundir):
    """Load samples ensuring ⟨ℎ∣ℎ⟩ > ``config.H_H_MIN``."""
    samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
    samples = samples[samples['weights'] > 0]

    # mchirp and q should be within injection prior, verify:
    if 'mchirp_range' in config.PRIOR_KWARGS:
        samples['mchirp'] = gw_utils.m1m2_to_mchirp(**samples[['m1', 'm2']])
        # +/- 1e-4 because we uploaded samples to zenodo in single precision
        np.testing.assert_array_compare(np.greater_equal,
                                        samples['mchirp'] + 1e-4,
                                        config.PRIOR_KWARGS['mchirp_range'][0])
        np.testing.assert_array_compare(np.less_equal,
                                        samples['mchirp'] - 1e-4,
                                        config.PRIOR_KWARGS['mchirp_range'][1])
    if 'q_min' in config.PRIOR_KWARGS:
        np.testing.assert_array_compare(np.greater_equal,
                                        samples['m2'] / samples['m1'],
                                        config.PRIOR_KWARGS['q_min'])

    # Some samples may have ⟨ℎ∣ℎ⟩ < config.H_H_MIN, discard them:
    return samples[(samples['h_h'] > config.H_H_MIN)]


def get_runtimes(config):
    """
    Return array of parameter estimation runtimes in hours.

    Note: excludes time spent in setting up each posterior object,
    which should be ~a couple minutes.
    """
    profilings = sorted(config.INJECTION_DIR.glob('runs/INJ*/run_*/profiling'))
    runtimes = [Stats(path.as_posix()).total_tt / 3600 for path in profilings]
    return np.array(runtimes)


def get_n_effectives(config):
    """
    Return array of effective sample size of the posteriors.
    There is one entry for each posterior.
    """
    feathers = sorted(
        config.INJECTION_DIR.glob('runs/INJ*/run_*/samples.feather'))
    n_effectives = [_n_effective(pd.read_feather(feather))
                    for feather in feathers]
    return np.array(n_effectives)


def _n_effective(samples):
    if utils.WEIGHTS_NAME in samples:
        return utils.n_effective(samples[utils.WEIGHTS_NAME])
    return len(samples)
