"""
Functions to perform injections in reproducible Gaussian noise and do
parameter estimation on them.

This code assumes that injection samples were already generated using
the ``generate_injections.py`` module.

This module can run as a script and it also provides a function
``submit_slurm`` to submit the equivalent job to a SLURM scheduler.
"""
import argparse
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cogwheel import data
from cogwheel import gw_prior
from cogwheel import gw_utils
from cogwheel import posterior
from cogwheel import sampling
from cogwheel import gw_plotting
from cogwheel import utils
from . import load_config


def _get_eventname(config, i_sample):
    """Return a standardized event name to identify the injection."""
    n_digits = len(str(config.N_INJECTIONS - 1))
    return f'INJ{i_sample:0>{n_digits}}'


def _get_event_data(config, i_sample):
    """
    Return instance of EventData with reproducible Gaussian noise.

    Parameters
    ----------
    config: module
        Output of ``cogwheel.validation.load_config()``, contains
        configuration parameters.

    i_sample: int
        Used to seed the random Gaussian noise. In this way, the
        event data objects don't need to be uploaded in the release.
    """
    samples = pd.read_feather(config.INJECTION_DIR/config.INJECTIONS_FILENAME)
    sample = samples.iloc[i_sample]

    event_data = data.EventData.gaussian_noise(
        eventname=_get_eventname(config, i_sample),
        seed=(config.SEED, i_sample),
        **config.EVENT_DATA_KWARGS)

    event_data.inject_signal(dict(sample), config.APPROXIMANT)
    return event_data


def _get_eventdir(config, i_sample):
    """Return ``pathlib.Path`` object with injection event directory."""
    eventname = _get_eventname(config, i_sample)
    return config.INJECTION_DIR/eventname


def get_rundir(config, i_sample):
    """Return ``pathlib.Path`` object with injection run directory."""
    return utils.get_rundir(_get_eventdir(config, i_sample))


def _get_inj_id(eventdir):
    """Return ``i_sample``, inverse of ``_get_eventdir``."""
    eventdir = pathlib.Path(eventdir)
    if not eventdir.match('INJ*'):
        raise ValueError(f'Could not parse `eventdir` {eventdir}.')

    return int(eventdir.name.removeprefix('INJ'))


INJECTION_DICT_FILENAME = 'injection.json'
CORNER_PLOT_FILENAME = 'corner_plot.pdf'


def make_corner_plot(config, rundir):
    """
    Make a corner plot of the parameter estimation results and save it
    to ``rundir/CORNER_PLOT_FILENAME``.

    Coordinate transformations are applied to emphasize well-measured
    parameters. The injected parameters are marked. The injection json
    and the samples feather are updated with the inverse-transformed
    parameter values.
    """
    rundir = pathlib.Path(rundir)

    # Instantiate a dummy prior to use its inverse transform
    # (for plotting well-measured parameters)
    _ias_prior = gw_prior.IASPrior(
        mchirp_range=(1, 2),  # Doesn't matter for this purpose
        detector_pair='HL',
        tgps=config.EVENT_DATA_KWARGS['tgps'],
        f_ref=config.PRIOR_KWARGS['f_ref'],
        f_avg=config.PRIOR_KWARGS['f_ref'],
        ref_det_name='L')

    # Inverse-transform and save injection:
    with open(rundir/INJECTION_DICT_FILENAME, encoding='utf-8') as file:
        injection = json.load(file)['par_dic']

    injection.update(_ias_prior.inverse_transform(
        **{par: injection[par] for par in _ias_prior.standard_params}))

    with open(rundir/INJECTION_DICT_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(injection, file)

    # Inverse-transform and save paremeter estimation samples:
    samples_filename = rundir/sampling.SAMPLES_FILENAME
    pe_samples = pd.read_feather(samples_filename)
    _ias_prior.inverse_transform_samples(pe_samples)
    pe_samples.to_feather(samples_filename)

    # Plot and save:
    plot_params = _ias_prior.sampled_params + ['lnl', 'h_h']
    corner_plot = gw_plotting.CornerPlot(pe_samples[plot_params])
    corner_plot.plot(title=rundir.parent.name, max_n_ticks=3)
    corner_plot.scatter_points(injection, adjust_lims=True,
                               colors=['C3'], marker='+', zorder=2, s=50)
    plt.savefig(rundir/'corner_plot.pdf', bbox_inches='tight')


def submit_slurm(config_filename, i_sample, n_hours_limit=12,
                 memory_per_task='16G', sbatch_cmds=()):
    """
    Submit a job to SLURM that performs an injection in reproducible
    Gaussian noise and estimates its parameters.

    Parameters
    ----------
    config_filename: str, os.PathLike
        Path to a config file.

    i_sample: int
        Index of the sample within the injection set.

    n_hours_limit: int
        Number of hours to allocate for the job.

    memory_per_task: str
        Determines the memory and number of CPUs.

    sbatch_cmds: tuple of str
        Strings with SBATCH commands.
    """
    config = load_config(config_filename)
    rundir = get_rundir(config, i_sample)
    job_name = '-'.join(rundir.parts[-2:])
    batch_path = rundir/'batchfile'
    stdout_path = rundir.joinpath('output.out').resolve()
    stderr_path = rundir.joinpath('errors.err').resolve()
    sbatch_cmds += (f'--mem-per-cpu={memory_per_task}',)
    args = f'{config_filename} {rundir.resolve().as_posix()}'

    rundir.mkdir(parents=True)
    utils.submit_slurm(job_name, n_hours_limit, stdout_path, stderr_path,
                       args, sbatch_cmds, batch_path)


def main(config_filename, rundir):
    """
    Perform an injection in reproducible Gaussian noise and estimate its
    parameters.

    Parameters
    ----------
    config_filename: str, os.PathLike
        Path to a config file.

    rundir: str, os.PathLike
        Run directory, output of ``get_rundir``.
    """
    rundir = pathlib.Path(rundir)
    config = load_config(config_filename)
    i_sample = _get_inj_id(rundir.parent)
    event_data = _get_event_data(config, i_sample)
    prior_kwargs = {par: config.PRIOR_KWARGS[par]
                    for par in config.PRIOR_KWARGS.keys() - {'mchirp_range'}}
    post = posterior.Posterior.from_event(
        event_data,
        mchirp_guess=None,
        approximant=config.APPROXIMANT,
        prior_class=config.PE_PRIOR_CLS,
        prior_kwargs=prior_kwargs,
        ref_wf_finder_kwargs={'time_range': (-.15, .15)})

    print('', flush=True)  # Flush maximization log before starting the sampler

    # Declare failure if the mchirp range doesn't include the truth:
    injected_mchirp = gw_utils.m1m2_to_mchirp(
        event_data.injection['par_dic']['m1'],
        event_data.injection['par_dic']['m2'])
    if (injected_mchirp < post.prior.range_dic['mchirp'][0]
            or injected_mchirp > post.prior.range_dic['mchirp'][1]):
        raise RuntimeError('Failed to find a good mchirp range')

    # Ensure the mchirp prior range is contained in the injection range:
    mchirp_range = np.clip(post.prior.range_dic['mchirp'],
                           *config.PRIOR_KWARGS['mchirp_range'])
    post.prior = post.prior.reinstantiate(mchirp_range=mchirp_range)

    post.likelihood.asd_drift = None  # Set to 1

    sampler = config.SAMPLER_CLS(post, run_kwargs=config.RUN_KWARGS)
    sampler.run(rundir)

    # Save the injection parameters in rundir for extra safety
    event_data.injection['par_dic']['lnl'] = post.likelihood.lnlike_fft(
        event_data.injection['par_dic'])
    with open(rundir/INJECTION_DICT_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(event_data.injection, file, cls=utils.NumpyEncoder, indent=2)

    make_corner_plot(config, rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('config_filename', help='Path to a config file.')
    parser.add_argument('rundir', help='''
        Run directory. Must be of the correct form, e.g. output of
        ``get_rundir``.''')
    parser_args = parser.parse_args()
    main(**vars(parser.parse_args()))
