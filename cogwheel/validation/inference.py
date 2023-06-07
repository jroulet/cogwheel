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
from cogwheel import posterior
from cogwheel import sampling
from cogwheel import gw_plotting
from cogwheel import utils

from . import config


def _get_eventname(i_inj_set, i_sample):
    """Return a standardized event name to identify the injection."""
    return f'INJ{i_inj_set}.{i_sample:0>4}'


def _get_event_data(i_inj_set, i_sample):
    """
    Return instance of EventData with reproducible Gaussian noise.

    i_inj_set, i_sample: int
        Used to seed the random Gaussian noise. In this way, the
        event data objects don't need to be uploaded in the release.
    """
    inj_dir = config.INJECTION_SET_DIRS[i_inj_set]
    samples = pd.read_feather(inj_dir/config.INJECTIONS_FILENAME)
    sample = samples.iloc[i_sample]

    event_data = data.EventData.gaussian_noise(
        eventname=_get_eventname(i_inj_set, i_sample),
        seed=(i_inj_set, i_sample),
        **config.EVENT_DATA_KWARGS)

    event_data.inject_signal(dict(sample), config.APPROXIMANT)
    return event_data


def _get_eventdir(i_inj_set, i_sample):
    """Return ``pathlib.Path`` object with injection event directory."""
    eventname = _get_eventname(i_inj_set, i_sample)
    return config.INJECTION_SET_DIRS[i_inj_set]/config.PRIOR_NAME/eventname


def _get_rundir(i_inj_set, i_sample):
    """Return ``pathlib.Path`` object with injection run directory."""
    return utils.get_rundir(_get_eventdir(i_inj_set, i_sample))


def _get_inj_id(eventdir):
    """Return ``i_inj_set, i_sample``, inverse of ``_get_eventdir``."""
    eventdir = pathlib.Path(eventdir)
    if not eventdir.match(_get_eventname('*', '****')):
        raise ValueError(f'Could not parse `eventdir` {eventdir}.')

    i_inj_set, i_sample = map(int,
                              eventdir.name.removeprefix('INJ').split('.'))

    assert eventdir.resolve() == _get_eventdir(i_inj_set, i_sample).resolve()

    return i_inj_set, i_sample


# Instantiate a dummy prior to use its inverse transform
# for plotting well-measured parameters
_ias_prior = gw_prior.IASPrior(
    mchirp_range=(1, 2),  # Doesn't matter for this purpose
    detector_pair='HL',
    tgps=config.EVENT_DATA_KWARGS['tgps'],
    f_ref=config.F_REF,
    f_avg=config.F_REF,
    ref_det_name='L')

INJECTION_DICT_FILENAME = 'injection.json'
CORNER_PLOT_FILENAME = 'corner_plot.pdf'


def make_corner_plot(rundir):
    """
    Make a corner plot of the parameter estimation results and save it
    to ``rundir/CORNER_PLOT_FILENAME``.

    Coordinate transformations are applied to emphasize well-measured
    parameters. The injected parameters are marked.
    """
    rundir = pathlib.Path(rundir)
    with open(rundir/INJECTION_DICT_FILENAME, encoding='utf-8') as file:
        injection = json.load(file)['par_dic']
    injection.update(_ias_prior.inverse_transform(
        **{par: injection[par] for par in _ias_prior.standard_params}))

    pe_samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
    _ias_prior.inverse_transform_samples(pe_samples)

    plot_params = _ias_prior.sampled_params + ['lnl', 'h_h']
    corner_plot = gw_plotting.CornerPlot(pe_samples[plot_params])
    corner_plot.plot(title=rundir.parent.name, max_n_ticks=3)
    corner_plot.scatter_points(injection, adjust_lims=True,
                               colors=['C3'], marker='+', zorder=2, s=50)
    plt.savefig(rundir/'corner_plot.pdf', bbox_inches='tight')


def submit_slurm(i_inj_set, i_sample, n_hours_limit=12,
                 memory_per_task='16G', sbatch_cmds=()):
    """
    Submit a job to SLURM that performs an injection in reproducible
    Gaussian noise and estimates its parameters.

    Parameters
    ----------
    i_inj_set: int
        Index to ``config.MCHIRP_RANGES`` determining the injection set.

    i_sample: int
        Index of the sample within the injection set.

    n_hours_limit: int
        Number of hours to allocate for the job.

    memory_per_task: str
        Determines the memory and number of CPUs.

    sbatch_cmds: tuple of str
        Strings with SBATCH commands.
    """
    rundir = _get_rundir(i_inj_set, i_sample)
    job_name = '-'.join(rundir.parts[-2:])
    batch_path = rundir/'batchfile'
    stdout_path = rundir.joinpath('output.out').resolve()
    stderr_path = rundir.joinpath('errors.err').resolve()
    sbatch_cmds += (f'--mem-per-cpu={memory_per_task}',)
    args = rundir.resolve().as_posix()

    rundir.mkdir(parents=True)
    utils.submit_slurm(job_name, n_hours_limit, stdout_path, stderr_path,
                       args, sbatch_cmds, batch_path)


def main(rundir):
    """
    Perform an injection in reproducible Gaussian noise and estimate its
    parameters.

    Parameters
    ----------
    rundir: str, os.PathLike
        Must be of the correct form, e.g. output of ``_get_rundir``.
    """
    rundir = pathlib.Path(rundir)
    i_inj_set, i_sample = _get_inj_id(rundir.parent)
    event_data = _get_event_data(i_inj_set, i_sample)
    post = posterior.Posterior.from_event(event_data,
                                          mchirp_guess=None,
                                          approximant=config.APPROXIMANT,
                                          prior_class=config.PRIOR_NAME,
                                          prior_kwargs={'f_ref': config.F_REF})

    # Ensure the mchirp prior range is contained in the injection range:
    mchirp_range = np.clip(post.prior.range_dic['mchirp'],
                           *config.MCHIRP_RANGES[i_inj_set])
    post.prior = post.prior.reinstantiate(mchirp_range=mchirp_range)

    post.likelihood.asd_drift = None  # Set to 1
    sampler = config.SAMPLER_CLS(post, run_kwargs=config.RUN_KWARGS)
    sampler.run(rundir)

    # Save the injection parameters in rundir for extra safety
    event_data.injection['par_dic']['lnl'] = post.likelihood.lnlike_fft(
        event_data.injection['par_dic'])
    with open(rundir/INJECTION_DICT_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(event_data.injection, file, cls=utils.NumpyEncoder, indent=2)

    make_corner_plot(rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('rundir', help='''
        Run directory. Must be of the correct form, e.g. output of
        ``_get_rundir``.''')
    parser_args = parser.parse_args()
    main(**vars(parser.parse_args()))
