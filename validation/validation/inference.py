"""
Functions to perform injections in reproducible Gaussian noise and do
parameter estimation on them.

This code assumes that injection samples were already generated using
the ``generate_injections.py`` module.

This module can run as a script and it also provides a function
``submit_slurm`` to submit the equivalent job to a SLURM scheduler.
"""
import argparse
import pathlib
import pandas as pd

from cogwheel import data
from cogwheel import posterior
from cogwheel import utils

from . import config


def _get_eventname(i_inj_set, i_sample):
    """Return a standardized event name to identify the injection."""
    return f'INJ_{i_inj_set}_{i_sample}'


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

    event_data.inject_signal({**sample, 't_geocenter': 0}, config.APPROXIMANT)
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
    if not eventdir.match(_get_eventname('*', '*')):
        raise ValueError(f'Could not parse `eventdir` {eventdir}.')

    i_inj_set, i_sample = eventdir.name.remove_prefix('INJ_').split('_')

    assert eventdir.resolve() == _get_eventdir(i_inj_set, i_sample).resolve()

    return int(i_inj_set), int(i_sample)


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
    job_name = '-'.join(rundir.parts[:-2])
    batch_path = rundir/'batchfile'
    stdout_path = rundir.joinpath('output.out').resolve()
    stderr_path = rundir.joinpath('errors.err').resolve()
    sbatch_cmds += (f'--mem-per-cpu={memory_per_task}',)
    args = rundir.resolve().as_posix()

    rundir.mkdir()
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
    post = posterior.Posterior.from_event(event_data=event_data,
                                          mchirp_guess=None,
                                          approximant=config.APPROXIMANT,
                                          prior_class=config.PRIOR_NAME)
    post.likelihood.asd_drift = None  # Set to 1
    sampler = config.SAMPLER_CLS(post, run_kwargs=config.RUN_KWARGS)
    sampler.run(rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('i_inj_set', help='''Index to the injection set''')
    parser.add_argument('i_sample',
                        help='''Sample index within the injection set.''')
    parser_args = parser.parse_args()
    main(**vars(parser.parse_args()))
