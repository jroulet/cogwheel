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
from cogwheel import waveform
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
    return config.INJECTION_DIR/'runs'/eventname


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
    plotting_prior_cls = getattr(config, 'PLOTTING_PRIOR_CLS',
                                 gw_prior.IASPrior)
    if isinstance(plotting_prior_cls, str):
        plotting_prior_cls = gw_prior.prior_registry[plotting_prior_cls]
    plotting_prior = plotting_prior_cls(
        mchirp_range=(1, 2),  # Doesn't matter for this purpose
        detector_pair='HL',
        tgps=config.EVENT_DATA_KWARGS['tgps'],
        f_ref=config.PRIOR_KWARGS['f_ref'],
        f_avg=config.PRIOR_KWARGS['f_ref'],
        ref_det_name='L')

    # Inverse-transform and save injection:
    with open(rundir/INJECTION_DICT_FILENAME, encoding='utf-8') as file:
        injection = json.load(file)
    par_dic = injection['par_dic']
    par_dic.update(plotting_prior.inverse_transform(
        **{par: par_dic[par] for par in plotting_prior.standard_params}))
    with open(rundir/INJECTION_DICT_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(injection, file)

    # Inverse-transform and save paremeter estimation samples:
    samples_filename = rundir/sampling.SAMPLES_FILENAME
    pe_samples = pd.read_feather(samples_filename)
    for par, val in waveform.DEFAULT_PARS.items():
        if par not in pe_samples:
            pe_samples[par] = val
    plotting_prior.inverse_transform_samples(pe_samples)
    pe_samples.to_feather(samples_filename)

    # Plot and save:
    plot_params = [
        par for par in plotting_prior.sampled_params + ['lnl', 'h_h']
        if par in pe_samples]
    corner_plot = gw_plotting.CornerPlot(pe_samples, params=plot_params,
                                         tail_probability=1e-4)
    corner_plot.plot(title=rundir.parent.name, max_n_ticks=3)
    corner_plot.scatter_points(par_dic, adjust_lims=True,
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
        likelihood_kwargs=getattr(config, 'LIKELIHOOD_KWARGS', None),
        ref_wf_finder_kwargs={**getattr(config, 'REF_WF_FINDER_KWARGS', {}),
                              'f_ref': config.PRIOR_KWARGS['f_ref']})
    post.likelihood.asd_drift = None  # Set to 1

    print('', flush=True)  # Flush maximization log before starting the sampler

    # Save the injection parameters in rundir for extra safety
    event_data.injection['par_dic']['lnl'] = post.likelihood.lnlike_fft(
        event_data.injection['par_dic'])
    with open(rundir/INJECTION_DICT_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(event_data.injection, file, cls=utils.NumpyEncoder, indent=2)

    # Declare failure if the range_dic, prior or likelihood don't
    # include the truth:
    sampled_inj = post.prior.inverse_transform(
        **event_data.injection['par_dic'])

    for par, val in sampled_inj.items():
        left, right = post.prior.range_dic[par]
        if val < left or val > right:
            raise RuntimeError(f'{par}={val} outside range {(left, right)}')

    if np.isneginf(post.prior.lnprior(**sampled_inj)):
        raise RuntimeError('prior = 0 at the injection.')

    if np.isneginf(post.likelihood.lnlike(event_data.injection['par_dic'])):
        raise RuntimeError('likelihood = 0 at the injection.')

    # Ensure the mchirp prior range is contained in the injection range:
    mchirp_range = post.prior.get_init_dict().get('mchirp_range')
    if mchirp_range is not None:
        mchirp_range = np.clip(mchirp_range,
                               *config.PRIOR_KWARGS['mchirp_range'])
        post.prior = post.prior.reinstantiate(mchirp_range=mchirp_range)


    sampler = config.SAMPLER_CLS(post, run_kwargs=config.RUN_KWARGS)
    sampler.run(rundir)

    make_corner_plot(config, rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('config_filename', help='Path to a config file.')
    parser.add_argument('rundir', help='''
        Run directory. Must be of the correct form, e.g. output of
        ``get_rundir``.''')
    main(**vars(parser.parse_args()))
