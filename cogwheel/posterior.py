"""
Define the Posterior class.
Can run as a script to make a Posterior instance from scratch and find
the maximum likelihood solution on the full parameter space.
"""

import argparse
import inspect
import json
import numpy as np

from cogwheel import data
from cogwheel import gw_prior
from cogwheel import utils
from cogwheel import waveform
from cogwheel.likelihood import RelativeBinningLikelihood, \
    ReferenceWaveformFinder


class PosteriorError(Exception):
    """Error raised by the Posterior class."""


class Posterior(utils.JSONMixin):
    """
    Class that instantiates a prior and a likelihood and provides
    methods for sampling the posterior distribution.

    Parameter space folding is implemented; this means that some
    ("folded") dimensions are sampled over half their original range,
    and a map to the other half of the range is defined by reflecting
    about the midpoint. The folded posterior distribution is defined as
    the sum of the original posterior over all `2**n_folds` mapped
    points. This is intended to reduce the number of modes in the
    posterior.
    """
    def __init__(self, prior, likelihood):
        """
        Parameters
        ----------
        prior:
            Instance of `prior.Prior`, provides coordinate
            transformations, priors, and foldable parameters.
        likelihood:
            Instance of `likelihood.RelativeBinningLikelihood`, provides
            likelihood computation.
        """
        if set(prior.standard_params) != set(
                likelihood.waveform_generator.params):
            raise PosteriorError('The prior and likelihood instances passed '
                                 'have incompatible parameters.')

        self.prior = prior
        self.likelihood = likelihood

        # Increase `n_cached_waveforms` to ensure fast moves remain fast
        fast_sampled_params = self.prior.get_fast_sampled_params(
            self.likelihood.waveform_generator.fast_params)
        n_slow_folded = len(set(self.prior.folded_params)
                            - set(fast_sampled_params))
        self.likelihood.waveform_generator.n_cached_waveforms \
            = 2 ** n_slow_folded

        # Match lnposterior signature to that of transform
        self.lnposterior.__func__.__signature__ = inspect.signature(
            self.prior.__class__.transform)

    def lnposterior(self, *args, **kwargs):
        """
        Natural logarithm of the posterior probability density in
        the space of sampled parameters (does not apply folding).
        """
        lnprior, standard_par_dic = self.prior.lnprior_and_transform(
            *args, **kwargs)
        return lnprior + self.likelihood.lnlike(standard_par_dic)

    @classmethod
    def from_event(
            cls, event, mchirp_guess, approximant, prior_class,
            likelihood_class=RelativeBinningLikelihood,
            prior_kwargs=None, likelihood_kwargs=None):
        """
        Instantiate a `Posterior` class from the strain data.
        Automatically find a good fit solution for relative binning.

        Parameters
        ----------
        event: Instance of `data.EventData` or string with event name,
               or path to npz file with `EventData` instance.
        approximant: str, approximant name.
        prior_class: string with key from `gw_prior.prior_registry`,
                     or subclass of `prior.Prior`.
        likelihood_class: subclass of likelihood.RelativeBinningLikelihood
        prior_kwargs: dict, kwargs for `prior_class` constructor.
        likelihood_kwargs: : dict, kwargs for `prior_class` constructor.

        Return
        ------
        Instance of `Posterior`.
        """
        prior_kwargs = prior_kwargs or {}
        likelihood_kwargs = likelihood_kwargs or {}

        if isinstance(prior_class, str):
            prior_class = gw_prior.prior_registry[prior_class]

        ref_wf_finder = ReferenceWaveformFinder.from_event(
            event, mchirp_guess, approximant=approximant)

        likelihood = likelihood_class.from_reference_waveform_finder(
            ref_wf_finder, approximant=approximant, **likelihood_kwargs)

        prior = prior_class.from_reference_waveform_finder(ref_wf_finder,
                                                           **prior_kwargs)
        return cls(prior, likelihood)

    def refine_reference_waveform(self, seed=None):
        """
        Reset relative-binning reference waveform, using differential
        evolution to find a good fit.
        It is guaranteed that the new waveform will have at least as
        good a fit as the current one.
        The likelihood maximization uses folded sampled parameters.
        """
        print('Maximizing likelihood over full parameter space...')
        folded_par_vals_0 = self.prior.fold(
            **self.prior.inverse_transform(**self.likelihood.par_dic_0))

        lnlike_unfolds = self.prior.unfold_apply(
            lambda *pars: self.likelihood.lnlike(self.prior.transform(*pars)))

        bestfit_folded = utils.differential_evolution_with_guesses(
            func=lambda pars: -max(lnlike_unfolds(*pars)),
            bounds=list(zip(self.prior.cubemin,
                            self.prior.cubemin + self.prior.folded_cubesize)),
            guesses=folded_par_vals_0, seed=seed, init='sobol').x
        i_fold = np.argmax(lnlike_unfolds(*bestfit_folded))

        self.likelihood.par_dic_0 = self.prior.transform(
            *self.prior.unfold(bestfit_folded)[i_fold])

        lnl = self.likelihood.lnlike(self.likelihood.par_dic_0)
        print(f'Found solution with lnl = {lnl}')

    def get_eventdir(self, parentdir):
        """
        Return directory name in which the Posterior instance
        should be saved, of the form
        {parentdir}/{prior_name}/{eventname}/
        """
        return utils.get_eventdir(parentdir, self.prior.__class__.__name__,
                                  self.likelihood.event_data.eventname)


_KWARGS_FILENAME = 'kwargs.json'


def submit_likelihood_maximization(
        eventname, approximant, prior_name, parentdir,
        scheduler='slurm', n_hours_limit=2, scheduler_cmds=(),
        overwrite=False, **kwargs):
    """
    Submit a job that runs `main()`, which maximizes the likelihood for
    an event.
    This will initialize `Posterior.from_event()` and
    `Posterior.refine_reference_waveform` and save the `Posterior` to
    JSON inside the appropriate `eventdir` (per
    `Posterior.get_eventdir`).

    Parameters
    ----------
    eventname: string with event name.
    approximant: string with approximant name.
    prior_name: string, key of `gw_prior.prior_registry`.
    parentdir: path to top directory where to save output.
    scheduler: 'slurm' or 'lsf'.
    n_hours_limit: int, hours until slurm jobs are terminated.
    sbatch_cmds: sequence of strings with commands for the scheduler.
        E.g., `('--mem-per-cpu=4G',)` for slurm,
        or `('-R "span[hosts=1] rusage[mem=4096]"')` for lsf.
    overwrite: bool, whether to overwrite preexisting files.
               `False` (default) raises an error if the file exists.
    **kwargs: optional keyword arguments to `Posterior.from_event()`.
              Must be JSON-serializable.
    """
    eventdir = utils.get_eventdir(parentdir, prior_name, eventname)
    filename = eventdir/'Posterior.json'
    if not overwrite and filename.exists():
        raise FileExistsError(
            f'{filename} exists, pass `overwrite=True` to overwrite.')

    utils.mkdirs(eventdir)

    job_name = f'{eventname}_posterior'
    stdout_path = (eventdir/'posterior_from_event.out').resolve()
    stderr_path = (eventdir/'posterior_from_event.err').resolve()

    args = ' '.join([eventname, mchirp_guess, approximant, prior_name,
                     parentdir])

    if kwargs:
        with open(eventdir/_KWARGS_FILENAME, 'w+') as kwargs_file:
            json.dump(kwargs, kwargs_file)
            args += f' {kwargs_file.name}'

    if overwrite:
        args += ' --overwrite'

    submit = {'slurm': utils.submit_slurm, 'lsf': utils.submit_lsf}[scheduler]
    submit(job_name, n_hours_limit, stdout_path, stderr_path, args,
           scheduler_cmds)


def main(eventname, mchirp_guess, approximant, prior_name, parentdir,
         overwrite, kwargs_filename=None):
    """
    Construct a Posterior instance, refine its reference waveform and
    save it to json.
    """
    kwargs = {}
    if kwargs_filename:
        with open(kwargs_filename) as kwargs_file:
            kwargs = json.load(kwargs_file)

    post = Posterior.from_event(eventname, mchirp_guess, approximant,
                                prior_name, **kwargs)
    post.refine_reference_waveform()
    post.to_json(post.get_eventdir(parentdir), overwrite=overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Construct a Posterior instance, refine its reference
                       waveform and save it to json.""")

    parser.add_argument('eventname', help='key from `data.event_registry`.')
    parser.add_argument('mchirp_guess', help='approximate chirp mass (Msun).',
                        type=float)
    parser.add_argument('approximant', help='key from `waveform.APPROXIMANTS`')
    parser.add_argument('prior_name',
                        help='key from `gw_prior.prior_registry`')
    parser.add_argument('parentdir', help='top directory to save output')
    parser.add_argument('kwargs_filename', nargs='?', default=None,
                        help='''optional json file with keyword arguments to
                                Posterior.from_event()''')
    parser.add_argument('--overwrite', action='store_true',
                        help='pass to overwrite existing json file')

    main(**vars(parser.parse_args()))
