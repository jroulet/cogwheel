"""
Define the Posterior class.
Can run as a script to make a Posterior instance from scratch and find
the maximum likelihood solution on the full parameter space.
"""

import argparse
import inspect
import json
import numpy as np

from cogwheel import gw_prior
from cogwheel import utils
from cogwheel.likelihood import (RelativeBinningLikelihood,
                                 ReferenceWaveformFinder)


class PosteriorError(Exception):
    """Error raised by the Posterior class."""


class Posterior(utils.JSONMixin):
    """
    Class that instantiates a prior and a likelihood and provides
    methods for sampling the posterior distribution.

    Parameter space folding is implemented; this means that some
    ("folded") dimensions are sampled over half their original range,
    and a map to the other half of the range is defined by reflecting or
    shifting about the midpoint. The folded posterior distribution is
    defined as the sum of the original posterior over all `2**n_folds`
    mapped points. This is intended to reduce the number of modes in the
    posterior.
    """
    def __init__(self, prior, likelihood):
        """
        Parameters
        ----------
        prior: Instance of `prior.Prior`
            Provides coordinate transformations, prior, and foldable
            parameters.

        likelihood: Instance of `likelihood.RelativeBinningLikelihood`
            Provides likelihood computation.
        """
        if set(prior.standard_params) != set(likelihood.params):
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
        signature = inspect.signature(self.prior.__class__.transform)
        self.lnposterior.__func__.__signature__ = signature
        self.lnposterior_pardic_and_metadata.__func__.__signature__ = signature

    def lnposterior(self, *args, **kwargs):
        """
        Natural logarithm of the posterior probability density in
        the space of sampled parameters (does not apply folding).
        """
        return self.lnposterior_pardic_and_metadata(*args, **kwargs)[0]

    def lnposterior_pardic_and_metadata(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Sampled parameters.

        Return
        ------
        lnposterior: float
            Natural logarithm of the posterior probability density in
            the space of sampled parameters (does not apply folding).

        standard_par_dic: dict
            Standard parameters.

        metadata: object
            Used to compute ancillary information about samples. This
            will vary depending on the likelihood implementation.
        """
        lnprior, standard_par_dic = self.prior.lnprior_and_transform(
            *args, **kwargs)

        if np.isneginf(lnprior):
            return -np.inf, standard_par_dic, None

        lnl, metadata = self.likelihood.lnlike_and_metadata(standard_par_dic)
        return lnprior + lnl, standard_par_dic, metadata

    @classmethod
    def from_event(
            cls, event, mchirp_guess, approximant, prior_class,
            likelihood_class=None, prior_kwargs=None,
            likelihood_kwargs=None, ref_wf_finder_kwargs=None):
        """
        Instantiate a `Posterior` class from the strain data.
        Automatically find a good fit solution for relative binning.

        Parameters
        ----------
        event: Instance of `data.EventData` or string with event name,
               or path to npz file with `EventData` instance.

        mchirp_guess: float
            Approximate chirp mass (Msun).

        approximant: str
            Approximant name.

        prior_class: string with key from `gw_prior.prior_registry`,
                     or subclass of `prior.Prior`.

        likelihood_class:
            subclass of likelihood.RelativeBinningLikelihood

        prior_kwargs: dict,
            Keyword arguments for `prior_class` constructor.

        likelihood_kwargs: dict
            Keyword arguments for `likelihood_class` constructor.

        Return
        ------
            Instance of `Posterior`.
        """
        prior_kwargs = prior_kwargs or {}
        likelihood_kwargs = likelihood_kwargs or {}
        ref_wf_finder_kwargs = ref_wf_finder_kwargs or {}

        if isinstance(prior_class, str):
            try:
                prior_class = gw_prior.prior_registry[prior_class]
            except KeyError as err:
                raise KeyError('Avaliable priors are: '
                               f'{", ".join(gw_prior.prior_registry)}.'
                              ) from err

        if likelihood_class is None:
            likelihood_class = getattr(prior_class,
                                       'default_likelihood_class',
                                       RelativeBinningLikelihood)

        ref_wf_finder = ReferenceWaveformFinder.from_event(
            event, mchirp_guess, approximant=approximant,
            **ref_wf_finder_kwargs)

        likelihood = likelihood_class.from_reference_waveform_finder(
            ref_wf_finder, approximant=approximant, **likelihood_kwargs)

        prior = prior_class.from_reference_waveform_finder(ref_wf_finder,
                                                           **prior_kwargs)
        return cls(prior, likelihood)

    def refine_reference_waveform(self, seed=None, params=None):
        """
        Reset relative-binning reference waveform, using differential
        evolution to find a good fit.
        It is guaranteed that the new waveform will have at least as
        good a fit as the current one.
        The likelihood maximization uses folded sampled parameters.

        Parameters
        ----------
        seed: {None, int, numpy.random.Generator,
               numpy.random.RandomState}, optional
            Passed to ``scipy.optimize.differential_evolution``

        params: list of str, optional
            Which parameters to maximize over. If provided, must be
            keys from ``self.prior.sampled_params``.
        """
        print(f'Old lnl = {self.likelihood.lnlike(self.likelihood.par_dic_0)}')

        params = params or self.prior.sampled_params
        inds = [self.prior.sampled_params.index(par) for par in params]

        folded_par_vals_0 = self.prior.fold(
            **self.prior.inverse_transform(**self.likelihood.par_dic_0))

        lnlike_unfolds = self.prior.unfold_apply(
            lambda *pars: self.likelihood.lnlike(self.prior.transform(*pars)))

        folded_par_vals = folded_par_vals_0.copy()
        def loss_function(pars):
            """
            Take parameter values on the folded space corresponding to
            ``params``, complete the remaining coordinates using the
            reference ``folded_par_vals``, return minus the maximum log
            likelihood over unfolds .
            """
            folded_par_vals[inds] = pars
            try:
                return -max(lnlike_unfolds(*folded_par_vals))
            except RuntimeError:
                return np.inf

        result = utils.differential_evolution_with_guesses(
            func=loss_function,
            bounds=list(zip(self.prior.cubemin[inds],
                            (self.prior.cubemin
                             + self.prior.folded_cubesize)[inds])),
            guesses=folded_par_vals_0[inds], seed=seed, init='sobol').x

        folded_par_vals[inds] = result
        i_fold = np.argmax(lnlike_unfolds(*folded_par_vals))

        par_dic_0 = self.prior.transform(
            *self.prior.unfold(folded_par_vals)[i_fold])

        self.likelihood.par_dic_0 = self.likelihood.par_dic_0 | par_dic_0

    def get_eventdir(self, parentdir):
        """
        Return directory name in which the Posterior instance should be
        saved, of the form '{parentdir}/{prior_name}/{eventname}/'.
        """
        return utils.get_eventdir(parentdir, self.prior.__class__.__name__,
                                  self.likelihood.event_data.eventname)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.prior.__class__.__name__}, '
                f'{self.likelihood.event_data.eventname})')


_KWARGS_FILENAME = 'kwargs.json'


def submit_likelihood_maximization(
        eventname, mchirp_guess, approximant, prior_name, parentdir,
        scheduler='slurm', n_hours_limit=2, scheduler_cmds=(),
        overwrite=False, **kwargs):
    """
    Submit a job that runs `main()`, which maximizes the likelihood for
    an event.
    This will initialize `Posterior.from_event()` and
    `Posterior.refine_reference_waveform` and save the `Posterior` to
    JSON in the appropriate `eventdir` (per `Posterior.get_eventdir`).

    Parameters
    ----------
    eventname: str
        Event name, e.g. 'GW150914'.

    mchirp_guess: float
        Approximate chirp mass (Msun).

    approximant: str
        Approximant name.

    prior_name: str
        Key in `gw_prior.prior_registry`.

    parentdir: os.PathLike
        Path to top directory where to save output.

    scheduler: {'slurm', 'lsf'}

    n_hours_limit: int
        Hours until jobs are terminated.

    sbatch_cmds: sequence of strings
        Commands for the scheduler. E.g., `('--mem-per-cpu=4G',)` for
        slurm, or `('-R "span[hosts=1] rusage[mem=4096]"')` for lsf.

    overwrite: bool
        Whether to overwrite preexisting files. `False` (default) raises
        an error if the file exists.

    **kwargs:
        Optional keyword arguments to `Posterior.from_event()`, must be
        JSON-serializable.
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

    args = f'{eventname} {mchirp_guess} {approximant} {prior_name} {parentdir}'

    if kwargs:
        with open(eventdir/_KWARGS_FILENAME, 'w+', encoding='utf-8'
                 ) as kwargs_file:
            json.dump(kwargs, kwargs_file)
            args += f' {kwargs_file.name}'

    if overwrite:
        args += ' --overwrite'

    submit = {'slurm': utils.submit_slurm, 'lsf': utils.submit_lsf}[scheduler]
    submit(job_name, n_hours_limit, stdout_path, stderr_path, args,
           scheduler_cmds)


def main(eventname, mchirp_guess, approximant, prior_name, parentdir,
         overwrite, kwargs_filename=None, refine=False):
    """
    Construct a Posterior instance, optionally refine its reference
    waveform and save it to JSON.

    Parameters
    ----------
    eventname: str
        Event name, e.g. 'GW150914'.

    mchirp_guess: float
        Approximate chirp mass (Msun).

    approximant: str
        Approximant name.

    prior_name: str
        Key in `gw_prior.prior_registry`.

    parentdir: os.PathLike
        Path to top directory where to save output.

    overwrite: bool
        Whether to overwrite preexisting files. `False` (default) raises
        an error if the file exists.

    kwargs_filename: PathLike or None
        Path to JSON file from which to load kwargs to
        ``Posterior.from_event()``.

    refine: bool
        Whether to apply an expensive likelihood maximization over all
        parameters.
    """
    kwargs = {}
    if kwargs_filename:
        with open(kwargs_filename, encoding='utf-8') as kwargs_file:
            kwargs = json.load(kwargs_file)

    post = Posterior.from_event(eventname, mchirp_guess, approximant,
                                prior_name, **kwargs)
    if refine:
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
    parser.add_argument('--refine', action='store_true',
                        help='pass to refine reference solution')

    main(**vars(parser.parse_args()))
