"""
Define the Posterior class.
Can run as a script to make and save a Posterior instance from scratch.
"""

import argparse
import inspect
import json
import numpy as np

from . import data
from . import gw_prior
from . import utils
from . import waveform
from .likelihood import RelativeBinningLikelihood, ReferenceWaveformFinder


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
    def from_event(cls, event, approximant, prior_class, fbin=None,
                   pn_phase_tol=.05, disable_precession=False,
                   harmonic_modes=None, seed=0, tc_rng=(-.1, .1),
                   f_ref_moment=1., **kwargs):
        """
        Instantiate a `Posterior` class from the strain data.
        Automatically find a good fit solution for relative binning.

        Parameters
        ----------
        event: Instance of `data.EventData` or string with
               event name.
        approximant: string with approximant name.
        prior_class: string with key from `gw_prior.prior_registry`,
                     or subclass of `prior.Prior`.
        fbin: Array with edges of the frequency bins used for relative
              binning [Hz]. Alternatively, pass `pn_phase_tol`.
        pn_phase_tol: Tolerance in the post-Newtonian phase [rad] used
                      for defining frequency bins. Alternatively, pass
                      `fbin`.
        disable_precession: bool, whether to set inplane spins to 0
                            when evaluating the waveform.
        harmonic_modes: list of 2-tuples with (l, m) of the harmonic
                        modes to include. Pass `None` to use
                        approximant-dependent defaults per
                        `waveform.APPROXIMANTS`.
        kwargs: Additional keyword arguments to instantiate the prior
                class.

        Return
        ------
        Instance of `Posterior`.
        """
        if isinstance(event, data.EventData):
            event_data = event
        elif isinstance(event, str):
            event_data = data.EventData.from_npz(event)
        else:
            raise ValueError('`event` must be of type `str` or `EventData`')

        if isinstance(prior_class, str):
            prior_class = gw_prior.prior_registry[prior_class]

        # Check required input before doing expensive maximization:
        required_pars = {par.name for par in
                         prior_class.init_parameters(include_optional=False)}
        event_data_keys = {'mchirp_range', 'tgps', 'q_min'}
        bestfit_keys = {'ref_det_name', 'detector_pair', 'f_ref', 'f_avg',
                        't0_refdet'}
        if missing_pars := (required_pars - event_data_keys - bestfit_keys
                            - kwargs.keys()):
            raise ValueError(f'Missing parameters: {", ".join(missing_pars)}')

        # Initialize likelihood:
        aux_waveform_generator = waveform.WaveformGenerator.from_event_data(
            event_data, approximant, harmonic_modes=[(2, 2)])
        bestfit = ReferenceWaveformFinder(
            event_data, aux_waveform_generator).find_bestfit_pars(tc_rng, seed)
        waveform_generator = waveform.WaveformGenerator.from_event_data(
            event_data, approximant, harmonic_modes, disable_precession)
        likelihood = RelativeBinningLikelihood(
            event_data, waveform_generator, bestfit['par_dic'], fbin,
            pn_phase_tol)
        assert likelihood._lnl_0 > 0

        bestfit['f_avg'] = likelihood.get_average_frequency(
            bestfit['par_dic'], bestfit['ref_det_name'])

        # Initialize prior:
        prior = prior_class(**
            {key: getattr(event_data, key) for key in event_data_keys}
            | bestfit | kwargs)

        # Initialize posterior and do second search:
        posterior = cls(prior, likelihood)
        posterior.refine_reference_waveform(seed)
        return posterior

    def refine_reference_waveform(self, seed=None):
        """
        Reset relative-binning reference waveform, using differential
        evolution to find a good fit.
        It is guaranteed that the new waveform will have at least as
        good a fit as the current one.
        The likelihood maximization uses folded sampled parameters.
        """
        folded_par_vals_0 = self.prior.fold(
            **self.prior.inverse_transform(**self.likelihood.par_dic_0))

        lnlike_unfolds = self.prior.unfold_apply(
            lambda *pars: self.likelihood.lnlike(self.prior.transform(*pars)))

        bestfit_folded = utils.differential_evolution_with_guesses(
            func=lambda pars: -max(lnlike_unfolds(*pars)),
            bounds=list(zip(self.prior.cubemin,
                            self.prior.cubemin + self.prior.folded_cubesize)),
            guesses=folded_par_vals_0,
            seed=seed).x
        i_fold = np.argmax(lnlike_unfolds(*bestfit_folded))

        self.likelihood.par_dic_0 = self.prior.transform(
            *self.prior.unfold(bestfit_folded)[i_fold])
        print(f'Found solution with lnl = {self.likelihood._lnl_0}')

    def get_eventdir(self, parentdir):
        """
        Return directory name in which the Posterior instance
        should be saved, of the form
        {parentdir}/{prior_name}/{eventname}/
        """
        return utils.get_eventdir(parentdir, self.prior.__class__.__name__,
                                  self.likelihood.event_data.eventname)


_KWARGS_FILENAME = 'kwargs.json'


def initialize_posterior_slurm(
        eventname, approximant, prior_name, parentdir, n_hours_limit=2,
        sbatch_cmds=('--mem-per-cpu=4G',), overwrite=False, **kwargs):
    """
    Submit jobs that run `main()` for each event.
    This will initialize `Posterior.from_event()` and save the
    `Posterior` to JSON inside the appropriate `eventdir` (per
    `Posterior.get_eventdir`).

    Parameters
    ----------
    eventname: string with event name.
    approximant: string with approximant name.
    prior_name: string, key of `gw_prior.prior_registry`.
    parentdir: path to top directory where to save output.
    n_hours_limit: int, hours until slurm jobs are terminated.
    sbatch_cmds: sequence of strings with SBATCH commands, e.g.
                 `('--mem-per-cpu=4G',)`
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

    args = ' '.join([eventname, approximant, prior_name, parentdir])

    if kwargs:
        with open(eventdir/_KWARGS_FILENAME, 'w+') as kwargs_file:
            json.dump(kwargs, kwargs_file)
            args += f' {kwargs_file.name}'

    if overwrite:
        args += ' --overwrite'

    utils.submit_slurm(job_name, n_hours_limit, stdout_path,
                       stderr_path, args, sbatch_cmds)


def initialize_posterior_lsf(
        eventname, approximant, prior_name, parentdir, n_hours_limit=2,
        bsub_cmds=('-R "span[hosts=1] rusage[mem=4096]"',),
        overwrite=False, **kwargs):
    """
    Submit jobs that run `main()` for each event.
    This will initialize `Posterior.from_event()` and save the
    `Posterior` to JSON inside the appropriate `eventdir` (per
    `Posterior.get_eventdir`).

    Parameters
    ----------
    eventname: string with event name.
    approximant: string with approximant name.
    prior_name: string, key of `gw_prior.prior_registry`.
    parentdir: path to top directory where to save output.
    n_hours_limit: int, hours until slurm jobs are terminated.
    bsub_cmds: sequence of strings with BSUB commands.
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

    args = ' '.join([eventname, approximant, prior_name, parentdir])

    if kwargs:
        with open(eventdir/_KWARGS_FILENAME, 'w+') as kwargs_file:
            json.dump(kwargs, kwargs_file)
            args += f' {kwargs_file.name}'

    if overwrite:
        args += ' --overwrite'

    utils.submit_lsf(job_name, n_hours_limit, stdout_path,
                       stderr_path, args, bsub_cmds)


def main(eventname, approximant, prior_name, parentdir, overwrite,
         kwargs_filename=None):
    """Construct a Posterior instance and save it to json."""
    kwargs = {}
    if kwargs_filename:
        with open(kwargs_filename) as kwargs_file:
            kwargs = json.load(kwargs_file)

    post = Posterior.from_event(eventname, approximant, prior_name, **kwargs)
    post.to_json(post.get_eventdir(parentdir), overwrite=overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Construct a Posterior instance and save it to json.''')

    parser.add_argument('eventname', help='key from `data.event_registry`.')
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
