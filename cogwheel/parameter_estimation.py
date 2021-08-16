"""Sample posterior distributions."""

import inspect
import itertools
import os
import numpy as np
from scipy import special

import ultranest
import ultranest.stepsampler
import pymultinest

from . import data
from . import gw_prior
from . import utils
from . import waveform
from . likelihood import RelativeBinningLikelihood, ReferenceWaveformFinder

posterior_registry = {}


class PosteriorError(Exception):
    """Error raised by the Posterior class."""


class Posterior(utils.JSONMixin):
    """
    Class that instantiates a prior and a likelihood and provides
    methods for sampling the posterior distribution.
    """
    def __init__(self, prior, likelihood):
        """
        Parameters
        ----------
        prior:
            Instance of `prior.Prior`, provides coordinate
            transformations and priors.
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

        self.cubemin = self.prior.cubemin.copy()
        self.cubesize = self.prior.cubesize.copy()
        self.periodic_params = self.prior.periodic_params

        # Transform signature is only known at init, so define lnposterior here
        def lnposterior(*args, **kwargs):
            """
            Natural logarithm of the posterior probability density in
            the space of sampled parameters.
            """
            lnprior, standard_par_dic = self.prior.lnprior_and_transform(
                *args, **kwargs)
            return lnprior + self.likelihood.lnlike(standard_par_dic)

        lnposterior.__signature__ = inspect.signature(self.prior.transform)
        self.lnposterior = lnposterior

    @classmethod
    def from_event(cls, event, approximant, prior_class, fbin=None,
                   pn_phase_tol=.05, disable_precession=False,
                   harmonic_modes=None, tolerance_params=None, seed=0,
                   **kwargs):
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
                        `waveform.HARMONIC_MODES`.
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

        # Check that we will have required parameters
        # before doing any expensive maximization
        sig = inspect.signature(prior_class.__init__)
        required_pars = {
            name for name, parameter in sig.parameters.items()
            if parameter.default is inspect._empty
            and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                       inspect.Parameter.VAR_KEYWORD)
            and name != 'self'}
        event_data_keys = {'mchirp_range', 'tgps', 'q_min'}
        bestfit_keys = {'ref_det_name', 'detector_pair', 'f_ref', 't0_refdet'}
        missing_pars = (required_pars - event_data_keys - bestfit_keys
                        - set(kwargs))
        if missing_pars:
            raise ValueError(f'Missing parameters: {", ".join(missing_pars)}')

        # Initialize likelihood
        aux_waveform_generator = waveform.WaveformGenerator(
            event_data.detector_names, event_data.tgps, event_data.tcoarse,
            approximant, f_ref=20., harmonic_modes=[(2, 2)])
        bestfit = ReferenceWaveformFinder(
            event_data, aux_waveform_generator).find_bestfit_pars()
        waveform_generator = waveform.WaveformGenerator(
            event_data.detector_names, event_data.tgps, event_data.tcoarse,
            approximant, bestfit['f_ref'], harmonic_modes, disable_precession)
        likelihood_instance = RelativeBinningLikelihood(
            event_data, waveform_generator, bestfit['par_dic'], fbin,
            pn_phase_tol, tolerance_params)

        # Initialize prior
        prior_kwargs = {key: getattr(event_data, key)
                        for key in event_data_keys}
        prior_kwargs.update({**bestfit, **kwargs})
        prior_instance = prior_class(**prior_kwargs)

        pe_instance = cls(prior_instance, likelihood_instance)

        # Refine relative binning solution over all space
        print('Performing a second search...')
        guess = prior_instance.inverse_transform(
            **likelihood_instance.par_dic_0)
        result = utils.differential_evolution_with_guesses(
            lambda pars: -pe_instance.likelihood.lnlike(
                pe_instance.prior.transform(*pars)),
            list(prior_instance.range_dic.values()),
            list(guess.values()),
            seed=seed)
        likelihood_instance.par_dic_0 = prior_instance.transform(*result.x)
        print(f'Found solution with lnl = {likelihood_instance._lnl_0}')

        return pe_instance


class FoldedPosterior(Posterior):
    """
    A posterior distribution that has been folded in parameter space.
    This means that some ("folded") dimensions are sampled over half
    their original range, and a map to the other half of the range is
    defined by reflecting about the midpoint.
    The folded posterior distribution is defined as the sum of the
    original posterior over all `2**n_folds` mapped points.
    This is intended to reduce the number of modes in the posterior.
    """
    def __init__(self, prior, likelihood):
        super().__init__(prior, likelihood)

        # Half range for folder parameters
        self._folded_inds = [self.prior.sampled_params.index(par)
                             for par in self.prior.foldable_params]
        self.cubesize[self._folded_inds] /= 2

        # Folded parameters lose their periodicity
        self.periodic_params = [par for par in self.periodic_params
                                if par not in self.prior.foldable_params]

        # Increase `n_cached_waveforms` to ensure fast moves remain fast
        fast_sampled_params = self.prior.get_fast_sampled_params(
            self.likelihood.waveform_generator.fast_params)
        n_slow_folded = len(set(self.prior.foldable_params)
                            - set(fast_sampled_params))
        self.likelihood.waveform_generator.n_cached_waveforms \
            = 2 ** n_slow_folded

        # Overwrite lnposterior method
        self._lnposterior_no_folding = self.lnposterior
        sig = inspect.signature(self.prior.transform)
        def lnposterior(*args, **kwargs):
            """
            Natural logarithm of the posterior probability density in
            the space of folded sampled parameters.
            """
            par_values = np.array(sig.bind(*args, **kwargs).args)
            unfolded = self.unfold(par_values)
            lnposts = [self._lnposterior_no_folding(*par_vals)
                       for par_vals in unfolded]
            return special.logsumexp(lnposts)

        lnposterior.__signature__ = sig
        self.lnposterior = lnposterior

    def unfold(self, par_values):
        """
        Return an array of shape `(2**n_folded_params, n_params)`
        with the different ways to unfold parameters.
        """
        original_values = par_values[self._folded_inds]
        unfolded_values = (2 * (self.cubemin[self._folded_inds]
                                + self.cubesize[self._folded_inds])
                           - original_values)

        unfolded = np.array([par_values] * 2**len(self._folded_inds))
        unfolded[:, self._folded_inds] = list(itertools.product(
            *zip(original_values, unfolded_values)))

        return unfolded


class Ultranest(utils.JSONMixin):
    """
    Sample a posterior using Ultranest.
    (Doesn't work well yet)
    """
    def __init__(self, posterior):
        self.posterior = posterior
        self.sampler = None
        print("Warning: I couldn't produce reasonable results with this class")

    def instantiate_sampler(self, run=False, *, prior_only=False,
                            n_fast_steps=8, **kwargs):
        """Set up `self.sampler`."""
        lnprob = (self._lnprior_ultranest if prior_only
                  else self._lnposterior_ultranest)

        wrapped_params = [par in self.posterior.periodic_params
                          for par in self.posterior.prior.sampled_params]

        self.sampler = ultranest.ReactiveNestedSampler(
            self.posterior.prior.sampled_params, lnprob, self._cubetransform,
            wrapped_params=wrapped_params)
        self.sampler.stepsampler \
            = ultranest.stepsampler.SpeedVariableRegionSliceSampler(
                self._get_step_matrix(n_fast_steps))
        if run:
            self.sampler.run(**kwargs)

    def _get_step_matrix(self, n_steps):
        """
        Return matrix with pattern of fast/slow steps.

        Parameters
        ----------
        n_steps: int, the cycle will have one slow step and
                 `n_steps - 1` fast steps.
        """
        fast_sampled_params = self.posterior.prior.get_fast_sampled_params(
            self.posterior.likelihood.waveform_generator.fast_params)

        step_matrix = np.ones(
            (n_steps, len(self.posterior.prior.sampled_params)), bool)
        step_matrix[1:] = [par in fast_sampled_params
                           for par in self.posterior.prior.sampled_params]
        return step_matrix

    def _cubetransform(self, cube):
        return self.posterior.cubemin + cube * self.posterior.cubesize

    def _lnprior_ultranest(self, par_vals):
        return self.posterior.prior.lnprior(*par_vals)

    def _lnposterior_ultranest(self, par_vals):
        return self.posterior.lnposterior(*par_vals)


class PyMultinest(utils.JSONMixin):
    """Sample a posterior using PyMultinest."""
    def __init__(self, posterior):
        self.posterior = posterior
        # PyMultinest doesn't know how to use fast parameters, don't cache
        self.posterior.likelihood.waveform_generator.n_cached_waveforms = 0

        self._nparams = len(self.posterior.cubemin)

    def run_pymultinest(self, rundir, prior_only=False, n_live_points=400,
                        tol=.1):
        """Make a directory to save results and run pymultinest."""
        rundir = rundir.rstrip('/') + '/'

        lnprob = (self._lnprior_pymultinest if prior_only
                  else self._lnposterior_pymultinest)

        wrapped_params = [par in self.posterior.periodic_params
                          for par in self.posterior.prior.sampled_params]

        if not os.path.exists(rundir):
            umask = os.umask(0o022)  # Change permissions to 755 (rwxr-xr-x)
            os.mkdir(rundir)
            os.umask(umask)  # Restore previous default permissions

        pymultinest.run(
            lnprob, self._cubetransform, self._nparams,
            outputfiles_basename=rundir, wrapped_params=wrapped_params,
            n_live_points=n_live_points, evidence_tolerance=tol)

        os.system(f'chmod 666 {rundir}*')

    def _cubetransform(self, cube, *_):
        for i in range(self._nparams):
            cube[i] = (self.posterior.cubemin[i]
                       + cube[i] * self.posterior.cubesize[i])

    def _lnprior_pymultinest(self, par_vals, *_):
        return self.posterior.prior.lnprior(
            *[par_vals[i] for i in range(self._nparams)])

    def _lnposterior_pymultinest(self, par_vals, *_):
        return self.posterior.lnposterior(
            *[par_vals[i] for i in range(self._nparams)])
