"""Sample posterior distributions."""

import inspect
import json
import os
import numpy as np

import ultranest
import ultranest.stepsampler

from . import bookkeeping
from . import gw_prior
from . import likelihood
from . import utils
from . import waveform


class CheckBoundsAndApplyTransform:
    """
    Decorator class that implements bounds checking and coordinate
    transformation to lnlike functions of the `ParameterEstimation`
    class.
    """
    def __init__(self, parameter_estimation):
        self.prior = parameter_estimation.prior

    def __call__(self, lnlike_func):
        sig = inspect.signature(self.prior.__class__.transform)

        def new_lnlike_func(*args, **kwargs):
            """
            Wrapper around `{}` that pre-applies the transformation
            from sampled parameters to standard parameters.
            Return -inf if parameters are outside bounds.
            """
            par_values = np.array(sig.bind(*args, **kwargs).args)

            if (np.any(par_values < self.prior.cubemin)
                    or np.any(par_values > self.prior.cubemax)):
                return -np.inf

            return lnlike_func(self.prior.transform(*par_values))

        # Fix new_lnlike_func signature and docstring:
        new_lnlike_func.__signature__ = sig
        new_lnlike_func.__doc__ = new_lnlike_func.__doc__.format(
            lnlike_func.__name__)
        return new_lnlike_func


class ParameterEstimation:
    """
    Class that instantiates a prior and a likelihood and provides
    methods for sampling the posterior distribution.
    """
    def __init__(self, prior_instance, likelihood_instance):
        """
        Parameters
        ----------
        prior_instance:
            Instance of `prior.Prior`, provides coordinate
            transformations and priors.
        likelihood_instance:
            Instance of `likelihood.RelativeBinningLikelihood`,
            provides likelihood computation.
        """
        self.prior = prior_instance
        self.likelihood = likelihood_instance

        # Transform signatures are only known at init so we define lnlike here
        @CheckBoundsAndApplyTransform(self)
        def lnlike(*args, **kwargs):
            return self.likelihood.lnlike(*args)
        self.lnlike = lnlike

        @CheckBoundsAndApplyTransform(self)
        def lnlike_fft(*args, **kwargs):
            return self.likelihood.lnlike_fft(*args)
        self.lnlike_fft = lnlike_fft

    @classmethod
    def from_event(cls, event, approximant, prior_class, fbin=None,
                   pn_phase_tol=.1, disable_precession=False,
                   harmonic_modes=None, tolerance_params=None, seed=0,
                   **kwargs):
        """
        Instantiate a `ParameterEstimation` class from the strain data.
        Automatically find a good fit solution for relative binning.

        Parameters
        ----------
        event: Instance of `bookkeeping.EventData` or string with
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
        Instance of `ParameterEstimation`.
        """
        if isinstance(event, bookkeeping.EventData):
            event_data = event
        elif isinstance(event, str):
            event_data = bookkeeping.EventData.from_npz(event)
        else:
            raise ValueError('`event` must be of type `str` or `EventData`')

        if isinstance(prior_class, str):
            prior_class = gw_prior.prior_registry[prior_class]

        # Check that we will have required parameters
        # before doing any expensive maximization
        sig = inspect.signature(prior_class.__init__)
        required_pars = {
            name for name, par in sig.parameters.items()
            if par.default is inspect._empty
            and par.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                 inspect.Parameter.VAR_KEYWORD)
            and name != 'self'}
        event_data_keys = {'mchirp_range', 'tgps', 'q_min'}
        bestfit_keys = {'ref_det_name', 'detector_pair', 'f_ref'}
        missing_pars = (required_pars - event_data_keys - bestfit_keys
                        - set(kwargs))
        if missing_pars:
            raise ValueError(f'Missing parameters: {", ".join(missing_pars)}')

        # Initialize likelihood
        aux_waveform_generator = waveform.WaveformGenerator(
            event_data.detector_names, event_data.tgps, event_data.tcoarse,
            approximant, f_ref=20., harmonic_modes=[(2, 2)])
        bestfit = likelihood.ReferenceWaveformFinder(
            event_data, aux_waveform_generator).find_bestfit_pars()
        waveform_generator = waveform.WaveformGenerator(
            event_data.detector_names, event_data.tgps, event_data.tcoarse,
            approximant, bestfit['f_ref'], harmonic_modes, disable_precession)
        likelihood_instance = likelihood.RelativeBinningLikelihood(
            event_data, waveform_generator, bestfit['par_dic'], fbin,
            pn_phase_tol, tolerance_params)

        # Initialize prior
        prior_kwargs = {
            key: getattr(event_data, key) for key in event_data_keys}
        prior_kwargs.update({**bestfit, **kwargs})
        prior_instance = prior_class(**prior_kwargs)

        pe_instance = cls(prior_instance, likelihood_instance)

        # Refine relative binning solution over all space
        print('Performing a second search...')
        guess = prior_instance.inverse_transform(
            **likelihood_instance.par_dic_0)
        result = utils.differential_evolution_with_guesses(
            lambda pars: -pe_instance.lnlike(*pars),
            list(prior_instance.range_dic.values()), list(guess.values()),
            seed=seed)
        likelihood_instance.par_dic_0 = prior_instance.transform(*result.x)
        print(f'Found solution with lnl = {likelihood_instance._lnl_0}')

        return pe_instance

    def to_json(self, outdir, overwrite=True, dir_permissions=755,
                file_permissions=644):
        """
        Save class instance to disk; files that can be loaded later
        using `ParameterEstimation.from_json()`.
        A directory `outdir` is created if missing, and two files are
        created in it: 'EventData_{eventname}.npz' and
        'ParameterEstimation_{eventname}.json'.
        The directory can later be moved since relative paths are saved.

        Parameters
        ----------
        outdir: string, path to output directory. Will be created if it
                does not exist.
        overwrite: bool, whether to overwrite existing files.
        dir_permissions: passed to `chmod`, permissions to give to the
                         output directory.
        file_permissions: permissions to give to the files.
        """
        relative_binning_kwargs = {
            key: getattr(self.likelihood, key) for key
            in ['par_dic_0', 'fbin', 'pn_phase_tol', 'tolerance_params']}
        if relative_binning_kwargs['pn_phase_tol']:
            relative_binning_kwargs.pop('fbin')

        eventname = self.likelihood.event_data.eventname

        dic = {'prior_class': self.prior.__class__.__name__,
               'prior_kwargs': self.prior.get_init_dic(),
               'waveform_generator_kwargs':  utils.get_init_dic(
                   self.likelihood.waveform_generator),
               'relative_binning_kwargs': relative_binning_kwargs,
               'event_data_filename': f'EventData_{eventname}.npz'}

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
            os.system(f'chmod {dir_permissions} {outdir}')

        json_filename = os.path.join(
            outdir, f'ParameterEstimation_{eventname}.json')
        with open(json_filename, 'w') as outfile:
            json.dump(dic, outfile, indent=2, cls=utils.NumpyEncoder)
            outfile.write('\n')
        os.system(f'chmod {file_permissions} {json_filename}')

        event_data_filename = os.path.join(outdir, dic['event_data_filename'])
        self.likelihood.event_data.to_npz(filename=event_data_filename,
                                          overwrite=overwrite)
        os.system(f'chmod {file_permissions} {event_data_filename}')

    @classmethod
    def from_json(cls, json_filename):
        """
        Load a `ParameterEstimation` instance previously saved with
        `to_json()`.

        Parameters
        ----------
        json_filename: string, path to a json file.
        """
        with open(json_filename, 'r') as json_file:
            dic = json.load(json_file)

        prior_instance = gw_prior.prior_registry[dic['prior_class']](
            **dic['prior_kwargs'])

        waveform_generator = waveform.WaveformGenerator(
            **dic['waveform_generator_kwargs'])
        event_data = bookkeeping.EventData.from_npz(
            filename=os.path.join(os.path.dirname(json_filename),
                                  dic['event_data_filename']))
        likelihood_instance = likelihood.RelativeBinningLikelihood(
            event_data, waveform_generator, **dic['relative_binning_kwargs'])

        return cls(prior_instance, likelihood_instance)


class ParameterEstimationUltranest(ParameterEstimation):
    def instatiate_sampler(self, run=False, *, prior_only=False,
                           n_fast_steps=8, **kwargs):
        if prior_only:
            lnprob = self._lnprior_ultranest
        else:
            lnprob = self._lnposterior_ultranest

        self.sampler = ultranest.ReactiveNestedSampler(
            self.prior.sampled_params, lnprob, self._cubetransform,
            wrapped_params=self.prior.periodic)
        self.sampler.stepsampler \
            = ultranest.stepsampler.SpeedVariableRegionSliceSampler(
                self._get_step_matrix(n_fast_steps))
        if run:
            return self.sampler.run(**kwargs)

    def _get_step_matrix(self, n_fast_steps):
        fast_sampled_params = self.prior.get_fast_sampled_params(
            self.likelihood.waveform_generator.fast_params)

        step_matrix = np.ones((n_fast_steps, len(self.prior.sampled_params)),
                              bool)
        step_matrix[1:] = [par in fast_sampled_params
                           for par in self.prior.sampled_params]
        return step_matrix

    def _cubetransform(self, cube):
        return self.prior.cubemin + cube * self.prior.cubesize

    def _lnprior_ultranest(self, par_vals):
        return self.prior.lnprior(*par_vals)

    def _lnposterior_ultranest(self, par_vals):
        lnprior, standard_par_dic = self.prior.lnprior_and_transform(*par_vals)
        return lnprior + self.likelihood.lnlike(standard_par_dic)
