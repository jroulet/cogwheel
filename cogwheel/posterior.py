"""
Define the Posterior class.
Can run as a script to make and save a Posterior instance from scratch.
"""

import argparse
import inspect
import pathlib
import subprocess
import sys
import os
import tempfile
import textwrap
import time
import numpy as np
import pandas as pd

from . import data
from . import gw_prior
from . import utils
from . import waveform
from . likelihood import RelativeBinningLikelihood, ReferenceWaveformFinder


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

    def test_relative_binning_accuracy(self, samples: pd.DataFrame,
                                       max_workers=None):
        """
        Compute likelihood with and without relative binning.
        Input a dataframe with samples, columns 'lnl_rb' and 'lnl_fft'
        get added in-place.

        Parameters
        ----------
        samples: pandas DataFrame with samples, its columns must contain
                 all `self.prior.sampled_params`.
        max_workers: maximum number of cores for parallelization,
                     defaults to all available cores.
        """
        package = os.path.join(os.path.dirname(__file__), '..')
        module = f'cogwheel.{os.path.basename(__file__)}'.rstrip('.py')

        n_samples = len(samples)
        n_workers = min(n_samples, max_workers or os.cpu_count() or 1)
        chunk_edges = np.linspace(0, n_samples, n_workers + 1, dtype=int)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.to_json(tmp_dir)

            # Divide samples into chunks and launch programs to process them:
            samples_paths = [os.path.join(tmp_dir, f'samples_{i}.pkl')
                             for i in range(n_workers)]
            processes = []
            for i, samples_path in enumerate(samples_paths):
                i_start, i_end = chunk_edges[[i, i+1]]
                samples[i_start : i_end].to_pickle(samples_path)

                processes.append(subprocess.Popen(
                    (f'PYTHONPATH={package} {sys.executable} -m {module} '
                     f'{tmp_dir} {samples_path}'),
                    shell=True))

            # Wait until they all finish:
            for process in processes:
                process.communicate()

            result = pd.concat(map(pd.read_pickle, samples_paths))

        # Add result to samples in-place
        samples['lnl_rb'] = result['lnl_rb']
        samples['lnl_fft'] = result['lnl_fft']

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
                        `waveform.APPROXIMANTS`.
        tolerance_params: dictionary
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
        if missing_pars := (required_pars - event_data_keys - bestfit_keys
                            - set(kwargs)):
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
        assert likelihood_instance._lnl_0 > 0

        # Initialize prior
        prior_kwargs = {key: getattr(event_data, key)
                        for key in event_data_keys} | bestfit | kwargs
        prior_instance = prior_class(**prior_kwargs)

        posterior_instance = cls(prior_instance, likelihood_instance)

        # Refine relative binning solution over all space
        print('Performing a second search...')
        guess = prior_instance.inverse_transform(
            **likelihood_instance.par_dic_0)
        result = utils.differential_evolution_with_guesses(
            lambda pars: -posterior_instance.likelihood.lnlike(
                posterior_instance.prior.transform(*pars)),
            list(prior_instance.range_dic.values()),
            list(guess.values()),
            seed=seed)
        likelihood_instance.par_dic_0 = prior_instance.transform(*result.x)
        print(f'Found solution with lnl = {likelihood_instance._lnl_0}')

        return posterior_instance

    def get_eventdir(self, parentdir):
        """
        Return directory name in which the Posterior instance
        should be saved, of the form
        {parentdir}/{prior_class}/{eventname}/
        """
        return utils.get_eventdir(parentdir, self.prior.__class__.__name__,
                                  self.likelihood.event_data.eventname)


def initialize_posteriors_slurm(eventnames, approximant, prior_class,
                                parentdir, n_hours_limit=2,
                                memory_per_task='4G'):
    """
    Submit jobs that initialize `Posterior.from_event()` for each event.
    """
    package = pathlib.Path(__file__).parents[1].resolve()
    module = f'cogwheel.{os.path.basename(__file__)}'.rstrip('.py')
    
    if isinstance(eventnames, str):
        eventnames = [eventnames]
    for eventname in eventnames:
        eventdir = utils.get_eventdir(parentdir, prior_class, eventname)
        utils.mkdirs(eventdir)

        job_name = f'{eventname}_posterior'
        stdout_path = (eventdir/'posterior_from_event.out').resolve()
        stderr_path = (eventdir/'posterior_from_event.err').resolve()
        args = ' '.join([eventname, approximant, prior_class, parentdir])

        with tempfile.NamedTemporaryFile('w+') as batchfile:
            batchfile.write(textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --output={stdout_path}
                #SBATCH --error={stderr_path}
                #SBATCH --mem-per-cpu={memory_per_task}
                #SBATCH --time={n_hours_limit:02}:00:00

                eval "$(conda shell.bash hook)"
                conda activate {os.environ['CONDA_DEFAULT_ENV']}

                cd {package}
                srun {sys.executable} -m {module} {args}
                """))
            batchfile.seek(0)

            os.system(f'chmod 777 {batchfile.name}')
            os.system(f'sbatch {batchfile.name}')
            time.sleep(.1)


def main(eventname, approximant, prior_class, parentdir):
    '''Construct a Posterior instance and save it to json.'''
    post = Posterior.from_event(eventname, approximant, prior_class)
    post.to_json(post.get_eventdir(parentdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Construct a Posterior instance and save it to json.''')

    parser.add_argument('eventname', help='key from `data.event_registry`.')
    parser.add_argument('approximant', help='key from `waveform.APPROXIMANTS`')
    parser.add_argument('prior_class',
                        help='key from `gw_prior.prior_registry`')
    parser.add_argument('parentdir', help='top directory to save output')

    main(**vars(parser.parse_args()))


# def _test_relative_binning_accuracy(posterior_path, samples_path):
#     """
#     Compute log likelihood of parameter samples with and without
#     relative binning. Results are stored as columns on the samples
#     DataFrame, whose pickle file is overwritten.

#     Parameters
#     ----------
#     posterior_path: path to a json file from a Posterior instance.
#     samples_path: path to a pickle file from a pandas DataFrame.
#     """
#     print('Analyzing', samples_path)
#     posterior = utils.read_json(posterior_path)
#     samples = pd.read_pickle(samples_path)[posterior.prior.sampled_params]
#     result = [posterior.likelihood.test_relative_binning_accuracy(
#         posterior.prior.transform(**sample))
#               for _, sample in samples.iterrows()]
#     samples['lnl_rb'], samples['lnl_fft'] = np.transpose(result)
#     samples.to_pickle(samples_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='''Compute likelihood with and without relative binning.
#                        Input a path to a dataframe with samples, results are
#                        saved as columns in the dataframe.''')

#     parser.add_argument('posterior_path',
#                         help='''path to json file from a `posterior.Posterior`
#                                 object.''')
#     parser.add_argument('samples_path',
#                         help='path to a `pandas` pickle file with samples.')

#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

#     _test_relative_binning_accuracy(**vars(parser.parse_args()))


