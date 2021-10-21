"""Sample posterior or prior distributions."""

import abc
import argparse
import pathlib
import os
import sys
import textwrap
from cProfile import Profile
from functools import wraps
import numpy as np
import pandas as pd
import scipy.special

import pymultinest
# import ultranest
# import ultranest.stepsampler

from . import postprocessing
from . import utils

SAMPLES_FILENAME = 'samples.feather'

class Sampler(abc.ABC, utils.JSONMixin):
    """
    Generic base class for sampling distributions.
    Subclasses implement the interface with specific sampling codes.
    """
    DEFAULT_RUN_KWARGS = {}  # Implemented by subclasses
    PROFILING_FILENAME = 'profiling'
    JSON_FILENAME = 'Sampler.json'

    def __init__(self, posterior, run_kwargs=None, sample_prior=False,
                 dir_permissions=utils.DIR_PERMISSIONS,
                 file_permissions=utils.FILE_PERMISSIONS):
        super().__init__()

        self.posterior = posterior
        nfolds = 2**len(self.posterior.prior.folded_params)

        self._lnprob_cols = [f'lnprob{i}' for i in range(nfolds)]
        self.params = (self.posterior.prior.sampled_params
                       + self._lnprob_cols)
        self._ndim = len(self.posterior.prior.sampled_params)
        self._nparams = len(self.params)

        self.run_kwargs = self.DEFAULT_RUN_KWARGS | (run_kwargs or {})

        self.dir_permissions = dir_permissions
        self.file_permissions = file_permissions

        self._get_lnprobs = None  # Set by sample_prior
        self.sample_prior = sample_prior

    @property
    def sample_prior(self):
        """Whether to sample the prior instead of the posterior."""
        return self._sample_prior

    @sample_prior.setter
    def sample_prior(self, sample_prior):
        self._sample_prior = sample_prior
        func = (self.posterior.prior.lnprior if sample_prior
                else self.posterior.lnposterior)
        self._get_lnprobs = self.posterior.prior.unfold_apply(func)

    def resample(self, samples: pd.DataFrame, seed=0):
        """
        Take a pandas DataFrame of folded samples and return another one
        with samples from the full phase space, drawn with the
        appropriate probabilities.

        Parameters
        ----------
        samples: pandas.DataFrame whose columns match `self.params`
        seed: Seed for the random number generator, determines to which
              unfolding the samples are assigned.

        Return
        ------
        pandas.DataFrame with columns per
        `self.posterior.prior.sampled_params`.
        """
        choice = np.random.default_rng(seed=seed).choice
        prior = self.posterior.prior

        resampled = []
        for _, sample in samples.iterrows():
            unfolded = prior.unfold(sample[prior.sampled_params].to_numpy())
            probabilities = np.exp(sample[self._lnprob_cols])
            probabilities /= probabilities.sum()
            resampled.append(choice(unfolded, p=probabilities))

        return pd.DataFrame(resampled, columns=prior.sampled_params)

    def get_rundir(self, parentdir):
        """
        Return a `pathlib.Path` object with a new run directory,
        following a standardized naming scheme for output directories.
        Directory will be of the form
        {parentdir}/{prior_name}/{eventname}/{RUNDIR_PREFIX}{i_run}

        Parameters
        ----------
        parentdir: str, path to a directory where to store parameter
                   estimation data.
        """
        eventdir = self.posterior.get_eventdir(parentdir)
        old_rundirs = [path for path in eventdir.iterdir() if path.is_dir()
                       and path.match(f'{utils.RUNDIR_PREFIX}*')]
        run_id = 0
        if old_rundirs:
            run_id = 1 + max(utils.rundir_number(rundir)
                             for rundir in old_rundirs)
        return eventdir.joinpath(f'{utils.RUNDIR_PREFIX}{run_id}')

    def submit_slurm(self, rundir, n_hours_limit=48,
                     memory_per_task='32G', resuming=False):
        """
        Parameters
        ----------
        rundir: path of run directory, e.g. from `self.get_rundir`
        n_hours_limit: Number of hours to allocate for the job
        memory_per_task: Determines the memory and number of cpus
        resuming: bool, whether to attempt resuming a previous run if
                  rundir already exists.
        """
        rundir = pathlib.Path(rundir)
        job_name = '_'.join([self.__class__.__name__,
                             self.posterior.prior.__class__.__name__,
                             self.posterior.likelihood.event_data.eventname,
                             rundir.name])
        stdout_path = rundir.joinpath('output.out').resolve()
        stderr_path = rundir.joinpath('errors.err').resolve()

        self.to_json(rundir, overwrite=resuming)

        package = pathlib.Path(__file__).parents[1].resolve()
        module = f'cogwheel.{os.path.basename(__file__)}'.rstrip('.py')

        batch_path = rundir/'batchfile'
        with open(batch_path, 'w+') as batchfile:
            batchfile.write(textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --output={stdout_path}
                #SBATCH --error={stderr_path}
                #SBATCH --open-mode=append
                #SBATCH --mem-per-cpu={memory_per_task}
                #SBATCH --time={n_hours_limit:02}:00:00

                eval "$(conda shell.bash hook)"
                conda activate {os.environ['CONDA_DEFAULT_ENV']}

                cd {package}
                srun {sys.executable} -m {module} {rundir.resolve()}
                """))
        batch_path.chmod(0o777)
        os.system(f'sbatch {batch_path.resolve()}')
        print(f'Submitted job {job_name!r}.')

    @abc.abstractmethod
    def _run(self):
        """Sample the distribution."""

    def run(self, rundir):
        """
        Make a directory to save results and run sampler.

        Parameters
        ----------
        rundir: directory where to save output, will create if needed.
        """

        rundir = pathlib.Path(rundir)
        self.to_json(rundir, dir_permissions=self.dir_permissions,
                     file_permissions=self.file_permissions, overwrite=True)

        with Profile() as profiler:
            self._run()
        profiler.dump_stats(rundir/self.PROFILING_FILENAME)

        self.load_samples().to_feather(rundir/SAMPLES_FILENAME)

        for path in rundir.iterdir():
            path.chmod(self.file_permissions)

    @abc.abstractmethod
    def load_samples(self):
        """
        Collect samples, resample from them to undo the parameter
        folding. Return a pandas.DataFrame with the samples.
        """

    @abc.abstractmethod
    def load_evidence(self):
        """
        Read evidence from sampling output.
        Return a dict with the following items:
          'log_ev' = log evidence from nested sampling
          'log_ev_std' = log standard deviation of evidence
        If using nested importance sampling (NIS), will also have:
          'log_ev_NIS' = log evidence from nested importance sampling
          'log_ev_std_NIS' = log standard deviation of NIS evidence
        """

    @staticmethod
    def completed(rundir) -> bool:
        """Return whether the run completed successfully."""
        return (pathlib.Path(rundir)/SAMPLES_FILENAME).exists()

    @wraps(utils.JSONMixin.to_json)
    def to_json(self, dirname, basename=None, **kwargs):
        """
        Make `basename` default to 'Sampler.json' even for subclasses.
        That way it is easier to locate the file if we don't know the
        sampler subclass.
        """
        super().to_json(dirname, basename or self.JSON_FILENAME, **kwargs)


class PyMultiNest(Sampler):
    """Sample a posterior or prior using PyMultiNest."""
    DEFAULT_RUN_KWARGS = {'n_iter_before_update': 1000,
                          'n_live_points': 2048,
                          'evidence_tolerance': 1/4}

    @wraps(Sampler.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_kwargs['wrapped_params'] = [
            par in self.posterior.prior.periodic_params
            for par in self.posterior.prior.sampled_params]

    def _run(self):
        pymultinest.run(self._lnprob_pymultinest, self._cubetransform,
                        self._ndim, self._nparams, **self.run_kwargs)

    def load_samples(self):
        """
        Collect pymultinest samples, resample from them to undo the
        parameter folding. Return a pandas.DataFrame with the samples.
        """
        fname = os.path.join(self.run_kwargs['outputfiles_basename'],
                             'post_equal_weights.dat')
        folded = pd.DataFrame(np.loadtxt(fname)[:, :-1], columns=self.params)
        return self.resample(folded)

    def load_evidence(self):
        evdic = {}
        with open(os.path.join(self.run_kwargs['outputfiles_basename'],
                               'stats.dat')) as stats_file:
            line = stats_file.readline()
            if 'Nested Sampling Global Log-Evidence' in line:
                evdic['log_ev'] = float(line.strip().split()[5])
                evdic['log_ev_std'] = float(line.strip().split()[7])
            line = stats_file.readline()
            if 'Nested Importance Sampling Global Log-Evidence' in line:
                evdic['log_ev_NIS'] = float(line.strip().split()[5])
                evdic['log_ev_std_NIS'] = float(line.strip().split()[7])
        return evdic

    def _lnprob_pymultinest(self, par_vals, *_):
        """
        Update the extra entries `par_vals[n_dim : n_params+1]` with the
        log posterior evaulated at each unfold. Return the logarithm of
        the folded posterior.
        """
        lnprobs = self._get_lnprobs(*[par_vals[i] for i in range(self._ndim)])
        for i, lnprob in enumerate(lnprobs):
            par_vals[self._ndim + i] = lnprob
        return scipy.special.logsumexp(lnprobs)

    def _cubetransform(self, cube, *_):
        for i in range(self._ndim):
            cube[i] = (self.posterior.prior.cubemin[i]
                       + cube[i] * self.posterior.prior.folded_cubesize[i])

    @wraps(utils.JSONMixin.to_json)
    def to_json(self, dirname, *args, **kwargs):
        """
        Update run_kwargs['outputfiles_basename'] before saving.
        Parameters are as in `utils.JSONMixin.to_json()`
        """
        self.run_kwargs['outputfiles_basename'] = os.path.join(dirname, '')
        super().to_json(dirname, *args, **kwargs)

# class Ultranest(Sampler):
#     """
#     Sample a posterior using Ultranest.
#     (Doesn't work well yet)
#     """
#     def __init__(self, posterior, run_kwargs):
#         super().__init__(posterior, run_kwargs)
#         self.sampler = None
#         print("Warning: I couldn't produce reasonable results with this class")

#     def instantiate_sampler(self, run=False, *, sample_prior=False,
#                             n_fast_steps=8, **kwargs):
#         """Set up `self.sampler`."""
#         lnprob = (self._lnprior_ultranest if sample_prior
#                   else self._lnposterior_ultranest)

#         wrapped_params = [par in self.posterior.periodic_params
#                           for par in self.posterior.prior.sampled_params]

#         self.sampler = ultranest.ReactiveNestedSampler(
#             self.posterior.prior.sampled_params, lnprob, self._cubetransform,
#             wrapped_params=wrapped_params)
#         self.sampler.stepsampler \
#             = ultranest.stepsampler.SpeedVariableRegionSliceSampler(
#                 self._get_step_matrix(n_fast_steps))
#         if run:
#             self.sampler.run(**kwargs)

#     def _get_step_matrix(self, n_steps: int):
#         """
#         Return matrix with pattern of fast/slow steps.

#         Parameters
#         ----------
#         n_steps: int, the cycle will have one slow step and
#                  `n_steps - 1` fast steps.
#         """
#         fast_sampled_params = self.posterior.prior.get_fast_sampled_params(
#             self.posterior.likelihood.waveform_generator.fast_params)

#         step_matrix = np.ones(
#             (n_steps, len(self.posterior.prior.sampled_params)), bool)
#         step_matrix[1:] = [par in fast_sampled_params
#                            for par in self.posterior.prior.sampled_params]
#         return step_matrix

#     def _cubetransform(self, cube):
#         return self.posterior.prior.cubemin + cube * self.posterior.cubesize

#     def _lnprior_ultranest(self, par_vals):
#         return self.posterior.prior.lnprior(*par_vals)

#     def _lnposterior_ultranest(self, par_vals):
#         return self.posterior.lnposterior(*par_vals)


def main(sampler_path, postprocess=True):
    """Load sampler and run it."""
    rundir = (sampler_path if os.path.isdir(sampler_path)
              else os.path.dirname(sampler_path))

    utils.read_json(sampler_path).run(rundir)

    if postprocess:
        postprocessing.postprocess_rundir(rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a distribution.')
    parser.add_argument('sampler_path', help='''path to a json file from a
                                                `sampling.Sampler` object.''')
    parser.add_argument('--no_postprocessing', action='store_true',
                        help='''Not postprocess the samples.''')
    parser_args = parser.parse_args()
    main(parser_args.sampler_path, not parser_args.no_postprocessing)
