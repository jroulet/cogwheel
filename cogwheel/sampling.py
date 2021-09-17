"""Sample posterior or prior distributions."""

import abc
import argparse
import pathlib
import os
import re
import sys
import tempfile
import textwrap
from cProfile import Profile
from functools import wraps
import numpy as np
import pandas as pd
import scipy.special

# import ultranest
# import ultranest.stepsampler
import pymultinest

from . import utils

PROFILING_FILENAME = 'profiling'


class Sampler(abc.ABC, utils.JSONMixin):
    """
    Generic base class for sampling distributions.
    Subclasses implement the interface with specific sampling codes.
    """
    DEFAULT_RUN_KWARGS = {}  # Implemented by subclasses
    RUNDIR_PREFIX = 'run_'

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
        self.samples = None

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
            unfolded = prior._unfold(sample[prior.sampled_params].to_numpy())
            probabilities = np.exp(sample[self._lnprob_cols])
            probabilities /= probabilities.sum()
            resampled.append(choice(unfolded, p=probabilities))

        return pd.DataFrame(resampled, columns=prior.sampled_params)

    def gen_rundir(self, parent_dir, mkdir=False,
                   dir_permissions=utils.DIR_PERMISSIONS):
        """
        Return a `pathlib.Path` object with a new run directory,
        following a standardized naming scheme for output directories.
        Directory will be of the form
        {parent_dir}/{prior_class}/{eventname}/{RUNDIR_PREFIX}{i_run}

        Parameters
        ----------
        parent_dir: str, path to a directory where to store parameter
                    estimation data.
        mkdir: bool, whether to actually make the rundir.
        dir_permissions: octal, directory permissions.
        """
        event_dir = pathlib.Path(parent_dir).joinpath(
            self.posterior.prior.__class__.__name__,
            self.posterior.likelihood.event_data.eventname)
        old_rundirs = [path for path in event_dir.iterdir() if path.is_dir()
                       and path.match(f'{self.RUNDIR_PREFIX}*')]
        run_id = 0
        if old_rundirs:
            run_id = max([int(re.search(r'\d+', rundir.name).group())
                          for rundir in old_rundirs]) + 1
        rundir = event_dir.joinpath(f'{self.RUNDIR_PREFIX}{run_id}')
        if mkdir:
            rundir.mkdir(dir_permissions)
        return rundir

    def submit_slurm(self, rundir, n_hours_limit=168,
                     memory_per_task='32G', resuming=False):
        """
        Parameters
        ----------
        rundir: path of run directory
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

        self.to_json(rundir, overwrite=resuming)

        with tempfile.TemporaryFile() as file:
            file.write(textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --output={stdout_path}
                #SBATCH --open-mode=append
                #SBATCH --mem-per-cpu={memory_per_task}
                #SBATCH --time={int(n_hours_limit)}:00:00

                eval "$(conda shell.bash hook)"
                conda activate {os.environ['CONDA_DEFAULT_ENV']}

                srun {sys.executable} {__file__} {rundir.resolve()}
                """))
            file.seek(0)  # Rewind
            os.system(f'srun {file}')
            print(f'Submitted job {job_name!r}.')

    @abc.abstractmethod
    def _run(self):
        """Sample the distribution."""

    def run(self, dirname):
        """
        Make a directory to save results and run sampler.

        Parameters
        ----------
        dirname: directory where to save output, will create if needed.
        dir_permissions: octal, permissions to give to the directory.
        file_permissions: octal, permissions to give to the files.
        n_iter_before_update: int, how often pymultinest should
                              checkpoint. Checkpointing often is slow
                              and increases the odds of getting
                              corrupted files (if the job gets killed).
        kwargs: passed to `pymultinest.run`, updates `self.run_kwargs`.
        """

        dirname = os.path.join(dirname, '')

        self.run_kwargs['outputfiles_basename'] = dirname

        self.to_json(dirname, dir_permissions=self.dir_permissions,
                     file_permissions=self.file_permissions, overwrite=True)

        with Profile() as profiler:
            self._run()
        profiler.dump_stats(os.path.join(dirname, PROFILING_FILENAME))

        for path in pathlib.Path(dirname).iterdir():
            path.chmod(self.file_permissions)



class PyMultiNest(Sampler):
    """Sample a posterior using PyMultinest."""
    DEFAULT_RUN_KWARGS = {'n_iter_before_update': 10**5,
                          'n_live_points': 2000,
                          'evidence_tolerance': .1}

    @wraps(Sampler.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_kwargs['wrapped_params'] = [
            par in self.posterior.prior.periodic_params
            for par in self.posterior.prior.sampled_params]

    def _run(self):
        pymultinest.run(self._lnprob_pymultinest, self._cubetransform,
                        self._ndim, self._nparams, **self.run_kwargs)

    def load_samples(self, test_relative_binning_accuracy=False):
        """
        Collect pymultinest samples, resample from them to undo the
        parameter folding, optionally compute the likelihood with and
        without relative binning for testing.
        A pandas DataFrame with the samples is stored as attribute
        `samples`.

        Parameters
        ----------
        dirname: Directory where pymultinest output is.
        """
        fname = os.path.join(self.run_kwargs['outputfiles_basename'],
                             'post_equal_weights.dat')
        folded = pd.DataFrame(np.loadtxt(fname)[:, :-1], columns=self.params)
        samples = self.posterior.resample(folded)
        if test_relative_binning_accuracy:
            self.posterior.test_relative_binning_accuracy(samples)
        self.samples = samples

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


def main(sampler_path):
    """Load sampler and run it."""
    dirname = (sampler_path if os.path.isdir(sampler_path)
               else os.path.dirname(sampler_path))
    utils.read_json(sampler_path).run(dirname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a distribution.')
    parser.add_argument('sampler_path', help='''path to a json file from a
                                                `sampling.Sampler` object.''')
    main(**vars(parser.parse_args()))
