"""Sample posterior or prior distributions."""

import pathlib
import os
from scipy import special
import numpy as np
import pandas as pd

import ultranest
import ultranest.stepsampler
import pymultinest

from . import utils

sampler_registry = {}


class Sampler(utils.JSONMixin):
    """
    Generic base class for sampling distributions.
    Subclasses implement the interface with specific sampling codes.
    """
    DEFAULT_RUN_KWARGS = {}

    def __init__(self, posterior, run_kwargs=None):
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

    def resample(self, samples: pd.DataFrame, seed=0):
        """
        Take a pandas DataFrame of folded samples and return another one
        with samples from the full phase space, drawn with the
        appropriate probabilities.
        """
        choice = np.random.default_rng(seed=seed).choice

        resampled = []
        for _, sample in samples.iterrows():
            unfolded = self.posterior.prior._unfold(
                sample[self.posterior.prior.sampled_params].to_numpy())
            probabilities = np.exp(sample[self._lnprob_cols])
            probabilities /= probabilities.sum()
            resampled.append(choice(unfolded, p=probabilities))

        return pd.DataFrame(resampled,
                            columns=self.posterior.prior.sampled_params)

    def __init_subclass__(cls):
        """Register subclass in `sampler_registry`."""
        super().__init_subclass__()
        sampler_registry[cls.__name__] = cls


class PyMultinest(Sampler):
    """Sample a posterior using PyMultinest."""
    DEFAULT_RUN_KWARGS = {'n_iter_before_update': 10**5,
                          'n_live_points': 2000,
                          'evidence_tolerance': .1}

    def __init__(self, posterior, run_kwargs=None):
        super().__init__(posterior, run_kwargs)
        self.run_kwargs['wrapped_params'] = [
            par in self.posterior.prior.periodic_params
            for par in self.posterior.prior.sampled_params]

    def run_pymultinest(self, dirname, dir_permissions=0o755,
                        file_permissions=0o644, **run_kwargs):
        """
        Make a directory to save results and run pymultinest.

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

        self.run_kwargs |= run_kwargs
        self.run_kwargs['outputfiles_basename'] = dirname

        self.to_json(dirname, dir_permissions=dir_permissions,
                     file_permissions=file_permissions, overwrite=True)

        pymultinest.run(self._lnposterior, self._cubetransform, self._ndim,
                        self._nparams, **self.run_kwargs)

        for path in pathlib.Path(dirname).iterdir():
            path.chmod(file_permissions)

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

    def _cubetransform(self, cube, *_):
        for i in range(self._ndim):
            cube[i] = (self.posterior.prior.cubemin[i]
                       + cube[i] * self.posterior.cubesize[i])

    def _lnposterior(self, par_vals, *_):
        """
        Update the extra entries `par_vals[n_dim : n_params+1]` with the
        log posterior evaulated at each unfold. Return the logarithm of
        the folded posterior.
        """
        lnposts = self.posterior.lnposteriors_folds(
            *[par_vals[i] for i in range(self._ndim)])
        for i, lnpost in enumerate(lnposts):
            par_vals[self._ndim + i] = lnpost
        return special.logsumexp(lnposts)


class Ultranest(utils.JSONMixin):
    """
    Sample a posterior using Ultranest.
    (Doesn't work well yet)
    """
    def __init__(self, posterior):
        self.posterior = posterior
        self.sampler = None
        print("Warning: I couldn't produce reasonable results with this class")

    def instantiate_sampler(self, run=False, *, sample_prior=False,
                            n_fast_steps=8, **kwargs):
        """Set up `self.sampler`."""
        lnprob = (self._lnprior_ultranest if sample_prior
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

    def _get_step_matrix(self, n_steps: int):
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
        return self.posterior.prior.cubemin + cube * self.posterior.cubesize

    def _lnprior_ultranest(self, par_vals):
        return self.posterior.prior.lnprior(*par_vals)

    def _lnposterior_ultranest(self, par_vals):
        return self.posterior.lnposterior(*par_vals)
