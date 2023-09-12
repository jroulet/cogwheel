"""Sample posterior or prior distributions."""

import abc
import argparse
import copy
import datetime
import inspect
import multiprocessing
import pathlib
import os
import sys
import textwrap
import warnings
from cProfile import Profile
from functools import wraps
import numpy as np
import pandas as pd
import scipy.special

import dynesty
import pymultinest
import zeus
import ultranest
import nautilus

from cogwheel import postprocessing
from cogwheel import utils

SAMPLES_FILENAME = 'samples.feather'
FINISHED_FILENAME = 'FINISHED.out'


class Sampler(abc.ABC, utils.JSONMixin):
    """
    Generic base class for sampling distributions.
    Subclasses implement the interface with specific sampling codes.

    Parameter space folding is used; this means that some ("folded")
    dimensions are sampled over half their original range, and a map to
    the other half of the range is defined by reflecting or shifting
    about the midpoint. The folded posterior distribution is defined as
    the sum of the original posterior over all `2**n_folds` mapped
    points. This is intended to reduce the number of modes in the
    posterior.
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

        self.run_kwargs = (copy.deepcopy(self.DEFAULT_RUN_KWARGS)
                           | (run_kwargs or {}))

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
        prior = self.posterior.prior

        unfold = np.vectorize(prior.unfold, signature='(n)->(m,n)')
        unfolded = unfold(samples[prior.sampled_params].to_numpy())

        probabilities = utils.exp_normalize(
            samples[self._lnprob_cols].to_numpy())
        cumprobs = np.cumsum(probabilities, axis=-1)

        searchsorted = np.vectorize(np.searchsorted, signature='(n),()->()')
        rng = np.random.default_rng(seed)
        inds = searchsorted(cumprobs, rng.uniform(size=len(samples)))

        return pd.DataFrame(unfolded[np.arange(len(inds)), inds],
                            columns=prior.sampled_params)

    def get_rundir(self, parentdir):
        """
        Return a `pathlib.Path` object with a new run directory,
        following a standardized naming scheme for output directories.
        Directory will be of the form
        {parentdir}/{prior_name}/{eventname}/{RUNDIR_PREFIX}{run_id}

        Parameters
        ----------
        parentdir: str, path to a directory where to store parameter
                   estimation data.
        """
        return utils.get_rundir(self.posterior.get_eventdir(parentdir))

    def submit_slurm(
            self, rundir, n_hours_limit=48, memory_per_task='32G',
            resuming=False, sbatch_cmds=(), postprocess=True):
        """
        Parameters
        ----------
        rundir: str, os.PathLike
            Run directory, e.g. from `self.get_rundir`

        n_hours_limit: int
            Number of hours to allocate for the job.

        memory_per_task: str
            Determines the memory and number of cpus.

        resuming: bool
            Whether to attempt resuming a previous run if rundir already
            exists.

        sbatch_cmds: tuple of str
            Strings with SBATCH commands.

        postprocess: bool
            Whether to perform convergence tests to the run after
            sampling. See ``postprocessing.postprocess_rundir``.
        """
        rundir = pathlib.Path(rundir)
        job_name = '_'.join([rundir.name,
                             self.posterior.likelihood.event_data.eventname,
                             self.posterior.prior.__class__.__name__,
                             self.__class__.__name__])
        batch_path = rundir/'batchfile'
        stdout_path = rundir.joinpath('output.out').resolve()
        stderr_path = rundir.joinpath('errors.err').resolve()

        self.to_json(rundir, overwrite=resuming)

        sbatch_cmds += (f'--mem-per-cpu={memory_per_task}',)
        args = str(rundir.resolve())

        if not postprocess:
            args += ' --no-postprocessing'

        utils.submit_slurm(job_name, n_hours_limit, stdout_path, stderr_path,
                           args, sbatch_cmds, batch_path)

    def submit_lsf(self, rundir, n_hours_limit=48,
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
        module = f'cogwheel.{os.path.basename(__file__)}'.removesuffix('.py')

        batch_path = rundir/'batchfile'
        with open(batch_path, 'w+', encoding='utf-8') as batchfile:
            batchfile.write(textwrap.dedent(f"""\
                #BSUB -J {job_name}
                #BSUB -o {stdout_path}
                #BSUB -e {stderr_path}
                #BSUB -M {memory_per_task}
                #BSUB -W {n_hours_limit:02}:00

                eval "$(conda shell.bash hook)"
                conda activate {os.environ['CONDA_DEFAULT_ENV']}

                cd {package}
                srun {sys.executable} -m {module} {rundir.resolve()}
                """))
        batch_path.chmod(0o777)
        os.system(f'bsub < {batch_path.resolve()}')
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
            exit_code = self._run()
            with open(rundir/FINISHED_FILENAME, 'w', encoding='utf-8') as file:
                file.write(f'{exit_code}\n{datetime.datetime.now()}')

            samples = self.load_samples()
            self.posterior.prior.transform_samples(samples)
            self.posterior.likelihood.postprocess_samples(samples)
            samples.to_feather(rundir/SAMPLES_FILENAME)

        profiler.dump_stats(rundir/self.PROFILING_FILENAME)

        for path in rundir.iterdir():
            path.chmod(self.file_permissions)

    @abc.abstractmethod
    def load_samples(self):
        """
        Collect samples, resample from them to undo the parameter
        folding. Return a pandas.DataFrame with the samples.
        """

    def load_evidence(self) -> dict:
        """
        Define for sampling classes which compute evidence.
        Return a dict with the following items:
          'log_ev' = log evidence from sampling
          'log_ev_std' = log standard deviation of evidence
        If using nested importance sampling (NIS), should also have:
          'log_ev_NIS' = log evidence from nested importance sampling
          'log_ev_std_NIS' = log standard deviation of NIS evidence
        """
        raise NotImplementedError(
            'Implement in subclass (if sampler computes evidence).')

    @staticmethod
    def completed(rundir) -> bool:
        """Return whether the run completed successfully."""
        return (pathlib.Path(rundir)/SAMPLES_FILENAME).exists()

    @wraps(utils.JSONMixin.to_json)
    def to_json(self, dirname, basename=None, **kwargs):
        # Make `basename` default to 'Sampler.json' even for subclasses.
        # That way it is easier to locate the file if we don't know the
        # sampler subclass.
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
        if not os.path.exists(fname):
            fname = fname[:100]  # Weird PyMultinest filename length limit

        folded = pd.DataFrame(np.loadtxt(fname)[:, :-1], columns=self.params)
        return self.resample(folded)

    def load_evidence(self):
        evdic = {}
        with open(os.path.join(self.run_kwargs['outputfiles_basename'],
                               'stats.dat'), encoding='utf-8') as stats_file:
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
        log posterior evaluated at each unfold. Return the logarithm of
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


class Dynesty(Sampler):
    """Sample a posterior or prior using ``dynesty``."""
    DEFAULT_RUN_KWARGS = {}

    @wraps(Sampler.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = None

    def _run(self):
        periodic = [self.posterior.prior.sampled_params.index(par)
                    for par in self.posterior.prior.periodic_params]
        reflective = [self.posterior.prior.sampled_params.index(par)
                      for par in self.posterior.prior.reflective_params]

        sampler_keys = (
            set(inspect.signature(dynesty.DynamicNestedSampler).parameters)
            & self.run_kwargs.keys())
        sampler_kwargs = {par: self.run_kwargs[par] for par in sampler_keys}

        run_keys = self.run_kwargs.keys() - sampler_keys
        run_kwargs = {par: self.run_kwargs[par] for par in run_keys}

        self.sampler = dynesty.DynamicNestedSampler(
            self._lnprob_dynesty,
            self._cubetransform,
            len(self.posterior.prior.sampled_params),
            rstate=np.random.default_rng(0),
            periodic=periodic or None,
            reflective=reflective or None,
            sample='rwalk',
            **sampler_kwargs)
        self.sampler.run_nested(**run_kwargs)

    def load_samples(self):
        """
        Collect dynesty samples, resample from them to undo the
        parameter folding. Return a ``pandas.DataFrame`` with samples.
        """
        folded = pd.DataFrame(self.sampler.results.samples,
                              columns=self.posterior.prior.sampled_params)

        # ``dynesty`` doesn't allow to save samples' metadata, so we
        # have to recompute ``lnprobs``:
        lnprobs = pd.DataFrame(
            [self._get_lnprobs(**row) for _, row in folded.iterrows()],
            columns=self._lnprob_cols)
        utils.update_dataframe(folded, lnprobs)

        samples = self.resample(folded)
        samples[utils.WEIGHTS_NAME] = np.exp(
            self.sampler.results.logwt - self.sampler.results.logwt.max())
        return samples

    def _lnprob_dynesty(self, par_vals):
        """Return the logarithm of the folded probability density."""
        return scipy.special.logsumexp(self._get_lnprobs(*par_vals))

    def _cubetransform(self, cube):
        return (self.posterior.prior.cubemin
                + cube * self.posterior.prior.folded_cubesize)


class Zeus(Sampler):
    """
    Sample a posterior or prior using ``zeus``.
    https://zeus-mcmc.readthedocs.io/en/latest/index.html.

    ``run_kwargs`` can take kwargs to
        * ``zeus.EnsembleSampler``
        * ``zeus.EnsembleSampler.run_mcmc``
        * ``zeus.EnsembleSampler.get_chain`` (discard, thin)
    with the caveat that they must be JSON-serializable. Therefore:
    To use ``callbacks``, pass a list of the form
    ``[(callback_name, kwargs), ...]``.
    To parallelize with a pool, pass a kwarg ``processes`` (int).
    See ``Zeus.DEFAULT_RUN_KWARGS`` for examples. Note you can
    override these by editing ``zeus_sampler.run_kwargs``.
    """
    DEFAULT_RUN_KWARGS = {
        'nwalkers': 50,
        'nsteps': 1000,
        'maxiter': 10**5,
        'light_mode': False,
        'discard': .1,
        'thin': 1,
        'processes': 1,  # For parallelization
        'callbacks': [
            # ('SplitRCallback', {}),
            # ('AutocorrelationCallback', {}),
            # ('MinIterCallback', {'nmin': 100}),
            ('SaveProgressCallback', {'filename': '{rundir}/chain.h5'})],
        'use_injection_value': False,  # "Cheat" and initialize walkers near injection
        }

    @wraps(Sampler.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampler = None
        self._rundir = None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    'invalid value encountered in add')
            self._folded_cubemax = (self.posterior.prior.cubemin
                                   + self.posterior.prior.folded_cubesize)
        self._folded_cubemax[np.isnan(self._folded_cubemax)] = np.inf

    @wraps(Sampler.run)
    def run(self, rundir):
        self._rundir = rundir
        super().run(rundir)

    def _run(self):
        sampler_kwargs = self._get_sampler_kwargs()
        run_mcmc_kwargs = self._get_run_mcmc_kwargs()

        processes = self.run_kwargs.get('processes', 1)
        if processes == 1:  # Serial
            self.sampler = zeus.EnsembleSampler(**sampler_kwargs)
            self.sampler.run_mcmc(**run_mcmc_kwargs)
        else:  # Parallel
            if processes > sampler_kwargs['nwalkers'] / 2:
                warnings.warn('Some processes will likely be idle, consider '
                              'increasing `nwalkers` to be `>= 2*processes` '
                              '[arxiv.org/abs/2002.06212].')
            with multiprocessing.Pool(processes) as pool:
                self.sampler = zeus.EnsembleSampler(pool=pool,
                                                    **sampler_kwargs)
                self.sampler.run_mcmc(**run_mcmc_kwargs)

    def _get_sampler_kwargs(self):
        sampler_keys = (
            set(inspect.signature(zeus.EnsembleSampler).parameters)
            & self.run_kwargs.keys())
        sampler_kwargs = {par: self.run_kwargs[par] for par in sampler_keys}

        global _lnprob_zeus  # Hack for multiprocessing to work
        def  _lnprob_zeus(*args, **kwargs):
            return self._lnprob_zeus(*args, **kwargs)

        sampler_kwargs.update(
            ndim=self._ndim,
            logprob_fn=_lnprob_zeus,
            blobs_dtype=[(col, float) for col in self._lnprob_cols])
        return sampler_kwargs

    def _get_run_mcmc_kwargs(self):
        run_mcmc_keys = (
            set(inspect.signature(zeus.EnsembleSampler.run_mcmc).parameters)
            & self.run_kwargs.keys())
        run_mcmc_kwargs = {par: self.run_kwargs[par] for par in run_mcmc_keys}

        if 'start' not in run_mcmc_kwargs:
            run_mcmc_kwargs['start'] = self._get_start()  # Initialize walkers

        # Instantiate callbacks from the JSON-serializable input:
        callbacks = self.run_kwargs.get('callbacks', [])
        save_progress_kwargs = dict(callbacks).get('SaveProgressCallback', {})
        if filename := save_progress_kwargs.get('filename'):
            # Dynamically interpolate ``rundir`` to filename:
            save_progress_kwargs['filename'] = filename.format(
                rundir=self._rundir)
        run_mcmc_kwargs['callbacks'] = [getattr(zeus.callbacks, name)(**kwargs)
                                        for name, kwargs in callbacks]
        return run_mcmc_kwargs

    def load_samples(self):
        """
        Collect zeus samples, resample from them to undo the parameter
        folding. Return a pandas.DataFrame with the samples.
        """
        kwargs = {par: self.run_kwargs[par]
                  for par in self.run_kwargs.keys() & {'discard', 'thin'}}
        folded = pd.DataFrame(self.sampler.get_chain(flat=True, **kwargs),
                              columns=self.posterior.prior.sampled_params)

        lnprobs = pd.DataFrame(self.sampler.get_blobs(flat=True, **kwargs))
        utils.update_dataframe(folded, lnprobs)
        return self.resample(folded)

    def _lnprob_zeus(self, par_vals):
        """
        Return the logarithm of the folded probability density and
        the individual log probabilities at each fold.
        """
        if (np.any(par_vals < self.posterior.prior.cubemin)
                or np.any(par_vals > self._folded_cubemax)):
            return (-np.inf,) * (1+2**len(self.posterior.prior.folded_params))

        lnprobs = self._get_lnprobs(*par_vals)
        return max(-1e100, scipy.special.logsumexp(lnprobs)), *lnprobs

    def _get_start(self, rscale=1e-2, max_lnprob_drop=10.):
        """
        Return initial position of the walkers in a Gaussian "cloud"
        around the reference solution `par_dic_0`.

        Parameters
        ----------
        rscale: float
            Standard deviation of the cloud of walkers in units of each
            parameter's range.

        max_lnprob_drop: float
            Keep redrawing samples if their log probability is below
            that of other samples by more than this.
        """
        if (self.run_kwargs.get('use_injection_value', False)
                and self.posterior.likelihood.event_data.injection):
            par_dic_0 = self.posterior.likelihood.event_data.injection['par_dic']
        else:
            par_dic_0 = self.posterior.likelihood.par_dic_0

        sampled_par_dic = self.posterior.prior.inverse_transform(**par_dic_0)

        loc = self.posterior.prior.fold(**sampled_par_dic)
        scale = rscale * self.posterior.prior.folded_cubesize
        scale[np.isinf(scale)] = rscale  # Interpret scale as absolute
                                         # if range is inf
        a = (self.posterior.prior.cubemin - loc) / scale
        b = (self._folded_cubemax - loc) / scale

        nparams = len(self.posterior.prior.sampled_params)
        mask = np.full(self.run_kwargs['nwalkers'], True)
        start = np.empty((self.run_kwargs['nwalkers'], nparams))
        log_prob = np.empty(self.run_kwargs['nwalkers'])

        while any(mask):  # Draw positions for these walkers
            start[mask] = scipy.stats.truncnorm.rvs(
                loc=loc,
                scale=scale,
                a=a,
                b=b,
                size=(np.count_nonzero(mask), nparams))
            log_prob[mask] = [self._lnprob_zeus(x)[0] for x in start[mask]]
            mask = log_prob < log_prob.max() - max_lnprob_drop

        return start


class UltraNest(Sampler):
    """Sample a posterior or prior using UltraNest."""
    DEFAULT_RUN_KWARGS = {'Lepsilon': 0.5,
                          'frac_remain': 1e-2,
                          'min_ess': 1000}

    def _cubetransform(self, cube):
        return (self.posterior.prior.cubemin
                + cube * self.posterior.prior.folded_cubesize)

    def _run(self):
        sampler_kwargs = self._get_sampler_kwargs()
        run_kwargs = self._get_run_kwargs()

        self.sampler = ultranest.ReactiveNestedSampler(**sampler_kwargs)
        self.sampler.run(**run_kwargs)

    def _get_sampler_kwargs(self):
        sampler_keys = (
            set(inspect.signature(ultranest.ReactiveNestedSampler).parameters)
            & self.run_kwargs.keys())
        sampler_kwargs = {par: self.run_kwargs[par] for par in sampler_keys}

        wrapped_params = [par in self.posterior.prior.periodic_params
                          for par in self.posterior.prior.sampled_params]

        sampler_kwargs.update(
            param_names=self.posterior.prior.sampled_params,
            wrapped_params=wrapped_params,
            transform=self._cubetransform,
            loglike=self._lnprob_ultranest)
        return sampler_kwargs

    def _get_run_kwargs(self):
        run_keys = (
            set(inspect.signature(ultranest.ReactiveNestedSampler.run).parameters)
            & self.run_kwargs.keys())
        run_kwargs = {par: self.run_kwargs[par] for par in run_keys}
        return run_kwargs

    def _lnprob_ultranest(self, par_vals):
        """Return the logarithm of the folded probability density."""
        lnprobs = self._get_lnprobs(*par_vals)
        return max(-1e100, scipy.special.logsumexp(lnprobs))

    def load_samples(self):
        """
        Collect ultranest samples, resample from them to undo the
        parameter folding. Return a ``pandas.DataFrame`` with samples.
        """
        log_dir = pathlib.Path(self.run_kwargs['log_dir'])
        resume = self.run_kwargs.get(
            'resume', inspect.signature(
                ultranest.ReactiveNestedSampler).parameters['resume'].default)
        if resume == 'subfolder':
            path = sorted(log_dir.glob('run*/chains/weighted_post.txt')
                          )[-1]
        else:
            path = log_dir.joinpath('chains', 'weighted_post.txt')

        result = pd.read_csv(path, sep='\s+')
        folded = result[self.posterior.prior.sampled_params]

        # ``ultranest`` doesn't allow to save samples' metadata, so we
        # have to recompute ``lnprobs``:
        lnprobs = pd.DataFrame(
            [self._get_lnprobs(**row) for _, row in folded.iterrows()],
            columns=self._lnprob_cols)
        utils.update_dataframe(folded, lnprobs)

        samples = self.resample(folded)
        samples[utils.WEIGHTS_NAME] = result['weight']
        return samples

    @wraps(utils.JSONMixin.to_json)
    def to_json(self, dirname, *args, **kwargs):
        """Update run_kwargs['log_dir'] before saving."""
        self.run_kwargs['log_dir'] = str(dirname)
        super().to_json(dirname, *args, **kwargs)


class Nautilus(Sampler):
    """Sample a posterior or prior using Nautilus."""
    DEFAULT_RUN_KWARGS = {'verbose': True}

    def _run(self):
        sampler_kwargs = self._get_sampler_kwargs()
        run_kwargs = self._get_run_kwargs()

        self.sampler = nautilus.Sampler(**sampler_kwargs)
        self.sampler.run(**run_kwargs)

    def _get_sampler_kwargs(self):
        sampler_keys = (
            set(inspect.signature(nautilus.Sampler).parameters)
            & self.run_kwargs.keys())
        sampler_kwargs = {par: self.run_kwargs[par] for par in sampler_keys}

        prior = nautilus.Prior()
        for par, rng in self.posterior.prior.range_dic.items():
            prior.add_parameter(par, tuple(rng))

        blobs_dtype=[(col, float) for col in self._lnprob_cols]

        return sampler_kwargs | {'prior': prior,
                                 'likelihood': self._lnprob_nautilus,
                                 'blobs_dtype': blobs_dtype}

    def _get_run_kwargs(self):
        run_keys = (
            set(inspect.signature(nautilus.Sampler.run).parameters)
            & self.run_kwargs.keys())
        run_kwargs = {par: self.run_kwargs[par] for par in run_keys}
        return run_kwargs

    def _lnprob_nautilus(self, par_dic):
        """Return the logarithm of the folded probability density."""
        lnprobs = self._get_lnprobs(**par_dic)
        return scipy.special.logsumexp(lnprobs), *lnprobs

    def load_samples(self):
        """
        Collect Nautilus samples, resample from them to undo the
        parameter folding. Return a ``pandas.DataFrame`` with samples.
        """
        points, log_w, log_l, blobs = self.sampler.posterior(return_blobs=True)
        lnprobs = pd.DataFrame(blobs)
        folded = pd.DataFrame(points, columns=self.posterior.prior.sampled_params)
        utils.update_dataframe(folded, lnprobs)

        samples = self.resample(folded)
        samples[utils.WEIGHTS_NAME] = np.exp(log_w)
        return samples


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
    parser.add_argument('--no-postprocessing', action='store_true',
                        help='''Not postprocess the samples.''')
    parser_args = parser.parse_args()
    main(parser_args.sampler_path, not parser_args.no_postprocessing)
