"""Sample posterior distributions."""

import abc
import argparse
import copy
import inspect
import multiprocessing
import pathlib
import os
import warnings
from cProfile import Profile
from functools import wraps
import numpy as np
import pandas as pd
import scipy.special

import dynesty
import nautilus
import zeus

from cogwheel import postprocessing
from cogwheel import utils

SAMPLES_FILENAME = 'samples.feather'


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

    @property
    @staticmethod
    @abc.abstractmethod
    def _SAMPLER_CLS():
        """E.g. ``dynesty.DynamicNestedSampler``."""

    _RUN_METHOD_NAME = 'run'  # Method of ``._SAMPLER_CLS``

    def __init__(self, posterior, run_kwargs=None, sample_prior=False,
                 dir_permissions=utils.DIR_PERMISSIONS,
                 file_permissions=utils.FILE_PERMISSIONS):
        """
        Parameters
        ----------
        posterior: cogwheel.posterior.Posterior
            Implements the prior and likelihood.

        run_kwargs: dict
            Keyword arguments for the sampler or its `run` method.
            Allowed keys depend on the particular sampler used.

        sample_prior: False
            Deprecated, will raise ValueError if it is not False.

        dir_permissions, file_permissions: octal
            Directory and file permissions.
        """
        super().__init__()

        if sample_prior:
            raise ValueError(
                'The functionality to sample the prior has been removed from '
                '``Sampler``. Use '
                '``cogwheel.prior.Prior.generate_random_samples`` instead.')
        self.sample_prior = False

        self.posterior = posterior

        self._ndim = len(self.posterior.prior.sampled_params)

        self.run_kwargs = (copy.deepcopy(self.DEFAULT_RUN_KWARGS)
                           | (run_kwargs or {}))

        self.dir_permissions = dir_permissions
        self.file_permissions = file_permissions

        self._get_lnprobs_pardics_metadatas \
            = self.posterior.prior.unfold_apply(
                self.posterior.lnposterior_pardic_and_metadata,
                otypes=[float, dict, object])

        self.sampler = None
        self._rng = np.random.default_rng()

        self._blobs_dtype = self._get_blobs_dtype()

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
        self._submit('slurm', rundir, n_hours_limit, memory_per_task,
                     resuming, sbatch_cmds, postprocess)

    def submit_lsf(
            self, rundir, n_hours_limit=48, memory_per_task='32G',
            resuming=False, bsub_cmds=(), postprocess=True):
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

        bsub_cmds: tuple of str
            Strings with BSUB commands.

        postprocess: bool
            Whether to perform convergence tests to the run after
            sampling. See ``postprocessing.postprocess_rundir``.
        """
        self._submit('lsf', rundir, n_hours_limit, memory_per_task,
                     resuming, bsub_cmds, postprocess)

    def _submit(self, scheduler, rundir, n_hours_limit=48,
                memory_per_task='32G', resuming=False, cmds=(),
                postprocess=True):
        """
        Implement `.submit_lsf` and `.submit_slurm`.

        Parameters
        ----------
        scheduler: {'slurm', 'lsf'}
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

        cmds += (f'--mem-per-cpu={memory_per_task}',)
        args = str(rundir.resolve())

        if not postprocess:
            args += ' --no-postprocessing'

        if scheduler == 'slurm':
            utils.submit_slurm(job_name, n_hours_limit, stdout_path,
                               stderr_path, args, cmds, batch_path)
        elif scheduler == 'lsf':
            utils.submit_lsf(job_name, n_hours_limit, stdout_path,
                             stderr_path, args, cmds, batch_path)
        else:
            raise ValueError('`scheduler` must be "slurm" or "lsf".')

    def submit_condor(self,
                      rundir,
                      request_cpus=1,
                      request_memory='8G',
                      request_disk='1G',
                      resuming=False,
                      postprocess=True,
                      **submit_kwargs):
        """
        Submit a parameter estimation run using the HTCondor scheduler.

        This method generates 'submit.sub' and 'executable.sh' files,
        the user should provide any instructions for the submit file as
        `**submit_kwargs`.

        Parameters
        ----------
        rundir: str, os.PathLike
            Run directory, e.g. from `self.get_rundir`

        request_cpus, request_memory, request_disk: int or str
            Specifications in the HTCondor submit file.

        resuming: bool
            Whether to attempt resuming a previous run if rundir already
            exists.

        postprocess: bool
            Whether to perform convergence tests to the run after
            sampling. See ``postprocessing.postprocess_rundir``.

        **submit_kwargs
            Further options to include in the HTCondor submit file. Do
            not pass `executable`, `output`, `error`, `log`, `queue`,
            which will be dealt with automatically.
        """
        rundir = pathlib.Path(rundir).resolve()

        submit_path = rundir/'submit.sub'

        submit_kwargs = {
            'executable': rundir/'executable.sh',
            'output': rundir/'output.out',
            'error': rundir/'errors.err',
            'log': rundir/'sampling.log',
            'request_cpus': request_cpus,
            'request_memory': request_memory,
            'request_disk': request_disk,
            } | submit_kwargs

        self.to_json(rundir, overwrite=resuming)

        args = rundir.as_posix()

        if not postprocess:
            args += ' --no-postprocessing'

        utils.submit_condor(submit_path, overwrite=resuming, args=args,
                            **submit_kwargs)

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
            samples = self.load_samples()
            samples.to_feather(rundir/SAMPLES_FILENAME)

        profiler.dump_stats(rundir/self.PROFILING_FILENAME)

        for path in rundir.iterdir():
            path.chmod(self.file_permissions)

    def run_kwargs_options(self):
        """
        Return list of possible parameters to configure sampler, along
        with their default values.

        Refer to each sampler's documentation for details.
        Sampler parameters not listed here are handled automatically by
        ``cogwheel``.

        Return
        ------
        list of inspect.Parameter
        """
        with utils.temporarily_change_attributes(self, run_kwargs={}):
            automatic_kw = self._get_run_kwargs() | self._get_sampler_kwargs()

        run_method = getattr(self._SAMPLER_CLS, self._RUN_METHOD_NAME)
        all_kwargs = (inspect.signature(run_method).parameters
                      | inspect.signature(self._SAMPLER_CLS).parameters)

        # Override defaults with any `run_kwargs` currently set:
        for key, value in self.run_kwargs.items():
            all_kwargs[key] = all_kwargs[key].replace(default=value)

        return [value for key, value in all_kwargs.items()
                if key not in automatic_kw]

    @abc.abstractmethod
    def _get_points_weights_blobs(self):
        """
        Collect samples, return arrays of posterior points, weights (if
        applicable, else None) and blobs (struct).
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

    def _lnfoldedprob_and_blob(self, folded_par_vals, as_dict=False):
        """
        Return log of the folded probability and a blob with extra
        information. The blob is provided by the likelihood class
        (self.posterior.likelihood.get_blob), and additionally this
        method chooses a random unfolding based on the probabilities and
        appends the unfolded parameter values to the blob.
        """
        lnprobs, par_dics, metadatas = self._get_lnprobs_pardics_metadatas(
            *folded_par_vals)

        ln_folded_prob = scipy.special.logsumexp(lnprobs)

        if np.isneginf(ln_folded_prob):
            i_unfold = 0
        else:
            probabilities = np.exp(lnprobs - ln_folded_prob)
            i_unfold = self._rng.choice(len(probabilities), p=probabilities)

        blob = (par_dics[i_unfold]
                | self.posterior.likelihood.get_blob(metadatas[i_unfold]))

        if as_dict:
            return ln_folded_prob, blob

        return ln_folded_prob, *blob.values()

    def _get_blobs_dtype(self):
        """Return list of 2-tuples with name and type of blob items."""
        folded_par_vals = (
            self.posterior.prior.cubemin
            + self._rng.uniform(0, self.posterior.prior.folded_cubesize))
        _, blob = self._lnfoldedprob_and_blob(folded_par_vals, as_dict=True)
        return [(key, type(val)) for key, val in blob.items()]

    def _get_sampler_kwargs(self):
        sampler_keys = (set(inspect.signature(self._SAMPLER_CLS).parameters)
                        & self.run_kwargs.keys())
        return {par: self.run_kwargs[par] for par in sampler_keys}

    def _get_run_kwargs(self):
        run_method = getattr(self._SAMPLER_CLS, self._RUN_METHOD_NAME)
        run_keys = (set(inspect.signature(run_method).parameters)
                    & self.run_kwargs.keys())
        return {par: self.run_kwargs[par] for par in run_keys}

    def _run(self):
        sampler_kwargs = self._get_sampler_kwargs()
        run_kwargs = self._get_run_kwargs()

        self.sampler = self._SAMPLER_CLS(**sampler_kwargs)
        run = getattr(self.sampler, self._RUN_METHOD_NAME)
        run(**run_kwargs)

    def load_samples(self):
        """Return a pandas.DataFrame with the posterior samples."""
        points, weights, blobs = self._get_points_weights_blobs()

        samples = pd.DataFrame(points, columns=self.sampled_params)

        if weights is not None:
            samples[utils.WEIGHTS_NAME] = weights

        utils.update_dataframe(samples, pd.DataFrame(blobs))

        return samples.dropna(ignore_index=True)

    @property
    def sampled_params(self):
        """
        Like ``.posterior.prior.sampled_params`` but the folded
        parameters have 'folded_' prepended.

        Return
        ------
        list of str
        """
        sampled_params = list(self.posterior.prior.sampled_params)

        for par in self.posterior.prior.folded_params:
            sampled_params[sampled_params.index(par)] = f'folded_{par}'
        return sampled_params

    def get_init_dict(self):
        """Keyword arguments to instantiate the class."""
        init_dict = super().get_init_dict()
        # Remove 'sample_prior' from the keys.
        assert not init_dict.pop('sample_prior', False)
        return init_dict


class PyMultiNest(Sampler):
    """Sample a posterior or prior using PyMultiNest."""
    DEFAULT_RUN_KWARGS = {'n_iter_before_update': 1000,
                          'n_live_points': 2048,
                          'evidence_tolerance': 1/4}

    class _SAMPLER_CLS:
        # Dummy sampler class for pymultinest (which doesn't have one).
        # A ``.run()`` method is implemented by ``PyMultiNest.__init__``
        # because we haven't imported ``pymultinest`` yet.
        run = None

    @wraps(Sampler.__init__)
    def __init__(self, *args, **kwargs):
        # Import optional dependency `pymultinest` on the fly
        try:
            import pymultinest
        except ImportError as err:
            raise ImportError('Missing optional dependency `pymultinest`, '
                              'install with `conda install pymultinest`.'
                              ) from err

        self._SAMPLER_CLS.run = staticmethod(pymultinest.run)

        super().__init__(*args, **kwargs)

    def _get_run_kwargs(self):
        run_kwargs = super()._get_run_kwargs()

        run_kwargs['wrapped_params'] = [
            par in self.posterior.prior.periodic_params
            for par in self.posterior.prior.sampled_params]

        run_kwargs['LogLikelihood'] = self._lnprob_pymultinest
        run_kwargs['Prior'] = self._cubetransform
        run_kwargs['n_dims'] = self._ndim
        run_kwargs['n_params'] = self._ndim + len(self._blobs_dtype)

        return run_kwargs

    def _get_points_weights_blobs(self):
        """
        Collect samples, return arrays of posterior points, weights (if
        applicable, else None) and blobs (struct).
        """
        fname = os.path.join(self.run_kwargs['outputfiles_basename'],
                             'post_equal_weights.dat')
        if not os.path.exists(fname):
            fname = fname[:100]  # Weird PyMultinest filename length limit

        points = np.loadtxt(fname, usecols=range(self._ndim))

        blob_cols = range(self._ndim, self._ndim + len(self._blobs_dtype))
        blobs = np.loadtxt(fname, usecols=blob_cols, dtype=self._blobs_dtype)

        return points, None, blobs

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

    def _lnprob_pymultinest(self, folded_par_vals, *_):
        """
        Update the extra entries `folded_par_vals[n_dim : n_params+1]`
        with the blob. Return the logarithm of the folded posterior.
        """
        lnfoldedprob, *blob = self._lnfoldedprob_and_blob(
            [folded_par_vals[i] for i in range(self._ndim)])

        for i, blob_value in enumerate(blob):
            folded_par_vals[self._ndim + i] = blob_value

        return lnfoldedprob

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
    _SAMPLER_CLS = dynesty.DynamicNestedSampler
    _RUN_METHOD_NAME = 'run_nested'
    DEFAULT_RUN_KWARGS = {'sample': 'rwalk'}

    def _get_sampler_kwargs(self):
        sampler_kwargs = super()._get_sampler_kwargs()

        sampler_kwargs['periodic'] = [
            self.posterior.prior.sampled_params.index(par)
            for par in self.posterior.prior.periodic_params] or None

        sampler_kwargs['reflective'] = [
            self.posterior.prior.sampled_params.index(par)
            for par in self.posterior.prior.reflective_params] or None

        sampler_kwargs['loglikelihood'] = self._lnprob_dynesty
        sampler_kwargs['prior_transform'] = self._cubetransform
        sampler_kwargs['ndim'] = self._ndim
        sampler_kwargs['blob'] = True

        return sampler_kwargs

    def _lnprob_dynesty(self, folded_par_vals):
        """Return the logarithm of the folded probability density."""
        lnfoldedprob, *blob = self._lnfoldedprob_and_blob(folded_par_vals)
        blob_rec = np.rec.array(blob, dtype=self._blobs_dtype)
        return lnfoldedprob, blob_rec

    def _cubetransform(self, cube):
        return (self.posterior.prior.cubemin
                + cube * self.posterior.prior.folded_cubesize)

    def _get_points_weights_blobs(self):
        """Collect dynesty samples."""
        results = self.sampler.results
        return results.samples, results.importance_weights(), results.blob


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
            ('SaveProgressCallback', {'filename': '{rundir}/chain.h5'})]
        }

    _SAMPLER_CLS = zeus.EnsembleSampler
    _RUN_METHOD_NAME = 'run_mcmc'

    @wraps(Sampler.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampler = None
        self._rundir = None
        self._folded_cubemax = (self.posterior.prior.cubemin
                                + self.posterior.prior.folded_cubesize)

        # Zeus won't stop at boundaries so we must intercept out-of-bound
        # proposals. Implement a blob of `nan`s to return in this case.
        # These moves are always rejected so this blob should never appear.
        self._nan_blob = []
        for _, dtype in self._blobs_dtype:
            try:
                self._nan_blob.append(dtype(np.nan))
            except (ValueError, TypeError):
                try:
                    self._nan_blob.append(dtype(None))
                except (ValueError, TypeError):
                    self._nan_blob.append(dtype(-1))

    @wraps(Sampler.run)
    def run(self, rundir):
        self._rundir = rundir
        super().run(rundir)

    def _run(self):
        sampler_kwargs = self._get_sampler_kwargs()
        run_kwargs = self._get_run_kwargs()

        processes = self.run_kwargs.get('processes', 1)
        if processes == 1:  # Serial
            self.sampler = zeus.EnsembleSampler(**sampler_kwargs)
            self.sampler.run_mcmc(**run_kwargs)
        else:  # Parallel
            if processes > sampler_kwargs['nwalkers'] / 2:
                warnings.warn('Some processes will likely be idle, consider '
                              'increasing `nwalkers` to be `>= 2*processes` '
                              '[arxiv.org/abs/2002.06212].')
            with multiprocessing.Pool(processes) as pool:
                self.sampler = zeus.EnsembleSampler(pool=pool,
                                                    **sampler_kwargs)
                self.sampler.run_mcmc(**run_kwargs)

    def _get_sampler_kwargs(self):
        sampler_kwargs = super()._get_sampler_kwargs()

        global _lnprob_zeus  # Hack for multiprocessing to work
        def  _lnprob_zeus(*args, **kwargs):
            return self._lnprob_zeus(*args, **kwargs)

        sampler_kwargs.update(
            ndim=self._ndim,
            logprob_fn=_lnprob_zeus,
            blobs_dtype=self._blobs_dtype)
        return sampler_kwargs

    def _get_run_kwargs(self):
        run_kwargs = super()._get_run_kwargs()

        if 'start' not in run_kwargs:
            run_kwargs['start'] = self._get_start()  # Initialize walkers

        # Instantiate callbacks from the JSON-serializable input:
        callbacks = self.run_kwargs.get('callbacks', [])
        save_progress_kwargs = dict(callbacks).get('SaveProgressCallback', {})
        if filename := save_progress_kwargs.get('filename'):
            # Dynamically interpolate ``rundir`` to filename:
            save_progress_kwargs['filename'] = filename.format(
                rundir=self._rundir)
        run_kwargs['callbacks'] = [getattr(zeus.callbacks, name)(**kwargs)
                                   for name, kwargs in callbacks]
        return run_kwargs

    def _get_points_weights_blobs(self):
        kwargs = {par: self.run_kwargs[par]
                  for par in self.run_kwargs.keys() & {'discard', 'thin'}}
        points = self.sampler.get_chain(flat=True, **kwargs)
        blobs = self.sampler.get_blobs(flat=True, **kwargs)
        return points, None, blobs

    def _lnprob_zeus(self, par_vals):
        """
        Return the logarithm of the folded probability density and blob.
        """
        # Intercept out-of-bound proposals:
        if (np.any(par_vals < self.posterior.prior.cubemin)
                or np.any(par_vals > self._folded_cubemax)):
            return -np.inf, *self._nan_blob

        return self._lnfoldedprob_and_blob(par_vals)

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
        par_dic_0 = self.posterior.likelihood.par_dic_0
        sampled_par_dic = self.posterior.prior.inverse_transform(**par_dic_0)

        loc = self.posterior.prior.fold(**sampled_par_dic)
        scale = rscale * self.posterior.prior.folded_cubesize
        a = (self.posterior.prior.cubemin - loc) / scale
        b = (self._folded_cubemax - loc) / scale

        mask = np.full(self.run_kwargs['nwalkers'], True)
        start = np.empty((self.run_kwargs['nwalkers'], self._ndim))
        log_prob = np.empty(self.run_kwargs['nwalkers'])

        while any(mask):  # Draw positions for these walkers
            start[mask] = scipy.stats.truncnorm.rvs(
                loc=loc,
                scale=scale,
                a=a,
                b=b,
                size=(np.count_nonzero(mask), self._ndim))
            log_prob[mask] = [self._lnprob_zeus(x)[0] for x in start[mask]]
            mask = log_prob < log_prob.max() - max_lnprob_drop

        return start


class Nautilus(Sampler):
    """Sample a posterior or prior using Nautilus."""
    _SAMPLER_CLS = nautilus.Sampler
    DEFAULT_RUN_KWARGS = {'verbose': True}

    def _get_sampler_kwargs(self):
        sampler_kwargs = super()._get_sampler_kwargs()

        prior = nautilus.Prior()
        ranges = zip(self.posterior.prior.cubemin,
                     self.posterior.prior.cubemin
                     + self.posterior.prior.folded_cubesize)
        for par, rng in zip(self.posterior.prior.sampled_params, ranges):
            prior.add_parameter(par, rng)

        sampler_kwargs['prior'] = prior
        sampler_kwargs['likelihood'] = self._lnfoldedprob_and_blob
        sampler_kwargs['blobs_dtype'] = self._blobs_dtype
        sampler_kwargs['pass_dict'] = False
        return sampler_kwargs

    def _get_points_weights_blobs(self):
        """Collect Nautilus samples."""
        points, log_w, _, blobs = self.sampler.posterior(return_blobs=True)
        return points, np.exp(log_w), blobs

    @wraps(utils.JSONMixin.to_json)
    def to_json(self, dirname, *args, **kwargs):
        """
        Update run_kwargs['filepath'] before saving.
        Parameters are as in `utils.JSONMixin.to_json()`
        """
        self.run_kwargs['filepath'] = os.path.join(dirname, 'checkpoint.hdf5')
        super().to_json(dirname, *args, **kwargs)

    @wraps(Sampler.load_evidence)
    def load_evidence(self):
        if self.sampler is None:
            log_z = self.read_log_z(self.run_kwargs['filepath'])
        else:
            try:
                log_z = self.sampler.log_z
            except AttributeError:  # Old nautilus version
                log_z = self.sampler.evidence()

        return {'log_ev': log_z}

    @staticmethod
    def read_log_z(filepath):
        """
        Read log of Bayesian evidence from the nautilus checkpoint file.

        Parameters
        ----------
        filepath: os.PathLike
            Typically, should be ``rundir/'checkpoint.hdf5'``.
        """
        dummy_kwargs = {'prior': lambda _: 0,
                        'likelihood': lambda _: 0,
                        'n_dim': 2}
        nautilus_sampler = nautilus.Sampler(**dummy_kwargs, filepath=filepath)
        try:
            return nautilus_sampler.log_z
        except AttributeError:  # Old nautilus version
            return nautilus_sampler.evidence()


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
