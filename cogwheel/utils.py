"""Utility functions."""

import functools
import importlib
import inspect
import json
import os
import pathlib
import re
import sys
import tempfile
import textwrap
import numpy as np
from scipy.optimize import _differentialevolution
from scipy.special import logsumexp
from numba import njit, vectorize


DIR_PERMISSIONS = 0o755
FILE_PERMISSIONS = 0o644

WEIGHTS_NAME = 'weights'


class ClassProperty:
    """
    Can be used like `@property` but for class attributes instead of
    instance attributes.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, inst, cls):
        return self.func(cls)


def differential_evolution_with_guesses(
        func, bounds, guesses, **kwargs):
    """
    Augmented differential_evolution solver that incorporates
    initial guesses passed by the user.

    Parameters
    ----------
    func, bounds: See `scipy.optimize.differential_evolution()` docs.
    guesses: nguesses x nparameters array with initial guesses.
             They will be appended to the initial population of
             differential evolution. Can be a 1d array for one guess.
    **kwargs: Passed to `scipy.optimize.differential_evolution()`.
    """
    with _DifferentialEvolutionSolverWithGuesses(func, bounds, guesses,
                                                 **kwargs) as solver:
        return solver.solve()


class _DifferentialEvolutionSolverWithGuesses(
        _differentialevolution.DifferentialEvolutionSolver):
    """
    Class that implements `differential_evolution_with_guesses()`.
    """
    def __init__(self, func, bounds, guesses, **kwargs):
        super().__init__(func, bounds, **kwargs)
        initial_pop = self._scale_parameters(self.population)
        population = np.vstack((initial_pop, guesses))
        self.init_population_array(population)


cached_functions_registry = []


def lru_cache(*args, **kwargs):
    """
    Decorator like `functools.lru_cache` that also registers the
    decorated function in ``cached_functions_registry`` so all caches
    can easily be cleared with ``clear_caches()``.
    """
    def decorator(function):
        function = functools.lru_cache(*args, **kwargs)(function)
        cached_functions_registry.append(function)
        return function
    return decorator


def clear_caches():
    """
    Clear caches of functions decorated with ``lru_cache``.
    """
    for function in cached_functions_registry:
        function.cache_clear()


def mod(value, start=0, period=2*np.pi):
    """
    Modulus operation, generalized so that the domain of the output can
    be specified.
    """
    return (value - start) % period + start


def weighted_avg_and_std(values, weights=None):
    """Return average and standard deviation of values with weights."""
    avg = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - avg) ** 2, weights=weights))
    return avg, std


def n_effective(weights):
    """Return effective sample size."""
    if weights.size == 0:
        return 0.
    return np.sum(weights)**2 / np.sum(weights**2)


def resample_equal(samples, weights_col=WEIGHTS_NAME, num=None):
    """
    Draw `num` samples from a DataFrame of weighted samples, so that the
    resulting samples have equal weights.
    Note: in general this does not produce independent samples, they may
    be repeated.

    Parameters
    ----------
    samples: pandas.DataFrame
        Rows correspond to samples from a distribution.

    weights_col: str
        Name of a column in `samples` to interpret as weights.

    num: int
        Length of the returned DataFrame, defaults to ``len(samples)``.

    Return
    ------
    equal_samples: pandas.DataFrame
        Contains `num` rows with equal-weight samples. The columns match
        those of `samples` except that `weights_col` is deleted.
    """
    if num is None:
        num = len(samples)
    weights = samples[weights_col]
    weights /= weights.sum()
    inds = np.random.choice(len(samples), num, p=weights)
    samples_equal = samples.iloc[inds].reset_index()
    del samples_equal[weights_col]
    return samples_equal


def exp_normalize(lnprob, axis=-1):
    """
    Return normalized probabilities from unnormalized log
    probabilities, safe to overflow.

    Parameters
    ----------
    lnprob: float array
        Natural log of the unnormalized probability.

    axis: int
        Axis of `lnprob` along which probabilities sum to 1.
    """
    return np.exp(lnprob - logsumexp(lnprob, axis=axis, keepdims=True))


@njit
def rand_choice_nb(arr, cprob, nvals):
    """
    Sample randomly from a list of probabilities

    Parameters
    ----------
    arr: np.ndarray
        A nD numpy array of values to sample from

    cprob: np.arrray
        A 1D numpy array of cumulative probabilities for the given samples

    nvals: int
        Number of samples desired

    Return
    ------
    nvals random samples from the given array with the given probabilities
    """
    rsamps = np.random.random(size=nvals)
    return arr[np.searchsorted(cprob, rsamps, side="right")]


@vectorize(nopython=True)
def abs_sq(x):
    """x.real^2 + x.imag^2"""
    return x.real**2 + x.imag**2


def real_matmul(a, b):
    """
    Return real part of complex matrix multiplication.

    Same as `(a @ b).real` but ~2 times faster.
    Note, `b` is not conjugated by this function.
    """
    a_real = a.real.copy()
    a_imag = a.imag.copy()
    b_real = b.real.copy()
    b_imag = b.imag.copy()
    return a_real @ b_real - a_imag @ b_imag


def merge_dictionaries_safely(*dics):
    """
    Merge multiple dictionaries into one.
    Accept repeated keys if values are consistent, otherwise raise
    `ValueError`.
    """
    merged = {}
    for dic in dics:
        for key in merged.keys() & dic.keys():
            if merged[key] != dic[key]:
                raise ValueError(f'Found incompatible values for {key}')
        merged |= dic
    return merged


def rm_suffix(string, suffix='.json', new_suffix=None):
    """
    Removes suffix from string if present, and appends a new suffix if
    requested.

    Parameters
    ----------
    string: Input string to modify.
    suffix: Suffix to remove if present.
    new_suffix: Suffix to add.
    """
    if string.endswith(suffix):
        outstr = string[:-len(suffix)]
    else:
        outstr = string
    if new_suffix is not None:
        outstr += new_suffix
    return outstr


def update_dataframe(df1, df2):
    """
    Modify `df1` in-place by adding the columns from `df2`, where `df1`
    and `df2` are pandas `DataFrame` objects.
    Caution: if (some of) the columns of `df1` are also in `df2` they
    get silently overwritten without checking for consistency.
    """
    for col, values in df2.items():
        df1[col] = values


def replace(sequence, *args):
    """
    Return a list like `sequence` with the first occurrence of `old0`
    replaced by `new0`, the first occurrence of `old1` replaced by
    `new1`, and so on, where ``old0, new0, old1, new1, ... = args``.
    Accepts an even number of arguments.
    """
    if len(args) % 2:
        raise ValueError('Pass an even number of args: '
                         'old1, new1, old2, new2, ...')

    out = list(sequence)
    for old, new in zip(args[::2], args[1::2]):
        out[out.index(old)] = new
    return out


def handle_scalars(function):
    """
    Decorator to change the behavior of functions that always return
    numpy arrays even if the input is scalar (e.g.
    ``scipy.interpolate.InterpolatedUnivariateSpline``).
    The decorated function will return a scalar for scalar input and
    array for array input like usual numpy ufuncs.
    """
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        return function(*args, **kwargs)[()]
    return new_function



def submit_slurm(job_name, n_hours_limit, stdout_path, stderr_path,
                 args='', sbatch_cmds=(), batch_path=None,
                 multithreading=False):
    """
    Generic function to submit a job using slurm.
    This function is intended to be called from other modules rather
    than used interactively. The job will run the calling module as
    script.

    Parameters
    ----------
    job_name: str
        Name of slurm job.

    n_hours_limit: int
        Number of hours to allocate for the job.

    stdout_path: str, os.PathLike
        File name, where to direct stdout.

    stderr_path: str, os.PathLike
        File name, where to direct stderr.

    args: str
        Command line arguments for the calling module's ``main()`` to
        parse.

    sbatch_cmds: sequence of str
        SBATCH commands, e.g. ``('--mem-per-cpu=8G',)``.

    batch_path: str, os.PathLike, optional
        File name where to save the batch script. If not provided, a
        temporary file will be used.

    multithreading: bool
        Whether to enable automatic OMP multithreading. Defaults to
        ``False`` because multithreading is found to be slower
        despite using more resources.
    """
    cogwheel_dir = pathlib.Path(__file__).parents[1].resolve()
    module = inspect.getmodule(inspect.stack()[1].frame).__name__

    sbatch_lines = """
        """.join(f'#SBATCH {cmd}' for cmd in sbatch_cmds)

    omp_line = ''
    if not multithreading:
        omp_line = 'export OMP_NUM_THREADS=1'

    batch_text = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={stdout_path}
        #SBATCH --error={stderr_path}
        #SBATCH --open-mode=append
        #SBATCH --time={n_hours_limit:02}:00:00
        {sbatch_lines}

        eval "$(conda shell.bash hook)"
        conda activate {os.environ['CONDA_DEFAULT_ENV']}

        {omp_line}

        cd {cogwheel_dir}
        srun {sys.executable} -m {module} {args}
        """)

    if batch_path:
        getfile = lambda: open(batch_path, 'w+', encoding='utf-8')
    else:
        getfile = lambda: tempfile.NamedTemporaryFile('w+', encoding='utf-8')

    with getfile() as batchfile:
        batchfile.write(batch_text)
        batchfile.seek(0)
        os.chmod(batchfile.name, 0o777)
        os.system(f'sbatch {os.path.abspath(batchfile.name)}')

    print(f'Submitted job {job_name!r}.')


def submit_lsf(job_name, n_hours_limit, stdout_path, stderr_path,
               args='', bsub_cmds=(), batch_path=None,
               multithreading=False):
    """
    Generic function to submit a job using IBM Spectrum LSF.
    This function is intended to be called from other modules rather
    than used interactively. The job will run the calling module as
    script.

    Parameters
    ----------
    job_name: str
        Name of LSF job.

    n_hours_limit: int
        Number of hours to allocate for the job.

    stdout_path: str, os.PathLike
        File name, where to direct stdout.

    stderr_path: str, os.PathLike
        File name, where to direct stderr.

    args: str
        Command line arguments for the calling module's ``main()`` to
        parse.

    bsub_cmds: sequence of str
        BSUB commands, e.g. ``('-M 8GB',)``

    batch_path: str, os.PathLike, optional
        File name where to save the batch script. If not provided, a
        temporary file will be used.

    multithreading: bool
        Whether to enable automatic OMP multithreading. Defaults to
        ``False`` because multithreading is found to be slower
        despite using more resources.
    """
    cogwheel_dir = pathlib.Path(__file__).parents[1].resolve()
    module = inspect.getmodule(inspect.stack()[1].frame).__name__

    bsub_lines = """
        """.join(f'#BSUB {cmd}' for cmd in bsub_cmds)

    omp_line = ''
    if not multithreading:
        omp_line = 'export OMP_NUM_THREADS=1'

    batch_text = textwrap.dedent(
        f"""\
        #!/bin/bash
        #BSUB -J {job_name}
        #BSUB -o {stdout_path}
        #BSUB -e {stderr_path}
        #BSUB -W {n_hours_limit:02}:00
        {bsub_lines}

        eval "$(conda shell.bash hook)"
        conda activate {os.environ['CONDA_DEFAULT_ENV']}

        {omp_line}

        cd {cogwheel_dir}
        {sys.executable} -m {module} {args}
        """)

    if batch_path:
        getfile = lambda: open(batch_path, 'w+', encoding='utf-8')
    else:
        getfile = lambda: tempfile.NamedTemporaryFile('w+', encoding='utf-8')

    with getfile() as batchfile:
        batchfile.write(batch_text)
        batchfile.seek(0)
        os.chmod(batchfile.name, 0o777)
        os.system(f'bsub < {os.path.abspath(batchfile.name)}')

    print(f'Submitted job {job_name!r}.')


# ----------------------------------------------------------------------
# Directory I/O:

RUNDIR_PREFIX = 'run_'


def get_eventdir(parentdir, prior_name, eventname):
    """
    Return `pathlib.Path` object for a directory of the form
    {parentdir}/{prior_name}/{eventname}/
    This directory is intended to contain a `Posterior` instance,
    and multiple rundir directories with parameter estimation
    output for different sampler settings.
    I.e. the file structure is as follows:

        <parentdir>
        └── <priordir>
            └── <eventdir>
                ├── Posterior.json
                └── <rundir>
                    ├── Sampler.json
                    └── <sampler_output_files>
    """
    return get_priordir(parentdir, prior_name)/eventname


def get_priordir(parentdir, prior_name):
    """
    Return `pathlib.Path` object for a directory of the form
    {parentdir}/{prior_name}
    This directory is intended to contain multiple eventdir
    directories, one for each event.
    I.e. the file structure is as follows:

        <parentdir>
        └── <priordir>
            └── <eventdir>
                ├── Posterior.json
                └── <rundir>
                    ├── Sampler.json
                    └── <sampler_output_files>
    """
    return pathlib.Path(parentdir)/prior_name


def mkdirs(dirname, dir_permissions=DIR_PERMISSIONS):
    """
    Create directory and its parents if needed, ensuring the
    whole tree has the same permissions. Existing directories
    are left unchanged.

    Parameters
    ----------
    dirname: path of directory to make.
    dir_permissions: octal with permissions.
    """
    dirname = pathlib.Path(dirname)
    for path in list(dirname.parents)[::-1] + [dirname]:
        path.mkdir(mode=dir_permissions, exist_ok=True)


def rundir_number(rundir) -> int:
    """Return first strech of numbers in `rundir` as `int`."""
    return int(re.search(r'\d+', os.path.basename(rundir)).group())


def sorted_rundirs(rundirs):
    """
    Return `rundirs` sorted by number (i.e. 'run_2' before 'run_10').
    """
    return sorted(rundirs, key=rundir_number)


# ----------------------------------------------------------------------
# JSON I/O:

class_registry = {}


def read_json(json_path):
    """
    Return a class instance that was saved to json.
    """
    # Accept a directory that contains a single json file
    json_path = pathlib.Path(json_path)
    if json_path.is_dir():
        jsons = list(json_path.glob('*.json'))
        if not jsons:
            raise ValueError(f'{json_path} contains no json files.')
        if len(jsons) > 1:
            raise ValueError(
                f'{json_path} contains multiple json files {jsons}')

        json_path = jsons[0]

    with open(json_path, encoding='utf-8') as json_file:
        return json.load(json_file, cls=CogwheelDecoder,
                         dirname=json_path.parent)


class JSONMixin:
    """
    Provide JSON output to subclasses.
    Register subclasses in `class_registry`.

    Define a method `get_init_dict` which works for classes that store
    their init parameters as attributes with the same names. If this is
    not the case, the subclass should override `get_init_dict`.

    Define a method `reinstantiate` that allows to safely modify
    attributes defined at init.
    """
    def to_json(self, dirname, basename=None, *,
                dir_permissions=DIR_PERMISSIONS,
                file_permissions=FILE_PERMISSIONS, overwrite=False):
        """
        Write class instance to json file.
        It can then be loaded with `read_json`.
        """
        basename = basename or f'{self.__class__.__name__}.json'
        filepath = pathlib.Path(dirname)/basename

        if not overwrite and filepath.exists():
            raise FileExistsError(
                f'{filepath.name} exists. Pass `overwrite=True` to overwrite.')

        mkdirs(dirname, dir_permissions)

        with open(filepath, 'w', encoding='utf-8') as outfile:
            json.dump(self, outfile, cls=CogwheelEncoder, dirname=dirname,
                      file_permissions=file_permissions, overwrite=overwrite,
                      indent=2)
        filepath.chmod(file_permissions)

    def __init_subclass__(cls):
        """Register subclasses."""
        super().__init_subclass__()
        class_registry[cls.__name__] = cls

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to `__init__`.
        Only works if the class stores its init parameters as attributes
        with the same names. Otherwise, the subclass should override
        this method.
        """
        keys = list(inspect.signature(self.__init__).parameters)
        if any(not hasattr(self, key) for key in keys):
            raise KeyError(
                f'`{self.__class__.__name__}` must override `get_init_dict` '
                '(or store its init parameters with the same names).')
        return {key: getattr(self, key) for key in keys}

    def reinstantiate(self, **new_init_kwargs):
        """
        Return an new instance of the current instance's class, with an
        option to update `init_kwargs`. Values not passed will be taken
        from the current instance.
        """
        init_kwargs = self.get_init_dict()

        if not new_init_kwargs.keys() <= init_kwargs.keys():
            raise ValueError(
                f'`new_init_kwargs` must be from ({", ".join(init_kwargs)})')

        return self.__class__(**init_kwargs | new_init_kwargs)


class NumpyEncoder(json.JSONEncoder):
    """
    Encoder for numpy data types.
    """
    def default(self, o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(o)

        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)

        if isinstance(o, (np.complex_, np.complex64, np.complex128, complex)):
            return {'real': o.real, 'imag': o.imag}

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, np.bool_):
            return bool(o)

        if isinstance(o, np.void):
            return None

        return super().default(o)


class CogwheelEncoder(NumpyEncoder):
    """
    Encoder for classes in the `cogwheel` package that subclass
    `JSONMixin`.
    """
    def __init__(self, dirname=None, file_permissions=FILE_PERMISSIONS,
                 overwrite=False, **kwargs):
        super().__init__(**kwargs)

        self.dirname = dirname
        self.file_permissions = file_permissions
        self.overwrite = overwrite

    @staticmethod
    def _get_module_name(obj):
        module = obj.__class__.__module__
        if module == '__main__' and (spec := inspect.getmodule(obj).__spec__):
            module = spec.name
        return module

    def default(self, o):
        if o.__class__.__name__ == 'EventData':
            filename = os.path.join(self.dirname, f'{o.eventname}.npz')
            o.to_npz(filename=filename, overwrite=self.overwrite,
                     permissions=self.file_permissions)
            return {'__cogwheel_class__': o.__class__.__name__,
                    '__module__': self._get_module_name(o),
                    'filename': os.path.basename(filename)}

        if o.__class__.__name__ in class_registry:
            return {'__cogwheel_class__': o.__class__.__name__,
                    '__module__': self._get_module_name(o),
                    'init_kwargs': o.get_init_dict()}

        return super().default(o)


class CogwheelDecoder(json.JSONDecoder):
    """
    Decoder for classes in the `cogwheel` package that subclass
    `JSONMixin`.
    """
    def __init__(self, dirname, **kwargs):
        self.dirname = dirname
        super().__init__(object_hook=self._object_hook, **kwargs)

    def _object_hook(self, obj):
        if isinstance(obj, dict) and '__cogwheel_class__' in obj:
            importlib.import_module(obj['__module__'])
            cls = class_registry[obj['__cogwheel_class__']]
            if cls.__name__ == 'EventData':
                return cls.from_npz(filename=os.path.join(self.dirname,
                                                          obj['filename']))
            return cls(**obj['init_kwargs'])
        return obj
