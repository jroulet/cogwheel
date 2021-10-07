"""Utility functions."""

import importlib
import inspect
import json
import os
import pathlib
import numpy as np
from scipy.optimize import _differentialevolution


DIR_PERMISSIONS = 0o755
FILE_PERMISSIONS = 0o644


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


def merge_dictionaries_safely(dics):
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


# ----------------------------------------------------------------------
# Directory I/O:

def get_eventdir(parentdir, prior_class, eventname):
    """
    Return `pathlib.Path` object for a directory of the form
    {parentdir}/{prior_class}/{eventname}/
    This directory is intended to contain a `Posterior` instance,
    and multiple rundir directories with parameter estimation
    output for different sampler settings.
    I.e. the file structure is as follows:

        <parentdir>
        └── <prior_class>
            └── <eventdir>
                ├── Posterior.json
                └── <rundir>
                    ├── Sampler.json
                    └── <sampler_output_files>
    """
    return pathlib.Path(parentdir)/prior_class/eventname


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
        if (njsons := len(jsons)) != 1:
            raise ValueError(f'{json_path} contains {njsons} json files.')
        json_path = jsons[0]

    with open(json_path) as json_file:
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

        with open(filepath, 'w') as outfile:
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

    def default(self, o):
        if o.__class__.__name__ == 'EventData':
            filename = os.path.join(self.dirname, f'{o.eventname}.npz')
            o.to_npz(filename=filename, overwrite=self.overwrite,
                     permissions=self.file_permissions)
            return {'__cogwheel_class__': o.__class__.__name__,
                    '__module__': o.__class__.__module__,
                    'filename': os.path.basename(filename)}

        if o.__class__.__name__ in class_registry:
            return {'__cogwheel_class__': o.__class__.__name__,
                    '__module__': o.__class__.__module__,
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
