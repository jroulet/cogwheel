"""Utility functions."""

import inspect
import json
import numpy as np
from scipy.optimize import _differentialevolution


class ClassProperty:
    """
    Can be used like `@property` but for class attributes instead of
    instance attributes.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, inst, cls):
        return self.func(cls)


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

        if isinstance(o, (np.complex_, np.complex64, np.complex128)):
            return {'real': o.real, 'imag': o.imag}

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, np.bool_):
            return bool(o)

        if isinstance(o, np.void):
            return None

        return super().default(o)


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
        ret = solver.solve()
    return ret


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


def get_init_dic(class_instance):
    """
    Return a dictionary that can be used to recreate a class instance.
    Requires that all init parameters are stored as instance attributes
    with the same name.
    """
    keys = list(inspect.signature(class_instance.__init__).parameters)
    return {key: getattr(class_instance, key) for key in keys}


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
        merged.update(dic)
    return merged
