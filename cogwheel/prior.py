"""
Implement the `Prior` and `CombinedPrior` classes.
These define Bayesian priors together with coordinate transformations.
There are two sets of coordinates: "sampled" parameters and "standard"
parameters. Standard parameters are physically interesting, sampled
parameters are chosen to minimize correlations or have convenient
priors.
It is possible to define multiple simple priors, each for a small subset
of the variables, and combine them with `CombinedPrior`.
If separate coordinate systems are not desired, a mix-in class
`IdentityTransformMixin` is provided to short-circuit these transforms.
Another mix-in `UniformPriorMixin` is provided to automatically define
uniform priors.
"""

from abc import ABC, abstractmethod
import inspect
import itertools
import pandas as pd
import numpy as np

from cogwheel import utils


class PriorError(Exception):
    """Base class for all exceptions in this module"""


class Prior(ABC, utils.JSONMixin):
    """"
    Abstract base class to define priors for Bayesian parameter
    estimation, together with coordinate transformations from "sampled"
    parameters to "standard" parameters.

    Schematically,
        lnprior(*sampled_par_vals, *conditioned_on_vals)
            = log P(sampled_par_vals | conditioned_on_vals)
    where P is the prior probability density in the space of sampled
    parameters;
        transform(*sampled_par_vals, *conditioned_on_vals)
            = standard_par_dic
    and
        inverse_transform(*standard_par_vals, *conditioned_on_vals)
            = sampled_par_dic.

    Subclassed by `CombinedPrior` and `FixedPrior`.

    Attributes
    ----------
    range_dic: dict
        Dictionary whose keys are sampled parameter names and whose
        values are pairs of floats defining their ranges.
        Needs to be defined by the subclass (either as a class attribute
        or instance attribute) before calling `Prior.__init__()`.

    sampled_params: list of str
        Names of sampled parameters (keys of `range_dic`).

    standard_params: list of str
        Names of standard parameters.

    conditioned_on: list of str
        Names of parameters on which this prior is conditioned on. To
        combine priors, conditioned-on parameters need to be among the
        standard parameters of another prior.

    periodic_params: list of str
        Names of sampled parameters that are periodic.

    reflective_params: list of str
        Names of sampled parameters that are reflective.

    folded_reflected_params: list of str
        Names of sampled parameters that are folded using reflection
        about the center.

    folded_shifted_params: list of str
        Names of sampled parameters that are folded using
        translation.

    folded_params: list of str
        ``folded_reflected_params + folded_shifted_params``.

    Methods
    -------
    lnprior:
        Take sampled and conditioned-on parameters and return a float
        with the natural logarithm of the prior probability density in
        the space of sampled parameters. Provided by the subclass.

    transform:
        Coordinate transformation, take sampled parameters and
        conditioned-on parameters and return a dict of standard
        parameters. Provided by the subclass.

    lnprior_and_transform:
        Take sampled parameters and return a tuple with the result of
        (lnprior, transform).

    inverse_transform:
        Inverse coordinate transformation, take standard parameters and
        conditioned-on parameters and return a dict of sampled
        parameters. Provided by the subclass.
    """

    conditioned_on = []
    periodic_params = []
    reflective_params = []
    folded_reflected_params = []
    folded_shifted_params = []

    def __init__(self, **kwargs):
        super().__init__()
        self._check_range_dic()

        self._folded_inds = [self.sampled_params.index(par)
                             for par in self.folded_params]

        self.cubemin = np.array([rng[0] for rng in self.range_dic.values()])
        cubemax = np.array([rng[1] for rng in self.range_dic.values()])
        self.cubesize = cubemax - self.cubemin
        self.folded_cubesize = self.cubesize.copy()
        self.folded_cubesize[self._folded_inds] /= 2
        self.signature = inspect.signature(self.transform)

        self.fold = None  # Set by ``self._setup_folding_transforms()``
        self.unfold = None  # Set by ``self._setup_folding_transforms()``
        self._setup_folding_transforms()

    @utils.ClassProperty
    def sampled_params(self):
        """List of sampled parameter names."""
        return list(self.range_dic)

    @utils.ClassProperty
    @abstractmethod
    def range_dic(self):
        """
        Dictionary whose keys are sampled parameter names and
        whose values are pairs of floats defining their ranges.
        Needs to be defined by the subclass.
        If the ranges are not known before class instantiation,
        define a class attribute as {'<par_name>': NotImplemented, ...}
        and populate the values at the subclass' `__init__()` before
        calling `Prior.__init__()`.
        """
        return {}

    @utils.ClassProperty
    @abstractmethod
    def standard_params(self):
        """
        List of standard parameter names.
        """
        return []

    @abstractmethod
    def lnprior(self, *par_vals, **par_dic):
        """
        Natural logarithm of the prior probability density.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a float.
        """

    @abstractmethod
    def transform(self, *par_vals, **par_dic):
        """
        Transform sampled parameter values to standard parameter values.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a dictionary with `self.standard_params` parameters.
        """

    @abstractmethod
    def inverse_transform(self, *par_vals, **par_dic):
        """
        Transform standard parameter values to sampled parameter values.
        Take `self.standard_params + self.conditioned_on` parameters and
        return a dictionary with `self.sampled_params` parameters.
        """

    def lnprior_and_transform(self, *par_vals, **par_dic):
        """
        Return a tuple with the results of `self.lnprior()` and
        `self.transform()`.
        The reason for this function is that for `CombinedPrior` it is
        already necessary to evaluate `self.transform()` in order to
        evaluate `self.lnprior()`. `CombinedPrior` overwrites this
        function so the user can get both `lnprior` and `transform`
        without evaluating `transform` twice.
        """
        return (self.lnprior(*par_vals, **par_dic),
                self.transform(*par_vals, **par_dic))

    @property
    def folded_params(self):
        """
        Names of folded parameters that are either reflected or shifted,
        in that order.
        """
        return self.folded_reflected_params + self.folded_shifted_params

    def _check_range_dic(self):
        """
        Ensure that range_dic values are stored as float arrays.
        Verify that ranges for all periodic, reflective and folded
        parameters were provided.
        """
        if missing := (set(self.periodic_params) - self.range_dic.keys()):
            raise PriorError('Periodic parameters are missing from '
                             f'`range_dic`: {", ".join(missing)}')

        if missing := (set(self.reflective_params) - self.range_dic.keys()):
            raise PriorError('Reflective parameters are missing from '
                             f'`range_dic`: {", ".join(missing)}')

        if missing := (set(self.folded_params) - self.range_dic.keys()):
            raise PriorError('Folded parameters are missing from '
                             f'`range_dic`: {", ".join(missing)}')

        for key, value in self.range_dic.items():
            if not hasattr(value, '__len__') or len(value) != 2:
                raise PriorError(f'`range_dic` {self.range_dic} must have '
                                 'ranges defined as pair of floats.')
            self.range_dic[key] = np.asarray(value, dtype=np.float_)

    @classmethod
    def get_fast_sampled_params(cls, fast_standard_params):
        """
        Return a list of parameter names that map to given "fast"
        standard parameters, useful for sampling fast-slow parameters.
        Updating fast sampling parameters is guaranteed to only
        change fast standard parameters.
        """
        if set(cls.standard_params) <= set(fast_standard_params):
            return cls.sampled_params
        return []

    def unfold_apply(self, func):
        """
        Return a function that unfolds its parameters and applies `func`
        to each unfolding. The returned function returns a list of
        length `2**n_folds`.
        """
        sig = inspect.signature(self.transform)
        vectorized_func = np.vectorize(func)

        def unfolding_func(*par_vals, **par_dic):
            par_values = np.array(sig.bind(*par_vals, **par_dic).args)
            unfolded = self.unfold(par_values)
            return vectorized_func(*unfolded.T)

        unfolding_func.__doc__ = f"""
            Return an array of {2**len(self.folded_params)} elements
            with the results of applying {func.__name__} to the
            different unfoldings of the parameters passed.
            """
        unfolding_func.__signature__ = sig
        return unfolding_func

    def _setup_folding_transforms(self):
        cubemin = self.cubemin[self._folded_inds]
        folded_cubesize = self.folded_cubesize[self._folded_inds]
        n_folds = 2 ** len(self.folded_params)
        n_reflect = len(self.folded_reflected_params)

        # Helper functions, vectorized:

        def normalize(sampled_par_values):
            """Map folded parameters from their range to (0, 2)."""
            return (sampled_par_values - cubemin) / folded_cubesize

        def unnormalize(normalized_values):
            """Inverse of ``normalize``."""
            return cubemin + normalized_values * folded_cubesize

        def reflect(normalized_value):
            """Linear interpolation of (0, 1, 2) -> (0, 1, 0)."""
            return 1 - np.abs(1 - normalized_value)

        def unreflect(normalized_value):
            """(0, 1) -> (2, 1)"""
            return 2 - normalized_value

        def shift(normalized_value):
            """(0, 1-, 1+, 2) -> (0, 1, 0, 1)."""
            return normalized_value % 1

        def unshift(normalized_value):
            """(0, 1) -> (1, 2)."""
            return normalized_value + 1

        # Folding / unfolding transforms:

        def unfold(folded_par_values):
            """
            Take an array of length `n_params` with parameter values in
            the space of folded sampled parameters, and return an array
            of shape `(2**n_folded_params, n_params)` with parameter
            values corresponding to the different ways of unfolding.
            """
            folded_values = folded_par_values[self._folded_inds]
            normalized = normalize(folded_values)
            norm_unfolded = np.r_[unreflect(normalized[:n_reflect]),
                                  unshift(normalized[n_reflect:])]
            unfolded_values = unnormalize(norm_unfolded)

            # Make 2**n copies of the original array and populate the
            # places corresponding to folded parameters with all the
            # combinations of unfold/no unfold:
            unfoldings = np.tile(folded_par_values, (n_folds, 1))
            unfoldings[:, self._folded_inds] = list(itertools.product(
                *zip(folded_values, unfolded_values)))

            return unfoldings

        self.unfold = unfold

        def fold(*sampled_par_values, **sampled_par_dic):
            """
            Take an array of length `n_params` with parameter values in
            the space of sampled parameters, and return an array of the
            same shape with folded parameter values.
            """
            out = np.array(self.signature.bind(*sampled_par_values,
                                               **sampled_par_dic).args)
            values = out[self._folded_inds]
            normalized = normalize(values)
            norm_folded = np.r_[reflect(normalized[:n_reflect]),
                                shift(normalized[n_reflect:])]
            out[self._folded_inds] = unnormalize(norm_folded)

            return out

        self.fold = fold

    @classmethod
    def init_parameters(cls, include_optional=True):
        """
        Return list of `inspect.Parameter` objects, for the parameters
        taken by the `__init__` of the class, sorted by parameter kind
        (i.e. positional arguments first, keyword arguments last).
        The `self` parameter is excluded.

        Parameters
        ----------
        include_optional: bool, whether to include parameters with
                          defaults in the returned list.
        """
        signature = inspect.signature(cls.__init__)
        all_parameters = list(signature.parameters.values())[1:]
        sorted_unique_parameters = sorted(
            dict.fromkeys(all_parameters),
            key=lambda par: (par.kind, par.default is not par.empty))

        if include_optional:
            return sorted_unique_parameters

        return [par for par in sorted_unique_parameters
                if par.default is par.empty
                and par.kind not in (par.VAR_POSITIONAL, par.VAR_KEYWORD)]

    def __init_subclass__(cls):
        """
        Check that subclasses that change the `__init__` signature also
        define their own `get_init_dict` method.
        """
        super().__init_subclass__()

        if (inspect.signature(cls.__init__)
                != inspect.signature(Prior.__init__)
                and cls.get_init_dict is Prior.get_init_dict):
            raise PriorError(
                f'{cls.__name__} must override `get_init_dict` method.')

    def __repr__(self):
        """
        Return a string of the form
        `Prior(sampled_params | conditioned_on) → standard_params`.
        """
        rep = self.__class__.__name__ + f'({", ".join(self.sampled_params)}'
        if self.conditioned_on:
            rep += f' | {", ".join(self.conditioned_on)}'
        rep += f') → [{", ".join(self.standard_params)}]'
        return rep

    @staticmethod
    def get_init_dict():
        """
        Return dictionary with keyword arguments to reproduce the class
        instance. Subclasses should override this method if they require
        initialization parameters.
        """
        return {}

    def transform_samples(self, samples: pd.DataFrame, force_update=True):
        """
        Add columns in-place for `self.standard_params` to `samples`.
        `samples` must include columns for `self.sampled_params` and
        `self.conditioned_on`.

        Parameters
        ----------
        samples: Dataframe with sampled params
        force_update: bool, whether to force an update if the transformed
                      standard samples already exist
        """
        if (not force_update) and \
                (set(self.standard_params) <= set(samples.columns)):
            return
        direct = samples[self.sampled_params + self.conditioned_on]
        standard = pd.DataFrame(list(np.vectorize(self.transform)(**direct)))
        utils.update_dataframe(samples, standard)

    def inverse_transform_samples(self, samples: pd.DataFrame):
        """
        Add columns in-place for `self.sampled_params` to `samples`.
        `samples` must include columns for `self.standard_params`.
        """
        inverse = samples[self.standard_params + self.conditioned_on]
        sampled = pd.DataFrame(list(
            np.vectorize(self.inverse_transform)(**inverse)))
        utils.update_dataframe(samples, sampled)


class CombinedPrior(Prior):
    """
    Make a new `Prior` subclass combining other `Prior` subclasses.

    Schematically, combine priors like [P(x), P(y|x)] → P(x, y).
    This class has a single abstract method `prior_classes` which is a
    list of `Prior` subclasses that we want to combine.
    Arguments to the `__init__` of the classes in `prior_classes` are
    passed by keyword, so it is important that those arguments have
    repeated names if and only if they are intended to have the same
    value.
    Also, the `__init__` of all classes in `prior_classes` need to
    accept `**kwargs` and forward them to `super().__init__()`.
    """
    @property
    @staticmethod
    @abstractmethod
    def prior_classes():
        """List of `Prior` subclasses with the priors to combine."""

    def __init__(self, *args, **kwargs):
        """
        Instantiate prior classes and define `range_dic`.

        The list of parameters to pass to a subclass `cls` can be found
        using `cls.init_parameters()`.
        """
        kwargs.update(dict(zip([par.name for par in self.init_parameters()],
                               args)))

        # Check for all required arguments at once:
        required = [
            par.name for par in self.init_parameters(include_optional=False)]
        if missing := [par for par in required if par not in kwargs]:
            raise TypeError(f'Missing {len(missing)} required arguments: '
                            f'{", ".join(missing)}')

        self.subpriors = [cls(**kwargs) for cls in self.prior_classes]

        self.range_dic = {}
        for subprior in self.subpriors:
            self.range_dic.update(subprior.range_dic)

        super().__init__(**kwargs)

    def __init_subclass__(cls):
        """
        Define the following attributes and methods from the combination
        of priors in `cls.prior_classes`:

            * `range_dic`
            * `standard_params`
            * `conditioned_on`
            * `periodic_params`
            * `reflective_params`
            * `folded_reflected_params`
            * `folded_shifted_params`
            * `transform`
            * `inverse_transform`
            * `lnprior_and_transform`
            * `lnprior`

        which are used to override the corresponding attributes and
        methods of the new `CombinedPrior` subclass.
        """
        super().__init_subclass__()

        cls._set_params()
        direct_params = cls.sampled_params + cls.conditioned_on
        inverse_params = cls.standard_params + cls.conditioned_on

        def transform(self, *par_vals, **par_dic):
            """
            Transform sampled parameter values to standard parameter
            values.
            Take `self.sampled_params + self.conditioned_on` parameters
            and return a dictionary with `self.standard_params`
            parameters.
            """
            par_dic.update(dict(zip(direct_params, par_vals)))
            for subprior in self.subpriors:
                input_dic = {par: par_dic[par]
                             for par in (subprior.sampled_params
                                         + subprior.conditioned_on)}
                par_dic.update(subprior.transform(**input_dic))
            return {par: par_dic[par] for par in self.standard_params}

        def inverse_transform(self, *par_vals, **par_dic):
            """
            Transform standard parameter values to sampled parameter values.
            Take `self.standard_params + self.conditioned_on` parameters and
            return a dictionary with `self.sampled_params` parameters.
            """
            par_dic.update(dict(zip(inverse_params, par_vals)))
            for subprior in self.subpriors:
                input_dic = {par: par_dic[par]
                             for par in (subprior.standard_params
                                         + subprior.conditioned_on)}
                par_dic.update(subprior.inverse_transform(**input_dic))
            return {par: par_dic[par] for par in self.sampled_params}

        def lnprior_and_transform(self, *par_vals, **par_dic):
            """
            Take sampled and conditioned-on parameters, and return a
            2-element tuple with the log of the prior and a dictionary
            with standard parameters.
            The reason for this function is that it is necessary to
            compute the transform in order to compute the prior, so if
            both are wanted it is efficient to compute them at once.
            """
            par_dic.update(dict(zip(direct_params, par_vals)))
            standard_par_dic = self.transform(**par_dic)
            par_dic.update(standard_par_dic)

            lnp = 0
            for subprior in self.subpriors:
                input_dic = {par: par_dic[par]
                             for par in (subprior.sampled_params
                                         + subprior.conditioned_on)}
                lnp += subprior.lnprior(**input_dic)
            return lnp, standard_par_dic

        def lnprior(self, *par_vals, **par_dic):
            """
            Natural logarithm of the prior probability density.
            Take `self.sampled_params + self.conditioned_on` parameters
            and return a float.
            """
            return self.lnprior_and_transform(*par_vals, **par_dic)[0]


        # Witchcraft to fix the functions' signatures:
        self_parameter = inspect.Parameter('self',
                                           inspect.Parameter.POSITIONAL_ONLY)
        direct_parameters = [self_parameter] + [
            inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for par in direct_params]
        inverse_parameters = [self_parameter] + [
            inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for par in inverse_params]
        cls._change_signature(transform, direct_parameters)
        cls._change_signature(inverse_transform, inverse_parameters)
        cls._change_signature(lnprior, direct_parameters)
        cls._change_signature(lnprior_and_transform, direct_parameters)

        cls.transform = transform
        cls.inverse_transform = inverse_transform
        cls.lnprior_and_transform = lnprior_and_transform
        cls.lnprior = lnprior

    @classmethod
    def _set_params(cls):
        """
        Set these class attributes:
            * `range_dic`
            * `standard_params`
            * `conditioned_on`
            * `periodic_params`
            * `reflective_params`
            * `folded_reflected_params`.
            * `folded_shifted_params`
        Raise `PriorError` if subpriors are incompatible.
        """
        cls.range_dic = {}
        for prior_class in cls.prior_classes:
            cls.range_dic.update(prior_class.range_dic)

        for params in ('standard_params', 'conditioned_on',
                       'periodic_params', 'reflective_params',
                       'folded_reflected_params', 'folded_shifted_params',):
            setattr(cls, params, [par for prior_class in cls.prior_classes
                                  for par in getattr(prior_class, params)])

        cls.conditioned_on = list(dict.fromkeys(
            [par for par in cls.conditioned_on
             if not par in cls.standard_params]))

        # Check that the provided prior_classes can be combined:
        if len(cls.sampled_params) != len(set(cls.sampled_params)):
            raise PriorError(
                f'Priors {cls.prior_classes} cannot be combined due to '
                f'repeated sampled parameters: {cls.sampled_params}')

        if len(cls.standard_params) != len(set(cls.standard_params)):
            raise PriorError(
                f'Priors {cls.prior_classes} cannot be combined due to '
                f'repeated standard parameters: {cls.standard_params}')

        for preceding, following in itertools.combinations(
                cls.prior_classes, 2):
            for conditioned_par in preceding.conditioned_on:
                if conditioned_par in following.standard_params:
                    raise PriorError(
                        f'{following} defines {conditioned_par}, which '
                        f'{preceding} requires. {following} should come before'
                        f' {preceding}.')

    @classmethod
    def init_parameters(cls, include_optional=True):
        """
        Return list of `inspect.Parameter` objects, for the aggregated
        parameters taken by the `__init__` of `prior_classes`, without
        duplicates and sorted by parameter kind (i.e. positional
        arguments first, keyword arguments last). The `self` parameter
        is excluded.

        Parameters
        ----------
        include_optional: bool, whether to include parameters with
                          defaults in the returned list.
        """
        signatures = [inspect.signature(prior_class.__init__)
                      for prior_class in cls.prior_classes]
        all_parameters = [par for signature in signatures
                          for par in list(signature.parameters.values())[1:]]
        sorted_unique_parameters = sorted(
            dict.fromkeys(all_parameters),
            key=lambda par: (par.kind, par.default is not par.empty))

        if include_optional:
            return sorted_unique_parameters

        return [par for par in sorted_unique_parameters
                if par.default is par.empty
                and par.kind not in (par.VAR_POSITIONAL, par.VAR_KEYWORD)]

    @staticmethod
    def _change_signature(func, parameters):
        """
        Change the signature of a function to explicitize the parameters
        it takes. Use with caution.

        Parameters
        ----------
        func: function.
        parameters: sequence of `inspect.Parameter` objects.
        """
        func.__signature__ = inspect.signature(func).replace(
            parameters=parameters)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        init_dicts = [subprior.get_init_dict() for subprior in self.subpriors]
        return utils.merge_dictionaries_safely(*init_dicts)

    @classmethod
    def get_fast_sampled_params(cls, fast_standard_params):
        """
        Return a list of parameter names that map to given "fast"
        standard parameters, useful for sampling fast-slow parameters.
        Updating fast sampling parameters is guaranteed to only change
        fast standard parameters.
        """
        return [par for prior_class in cls.prior_classes
                for par in prior_class.get_fast_sampled_params(
                    fast_standard_params)]


class FixedPrior(Prior):
    """
    Abstract class to set standard parameters to fixed values.
    Usage: Subclass `FixedPrior` and define a `standard_par_dic`
    attribute.
    """
    def __init__(self, **kwargs):
        """Check that `self.standard_par_dic` has the correct keys."""
        super().__init__(**kwargs)

        if missing := (self.__class__.standard_par_dic.keys()
                       - self.standard_par_dic.keys()):
            raise ValueError(f'`standard_par_dic` is missing keys: {missing}')

        if extra := (self.standard_par_dic.keys()
                     - self.__class__.standard_par_dic.keys()):
            raise ValueError(f'`standard_par_dic` has extra keys: {extra}')

    @property
    @staticmethod
    @abstractmethod
    def standard_par_dic():
        """Dictionary with fixed parameter names and values."""

    @utils.ClassProperty
    def standard_params(self):
        return list(self.standard_par_dic)

    range_dic = {}

    @staticmethod
    def lnprior():
        """Natural logarithm of the prior probability density."""
        return 0

    def transform(self):
        """Return a fixed dictionary of standard parameters."""
        return self.standard_par_dic

    def inverse_transform(self, **standard_par_dic):
        """
        Return an empty dictionary of sampled parameters.
        If `require_consistency` is set to `True`, verify that the
        `standard_par_dic` passed matches the one stored and raise
        `PriorError` if it does not.
        """
        if mismatched := [par for par, value in self.standard_par_dic.items()
                          if value != standard_par_dic[par]]:
            raise PriorError(
                'Cannot invert `standard_par_dic` because it does not '
                f'match the entries for {", ".join(mismatched)} in the '
                'fixed prior.')

        return {}


class UniformPriorMixin:
    """
    Define `lnprior` for uniform priors.
    It must be inherited before `Prior` (otherwise a `PriorError` is
    raised) so that abstract methods get overriden.
    """
    @utils.lru_cache()
    def lnprior(self, *par_vals, **par_dic):
        """
        Natural logarithm of the prior probability density.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a float.
        """
        return - np.log(np.prod(self.cubesize))

    def __init_subclass__(cls):
        """
        Check that UniformPriorMixin comes before Prior in the MRO.
        """
        super().__init_subclass__()
        check_inheritance_order(cls, UniformPriorMixin, Prior)


class IdentityTransformMixin:
    """
    Define `standard_params`, `transform` and `inverse_transform` for
    priors whose sampled and standard parameters are the same.
    It must be inherited before `Prior` (otherwise a `PriorError` is
    raised) so that abstract methods get overriden.
    """
    def __init_subclass__(cls):
        """
        Set ``standard_params`` to match ``sampled_params``, and check
        that ``IdentityTransformMixin`` comes before ``Prior`` in the
        MRO.
        """
        super().__init_subclass__()

        check_inheritance_order(cls, IdentityTransformMixin, Prior)
        cls.standard_params = cls.sampled_params

    @utils.lru_cache()
    def transform(self, *par_vals, **par_dic):
        """
        Transform sampled parameter values to standard parameter values.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a dictionary with `self.standard_params` parameters.
        """
        par_dic.update(dict(zip(self.sampled_params + self.conditioned_on,
                                par_vals)))
        return {par: par_dic[par] for par in self.standard_params}

    inverse_transform = transform


def check_inheritance_order(subclass, base1, base2):
    """
    Check that class `subclass` subclasses `base1` and `base2`, in that
    order. If it doesn't, raise `PriorError`.
    """
    for base in base1, base2:
        if not issubclass(subclass, base):
            raise PriorError(
                f'{subclass.__name__} must subclass {base.__name__}')

    if subclass.mro().index(base1) > subclass.mro().index(base2):
        raise PriorError(f'Wrong inheritance order: `{subclass.__name__}` '
                         f'must inherit from `{base1.__name__}` before '
                         f'`{base2.__name__}` (or their subclasses).')
