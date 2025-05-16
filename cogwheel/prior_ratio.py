"""Compute ratio between priors."""
import numpy as np


class PriorRatio:
    """
    Compute the ratio of two priors.

    The ratio of two distributions is coordinate invariant (so long as
    the two distributions use the same coordinates); we will take
    the ratio in the space of
    :py:attr:`~cogwheel.prior.Prior.standard_params`, which are expected
    to be the same between different priors. However, prior densities in
    ``cogwheel`` are naturally defined in the space of
    :py:attr:`~cogwheel.prior.Prior.sampled_params`, so we need the
    Jacobian of the :py:meth:`~cogwheel.prior.Prior.inverse_transform`.
    This is implemented as an optional method of the ``cogwheel``
    priors, :py:meth:`~cogwheel.prior.Prior.ln_jacobian_determinant`.

    For convenience and efficiency, we skip computation of some
    Jacobians in the following situation: if the priors are instances of
    :py:class:`cogwheel.prior.CombinedPrior` and they have some
    subpriors in common, then they can be "canceled out" without
    computing them. In this case it is allowed to leave some Jacobians
    undefined, only the subpriors that differ need to have their
    Jacobians defined. In particular this allows to deal with cases
    where the number of sampled and standard parameters are not the same
    (e.g. :py:class:`~cogwheel.prior.FixedPrior`).
    """
    def __init__(self, numerator, denominator):
        """
        Parameters
        ----------
        numerator, denominator : cogwheel.prior.Prior
            Priors that define the ratio.
        """
        if set(numerator.standard_params) != set(denominator.standard_params):
            raise ValueError('Incompatible standard_params.')

        self.numerator = numerator
        self.denominator = denominator

        # Remove matching subpriors so that we can get away with not
        # defining Jacobians for them.
        self._numerator_subpriors = _get_subpriors(numerator)
        self._denominator_subpriors = _get_subpriors(denominator)

        _remove_matching_items(self._numerator_subpriors,
                               self._denominator_subpriors)

    def ln_prior_ratio(self, **parameters):
        """
        Return natural log of the ratio of prior densities.

        Parameters
        ----------
        **parameters
            Values for standard and conditioned-on parameters, they
            should contain (at least) the parameters required by the
            subpriors that are not shared by the numerator and
            denominator.
        """
        lnp_numerator = sum((prior.standard_lnprior(**parameters)
                             for prior in self._numerator_subpriors),
                            start=0.0)

        lnp_denominator = sum(prior.standard_lnprior(**parameters)
                              for prior in self._denominator_subpriors)

        return lnp_numerator - lnp_denominator


def _get_subpriors(prior):
    """Break a (combined) prior into its smallest components."""
    if hasattr(prior, 'subpriors'):  # See cogwheel.prior.CombinedPrior
        subpriors = []
        for subprior in prior.subpriors:
            subpriors.extend(_get_subpriors(subprior))
        return subpriors
    return [prior]


def _remove_matching_items(list1, list2):
    """
    Remove items from both lists if they have an equal counterpart.
    """
    i = 0
    while i < len(list1):
        item1 = list1[i]
        for j, item2 in enumerate(list2):
            if _deep_eq(item1, item2):
                del list1[i]
                del list2[j]
                i -= 1  # Stay on the same index after removal
                break
        i += 1


def _deep_eq(obj1, obj2):
    """
    Recursively check equality of two objects, handling NumPy arrays.
    """
    if type(obj1) is not type(obj2):
        return False

    if isinstance(obj1, np.ndarray):
        return np.array_equal(obj1, obj2)

    if isinstance(obj1, dict):
        return (obj1.keys() == obj2.keys()
                and all(_deep_eq(obj1[k], obj2[k]) for k in obj1))

    if isinstance(obj1, (list, tuple)):
        return (len(obj1) == len(obj2)
                and all(_deep_eq(a, b) for a, b in zip(obj1, obj2)))

    if hasattr(obj1, '__dict__'):
        return (hasattr(obj2, '__dict__')
                and _deep_eq(obj1.__dict__, obj2.__dict__))

    return obj1 == obj2
