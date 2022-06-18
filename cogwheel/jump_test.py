"""Tests that samples actually follow some probability distribution."""

from abc import ABC, abstractmethod
import scipy.special
import numpy as np
import pandas as pd

from cogwheel import gw_plotting


class BaseJumpTest(ABC):
    """
    Abstract class for performing tests for detailed balance.

    The purpose of this class is to test whether a set of samples
    comes from a distribution that we know how to evaluate. The test
    consists in applying a transformation to the samples that produces a
    second set of samples. Any histograms of either set of samples
    should remain unchanged if the samples indeed come from the
    distribution.
    The test requires the user to provide a ``jump_function``, this can
    be any bijective function on the space of folded sampled parameters
    with the following properties:
        * Unit Jacobian determinant
        * Self inverse, i.e. f(f(x)) = x
    Each sample ``x`` is randomly reassigned between ``x`` and ``f(x)``,
    depending on the relative probabilities ``P(x)`` and ``P(f(x))``.
    A good jump function would connect regions of the parameter space
    with similar probabilities, and if you suspect that some particular
    parameter may have been mis-sampled you should try to involve it in
    the jump function. It is usually hard to devise a good jump function
    that involves correlated parameters.
    If a set of samples passes the jump test it does not mean it is
    correct. If it does not pass the jump test it means it is incorrect.
    """
    def __init__(self, posterior, samples):
        """
        Parameters
        ----------
        posterior: cogwheel.posterior.Posterior

        samples: pandas.DataFrame
            Columns must contain ``posterior.prior.sampled_params``,
            rows correspond to samples that allegedly come from
            ``posterior.lnposterior``.
        """
        self.posterior = posterior
        self.folded_samples = pd.DataFrame(
            posterior.prior.fold(**samples[posterior.prior.sampled_params]),
            columns=posterior.prior.sampled_params)

        lnposterior_unfolds = posterior.prior.unfold_apply(
            posterior.lnposterior)

        @np.vectorize
        def ln_folded_posterior(*folded_par_vals, **folded_par_dic):
            """
            Return the natural logarithm of the folded posterior
            density. Inputs must live in the space of folded sampled
            parameters.
            """
            return scipy.special.logsumexp(
                lnposterior_unfolds(*folded_par_vals, **folded_par_dic))

        self.ln_folded_posterior = ln_folded_posterior
        self.ln_folded_posterior.__signature__ = self.posterior.prior.signature

    @abstractmethod
    def jump_function(self, samples):
        """
        Provided by the subclass.
        A function that takes a DataFrame of samples and returns another
        one with the same columns. Needs to have unit Jacobian and be
        its own inverse.
        """

    def __call__(self):
        """
        Perform the jump test. This will:

            * Print the percentage of samples that jumped; this depends
              on how well the jump function was chosen. Ideally a
              sizeable fraction should jump and the higher the better,
              in the sense that if very few samples jump then the test
              does not have a chance to fail.

            * Make a corner plot of the samples before and after the
              jump. If these are mismatched the samples do not follow
              the allefed distribution.

            * Return two `DataFrame`s, with samples before and after the
              jump. Only samples for which the jump function is not the
              identity are included.
        """
        params = self.posterior.prior.sampled_params
        jumped_samples = self.jump_function(self.folded_samples)

        np.testing.assert_allclose(self.folded_samples,
                                   self.jump_function(jumped_samples),
                                   err_msg='Jump function is not self inverse.')

        # Discard samples for which the jump function is the identity:
        relevant_rows = ~np.all(self.folded_samples == jumped_samples, axis=1)
        samples = self.folded_samples[relevant_rows]
        jumped_samples = jumped_samples[relevant_rows]

        lnprob = self.ln_folded_posterior(**samples[params])
        lnprob_jumped = self.ln_folded_posterior(**jumped_samples[params])

        jump_probability = np.exp(lnprob_jumped - lnprob)
        jump = np.random.uniform(0, 1, len(samples)) < jump_probability

        print(f'{100 * np.count_nonzero(jump) / len(self.folded_samples):.1f}%'
              ' of samples jumped.')

        test_samples = samples.copy()
        test_samples[jump] = jumped_samples[jump]

        relevant_pars = list(
            samples.columns[~np.all(samples == jumped_samples, axis=0)])
        gw_plotting.MultiCornerPlot([samples, test_samples],
                                    labels=['Original', 'Jump test'],
                                    params=relevant_pars
                                   ).plot()
        return samples, test_samples


class AutomaticJumpTest(BaseJumpTest):
    """
    Abstract class for jump tests whose range is set automatically from
    the samples.
    Defines attributes ``params`` and ``range_dic``.
    Provides a method ``jump_function`` and an abstract method
    ``_aux_jump``.
    """
    def __init__(self, posterior, samples, params, quantile=(.01, .99)):
        """
        Parameters
        ----------
        posterior: cogwheel.posterior.Posterior

        samples: pandas.DataFrame
            Columns must contain ``posterior.prior.sampled_params``,
            rows correspond to samples that allegedly come from
            ``posterior.lnposterior``.

        params: list of str
            Which parameters to involve in the jump function.
            If you want to jump-test some parameter that has
            correlations, try reflecting all correlated parameters at
            once. All parameters must be in
            ``posterior.prior.sampled_params``.

        quantile: (float, float) between 0 and 1
            Used to automatically detect the range of values for
            parameters in ``params`` for which the jump function will be
            applied, in terms of quantiles of ``samples``.
        """
        super().__init__(posterior, samples)

        self.params = params
        self.range_dic = {par: self.folded_samples[par].quantile(quantile)
                          for par in params}

    def jump_function(self, samples):
        jumped_samples = samples.copy()

        for par, rng in self.range_dic.items():
            jumped_samples[par] = self._aux_jump(samples[par], *rng)

        return jumped_samples

    @staticmethod
    @abstractmethod
    @np.vectorize
    def _aux_jump(value, low, high):
        """
        Provided by subclass.
        Return the value at which the parameter will jump given the
        automatically defined range.
        """


class ReflectionJumpTest(AutomaticJumpTest):
    """
    Generate a jump function automatically that inverts a range of
    values in the bulk of the distribution.
    Should work best if ``params`` are unimodal parameters.
    """
    @staticmethod
    @np.vectorize
    def _aux_jump(value, low, high):
        if low < value < high:
            return low + high - value
        return value


class ShiftJumpTest(AutomaticJumpTest):
    """
    Generate a jump function automatically that shifts a range of
    values in the bulk of the distribution.
    Should work best if ``params`` are unimodal parameters.
    """
    @staticmethod
    @np.vectorize
    def _aux_jump(value, low, high):
        if low < value < high:
            return (value - low + (high-low)/2) % (high-low) + low
        return value
