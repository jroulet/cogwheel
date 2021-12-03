"""Tests for the `gw_prior` module."""

import itertools
from unittest import TestCase, main
import numpy as np

from cogwheel import gw_prior
from cogwheel.gw_utils import DETECTORS


DETECTOR_PAIRS = [''.join(pair)
                  for pair in itertools.combinations(DETECTORS, 2)]


def get_random_init_parameters():
    """Return dictionary of keyword arguments to initialize priors."""
    return dict(
        mchirp_range=np.sort(np.random.uniform(2, 40, 2)),
        mtot_range=np.sort(np.random.uniform(2, 40, 2)),
        mtot_source_range=np.sort(np.random.uniform(2, 40, 2)),
        q_min=np.random.uniform(.01, .9),
        detector_pair=np.random.choice(DETECTOR_PAIRS),
        tgps=np.random.uniform(0, 1e9),
        t_range=np.sort(np.random.uniform([-.1, .1])),
        ref_det_name=np.random.choice(list(DETECTORS)),
        f_ref=np.random.uniform(20, 100),
        d_hat_max=np.random.uniform(1e2, 1e4),
        symmetrize_lnq=False,  # Note `symmetrize_lnq=True` is not invertible
        )


def gen_random_par_dic(prior):
    """Return dirctionary of sampled parameter values."""
    r = np.random.uniform(size=len(prior.sampled_params))
    return dict(zip(prior.sampled_params, prior.cubemin + r * prior.cubesize))


class PriorTestCase(TestCase):
    """Class to test `Prior` subclasses."""
    @staticmethod
    def test_inverse_transform():
        """
        Test that `prior.transform()` and `prior.inverse_transform()`
        are mutual inverses.
        """
        for prior_class in gw_prior.prior_registry.values():
            init_params = get_random_init_parameters()
            prior = prior_class(**init_params)
            par_dic = gen_random_par_dic(prior)
            par_dic_ = prior.inverse_transform(**prior.transform(**par_dic))
            assert np.allclose(list(par_dic.values()),
                               list(par_dic.values())), (
                f'{prior} initialized with {init_params} does not have '
                '`transform` inverse to `inverse_transform`:\n'
                f'{par_dic} != {par_dic_}.')

    @staticmethod
    def test_periodicity():
        """
        Test that sampled parameters and sampled parameters shifted by
        their period produced the same standard parameters.
        """
        for prior_class in gw_prior.prior_registry.values():
            init_params = get_random_init_parameters()
            prior = prior_class(**init_params)
            for par in prior.periodic_params:
                if par in prior.standard_params:
                    continue  # Identity transforms don't apply mod period
                par_dic = gen_random_par_dic(prior)
                par_dic_shifted = par_dic.copy()
                period = prior.cubesize[prior.sampled_params.index(par)]
                par_dic_shifted[par] += period

                standard_par_dic = prior.transform(**par_dic)
                standard_par_dic_shifted = prior.transform(**par_dic_shifted)

                assert np.allclose(list(standard_par_dic.values()),
                                   list(standard_par_dic_shifted.values())), (
                    f'Parameter {par} of {prior} does not have period {period}'
                    f'\n\nSampled parameters: {par_dic}\n\n'
                    f'Sampled parameters shifted: {par_dic_shifted}\n\n'
                    f'Standard parameters: {standard_par_dic}\n\n'
                    f'Standard parameters shifted: {standard_par_dic_shifted}')


if __name__ == '__main__':
    main()
