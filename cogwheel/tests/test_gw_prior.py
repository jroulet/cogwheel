"""Tests for the `gw_prior` module."""

import itertools
from unittest import TestCase, main
import numpy as np

import lal

from cogwheel import gw_prior, DETECTORS


DETECTOR_PAIRS = [''.join(pair)
                  for pair in itertools.combinations(DETECTORS, 2)]


def get_random_init_parameters():
    return dict(
        mchirp_range=np.sort(np.random.uniform(2, 40, 2)),
        q_min=np.random.uniform(.01, .9),
        detector_pair=np.random.choice(DETECTOR_PAIRS),
        tgps=np.random.uniform(0, 1e9),
        t_range=np.sort(np.random.uniform([-.1, .1])),
        ref_det_name=np.random.choice(list(DETECTORS)),
        f_ref=np.random.uniform(20, 100),
        d_hat_max=np.random.uniform(1e2, 1e4),
        symmetrize_lnq=False,  # Note `symmetrize_lnq=True` is not invertible
        )


def gen_random_pars(prior):
    return prior.cubemin + prior.cubesize * np.random.uniform(size=prior.ndim)


class PriorTestCase(TestCase):
    """Class to test `Prior` subclasses."""
    def test_inverse_transform(self):
        """
        Test that `prior.transform()` and `prior.inverse_transform()`
        are mutual inverses.
        """
        for prior_class in gw_prior.prior_registry.values():
            init_params = get_random_init_parameters()
            prior = prior_class(**init_params)
            par_vals = gen_random_pars(prior)
            par_vals_ = list(
                prior.inverse_transform(**prior.transform(*par_vals)).values())
            assert np.allclose(par_vals, par_vals_), (
                f'{prior} initialized with {init_params} does not have '
                '`transform` inverse to `inverse_transform`:\n'
                f'{par_vals} != {par_vals_}.')

    def test_periodicity(self):
        """
        Test that sampled parameters and sampled parameters shifted by
        their period produced the same standard parameters.
        """
        for prior_class in gw_prior.prior_registry.values():
            init_params = get_random_init_parameters()
            prior = prior_class(**init_params)
            for i_par in prior.periodic_inds:
                if prior.sampled_params[i_par] in prior.standard_params:
                    continue  # Identity transforms don't apply mod period
                sampled_par_vals = gen_random_pars(prior)
                sampled_par_vals_shifted = sampled_par_vals.copy()
                sampled_par_vals_shifted[i_par] += prior.cubesize[i_par]

                standard_par_dic = prior.transform(*sampled_par_vals)
                standard_par_dic_shifted = prior.transform(*sampled_par_vals_shifted)

                assert np.allclose(list(standard_par_dic.values()),
                                   list(standard_par_dic_shifted.values())), (
                    f'Parameter {i_par} of {prior} ({prior.sampled_params[i_par]}) '
                    f'does not have period {prior.cubesize[i_par]}:\n\n'
                    'Sampled parameters: '
                    f'{dict(zip(prior.sampled_params, sampled_par_vals))}\n\n'
                    'Sampled parameters shifted: '
                    f'{dict(zip(prior.sampled_params, sampled_par_vals_shifted))}\n\n'
                    f'Standard parameters: {standard_par_dic}\n\n'
                    f'Standard parameters shifted: {standard_par_dic_shifted}')


if __name__ == '__main__':
    main()
