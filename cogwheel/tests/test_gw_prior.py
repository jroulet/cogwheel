"""Tests for the `gw_prior` module."""

import itertools
import textwrap
from unittest import TestCase, main
import numpy as np

from cogwheel import gw_prior, gw_utils
from cogwheel.tests import test_waveform
from cogwheel.prior_ratio import PriorRatio


DETECTOR_PAIRS = [''.join(pair)
                  for pair in itertools.combinations(gw_utils.DETECTORS, 2)]


def get_random_init_parameters():
    """Return dictionary of keyword arguments to initialize priors."""
    par_dic_0 = test_waveform.get_random_par_dic()
    intrinsic_params = (gw_prior.miscellaneous.FixedIntrinsicParametersPrior
                        .standard_par_dic.keys())
    standard_par_dic = {key: value for key, value in par_dic_0.items()
                        if key in intrinsic_params}
    mchirp = gw_utils.m1m2_to_mchirp(par_dic_0['m1'], par_dic_0['m2'])
    q = par_dic_0['m2'] / par_dic_0['m1']
    return {'mchirp_range': gw_utils.estimate_mchirp_range(mchirp),
            'q_min': np.random.uniform(q, 0.5 * q),
            'detector_pair': np.random.choice(DETECTOR_PAIRS),
            'tgps': np.random.uniform(0, 1e9),
            't_range': np.sort(np.random.uniform(-.1, .1, 2)),
            'ref_det_name': np.random.choice(list(gw_utils.DETECTORS)),
            'f_ref': np.random.uniform(20, 100),
            'd_hat_max': np.random.uniform(1e2, 1e4),
            'symmetrize_lnq': False,  # `symmetrize_lnq=True` is not invertible
            'standard_par_dic': standard_par_dic,
            'f_avg': np.random.uniform(10, 200),
            'par_dic_0': par_dic_0,
            'eigvecs': np.array([[1.24880323, 0.01576206],
                                 [0.3499357 , -0.04291714],
                                 [0.15682589, -0.02974939]])}


def gen_random_par_dic(prior):
    """Return dictionary of sampled parameter values."""
    return prior.generate_random_samples(1)[prior.sampled_params
        ].iloc[0].to_dict()


class PriorTestCase(TestCase):
    """Class to test `Prior` subclasses."""
    def test_inverse_transform(self):
        """
        Test that `prior.transform()` and `prior.inverse_transform()`
        are mutual inverses.
        """
        for prior_class in gw_prior.prior_registry.values():
            with self.subTest(prior_class):
                init_params = get_random_init_parameters()
                prior = prior_class(**init_params)
                par_dic = gen_random_par_dic(prior)
                par_dic_ = prior.inverse_transform(
                    **prior.transform(**par_dic))
                err_msg = textwrap.dedent(f"""
                    {prior}
                    initialized with
                    {init_params}
                    does not have `transform` inverse to `inverse_transform`:
                    {par_dic}
                    !=
                    {par_dic_}.""")
                np.testing.assert_allclose(list(par_dic.values()),
                                           list(par_dic_.values()),
                                           rtol=1e-4, err_msg=err_msg)

    def test_periodicity(self):
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
                with self.subTest((prior, par)):
                    par_dic = gen_random_par_dic(prior)
                    par_dic_shifted = par_dic.copy()
                    period = prior.cubesize[prior.sampled_params.index(par)]
                    par_dic_shifted[par] += period

                    standard_par_dic = prior.transform(**par_dic)
                    standard_par_dic_shifted = prior.transform(
                        **par_dic_shifted)
                    err_msg = textwrap.dedent(f"""
                        Parameter {par} of {prior} does not have period
                        {period}

                        Sampled parameters: {par_dic}

                        Sampled parameters shifted: {par_dic_shifted}

                        Standard parameters: {standard_par_dic}

                        Standard parameters shifted: {standard_par_dic_shifted}
                        """)
                    np.testing.assert_allclose(
                        list(standard_par_dic.values()),
                        list(standard_par_dic_shifted.values()),
                        err_msg=err_msg)

    def test_prior_ratio_ias_lvc(self):
        """
        Test that PriorRatio can be computed between IASPrior and LVCPrior.
        """
        init_params = get_random_init_parameters()
        ias_prior = gw_prior.combined.IASPrior(**init_params)
        lvc_prior = gw_prior.combined.LVCPrior(**init_params)

        prior_ratio = PriorRatio(numerator=ias_prior, denominator=lvc_prior)

        ias_par_dic = gen_random_par_dic(ias_prior)
        standard_par_dic = ias_prior.transform(**ias_par_dic)

        ln_prior_ratio = prior_ratio.ln_prior_ratio(**standard_par_dic)
        self.assertTrue(np.isfinite(ln_prior_ratio),
                        f"ln_prior_ratio should be finite, got {ln_prior_ratio}")


if __name__ == '__main__':
    main()
