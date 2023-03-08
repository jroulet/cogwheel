"""
Default modular priors for mass parameters, for convenience.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""
import numpy as np
from scipy.integrate import dblquad

from cogwheel import utils
from cogwheel.prior import Prior


class UniformDetectorFrameMassesPrior(Prior):
    """
    Uniform prior for detector frame masses.
    Sampled variables are mchirp, lnq. These are transformed to m1, m2.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'mchirp': None,
                 'lnq': None}
    reflective_params = ['lnq']

    def __init__(self, *, mchirp_range, q_min=.05, symmetrize_lnq=False,
                 **kwargs):
        lnq_min = np.log(q_min)
        self.range_dic = {'mchirp': mchirp_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq)}
        super().__init__(**kwargs)

        self.prior_norm = 1
        self.prior_norm = dblquad(
            lambda mchirp, lnq: np.exp(self.lnprior(mchirp, lnq)),
            *self.range_dic['lnq'], *self.range_dic['mchirp'])[0]

    @staticmethod
    @utils.lru_cache()
    def transform(mchirp, lnq):
        """(mchirp, lnq) to (m1, m2)."""
        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q)**.2 / q**.6
        return {'m1': m1,
                'm2': m1 * q}

    @staticmethod
    def inverse_transform(m1, m2):
        """
        (m1, m2) to (mchirp, lnq). Note that if symmetrize_lnq==True the
        transformation is not invertible. Here, lnq <= 0 always.
        """
        q = m2 / m1
        return {'mchirp': m1 * q**.6 / (1 + q)**.2,
                'lnq': np.log(q)}

    @utils.lru_cache()
    def lnprior(self, mchirp, lnq):
        """
        Natural logarithm of the prior probability for `mchirp, lnq`
        under a prior flat in detector-frame masses.
        """
        return np.log(mchirp * np.cosh(lnq/2)**.4 / self.prior_norm)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'mchirp_range': self.range_dic['mchirp'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}
