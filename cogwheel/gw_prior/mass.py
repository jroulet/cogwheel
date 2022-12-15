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
from cogwheel.cosmology import z_of_d_luminosity
from cogwheel.prior import Prior


class UniformDetectorFrameMassesPrior(Prior):
    """
    Uniform prior for detector frame masses.
    Sampled variables are mchirp, lnq. These are transformed to m1, m2.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'mchirp': NotImplemented,
                 'lnq': NotImplemented}
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


class UniformDetectorFrameTotalMassInverseMassRatioPrior(Prior):
    """
    Uniform in detector-frame total mass and inverse mass ratio,
        mtot = m1 + m2
        1 / q = m1 / m2.
    Sampled params are mchirp, lnq, these are transformed to m1, m2.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'mchirp': NotImplemented,
                 'lnq': NotImplemented}
    reflective_params = ['lnq']

    def __init__(self, *, mchirp_range, q_min, symmetrize_lnq=False,
                 **kwargs):
        if not 0 < q_min <= 1:
            raise ValueError('`q_min` should be between 0 and 1.')

        lnq_min = np.log(q_min)
        self.range_dic = {'mchirp': mchirp_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq)}
        super().__init__(**kwargs)

        self.prior_lognorm = 0
        self.prior_lognorm = np.log(dblquad(
            lambda mchirp, lnq: np.exp(self.lnprior(mchirp, lnq)),
            *self.range_dic['lnq'], *self.range_dic['mchirp'])[0])

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
        """(m1, m2) to (mchirp, lnq)."""
        return {'mchirp': (m1 * m2)**.6 / (m1 + m2)**.2,
                'lnq': np.log(m2 / m1)}

    def lnprior(self, mchirp, lnq):
        """
        Uniform in 1/q and mtot ==>
        (using mchirp = eta**(3/5) * mtot, eta = q / (1+q)**2)
        P(lnq, mchirp) = (C/q) * q**(3/5) / (1+q)**(6/5)
                       = C / q**.4 / (1+q)**1.2
                       = C / q / cosh(lnq / 2)**1.2  ==>
        lnP - lnC = -.4*lnq - 1.2*ln(1+q)
                  = -lnq - 1.2*ln(2*cosh(.5*lnq))
        """
        true_lnq = -np.abs(lnq)
        q = np.exp(true_lnq)
        return -.4*true_lnq - 1.2*np.log(1 + q) - self.prior_lognorm

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'mchirp_range': self.range_dic['mchirp'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}


class UniformSourceFrameTotalMassInverseMassRatioPrior(Prior):
    """
    Uniform in source-frame total mass and inverse mass ratio
        mtot_source = (m1 + m2) / (1 + z),
        1/q = m1/m2.
    Sampled params are mtot_source, lnq, these are transformed to m1, m2
    conditioned on d_luminosity.
    Note: cannot be combined with distance prior that requires mass
    conditioning.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'mtot_source': NotImplemented,
                 'lnq': NotImplemented}
    reflective_params = ['lnq']
    conditioned_on = ['d_luminosity']

    def __init__(self, *, mtot_source_range, q_min,
                 symmetrize_lnq=False, **kwargs):
        if not 0 < q_min <= 1:
            raise ValueError('`q_min` should be between 0 and 1.')

        lnq_min = np.log(q_min)
        self.range_dic = {'mtot_source': mtot_source_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq)}
        super().__init__(**kwargs)

        self.prior_lognorm = 0
        self.prior_lognorm = np.log(dblquad(
            lambda mtot_source, lnq: np.exp(self.lnprior(mtot_source, lnq)),
            *self.range_dic['lnq'], *self.range_dic['mtot_source'])[0])

    @staticmethod
    def transform(mtot_source, lnq, d_luminosity):
        """(mtot_source, lnq, d_luminosity) to (m1, m2)"""
        q = np.exp(-np.abs(lnq))
        m1 = (1 + z_of_d_luminosity(d_luminosity)) * mtot_source / (1 + q)

        return {'m1': m1,
                'm2': q * m1}

    @staticmethod
    def inverse_transform(m1, m2, d_luminosity):
        """(m1, m2, d_luminosity) to (mtot_source, lnq)"""
        mtot_source = (m1 + m2) / (1 + z_of_d_luminosity(d_luminosity))
        return {'mtot_source': mtot_source,
                'lnq': np.log(m2 / m1)}

    def lnprior(self, mtot_source, lnq):
        """Uniform in 1/q."""
        return -np.abs(lnq) - self.prior_lognorm

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'mtot_source_range': self.range_dic['mtot_source'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}