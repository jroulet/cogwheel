"""
Default modular priors for miscellaneous parameters, for convenience.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""
import numpy as np

from cogwheel import waveform
from cogwheel.prior import Prior, FixedPrior, UniformPriorMixin


class ZeroTidalDeformabilityPrior(FixedPrior):
    """Set tidal deformability parameters to zero."""
    standard_par_dic = {'l1': 0,
                        'l2': 0}


class FixedIntrinsicParametersPrior(FixedPrior):
    """Fix masses, spins and tidal deformabilities."""
    standard_par_dic = {'m1': None,
                        'm2': None,
                        's1x_n': None,
                        's1y_n': None,
                        's1z': None,
                        's2x_n': None,
                        's2y_n': None,
                        's2z': None,
                        'l1': None,
                        'l2': None}

    def __init__(self, *, standard_par_dic, **kwargs):
        """
        Parameters
        ----------
        standard_par_dic:
            dictionary containing entries for
            `m1, m2, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, l1, l2`.
            Spins and tidal deformabilities would default to `0.` if not
            passed. Passing a `standard_par_dic` with other missing keys
            will raise a `ValueError`. Extra keys are silently ignored
            and passed to `super().__init__()`.
        """
        self._original_par_dic = standard_par_dic
        relevant_pars = self.standard_par_dic.keys() & standard_par_dic.keys()
        relevant_dic = {par: standard_par_dic[par] for par in relevant_pars}
        self.standard_par_dic = waveform.DEFAULT_PARS | relevant_dic
        super().__init__(standard_par_dic=standard_par_dic, **kwargs)

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'standard_par_dic': self._original_par_dic}


class FixedReferenceFrequencyPrior(FixedPrior):
    """Fix reference frequency `f_ref`."""
    standard_par_dic = {'f_ref': None}

    def __init__(self, *, f_ref, **kwargs):
        super().__init__(**kwargs)
        self.standard_par_dic = {'f_ref': f_ref}

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return self.standard_par_dic


class LogarithmicReferenceFrequencyPrior(UniformPriorMixin, Prior):
    """
    Promote `f_ref` to a sampled parameter to explore its effect.
    Not intended for actual parameter estimation.
    """
    standard_params = ['f_ref']
    range_dic = {'ln_f_ref': None}

    def __init__(self, f_ref_rng=(15, 300), **kwargs):
        """
        Parameters
        ----------
        `f_ref_rng`: minimum and maximum reference frequencies in Hz.
        """
        self.range_dic = {'ln_f_ref': np.log(f_ref_rng)}
        super().__init__(**kwargs)

    def transform(self, ln_f_ref):
        """`ln_f_ref` to `f_ref`."""
        return {'f_ref': np.exp(ln_f_ref)}

    def inverse_transform(self, f_ref):
        """`f_ref` to `ln_f_ref`."""
        return {'ln_f_ref': np.log(f_ref)}

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'f_ref_rng': np.exp(self.range_dic['ln_f_ref'])}
