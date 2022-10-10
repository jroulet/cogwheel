"""
Default modular priors for tidal parameters, for convenience.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""

import numpy as np
from cogwheel import utils
from cogwheel import prior

class UniformTidalDeformabilitiesBNSPrior(prior.UniformPriorMixin,
                                          prior.IdentityTransformMixin,
                                          prior.Prior):
    range_dic = {'l1': NotImplemented,
                 'l2': NotImplemented}

    def __init__(self, *, max_tidal_deformability=5e3, **kwargs):
        """
        Parameters
        ----------
        max_tidal_deformability: float
            Maximum dimensionless tidal deformability for both stars.
        """
        self.range_dic = {'l1': (0, max_tidal_deformability),
                          'l2': (0, max_tidal_deformability)}
        super().__init__(**kwargs)

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'max_tidal_deformability': self.range_dic['l1'][1]}