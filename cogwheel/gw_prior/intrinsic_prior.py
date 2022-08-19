"""
Priors for estimating only the intrinsic parameter of compact binary mergers
"""
from cogwheel.prior import CombinedPrior
from .combined import RegisteredPriorMixin
from .mass import UniformDetectorFrameMassesPrior
from .spin import UniformEffectiveSpinPrior, ZeroInplaneSpinsPrior
from .miscellaneous import ZeroTidalDeformabilityPrior, FixedReferenceFrequencyPrior

class IntrinsicParametersPrior(RegisteredPriorMixin, CombinedPrior):
    
    prior_classes = [UniformDetectorFrameMassesPrior,
                    UniformEffectiveSpinPrior,
                    ZeroInplaneSpinsPrior,
                    ZeroTidalDeformabilityPrior,
                    FixedReferenceFrequencyPrior]