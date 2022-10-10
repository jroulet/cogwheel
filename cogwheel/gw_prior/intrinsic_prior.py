"""
Priors for estimating only the intrinsic parameter of compact binary mergers
"""
from cogwheel.prior import CombinedPrior
from .combined import RegisteredPriorMixin
from .mass import UniformDetectorFrameMassesPrior
from .tides import UniformTidalDeformabilitiesBNSPrior
from .spin import UniformEffectiveSpinPrior, ZeroInplaneSpinsPrior
from .miscellaneous import ZeroTidalDeformabilityPrior, FixedReferenceFrequencyPrior

# for BBH systems
class IntrinsicParametersPrior(RegisteredPriorMixin, CombinedPrior):
    prior_classes = [UniformDetectorFrameMassesPrior,
                    UniformEffectiveSpinPrior,
                    ZeroInplaneSpinsPrior,
                    ZeroTidalDeformabilityPrior,
                    FixedReferenceFrequencyPrior]
    
# for BNS and NSBH systems
class IntrinsicTidalPrior(RegisteredPriorMixin, CombinedPrior):
    prior_classes = [UniformTidalDeformabilitiesBNSPrior,
                     UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     FixedReferenceFrequencyPrior]