"""
Priors for estimating only the intrinsic parameter of compact binary mergers
"""
from cogwheel.prior import CombinedPrior
from cogwheel.gw_prior.combined import RegisteredPriorMixin
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior
from cogwheel.gw_prior.tides import UniformTidalDeformabilitiesBNSPrior
from cogwheel.gw_prior.spin import UniformEffectiveSpinPrior, ZeroInplaneSpinsPrior
from cogwheel.gw_prior.miscellaneous import ZeroTidalDeformabilityPrior, FixedReferenceFrequencyPrior

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