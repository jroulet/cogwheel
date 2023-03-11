"""
Define some commonly used priors for the full set of parameters, for
convenience.

Prior classes defined here can be used for parameter estimation and
are registered in a dictionary ``prior_registry``.
"""

from cogwheel import utils
from cogwheel.prior import CombinedPrior, Prior, check_inheritance_order
from cogwheel.likelihood import (RelativeBinningLikelihood,
                                 MarginalizedDistanceLikelihood,
                                 MarginalizedExtrinsicLikelihood,
                                 MarginalizedExtrinsicLikelihoodQAS)

from .extrinsic import (UniformPhasePrior,
                        IsotropicInclinationPrior,
                        IsotropicSkyLocationPrior,
                        UniformTimePrior,
                        UniformPolarizationPrior,
                        UniformLuminosityVolumePrior,
                        UniformComovingVolumePrior)

from .mass import UniformDetectorFrameMassesPrior

from .tides import UniformTidalDeformabilitiesBNSPrior

from .miscellaneous import (ZeroTidalDeformabilityPrior,
                            FixedIntrinsicParametersPrior,
                            FixedReferenceFrequencyPrior)

from .spin import (
    UniformEffectiveSpinPrior,
    IsotropicSpinsAlignedComponentsPrior,
    UniformDiskInplaneSpinsIsotropicInclinationPrior,
    IsotropicSpinsInplaneComponentsIsotropicInclinationPrior,
    UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior,
    IsotropicSpinsInplaneComponentsIsotropicInclinationSkyLocationPrior,
    ZeroInplaneSpinsPrior)

prior_registry = {}


class ConditionedPriorError(Exception):
    """Indicates that a Prior is conditioned on some parameters."""


class ReferenceWaveformFinderMixin:
    """
    Provide a constructor based on a `likelihood.ReferenceWaveformFinder`
    instance to provide initialization arguments.
    """
    @classmethod
    def from_reference_waveform_finder(
            cls, reference_waveform_finder, **kwargs):
        """
        Instantiate `prior.Prior` subclass with help from a
        `likelihood.ReferenceWaveformFinder` instance.
        This will generate kwargs for:
            * tgps
            * par_dic_0
            * f_avg
            * f_ref
            * ref_det_name
            * detector_pair
            * t0_refdet
            * mchirp_range
        Additional `**kwargs` can be passed to complete missing entries
        or override these.
        """
        return cls(**reference_waveform_finder.get_coordinate_system_kwargs()
                   | kwargs)


class RegisteredPriorMixin(ReferenceWaveformFinderMixin):
    """
    Register existence of a `Prior` subclass in `prior_registry`.
    Intended usage is to only register the final priors (i.e., for the
    full set of GW parameters).
    `RegisteredPriorMixin` should be inherited before `Prior` (otherwise
    `PriorError` is raised) in order to test for conditioned-on
    parameters.
    """
    def __init_subclass__(cls):
        """Validate subclass and register it in prior_registry."""
        super().__init_subclass__()
        check_inheritance_order(cls, RegisteredPriorMixin, Prior)

        if cls.conditioned_on:
            raise ConditionedPriorError('Only register fully defined priors.')

        prior_registry[cls.__name__] = cls


# ----------------------------------------------------------------------
# Default priors for the full set of variables, for convenience.

class IASPrior(RegisteredPriorMixin, CombinedPrior):
    """Precessing, flat in chieff, uniform luminosity volume."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = [
        FixedReferenceFrequencyPrior,
        UniformDetectorFrameMassesPrior,
        UniformEffectiveSpinPrior,
        UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior,
        UniformPolarizationPrior,
        UniformTimePrior,
        UniformPhasePrior,
        UniformLuminosityVolumePrior,
        ZeroTidalDeformabilityPrior]


class AlignedSpinIASPrior(RegisteredPriorMixin, CombinedPrior):
    """Aligned spin, flat in chieff, uniform luminosity volume."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = [UniformDetectorFrameMassesPrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     UniformLuminosityVolumePrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class TidalIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Aligned spin, flat in tidal parameters, flat in chieff, uniform
    luminosity volume.
    """
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = [UniformDetectorFrameMassesPrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     UniformLuminosityVolumePrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     UniformTidalDeformabilitiesBNSPrior,
                     FixedReferenceFrequencyPrior]


class LVCPrior(RegisteredPriorMixin, CombinedPrior):
    """Precessing, isotropic spins, uniform luminosity volume."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = [
        FixedReferenceFrequencyPrior,
        UniformDetectorFrameMassesPrior,
        IsotropicSpinsAlignedComponentsPrior,
        UniformPolarizationPrior,
        IsotropicSpinsInplaneComponentsIsotropicInclinationSkyLocationPrior,
        UniformTimePrior,
        UniformPhasePrior,
        UniformLuminosityVolumePrior,
        ZeroTidalDeformabilityPrior]


class AlignedSpinLVCPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Aligned spin components from isotropic distribution, uniform
    luminosity volume.
    """
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = [UniformDetectorFrameMassesPrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     UniformLuminosityVolumePrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class IASPriorComovingVT(RegisteredPriorMixin, CombinedPrior):
    """Precessing, flat in chieff, uniform comoving VT."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = utils.replace(IASPrior.prior_classes,
                                  UniformLuminosityVolumePrior,
                                  UniformComovingVolumePrior)


class AlignedSpinIASPriorComovingVT(RegisteredPriorMixin,
                                    CombinedPrior):
    """Aligned spin, flat in chieff, uniform comoving VT."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = utils.replace(AlignedSpinIASPrior.prior_classes,
                                  UniformLuminosityVolumePrior,
                                  UniformComovingVolumePrior)


class LVCPriorComovingVT(RegisteredPriorMixin, CombinedPrior):
    """Precessing, isotropic spins, uniform comoving VT."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = utils.replace(LVCPrior.prior_classes,
                                  UniformLuminosityVolumePrior,
                                  UniformComovingVolumePrior)


class AlignedSpinLVCPriorComovingVT(RegisteredPriorMixin,
                                    CombinedPrior):
    """
    Aligned spins from isotropic distribution, uniform comoving VT.
    """
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = utils.replace(AlignedSpinLVCPrior.prior_classes,
                                  UniformLuminosityVolumePrior,
                                  UniformComovingVolumePrior)


class ExtrinsicParametersPrior(RegisteredPriorMixin, CombinedPrior):
    """Uniform luminosity volume, fixed intrinsic parameters."""
    default_likelihood_class = RelativeBinningLikelihood

    prior_classes = [FixedIntrinsicParametersPrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     UniformLuminosityVolumePrior,
                     FixedReferenceFrequencyPrior]


class MarginalizedDistanceIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior for usage with ``MarginalizedDistanceLikelihood``.
    Similar to ``IASPrior`` except it does not include distance.
    Uniform in effective spin and detector-frame component masses.
    """
    default_likelihood_class = MarginalizedDistanceLikelihood

    prior_classes = IASPrior.prior_classes.copy()
    prior_classes.remove(UniformLuminosityVolumePrior)


class MarginalizedDistanceLVCPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior for usage with ``MarginalizedDistanceLikelihood``.
    Similar to ``LVCPrior`` except it does not include distance.
    Isotropic spin orientations, uniform in component spin magnitudes
    and detector-frame component masses.
    """
    default_likelihood_class = MarginalizedDistanceLikelihood

    prior_classes = LVCPrior.prior_classes.copy()
    prior_classes.remove(UniformLuminosityVolumePrior)


class IntrinsicAlignedSpinIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior for usage with ``MarginalizedExtrinsicLikelihoodQAS``.
    Intrinsic parameters only, aligned spins, uniform in effective spin
    and detector frame component masses, no tides.
    """
    default_likelihood_class = MarginalizedExtrinsicLikelihoodQAS

    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class IntrinsicAlignedSpinLVCPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior for usage with ``MarginalizedExtrinsicLikelihoodQAS``.
    Intrinsic parameters only, aligned spins, uniform in effective spin
    and detector frame component masses, no tides.
    """
    default_likelihood_class = MarginalizedExtrinsicLikelihoodQAS

    prior_classes = [UniformDetectorFrameMassesPrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class IntrinsicIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior for usage with ``MarginalizedExtrinsicLikelihood``.
    Intrinsic parameters only, precessing, uniform in effective spin
    and detector frame component masses, no tides.
    """
    default_likelihood_class = MarginalizedExtrinsicLikelihood

    prior_classes = [FixedReferenceFrequencyPrior,
                     UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     UniformDiskInplaneSpinsIsotropicInclinationPrior,
                     ZeroTidalDeformabilityPrior]


class IntrinsicLVCPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior for usage with ``MarginalizedExtrinsicLikelihood``.
    Intrinsic parameters only, precessing, isotropic spins, uniform in
    component spin magnitudes and detector frame masses, no tides.
    """
    default_likelihood_class = MarginalizedExtrinsicLikelihood

    prior_classes = [FixedReferenceFrequencyPrior,
                     UniformDetectorFrameMassesPrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     IsotropicSpinsInplaneComponentsIsotropicInclinationPrior,
                     ZeroTidalDeformabilityPrior]
