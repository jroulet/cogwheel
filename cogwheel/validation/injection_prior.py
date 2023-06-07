"""Define the class ``InjectionPrior``."""
import numpy as np

from cogwheel import gw_prior
from cogwheel import prior


class _UniformDimensionlessVolumePrior(prior.UniformPriorMixin,
                                       prior.Prior):
    """
    Volumetric prior for the dimensionless distance
        p(d) = 3 * d**2,  0 < d < 1.
    Intended to choose the distance scale after samples have been
    generated.
    """
    range_dic = {'dimensionless_volume': (0, 1)}
    standard_params = ['dimensionless_distance']

    @staticmethod
    def transform(dimensionless_volume):
        return {'dimensionless_distance': dimensionless_volume ** (1/3)}

    @staticmethod
    def inverse_transform(dimensionless_distance):
        return {'dimensionless_volume': dimensionless_distance ** 3}


class _SimplePhasePrior(prior.UniformPriorMixin,
                        prior.IdentityTransformMixin, prior.Prior):
    """Uniform prior for the phase."""
    range_dic = {'phi_ref': (0, 2*np.pi)}


class _ZeroTimePrior(prior.FixedPrior):
    """Set t_geocenter = 0."""
    standard_par_dic = {'t_geocenter': 0.}


class InjectionPrior(prior.CombinedPrior):
    """
    Prior for making injections. Its density is similar to the IAS
    prior, except that t_geocenter is fixed to 0, and the distance
    standard parameter is dimensionless (between 0 and 1).

    This does not include any cut in ⟨ℎ∣ℎ⟩, it must be applied a
    posteriori if desired.
    """
    prior_classes = [
        gw_prior.FixedReferenceFrequencyPrior,
        gw_prior.UniformDetectorFrameMassesPrior,
        gw_prior.UniformEffectiveSpinPrior,
        gw_prior.UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior,
        gw_prior.UniformPolarizationPrior,
        _SimplePhasePrior,
        _UniformDimensionlessVolumePrior,
        gw_prior.ZeroTidalDeformabilityPrior,
        _ZeroTimePrior]
