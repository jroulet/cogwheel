"""
Define the class ``IASInjectionPrior``, intended for generating
injections rather than parameter estimation.
"""
import numpy as np

from cogwheel import gw_prior
from cogwheel import prior
from cogwheel import utils

# pylint: disable=arguments-differ


class _UniformDimensionlessVolumePrior(prior.UniformPriorMixin,
                                       prior.Prior):
    """
    Volumetric prior for the dimensionless distance
        p(d) = 3 * d**2,  0 < d < 1.
    Intended for choosing the distance scale after samples have been
    generated.
    """
    range_dic = {'dimensionless_volume': (0, 1)}
    standard_params = ['dimensionless_distance']

    def transform(self, dimensionless_volume):
        return {'dimensionless_distance': dimensionless_volume ** (1/3)}

    def inverse_transform(self, dimensionless_distance):
        return {'dimensionless_volume': dimensionless_distance ** 3}


class _SimpleSkyLocationPrior(prior.UniformPriorMixin, prior.Prior):
    """
    Isotropic prior for sky location.
    Intended for injections rather than parameter estimation.
    """
    range_dic = {'ra': (0, 2*np.pi),
                 'sindec': (-1, 1)}
    standard_params = ['ra', 'dec']

    def transform(self, ra, sindec):
        return {'ra': ra,
                'dec': np.arcsin(sindec)}

    def inverse_transform(self, ra, dec):
        return {'ra': ra,
                'sindec': np.sin(dec)}


class _SimplePhasePrior(prior.UniformPriorMixin,
                        prior.IdentityTransformMixin, prior.Prior):
    """Uniform prior for the phase."""
    range_dic = {'phi_ref': (0, 2*np.pi)}


class _ZeroTimePrior(prior.FixedPrior):
    """Set t_geocenter = 0."""
    standard_par_dic = {'t_geocenter': 0.}


class IASInjectionPrior(prior.CombinedPrior):
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
        _SimpleSkyLocationPrior,
        gw_prior.UniformDiskInplaneSpinsIsotropicInclinationPrior,
        gw_prior.UniformPolarizationPrior,
        _SimplePhasePrior,
        _UniformDimensionlessVolumePrior,
        gw_prior.ZeroTidalDeformabilityPrior,
        _ZeroTimePrior]


class VolumetricInjectionPrior(prior.CombinedPrior):
    """
    Prior for making injections. Similar to ``IASInjectionPrior`` above
    but volumetric in component spins rather than uniform in effective
    spin.
    """
    prior_classes = utils.replace(
        IASInjectionPrior.prior_classes,
        gw_prior.UniformEffectiveSpinPrior,
        gw_prior.VolumetricSpinsAlignedComponentsPrior)


class AlignedSpinIASInjectionPrior(prior.CombinedPrior):
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
        _SimpleSkyLocationPrior,
        gw_prior.IsotropicInclinationPrior,
        gw_prior.UniformPolarizationPrior,
        _SimplePhasePrior,
        _UniformDimensionlessVolumePrior,
        gw_prior.ZeroTidalDeformabilityPrior,
        gw_prior.ZeroInplaneSpinsPrior,
        _ZeroTimePrior]
