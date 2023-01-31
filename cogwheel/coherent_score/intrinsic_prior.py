"""
Priors on intrinsic parameters suitable for use with extrinsic parameter
marginalization.
"""
import numpy as np

import lal
import lalsimulation

from cogwheel import gw_prior
from cogwheel import prior
from cogwheel import utils


class UniformDiskInplaneSpinsIsotropicInclinationPrior(
        prior.UniformPriorMixin, prior.Prior):
    """
    Prior for in-plane spins and inclination that is uniform in the disk
        sx^2 + sy^2 < 1 - sz^2
    for each of the component spins and isotropic in the inclination.
    It corresponds to the IAS spin prior when combined with
    `UniformEffectiveSpinPrior`.
    """
    standard_params = ['iota', 's1x_n', 's1y_n', 's2x_n', 's2y_n',
                       ]
    range_dic = {'costheta_jn': (-1, 1),
                 'phi_jl_hat': (0, 2*np.pi),
                 'phi12': (0, 2*np.pi),
                 'cums1r_s1z': (0, 1),
                 'cums2r_s2z': (0, 1)}
    periodic_params = ['phi_jl_hat', 'phi12']
    folded_reflected_params = ['costheta_jn']
    conditioned_on = ['s1z', 's2z', 'm1', 'm2', 'f_ref']

    @staticmethod
    def _spin_transform(cumsr_sz, sz):
        """
        The in-plane spin prior is flat in the disk.
        Subclasses can override to change the spin prior.
        """
        sr = np.sqrt(cumsr_sz * (1 - sz ** 2))
        chi = np.sqrt(sr**2 + sz**2)
        tilt = np.arctan2(sr, sz)
        return chi, tilt

    @staticmethod
    def _inverse_spin_transform(chi, tilt, sz):
        """
        Return value of `cumsr_sz`, the cumulative of the prior on
        in-plane spin magnitude given the aligned spin magnitude `sz`,
        for either companion.
        The in-plane spin prior is flat in the disk. Subclasses can
        override to change the spin prior.
        """
        cumsr_sz = (chi*np.sin(tilt))**2 / (1-sz**2)
        return cumsr_sz

    @utils.lru_cache()
    def transform(self, costheta_jn, phi_jl_hat, phi12,
                  cums1r_s1z, cums2r_s2z, s1z, s2z, m1, m2, f_ref):
        """
        Return dictionary with inclination, inplane spins, right
        ascension and declination. Spin components are defined in a
        coordinate system where `z` is parallel to the orbital angular
        momentum `L` and the direction of propagation `N` lies in the
        `y-z` plane.
        """
        chi1, tilt1 = self._spin_transform(cums1r_s1z, s1z)
        chi2, tilt2 = self._spin_transform(cums2r_s2z, s2z)
        theta_jn = np.arccos(costheta_jn)
        phi_jl = (phi_jl_hat + np.pi * (costheta_jn < 0)) % (2*np.pi)

        # Use `phi_ref=0` as a trick to define azimuths based on the
        # line of sight rather than orbital separation.
        iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z \
            = lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2,
                m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_ref, phiRef=0.)

        return {'iota': iota,
                's1x_n': s1x_n,
                's1y_n': s1y_n,
                's2x_n': s2x_n,
                's2y_n': s2y_n}

    def inverse_transform(self, iota, s1x_n, s1y_n, s2x_n, s2y_n,
                          s1z, s2z, m1, m2, f_ref):
        """`standard_params` to `sampled_params`."""
        theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 \
            = lalsimulation.SimInspiralTransformPrecessingWvf2PE(
                iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, m1, m2, f_ref,
                phiRef=0.)

        cums1r_s1z = self._inverse_spin_transform(chi1, tilt1, s1z)
        cums2r_s2z = self._inverse_spin_transform(chi2, tilt2, s2z)

        costheta_jn = np.cos(theta_jn)
        phi_jl_hat = (phi_jl + np.pi * (costheta_jn < 0)) % (2*np.pi)

        return {'costheta_jn': costheta_jn,
                'phi_jl_hat': phi_jl_hat,
                'phi12': phi12,
                'cums1r_s1z': cums1r_s1z,
                'cums2r_s2z': cums2r_s2z}


class IntrinsicIASPrior(gw_prior.RegisteredPriorMixin,
                        gw_prior.CombinedPrior):
    """Precessing, flat in chieff, uniform luminosity volume."""
    prior_classes = [
        gw_prior.FixedReferenceFrequencyPrior,
        gw_prior.UniformDetectorFrameMassesPrior,
        gw_prior.UniformEffectiveSpinPrior,
        UniformDiskInplaneSpinsIsotropicInclinationPrior,
        gw_prior.ZeroTidalDeformabilityPrior]
