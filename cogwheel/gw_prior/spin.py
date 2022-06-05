"""
Default modular priors for spin parameters, for convenience.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""
import numpy as np
from scipy.interpolate import interp1d

import lal
import lalsimulation

from cogwheel.prior import Prior, FixedPrior, UniformPriorMixin
from .extrinsic import (UniformPhasePrior,
                        UniformTimePrior,
                        IsotropicSkyLocationPrior)


class UniformEffectiveSpinPrior(UniformPriorMixin, Prior):
    """
    Spin prior for aligned spins that is flat in effective spin chieff.
    The sampled parameters are `chieff` and `cumchidiff`.
    `cumchidiff` ranges from 0 to 1 and is typically poorly measured.
    `cumchidiff` is the cumulative of a prior on the spin difference
    that is uniform conditioned on `chieff`, `q`.
    """
    standard_params = ['s1z', 's2z']
    range_dic = {'chieff': (-1, 1),
                 'cumchidiff': (0, 1)}
    conditioned_on = ['m1', 'm2']

    @staticmethod
    def _get_s1z(chieff, q, s2z):
        return (1+q)*chieff - q*s2z

    def _s1z_lim(self, chieff, q):
        s1z_min = np.maximum(self._get_s1z(chieff, q, s2z=1), -1)
        s1z_max = np.minimum(self._get_s1z(chieff, q, s2z=-1), 1)
        return s1z_min, s1z_max

    def transform(self, chieff, cumchidiff, m1, m2):
        """(chieff, cumchidiff) to (s1z, s2z)."""
        q = m2 / m1
        s1z_min, s1z_max = self._s1z_lim(chieff, q)
        s1z = s1z_min + cumchidiff * (s1z_max - s1z_min)
        s2z = ((1+q)*chieff - s1z) / q
        return {'s1z': s1z,
                's2z': s2z}

    def inverse_transform(self, s1z, s2z, m1, m2):
        """(s1z, s2z) to (chieff, cumchidiff)."""
        q = m2 / m1
        chieff = (s1z + q*s2z) / (1 + q)
        s1z_min, s1z_max = self._s1z_lim(chieff, q)
        cumchidiff = (s1z - s1z_min) / (s1z_max - s1z_min)
        return {'chieff': chieff,
                'cumchidiff': cumchidiff}


class UniformDiskInplaneSpinsInclinationPhaseSkyLocationTimePrior(
        UniformPhasePrior,
        IsotropicSkyLocationPrior,
        UniformTimePrior):
    """
    Prior for in-plane spins, inclination, reference phase, sky location
    and time that is uniform in the disk
        sx^2 + sy^2 < 1 - sz^2
    for each of the component spins, isotropic in the inclination and
    sky location and uniform in reference phase and time.
    It corresponds to the IAS spin prior when combined with
    `UniformEffectiveSpinPrior`.
    # TODO break into little pieces
    """
    standard_params = ['iota', 's1x_n', 's1y_n', 's2x_n', 's2y_n', 'phi_ref',
                       'ra', 'dec', 't_geocenter']
    range_dic = {'costheta_jn': (-1, 1),
                 'phi_jl_hat': (0, 2*np.pi),
                 'phi12': (0, 2*np.pi),
                 'cums1r_s1z': (0, 1),
                 'cums2r_s2z': (0, 1),
                 'phi_ref_hat': (-np.pi/2, 3*np.pi/2),
                 'costhetanet': (-1, 1),
                 'phinet_hat': (0, 2*np.pi),
                 't_refdet': NotImplemented}
    periodic_params = ['phi_jl_hat', 'phi12']
    folded_reflected_params = ['costheta_jn', 'phinet_hat']
    folded_shifted_params = ['phi_ref_hat']
    conditioned_on = ['s1z', 's2z', 'm1', 'm2', 'f_ref', 'psi']

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

    def transform(self, costheta_jn, phi_jl_hat, phi12, cums1r_s1z,
                  cums2r_s2z, phi_ref_hat, costhetanet, phinet_hat,
                  t_refdet, s1z, s2z, m1, m2, f_ref, psi):
        """
        Return dictionary of standard parameters.
        """
        # Find `iota`, remember auxiliary spins for later:
        iota, inplane_spins = self._iota_inplane_spins(
            costheta_jn, phi_jl_hat, phi12, cums1r_s1z, cums2r_s2z, s1z, s2z,
            m1, m2, f_ref)

        # Use `iota` to get `ra, dec`:
        ra_dec = IsotropicSkyLocationPrior.transform(
            self, costhetanet, phinet_hat, iota)

        # Use `t_refdet, ra, dec` to get `t_geocenter`:
        t_geocenter = UniformTimePrior.transform(self, t_refdet, **ra_dec)

        # Use `iota, ra, dec, t_geocenter` to get `phi_ref`:
        phi_ref = UniformPhasePrior.transform(
            self, phi_ref_hat, iota, **ra_dec, psi=psi, **t_geocenter)

        return {'iota': iota} | inplane_spins | phi_ref | ra_dec | t_geocenter

    def _iota_inplane_spins(self, costheta_jn, phi_jl_hat, phi12,
                            cums1r_s1z, cums2r_s2z, s1z, s2z, m1, m2,
                            f_ref):
        """
        Return inclination and dictionary of inplane spins, defined in a
        coordinate system where `z` is parallel to the orbital angular
        momentum `L` and the line of sight `N` lies in the `y-z` plane.
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

        return iota, {'s1x_n': s1x_n,
                      's1y_n': s1y_n,
                      's2x_n': s2x_n,
                      's2y_n': s2y_n}

    def inverse_transform(self, iota, s1x_n, s1y_n, s2x_n, s2y_n,
                          s1z, s2z, phi_ref, ra, dec, m1, m2, f_ref,
                          psi, t_geocenter):
        """
        Return dictionary of sampled parameters.
        """
        costheta_jn_inplane_spins = self._invert_iota_inplane_spins(
            iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, m1, m2, f_ref)

        sky_angles = IsotropicSkyLocationPrior.inverse_transform(
            self, ra, dec, iota)

        phi_ref_hat = UniformPhasePrior.inverse_transform(
            self, psi, iota, ra, dec, phi_ref, t_geocenter)

        t_refdet = UniformTimePrior.inverse_transform(
            self, t_geocenter, ra, dec)

        return costheta_jn_inplane_spins | sky_angles | phi_ref_hat | t_refdet

    def _invert_iota_inplane_spins(self, iota, s1x_n, s1y_n, s1z,
                                   s2x_n, s2y_n, s2z, m1, m2, f_ref):
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


class IsotropicSpinsAlignedComponentsPrior(UniformPriorMixin, Prior):
    """
    Spin prior for aligned spin components that can be combined with
    IsotropicSpinsInplaneComponentsPrior to give constituent spin priors
    that are independently uniform in magnitude and solid angle.
    The sampled parameters are `cums1z` and `cums2z`, both from U(0, 1).
    """
    standard_params = ['s1z', 's2z']
    range_dic = {'cums1z': (0, 1),
                 'cums2z': (0, 1)}
    # Spin coordinates that are convenient for the
    # LVC isotropic spin prior (0 <= cumsz <= 1)
    sz_grid = np.linspace(-1, 1, 4000)
    cumsz_grid = (1 + sz_grid - sz_grid * np.log(np.abs(sz_grid))) / 2
    sz_interp = interp1d(cumsz_grid, sz_grid, bounds_error=True)

    @classmethod
    def _spin_transform(cls, cumsz):
        return cls.sz_interp(cumsz)[()]

    def transform(self, cums1z, cums2z):
        """(cums1z, cums2z) to (s1z, s2z)."""
        return {'s1z': self._spin_transform(cums1z),
                's2z': self._spin_transform(cums2z)}

    @staticmethod
    def _inverse_spin_transform(sz):
        return (1 + sz - sz * np.log(np.abs(sz))) / 2

    def inverse_transform(self, s1z, s2z):
        """(s1z, s2z) to (cums1z, cums2z)."""
        return {'cums1z': self._inverse_spin_transform(s1z),
                'cums2z': self._inverse_spin_transform(s2z)}


class IsotropicSpinsInplaneComponentsInclinationPhaseSkyLocationTimePrior(
        UniformDiskInplaneSpinsInclinationPhaseSkyLocationTimePrior):
    """
    Like `UniformDiskInplaneSpinsInclinationPhaseSkyLocationTimePrior`
    except it gives a spin prior uniform in magnitude and solid angle
    (isotropic) for each of the constituent spins independently when
    combined with `IsotropicSpinsAlignedComponentsPrior`.
    """
    @staticmethod
    def _spin_transform(cumsr_sz, sz):
        sr = np.sqrt(sz**2 * (1 / (sz**2)**cumsr_sz - 1))
        chi = np.sqrt(sr**2 + sz**2)
        tilt = np.arctan2(sr, sz)
        return chi, tilt

    @staticmethod
    def _inverse_spin_transform(chi, tilt, sz):
        """(cumsr_sz, sphi_hat) from (sx, sy, sz, phi_ref, iota)."""
        sz_sq = sz**2
        sr = np.tan(tilt) * sz
        cumsr_sz = np.log(sz_sq / (sr**2 + sz_sq)) / np.log(sz_sq)
        return cumsr_sz


class ZeroInplaneSpinsPrior(FixedPrior):
    """Set inplane spins to zero."""
    standard_par_dic = {'s1x_n': 0,
                        's1y_n': 0,
                        's2x_n': 0,
                        's2y_n': 0}
