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


class UniformDiskInplaneSpinsPrior(UniformPriorMixin, Prior):
    """
    Spin prior for in-plane spins that is uniform in the disk
        sx^2 + sy^2 < 1 - sz^2
    for each of the component spins.
    """
    standard_params = ['s1x', 's1y', 's2x', 's2y']
    range_dic = {'cums1r_s1z': (0, 1),
                 's1phi_hat': (0, 2*np.pi),
                 'cums2r_s2z': (0, 1),
                 's2phi_hat': (0, 2*np.pi)}
    periodic_params = ['s1phi_hat', 's2phi_hat']
    conditioned_on = ['s1z', 's2z', 'phi_ref', 'iota']

    @staticmethod
    def _spin_transform(cumsr_sz, sphi_hat, sz, phi_ref, iota):
        sphi = (sphi_hat - phi_ref - np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        sr = np.sqrt(cumsr_sz * (1 - sz ** 2))
        sx = sr * np.cos(sphi)
        sy = sr * np.sin(sphi)
        return sx, sy

    def transform(self, cums1r_s1z, s1phi_hat, cums2r_s2z, s2phi_hat,
                  s1z, s2z, phi_ref, iota):
        """Spin prior cumulatives to spin components."""
        s1x, s1y = self._spin_transform(cums1r_s1z, s1phi_hat, s1z, phi_ref,
                                        iota)
        s2x, s2y = self._spin_transform(cums2r_s2z, s2phi_hat, s2z, phi_ref,
                                        iota)
        return {'s1x': s1x,
                's1y': s1y,
                's2x': s2x,
                's2y': s2y}

    @staticmethod
    def _inverse_spin_transform(sx, sy, sz, phi_ref, iota):
        sr = np.sqrt(sx**2 + sy**2)
        sphi = np.arctan2(sy, sx)
        cumsr_sz = sr**2 / (1-sz**2)
        sphi_hat = (sphi + phi_ref + np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        return cumsr_sz, sphi_hat

    def inverse_transform(
            self, s1x, s1y, s2x, s2y, s1z, s2z, phi_ref, iota):
        """Spin components to spin prior cumulatives."""
        cums1r_s1z, s1phi_hat = self._inverse_spin_transform(s1x, s1y, s1z,
                                                             phi_ref, iota)
        cums2r_s2z, s2phi_hat = self._inverse_spin_transform(s2x, s2y, s2z,
                                                             phi_ref, iota)
        return {'cums1r_s1z': cums1r_s1z,
                's1phi_hat': s1phi_hat,
                'cums2r_s2z': cums2r_s2z,
                's2phi_hat': s2phi_hat}


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
    sz_grid = np.linspace(-1, 1, 2000)
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


class IsotropicSpinsInplaneComponentsPrior(UniformPriorMixin, Prior):
    """
    Spin prior uniform in magnitude and solid angle (isotropic)
    for each of the constituent spins independently when combined
    with IsotropicSpinsAlignedComponentsPrior.
    The sampled parameters are `cums1r_s1z`, `cums1r_s1z` ~ U(0, 1)
    and (periodic) `s1phi_hat', `s2phi_hat' ~ U(0, 2*np.pi),
    conditioned on s1z, s2z, phi_ref, iota.
    """
    standard_params = ['s1x', 's1y', 's2x', 's2y']
    range_dic = {'cums1r_s1z': (0, 1),
                 's1phi_hat': (0, 2*np.pi),
                 'cums2r_s2z': (0, 1),
                 's2phi_hat': (0, 2*np.pi)}
    periodic_params = ['s1phi_hat', 's2phi_hat']
    conditioned_on = ['s1z', 's2z', 'phi_ref', 'iota']

    @staticmethod
    def _spin_transform(cumsr_sz, sphi_hat, sz, phi_ref, iota):
        """get (sx, sy) from (cumsr_sz, sphi_hat, sz, phi_ref, iota)"""
        sphi = (sphi_hat - phi_ref - np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        sr = np.sqrt(sz**2 * (1 / (sz**2)**cumsr_sz - 1))
        sx = sr * np.cos(sphi)
        sy = sr * np.sin(sphi)
        return sx, sy

    def transform(self, cums1r_s1z, s1phi_hat, cums2r_s2z, s2phi_hat,
                  s1z, s2z, phi_ref, iota):
        """Spin prior cumulatives to spin components."""
        s1x, s1y = self._spin_transform(cums1r_s1z, s1phi_hat, s1z, phi_ref,
                                        iota)
        s2x, s2y = self._spin_transform(cums2r_s2z, s2phi_hat, s2z, phi_ref,
                                        iota)
        return {'s1x': s1x,
                's1y': s1y,
                's2x': s2x,
                's2y': s2y}

    @staticmethod
    def _inverse_spin_transform(sx, sy, sz, phi_ref, iota):
        """(cumsr_sz, sphi_hat) from (sx, sy, sz, phi_ref, iota)."""
        sz_sq = sz**2
        cumsr_sz = np.log(sz_sq / (sx**2 + sy**2 + sz_sq)) / np.log(sz_sq)
        sphi = np.arctan2(sy, sx)
        sphi_hat = (sphi + phi_ref + np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        return cumsr_sz, sphi_hat

    def inverse_transform(
            self, s1x, s1y, s2x, s2y, s1z, s2z, phi_ref, iota):
        """Spin components to spin prior cumulatives."""
        cums1r_s1z, s1phi_hat = self._inverse_spin_transform(s1x, s1y, s1z,
                                                             phi_ref, iota)
        cums2r_s2z, s2phi_hat = self._inverse_spin_transform(s2x, s2y, s2z,
                                                             phi_ref, iota)
        return {'cums1r_s1z': cums1r_s1z,
                's1phi_hat': s1phi_hat,
                'cums2r_s2z': cums2r_s2z,
                's2phi_hat': s2phi_hat}


class IsotropicInclinationUniformDiskInplaneSpinsPrior(
        UniformPriorMixin, Prior):
    """
    Prior for in-plane spins and inclination that is uniform in the disk
        sx^2 + sy^2 < 1 - sz^2
    for each of the component spins and isotropic in the inclination.
    """
    standard_params = ['iota', 's1x', 's1y', 's2x', 's2y']
    range_dic = {'costheta_jn': (-1, 1),
                 'phi_jl_hat': (0, 2*np.pi),
                 'phi12': (0, 2*np.pi),
                 'cums1r_s1z': (0, 1),
                 'cums2r_s2z': (0, 1)}
    periodic_params = ['phi_jl_hat', 'phi12']
    folded_params = ['costheta_jn']
    conditioned_on = ['s1z', 's2z', 'phi_ref', 'm1', 'm2', 'f_ref']

    @staticmethod
    def _spin_transform(cumsr_sz, sz):
        sr = np.sqrt(cumsr_sz * (1 - sz ** 2))
        chi = np.sqrt(sr**2 + sz**2)
        tilt = np.arctan2(sr, sz)
        return chi, tilt

    def transform(self, costheta_jn, phi_jl_hat, phi12, cums1r_s1z,
                  cums2r_s2z, s1z, s2z, phi_ref, m1, m2, f_ref):
        """Spin prior cumulatives to spin components."""
        chi1, tilt1 = self._spin_transform(cums1r_s1z, s1z)
        chi2, tilt2 = self._spin_transform(cums2r_s2z, s2z)
        theta_jn = np.arccos(costheta_jn)
        phi_jl = (phi_jl_hat + np.pi * (costheta_jn < 0)) % (2*np.pi)

        iota, s1x, s1y, s1z, s2x, s2y, s2z \
            = lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2,
                m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_ref, phi_ref)

        return {'iota': iota,
                's1x': s1x,
                's1y': s1y,
                's2x': s2x,
                's2y': s2y}

    @staticmethod
    def _inverse_spin_transform(chi, tilt, sz):
        """
        Return value of `cumsr_sz`, the cumulative of the prior on
        in-plane spin magnitude given the aligned spin magnitude `sz`,
        for either companion.
        The in-plane spin prior is flat in the disk.
        """
        cumsr_sz = (chi*np.sin(tilt))**2 / (1-sz**2)
        return cumsr_sz

    def inverse_transform(self, iota, s1x, s1y, s2x, s2y, s1z, s2z,
                          phi_ref, m1, m2, f_ref):
        """
        Inclination and spin components to theta_jn, phi_jl, phi12 and
        inplane-spin-magnitude prior cumulatives.
        """
        theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 \
            = lalsimulation.SimInspiralTransformPrecessingWvf2PE(
                iota, s1x, s1y, s1z, s2x, s2y, s2z, m1, m2, f_ref, phi_ref)

        cums1r_s1z = self._inverse_spin_transform(chi1, tilt1, s1z)
        cums2r_s2z = self._inverse_spin_transform(chi2, tilt2, s2z)

        costheta_jn = np.cos(theta_jn)
        phi_jl_hat = (phi_jl + np.pi * (costheta_jn < 0)) % (2*np.pi)

        return {'costheta_jn': costheta_jn,
                'phi_jl_hat': phi_jl_hat,
                'phi12': phi12,
                'cums1r_s1z': cums1r_s1z,
                'cums2r_s2z': cums2r_s2z}


class ZeroInplaneSpinsPrior(FixedPrior):
    """Set inplane spins to zero."""
    standard_par_dic = {'s1x': 0,
                        's1y': 0,
                        's2x': 0,
                        's2y': 0}
