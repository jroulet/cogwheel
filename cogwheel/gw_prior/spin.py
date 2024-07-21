"""
Default modular priors for spin parameters, for convenience.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""
from abc import abstractmethod
from scipy.interpolate import interp1d
import numpy as np

import lal
import lalsimulation

from cogwheel import skyloc_angles
from cogwheel import utils
from cogwheel.prior import Prior, FixedPrior, UniformPriorMixin
from .twosquircle import TwoSquircularMapping

# pylint: disable=arguments-differ


# ----------------------------------------------------------------------
# Aligned spin components

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

    @classmethod
    def _s1z_lim(cls, chieff, q):
        s1z_min = np.maximum(cls._get_s1z(chieff, q, s2z=1), -1)
        s1z_max = np.minimum(cls._get_s1z(chieff, q, s2z=-1), 1)
        return s1z_min, s1z_max

    @classmethod
    def transform(cls, chieff, cumchidiff, m1, m2):
        """(chieff, cumchidiff) to (s1z, s2z)."""
        q = m2 / m1
        s1z_min, s1z_max = cls._s1z_lim(chieff, q)
        s1z = s1z_min + cumchidiff * (s1z_max - s1z_min)
        s2z = ((1+q)*chieff - s1z) / q
        return {'s1z': s1z,
                's2z': s2z}

    @classmethod
    def inverse_transform(cls, s1z, s2z, m1, m2):
        """(s1z, s2z) to (chieff, cumchidiff)."""
        q = m2 / m1
        chieff = (s1z + q*s2z) / (1 + q)
        s1z_min, s1z_max = cls._s1z_lim(chieff, q)
        cumchidiff = (s1z - s1z_min) / (s1z_max - s1z_min)
        return {'chieff': chieff,
                'cumchidiff': cumchidiff}


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

    @utils.lru_cache()
    def transform(self, cums1z, cums2z):
        """(cums1z, cums2z) to (s1z, s2z)."""
        return {'s1z': self._spin_transform(cums1z),
                's2z': self._spin_transform(cums2z)}

    def inverse_transform(self, s1z, s2z):
        """(s1z, s2z) to (cums1z, cums2z)."""
        return {'cums1z': self._inverse_spin_transform(s1z),
                'cums2z': self._inverse_spin_transform(s2z)}

    @classmethod
    def _spin_transform(cls, cumsz):
        return cls.sz_interp(cumsz)[()]

    @staticmethod
    def _inverse_spin_transform(sz):
        return (1 + sz - sz * np.log(np.abs(sz))) / 2


class VolumetricSpinsAlignedComponentsPrior(UniformPriorMixin, Prior):
    """
    Prior for aligned spin components corresponding to a density uniform
    in the ball |s1,2| < 1. I.e.:
        p(s1z) = p(s2z) = 3/4 * (1 - sz**2), |sz| < 1.
    """
    standard_params = ['s1z', 's2z']
    range_dic = {'cums1z': (0, 1),
                 'cums2z': (0, 1)}

    def transform(self, cums1z, cums2z):
        """Sampled parameters to standard parameters."""
        return {'s1z': self._spin_transform(cums1z),
                's2z': self._spin_transform(cums2z)}

    def inverse_transform(self, s1z, s2z):
        """Standard parameters to sampled parameters."""
        return {'cums1z': self._inverse_spin_transform(s1z),
                'cums2z': self._inverse_spin_transform(s2z)}

    @staticmethod
    def _spin_transform(cumsz):
        cumsz = np.complex128(cumsz)
        return np.real(
            (-1 + np.sqrt(3)*1j
             - (1 + np.sqrt(3)*1j)
             * (1-2*cumsz+2*((-1+cumsz)*cumsz)**.5)**(2/3))
            / (2 * (1 - 2*cumsz
                    + 2 * ((-1 + cumsz) * cumsz)**.5)**(1/3)))

    @staticmethod
    def _inverse_spin_transform(sz):
        return 1/4 * (2 + 3*sz - sz**3)


# ----------------------------------------------------------------------
# Inplane spin components + inclination (+ sky location)

class _BaseInplaneSpinsInclinationPrior(UniformPriorMixin, Prior):
    """
    Abstract base class for defining a prior on inplane spins and
    inclination, conditioned on masses, aligned spins and reference
    frequency.

    Subclasses
    ----------
    UniformDiskInplaneSpinsIsotropicInclinationPrior
    IsotropicSpinsInplaneComponentsIsotropicInclinationPrior
    """
    standard_params = ['iota', 's1x_n', 's1y_n', 's2x_n', 's2y_n']
    range_dic = {'costheta_jn': (-1, 1),
                 'phi_jl_hat': (0, 2*np.pi),
                 'phi12': (0, 2*np.pi),
                 'cums1r_s1z': (0, 1),
                 'cums2r_s2z': (0, 1)}
    periodic_params = ['phi_jl_hat', 'phi12']
    folded_reflected_params = ['costheta_jn']
    conditioned_on = ['s1z', 's2z', 'm1', 'm2', 'f_ref']

    @staticmethod
    @abstractmethod
    def _spin_transform(cumsr_sz, sz):
        """
        Subclasses must override to set the spin prior.

        Parameters
        ----------
        cumsr_sz: float
            Cumulative of the prior on in-plane spin magnitude given the
            aligned spin magnitude `sz`, for either companion.

        sz: float
            Aligned spin magnitude for either companion.

        Return
        ------
        chi: float
            Dimensionless spin magnitude between 0 and 1.

        tilt: float
            Zenithal angle between spin and orbital angular momentum
            between 0 and pi.
        """

    @staticmethod
    @abstractmethod
    def _inverse_spin_transform(chi, tilt, sz):
        """
        Inverse of `._spin_transform`. Subclasses must override to set
        the spin prior.

        Parameters
        ----------
        chi: float
            Dimensionless spin magnitude between 0 and 1.

        tilt: float
            Zenithal angle between spin and orbital angular momentum
            between 0 and pi.

        sz: float
            Aligned spin magnitude for either companion.

        Return
        ------
        cumsr_sz: float
            Cumulative of the prior on in-plane spin magnitude given the
            aligned spin magnitude `sz`, for either companion.
        """

    @utils.lru_cache()
    def transform(self, costheta_jn, phi_jl_hat, phi12, cums1r_s1z,
                  cums2r_s2z, s1z, s2z, m1, m2, f_ref) -> dict:
        """
        Return dictionary with inclination and inplane spins.
        Spin components are defined in a coordinate system where `z` is
        parallel to the orbital angular momentum `L` and the direction
        of propagation `N` lies in the `y-z` plane.
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
                          s1z, s2z, m1, m2, f_ref) -> dict:
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


class UniformDiskInplaneSpinsIsotropicInclinationPrior(
        _BaseInplaneSpinsInclinationPrior):
    """
    Prior for in-plane spins and inclination that is uniform in the disk
        sx^2 + sy^2 < 1 - sz^2
    for each of the component spins and isotropic in the inclination.
    It corresponds to the IAS spin prior when combined with
    `UniformEffectiveSpinPrior`.
    """
    @staticmethod
    def _spin_transform(cumsr_sz, sz):
        sr = np.sqrt(cumsr_sz * (1 - sz ** 2))
        chi = np.sqrt(sr**2 + sz**2)
        tilt = np.arctan2(sr, sz)
        return chi, tilt

    @staticmethod
    def _inverse_spin_transform(chi, tilt, sz):
        cumsr_sz = (chi*np.sin(tilt))**2 / (1-sz**2)
        return cumsr_sz


class IsotropicSpinsInplaneComponentsIsotropicInclinationPrior(
        _BaseInplaneSpinsInclinationPrior):
    """
    Prior for in-plane spins and inclination that is uniform in
    magnitude and solid angle (isotropic) for each of the constituent
    spins independently when combined with
    `IsotropicSpinsAlignedComponentsPrior`.
    """
    @staticmethod
    def _spin_transform(cumsr_sz, sz):
        sr = np.sqrt(sz**2 * (1 / (sz**2)**cumsr_sz - 1))
        chi = np.sqrt(sr**2 + sz**2)
        tilt = np.arctan2(sr, sz)
        return chi, tilt

    @staticmethod
    def _inverse_spin_transform(chi, tilt, sz):
        sz_square = sz ** 2
        sr = np.tan(tilt) * sz
        cumsr_sz = np.log(sz_square / (sr**2 + sz_square)) / np.log(sz_square)
        return cumsr_sz


class _BaseSkyLocationPrior(UniformPriorMixin, Prior):
    """
    Abstract base class for adding sky location parameters to a prior
    that already describes inplane spins and inclination.
    The need for this class arises because the coordinates we use for
    sky location depend on the sign of cos(theta_jn).
    Subclasses must override the class attribute
    ``._inplane_spin_inclination_prior_class``.

    Subclasses
    ----------
    UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior
    IsotropicSpinsInplaneComponentsIsotropicInclinationSkyLocationPrior
    """
    standard_params = ['iota', 's1x_n', 's1y_n', 's2x_n', 's2y_n', 'ra', 'dec']
    range_dic = {'costheta_jn': (-1, 1),
                 'phi_jl_hat': (0, 2*np.pi),
                 'phi12': (0, 2*np.pi),
                 'cums1r_s1z': (0, 1),
                 'cums2r_s2z': (0, 1),
                 'costhetanet': (-1, 1),
                 'phinet_hat': (0, 2*np.pi)}
    periodic_params = ['phi_jl_hat', 'phi12']
    folded_reflected_params = ['costheta_jn', 'phinet_hat']
    conditioned_on = ['s1z', 's2z', 'm1', 'm2', 'f_ref']

    @staticmethod
    @utils.ClassProperty
    @abstractmethod
    def _inplane_spin_inclination_prior_class():
        """
        ``UniformDiskInplaneSpinsIsotropicInclinationPrior`` or
        ``IsotropicSpinsInplaneComponentsIsotropicInclinationPrior``.
        """

    def __init__(self, *, detector_pair, tgps, **kwargs):
        super().__init__(detector_pair=detector_pair, tgps=tgps,
                         **kwargs)
        self._inplane_spin_inclination_prior \
            = self._inplane_spin_inclination_prior_class()

        self.skyloc = skyloc_angles.SkyLocAngles(detector_pair, tgps)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return self.skyloc.get_init_dict()

    @utils.lru_cache()
    def transform(self, costheta_jn, phi_jl_hat, phi12,
                  cums1r_s1z, cums2r_s2z, costhetanet, phinet_hat,
                  s1z, s2z, m1, m2, f_ref):
        """
        Return dictionary with inclination, inplane spins, right
        ascension and declination. Spin components are defined in a
        coordinate system where `z` is parallel to the orbital angular
        momentum `L` and the direction of propagation `N` lies in the
        `y-z` plane.
        """
        iota_inplane_spins = self._inplane_spin_inclination_prior.transform(
            costheta_jn, phi_jl_hat, phi12, cums1r_s1z, cums2r_s2z, s1z, s2z,
            m1, m2, f_ref)

        thetanet = np.arccos(costhetanet)
        phinet = (phinet_hat - np.pi*(costheta_jn > 0)) % (2*np.pi)
        ra, dec = self.skyloc.thetaphinet_to_radec(thetanet, phinet)

        return iota_inplane_spins | {'ra': ra, 'dec': dec}

    def inverse_transform(self, iota, s1x_n, s1y_n, s2x_n, s2y_n,
                          ra, dec, s1z, s2z, m1, m2, f_ref):
        """`standard_params` to `sampled_params`."""
        costheta_jn_sampled_inplane_spins \
            = self._inplane_spin_inclination_prior.inverse_transform(
                iota, s1x_n, s1y_n, s2x_n, s2y_n, s1z, s2z, m1, m2, f_ref)

        costheta_jn = costheta_jn_sampled_inplane_spins['costheta_jn']

        thetanet, phinet = self.skyloc.radec_to_thetaphinet(ra, dec)
        costhetanet = np.cos(thetanet)
        phinet_hat = (phinet + np.pi*(costheta_jn > 0)) % (2*np.pi)

        return costheta_jn_sampled_inplane_spins | {'costhetanet': costhetanet,
                                                    'phinet_hat': phinet_hat}


class UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior(
        _BaseSkyLocationPrior):
    """
    Prior for in-plane spins, inclination and sky location.
    It is uniform in the disk
        sx^2 + sy^2 < 1 - sz^2
    for each of the component spins and isotropic in the inclination
    and sky location.
    It corresponds to the IAS spin prior when combined with
    `UniformEffectiveSpinPrior`.
    """
    _inplane_spin_inclination_prior_class \
         = UniformDiskInplaneSpinsIsotropicInclinationPrior


class IsotropicSpinsInplaneComponentsIsotropicInclinationSkyLocationPrior(
        _BaseSkyLocationPrior):
    """
    Prior for in-plane spins, inclination and sky location that is
    uniform in magnitude and solid angle (isotropic) for each of the
    constituent spins independently when combined with
    `IsotropicSpinsAlignedComponentsPrior`. It is isotropic in
    inclination and sky location.
    """
    _inplane_spin_inclination_prior_class \
         = IsotropicSpinsInplaneComponentsIsotropicInclinationPrior


class CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior(Prior):
    """
    Similar to UniformDiskInplaneSpinsIsotropicInclinationPrior except
    it uses Cartesian rather than polar coordinates.
    It might behave better near the origin (i.e. aligned spins) as it
    naturally enforces the density to be azimuth-independent there.

    The variables are defined with the following meaning:
        (u1, v1) cartesian ≡ (sqrt(cums1r_s1z), phi_jl_hat) polar
        (u2, v2) cartesian ≡ (sqrt(cums2r_s2z), phi12) polar
    u, v pairs live in the disk u^2+v^2 < 1.
    These are in turn mapped to squares (x1, y1) and (x2, y2) via
    2-squircular mappings.
    """
    standard_params = ['iota', 's1x_n', 's1y_n', 's2x_n', 's2y_n']
    range_dic = {'costheta_jn': (-1, 1),
                 'x1': (-1, 1),
                 'y1': (-1, 1),
                 'x2': (-1, 1),
                 'y2': (-1, 1)}
    folded_reflected_params = ['costheta_jn']
    conditioned_on = ['s1z', 's2z', 'm1', 'm2', 'f_ref']

    _polar_prior = UniformDiskInplaneSpinsIsotropicInclinationPrior()
    _mapping = TwoSquircularMapping()

    def transform(self, costheta_jn, x1, y1, x2, y2,
                  s1z, s2z, m1, m2, f_ref):
        """`sampled_params` to `standard_params`."""
        cums1r_s1z, phi_jl_hat = self._cartesian_to_polar(x1, y1)
        cums2r_s2z, phi12 = self._cartesian_to_polar(x2, y2)

        return self._polar_prior.transform(costheta_jn, phi_jl_hat, phi12,
                                           cums1r_s1z, cums2r_s2z,
                                           s1z, s2z, m1, m2, f_ref)

    def inverse_transform(self, iota, s1x_n, s1y_n, s2x_n, s2y_n,
                          s1z, s2z, m1, m2, f_ref):
        """`standard_params` to `sampled_params`."""
        polar_dic = self._polar_prior.inverse_transform(
            iota, s1x_n, s1y_n, s2x_n, s2y_n, s1z, s2z, m1, m2, f_ref)
        x1, y1 = self._polar_to_cartesian(polar_dic['cums1r_s1z'],
                                          polar_dic['phi_jl_hat'])
        x2, y2 = self._polar_to_cartesian(polar_dic['cums2r_s2z'],
                                          polar_dic['phi12'])
        return {'costheta_jn': polar_dic['costheta_jn'],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2}

    def lnprior(self, costheta_jn, x1, y1, x2, y2,
                s1z, s2z, m1, m2, f_ref):
        """Log prior density in the space of sampled parameters."""
        del costheta_jn, s1z, s2z, m1, m2, f_ref
        return np.log(0.5
                      * self._mapping.jacobian_determinant(x1, y1)
                      * self._mapping.jacobian_determinant(x2, y2))

    def _cartesian_to_polar(self, x, y):
        u, v = self._mapping.square_to_disk(x, y)
        cumr = u**2 + v**2
        phi = np.arctan2(v, u)
        return cumr, phi

    def _polar_to_cartesian(self, cumr, phi):
        r = np.sqrt(cumr)
        u = r * np.cos(phi)
        v = r * np.sin(phi)
        return self._mapping.disk_to_square(u, v)


class ZeroInplaneSpinsPrior(FixedPrior):
    """Set inplane spins to zero."""
    standard_par_dic = {'s1x_n': 0.,
                        's1y_n': 0.,
                        's2x_n': 0.,
                        's2y_n': 0.}
