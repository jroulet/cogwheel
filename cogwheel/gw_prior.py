"""
Priors for parameter estimation of compact binary mergers.
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import interp1d

import lal

from . import gw_utils
from . import skyloc_angles
from . import cosmology as cosmo

from . prior import Prior, CombinedPrior, FixedPrior, UniformPriorMixin, \
    IdentityTransformMixin, check_inheritance_order


prior_registry = {}


class GWPriorError(Exception):
    """Base class for all exceptions in this module"""


class RegisteredPriorMixin:
    """
    Register existence of a `Prior` subclass in `prior_registry`.
    Intended usage is to only register the final priors (i.e., for the
    full set of GW parameters).
    `RegisteredPriorMixin` should be inherited after `Prior` (otherwise
    `PriorError` is raised) so that abstract methods get overriden.
    """
    def __init_subclass__(cls):
        """Validate subclass and register it in prior_registry."""
        super().__init_subclass__()
        check_inheritance_order(cls, Prior, RegisteredPriorMixin)

        if cls.conditioned_on:
            raise GWPriorError('Only register fully defined priors.')

        prior_registry[cls.__name__] = cls


class ReferenceDetectorMixin:
    """
    Methods for priors that need to know about the reference detector.
    They must have `tgps` and `ref_det_name` attributes.
    """
    @property
    def ref_det_location(self):
        """3d location of the reference detector on Earth [meters]."""
        return gw_utils.DETECTORS[self.ref_det_name].location

    def time_delay_refdet(self, ra, dec):
        """Time delay from Earth center to the reference detector."""
        return lal.TimeDelayFromEarthCenter(self.ref_det_location, ra, dec,
                                            self.tgps)

    def fplus_fcross_refdet(self, ra, dec, psi):
        """Antenna coefficients (F+, Fx) at the reference detector."""
        return gw_utils.fplus_fcross(self.ref_det_name, ra, dec, psi,
                                     self.tgps)[:, 0]

    def geometric_factor_refdet(self, ra, dec, psi, iota):
        """
        Return the complex geometric factor
            R = (1+cos^2(iota)) Fp / 2 + i cos(iota) Fc
        that relates a waveform with generic orientation to an overhead
        face-on one to leading post-Newtonian order.
        """
        fplus, fcross = self.fplus_fcross_refdet(ra, dec, psi)
        return (1 + np.cos(iota)**2) / 2 * fplus + 1j * np.cos(iota) * fcross

    def psi_refdet(self, vphi, iota, ra, dec):
        """
        Find psi such that arg(R e^(-i 2 vphi)) == 0 at the reference
        detector, where R = (1+cos^2(iota)) Fp / 2 + i cos(iota) Fc.
        """
        fp0, fc0 = self.fplus_fcross_refdet(ra, dec, psi=0)
        cosiota = np.cos(iota)
        a = 2 * cosiota * np.cos(2*vphi)
        b = (1+cosiota**2) * np.sin(2*vphi)
        delta = np.pi * (cosiota * (fp0*a + fc0*b) < 0)  # 0 or pi
        return .5 * (np.arctan((fc0*a - fp0*b) / (fp0*a + fc0*b)) + delta)


# ----------------------------------------------------------------------
# Default modular priors for a subset of the variables, for convenience.
# They can be combined later just by subclassing `CombinedPrior` and
# defining an attribute `prior_classes` that is a list of these.
# Each may consume some arguments in the __init__(), but should forward
# as `**kwargs` any arguments that other priors may need.

class UniformDetectorFrameMassesPrior(Prior):
    standard_params = ['m1', 'm2']
    range_dic = {'mchirp': NotImplemented,
                 'lnq': NotImplemented}

    def __init__(self, *, mchirp_range, q_min, symmetrize_lnq=False,
                 **kwargs):
        lnq_min = np.log(q_min)
        self.range_dic = {'mchirp': mchirp_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq)}
        super().__init__(**kwargs)

        self.prior_norm = 1
        self.prior_norm = dblquad(
            lambda mchirp, lnq: np.exp(self.lnprior(mchirp, lnq)),
            *self.range_dic['lnq'], *self.range_dic['mchirp'])[0]

    @staticmethod
    def transform(mchirp, lnq):
        """(mchirp, lnq) to (m1, m2)."""
        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q)**.2 / q**.6
        return {'m1': m1,
                'm2': m1 * q}

    @staticmethod
    def inverse_transform(m1, m2):
        """
        (m1, m2) to (mchirp, lnq). Note that if symmetrize_lnq==True the
        transformation is not invertible. Here, lnq <= 0 always.
        """
        q = m2 / m1
        lnq = np.log(q)
        mchirp = m1 * q**.6 / (1 + q)**.2
        return {'mchirp': mchirp,
                'lnq': lnq}

    def lnprior(self, mchirp, lnq):
        return np.log(mchirp * np.cosh(lnq/2)**.4 / self.prior_norm)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'mchirp_range': self.range_dic['mchirp'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}


class UniformPhasePrior(UniformPriorMixin, IdentityTransformMixin, Prior):
    """Uniform prior for the orbital phase. No change of coordinates."""
    standard_params = ['vphi']
    range_dic = {'vphi': (0, 2*np.pi)}
    periodic_params = ['vphi']


class IsotropicInclinationPrior(UniformPriorMixin, Prior):
    """Uniform-in-cosine prior for the binary's inclination."""
    standard_params = ['iota']
    range_dic = {'cosiota': (-1, 1)}
    folded_params = ['cosiota']

    def transform(self, cosiota):
        """cos(inclination) to inclination."""
        return {'iota': np.arccos(cosiota)}

    def inverse_transform(self, iota):
        """Inclination to cos(inclination)."""
        return {'cosiota': np.cos(iota)}


class IsotropicSkyLocationPrior(UniformPriorMixin, Prior):
    """
    Isotropic prior for the sky localization angles.
    The angles sampled are fixed to Earth and defined with respect to a
    pair of detectors: the polar angle `thetanet` is with respect to the
    line connecting the two detectors, and the azimuthal angle `phinet`
    with respect to the horizon at the midpoint between the two
    detectors. These are transformed to the standard `(ra, dec)`
    """
    standard_params = ['ra', 'dec']
    range_dic = {'costhetanet': (-1, 1),
                 'phinet_hat': (0, 2*np.pi)}
    conditioned_on = ['iota']
    folded_params = ['phinet_hat']

    def __init__(self, *, detector_pair, tgps, **kwargs):
        super().__init__(detector_pair=detector_pair, tgps=tgps,
                         **kwargs)
        self.skyloc = skyloc_angles.SkyLocAngles(detector_pair, tgps)

    def transform(self, costhetanet, phinet_hat, iota):
        """Network sky angles to right ascension and declination."""
        thetanet = np.arccos(costhetanet)
        phinet = (phinet_hat - np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        ra, dec = self.skyloc.thetaphinet_to_radec(thetanet, phinet)
        return {'ra': ra,
                'dec': dec}

    def inverse_transform(self, ra, dec, iota):
        """Right ascension and declination to network sky angles."""
        thetanet, phinet = self.skyloc.radec_to_thetaphinet(ra, dec)
        costhetanet = np.cos(thetanet)
        phinet_hat = (phinet + np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        return {'costhetanet': costhetanet,
                'phinet_hat': phinet_hat}

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return self.skyloc.get_init_dict()


class UniformTimePrior(ReferenceDetectorMixin, UniformPriorMixin, Prior):
    """Prior for the time of arrival at a reference detector."""
    standard_params = ['t_geocenter']
    range_dic = {'t_refdet': NotImplemented}
    conditioned_on = ['ra', 'dec']

    def __init__(self, *, tgps, ref_det_name, t0_refdet=0, dt0=.07,
                 **kwargs):
        self.range_dic = {'t_refdet': (t0_refdet - dt0, t0_refdet + dt0)}
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)

        self.tgps = tgps
        self.ref_det_name = ref_det_name

    def transform(self, t_refdet, ra, dec):
        return {'t_geocenter': t_refdet - self.time_delay_refdet(ra, dec)}

    def inverse_transform(self, t_geocenter, ra, dec):
        return {'t_refdet': t_geocenter + self.time_delay_refdet(ra, dec)}

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'t0_refdet': np.mean(self.range_dic['t_refdet']),
                'dt0': np.diff(self.range_dic['t_refdet'])[0] / 2,
                'tgps': self.tgps,
                'ref_det_name': self.ref_det_name}


class UniformPolarizationPrior(ReferenceDetectorMixin, UniformPriorMixin,
                               Prior):
    """
    Prior for the polarization.
    The sampled variable `psi_hat` differs from the standard
    polarization `psi` by an inclination-dependent sign and an additive
    function of `vphi, iota, ra, dec`, such that it describes the well-
    measured phase of the waveform at a reference detector.
    """
    standard_params = ['psi']
    range_dic = {'psi_hat': (0, np.pi)}
    periodic_params = ['psi_hat']
    conditioned_on = ['ra', 'dec', 'iota', 'vphi', 't_geocenter']

    def __init__(self, *, tgps, ref_det_name, f_ref, **kwargs):
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, f_ref=f_ref,
                         **kwargs)
        self.tgps = tgps
        self.ref_det_name = ref_det_name
        self.f_ref = f_ref

    def transform(self, psi_hat, ra, dec, iota, vphi, t_geocenter):
        """psi_hat to psi."""
        psi_refdet = self.psi_refdet(vphi, iota, ra, dec)
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        psi = ((psi_hat + np.pi*self.f_ref*t_refdet) * np.sign(np.cos(iota))
               + psi_refdet) % np.pi
        return {'psi': psi}

    def inverse_transform(self, psi, ra, dec, iota, vphi, t_geocenter):
        """psi to psi_hat"""
        psi_refdet = self.psi_refdet(vphi, iota, ra, dec)
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        psi_hat = ((psi - psi_refdet) * np.sign(np.cos(iota))
                   - np.pi*self.f_ref*t_refdet) % np.pi
        return {'psi_hat': psi_hat}

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'tgps': self.tgps,
                'ref_det_name': self.ref_det_name,
                'f_ref': self.f_ref}


class UniformLuminosityVolumePrior(ReferenceDetectorMixin, Prior):
    """
    Distance prior uniform in luminosity volume and detector-frame time.
    The sampled parameter is
        d_hat := d_effective / mchirp
    where the effective distance is defined in one "reference" detector.
    """
    standard_params = ['d_luminosity']
    range_dic = {'d_hat': NotImplemented}
    conditioned_on = ['ra', 'dec', 'psi', 'iota', 'm1', 'm2']

    def __init__(self, *, tgps, ref_det_name, d_hat_max=500, **kwargs):
        self.range_dic = {'d_hat': (0, d_hat_max)}
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)

        self.tgps = tgps
        self.ref_det_name = ref_det_name

    def _conversion_factor(self, ra, dec, psi, iota, m1, m2):
        """
        Return conversion factor such that
            d_luminosity = d_hat * conversion_factor.
        """
        mchirp = (m1*m2)**.6 / (m1+m2)**.2
        amplitude = np.abs(self.geometric_factor_refdet(ra, dec, psi, iota))
        return mchirp * amplitude

    def transform(self, d_hat, ra, dec, psi, iota, m1, m2):
        """d_hat to d_luminosity"""
        return {'d_luminosity': d_hat * self._conversion_factor(ra, dec, psi,
                                                                iota, m1, m2)}

    def inverse_transform(self, d_luminosity, ra, dec, psi, iota, m1, m2):
        """d_luminosity to d_hat"""
        return {'d_hat': d_luminosity / self._conversion_factor(ra, dec, psi,
                                                                iota, m1, m2)}

    def lnprior(self, d_hat, ra, dec, psi, iota, m1, m2):
        """
        Natural log of the prior probability density for d_hat.
        This prior is not normalized, as that would need to know
        the masses' integration region.
        """
        return np.log(self._conversion_factor(ra, dec, psi, iota, m1, m2)**3
                      * d_hat**2)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'tgps': self.tgps,
                'ref_det_name': self.ref_det_name,
                'd_hat_max': self.range_dic['d_hat'][1]}

class UniformComovingVolumePrior(UniformLuminosityVolumePrior):
    """
    Distance prior uniform in comoving volume-time.
    The sampled parameter is
        d_hat := d_effective / mchirp
    where the effective distance is defined in one "reference" detector.
    """
    def lnprior(self, d_hat, ra, dec, psi, iota, m1, m2):
        """
        Natural log of the prior probability density for d_hat.
        This prior is not normalized, as that would need to know
        the masses' integration region.
        """
        d_luminosity = d_hat * self._conversion_factor(ra, dec, psi, iota, m1, m2)
        z = cosmo.z_of_DL_Mpc(d_luminosity)
        cosmo_weight = ((1 - d_luminosity * cosmo.dz_dDL(d_luminosity) / (1 + z))
                        / (1 + z)**4)
        return np.log(cosmo_weight * d_luminosity**3 / d_hat)


class FlatChieffPrior(UniformPriorMixin, Prior):
    """
    Spin prior for aligned spins that is flat in effective spin chieff.
    The sampled parameters are `chieff` and `cumchidiff`.
    `cumchidiff` ranges from 0 to 1 and is typically poorly measured.
    cumchidiff is derived from
        chidiff = (q*s1z-s2z) / (1+q),
    but rescaled to [0, 1].
    In other words, cumchidiff is the cumulative of a uniform
    prior(chidiff | chieff, q).
    """
    standard_params = ['s1z', 's2z']
    range_dic = {'chieff': (-1, 1),
                 'cumchidiff': (0, 1)}
    conditioned_on = ['m1', 'm2']

    @staticmethod
    def _aux_chidiff_lim(s1, s2, s3, s4, chieff, q):
        chi0 = (1-q) / (1+q)
        return ((s2*chi0 - s4)/(s1 - s3*chi0)*chieff + s2*chi0
                - s1*(s2*chi0 - s4)/(s1 - s3*chi0))

    def _chidiff_lim(self, chieff, q):
        chidiffmin = np.maximum(self._aux_chidiff_lim(1, -1, -1, -1, chieff, q),
                                self._aux_chidiff_lim(-1, 1, -1, -1, chieff, q))
        chidiffmax = np.minimum(self._aux_chidiff_lim(1, -1, 1, 1, chieff, q),
                                self._aux_chidiff_lim(-1, 1, 1, 1, chieff, q))
        return chidiffmin, chidiffmax

    def transform(self, chieff, cumchidiff, m1, m2):
        """(chieff, cumchidiff) to (s1z, s2z)."""
        q = m2 / m1
        chidiffmin, chidiffmax = self._chidiff_lim(chieff, q)
        chidiff = chidiffmin + cumchidiff * (chidiffmax - chidiffmin)
        s1z = (1 + q) / (1 + q**2) * (q*chidiff + chieff)
        s2z = (1 + q) / (1 + q**2) * (q*chieff - chidiff)
        return {'s1z': s1z,
                's2z': s2z}

    def inverse_transform(self, s1z, s2z, m1, m2):
        """(s1z, s2z) to (chieff, cumchidiff)."""
        q = m2 / m1
        chieff = (s1z + q*s2z) / (1 + q)
        chidiffmin, chidiffmax = self._chidiff_lim(chieff, q)
        chidiff = (q*s1z - s2z) / (1 + q)
        cumchidiff = (chidiff - chidiffmin) / (chidiffmax - chidiffmin)
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
    conditioned_on = ['s1z', 's2z', 'vphi', 'iota']

    @staticmethod
    def _spin_transform(cumsr_sz, sphi_hat, sz, vphi, iota):
        sphi = (sphi_hat - vphi - np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        sr = np.sqrt(cumsr_sz * (1 - sz ** 2))
        sx = sr * np.cos(sphi)
        sy = sr * np.sin(sphi)
        return sx, sy

    def transform(self, cums1r_s1z, s1phi_hat, cums2r_s2z, s2phi_hat,
                  s1z, s2z, vphi, iota):
        """Spin prior cumulatives to spin components."""
        s1x, s1y = self._spin_transform(cums1r_s1z, s1phi_hat, s1z, vphi, iota)
        s2x, s2y = self._spin_transform(cums2r_s2z, s2phi_hat, s2z, vphi, iota)
        return {'s1x': s1x,
                's1y': s1y,
                's2x': s2x,
                's2y': s2y}

    @staticmethod
    def _inverse_spin_transform(sx, sy, sz, vphi, iota):
        sr = np.sqrt(sx**2 + sy**2)
        sphi = np.arctan2(sy, sx)
        cumsr_sz = sr**2 / (1-sz**2)
        sphi_hat = (sphi + vphi + np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        return cumsr_sz, sphi_hat

    def inverse_transform(self, s1x, s1y, s2x, s2y, s1z, s2z, vphi, iota):
        """Spin components to spin prior cumulatives."""
        cums1r_s1z, s1phi_hat = self._inverse_spin_transform(s1x, s1y, s1z,
                                                             vphi, iota)
        cums2r_s2z, s2phi_hat = self._inverse_spin_transform(s2x, s2y, s2z,
                                                             vphi, iota)
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
    conditioned on s1z, s2z, vphi, iota.
    """
    standard_params = ['s1x', 's1y', 's2x', 's2y']
    range_dic = {'cums1r_s1z': (0, 1),
                 's1phi_hat': (0, 2*np.pi),
                 'cums2r_s2z': (0, 1),
                 's2phi_hat': (0, 2*np.pi)}
    periodic_params = ['s1phi_hat', 's2phi_hat']
    conditioned_on = ['s1z', 's2z', 'vphi', 'iota']

    @staticmethod
    def _spin_transform(cumsr_sz, sphi_hat, sz, vphi, iota):
        """get (sx, sy) from (cumsr_sz, sphi_hat, sz, vphi, iota)"""
        sphi = (sphi_hat - vphi - np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        sr = np.sqrt(sz**2 * (1 / (sz**2)**cumsr_sz - 1))
        sx = sr * np.cos(sphi)
        sy = sr * np.sin(sphi)
        return sx, sy

    def transform(self, cums1r_s1z, s1phi_hat, cums2r_s2z, s2phi_hat,
                  s1z, s2z, vphi, iota):
        """Spin prior cumulatives to spin components."""
        s1x, s1y = self._spin_transform(cums1r_s1z, s1phi_hat, s1z, vphi, iota)
        s2x, s2y = self._spin_transform(cums2r_s2z, s2phi_hat, s2z, vphi, iota)
        return {'s1x': s1x,
                's1y': s1y,
                's2x': s2x,
                's2y': s2y}

    @staticmethod
    def _inverse_spin_transform(sx, sy, sz, vphi, iota):
        """get (cumsr_sz, sphi_hat) from (sx, sy, sz, vphi, iota)"""
        sz_sq = sz**2
        cumsr_sz = np.log(sz_sq / (sx**2 + sy**2 + sz_sq)) / np.log(sz_sq)
        sphi = np.arctan2(sy, sx)
        sphi_hat = (sphi + vphi + np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        return cumsr_sz, sphi_hat

    def inverse_transform(self, s1x, s1y, s2x, s2y, s1z, s2z, vphi, iota):
        """Spin components to spin prior cumulatives."""
        cums1r_s1z, s1phi_hat = self._inverse_spin_transform(s1x, s1y, s1z,
                                                             vphi, iota)
        cums2r_s2z, s2phi_hat = self._inverse_spin_transform(s2x, s2y, s2z,
                                                             vphi, iota)
        return {'cums1r_s1z': cums1r_s1z,
                's1phi_hat': s1phi_hat,
                'cums2r_s2z': cums2r_s2z,
                's2phi_hat': s2phi_hat}



class ZeroInplaneSpinsPrior(FixedPrior):
    """Set inplane spins to zero."""
    standard_par_dic = {'s1x': 0,
                        's1y': 0,
                        's2x': 0,
                        's2y': 0}


class ZeroTidalDeformabilityPrior(FixedPrior):
    """Set tidal deformability parameters to zero."""
    standard_par_dic = {'l1': 0,
                        'l2': 0}


# ----------------------------------------------------------------------
# Default priors for the full set of variables, for convenience.

class IASPrior(CombinedPrior, RegisteredPriorMixin):
    """Precessing, flat in chieff, uniform luminosity volume."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     FlatChieffPrior,
                     UniformDiskInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior]


class AlignedSpinIASPrior(CombinedPrior, RegisteredPriorMixin):
    """Aligned spin, flat in chieff, uniform luminosity volume."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     FlatChieffPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior]


class LVCPrior(CombinedPrior, RegisteredPriorMixin):
    """Precessing, isotropic spins, uniform luminosity volume."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     IsotropicSpinsInplaneComponentsPrior,
                     ZeroTidalDeformabilityPrior]

class AlignedSpinLVCPrior(CombinedPrior, RegisteredPriorMixin):
    """Aligned spins from isotropic distribution, uniform luminosity volume."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior]


class IASPriorComovingVT(IASPrior):
    """Precessing, flat in chieff, uniform comoving volume-time."""
    prior_classes = IASPrior.prior_classes.copy()
    prior_classes[-4] = UniformComovingVolumePrior


class AlignedSpinIASPriorComovingVT(AlignedSpinIASPrior):
    """Aligned spin, flat in chieff, uniform comoving volume-time."""
    prior_classes = AlignedSpinIASPrior.prior_classes.copy()
    prior_classes[-4] = UniformComovingVolumePrior


class LVCPriorComovingVT(LVCPrior):
    """Precessing, isotropic spins, uniform comoving volume-time."""
    prior_classes = LVCPrior.prior_classes.copy()
    prior_classes[-4] = UniformComovingVolumePrior


class AlignedSpinLVCPriorComovingVT(AlignedSpinLVCPrior):
    """Aligned spins from isotropic distribution, uniform comoving volume-time."""
    prior_classes = AlignedSpinLVCPrior.prior_classes.copy()
    prior_classes[-4] = UniformComovingVolumePrior