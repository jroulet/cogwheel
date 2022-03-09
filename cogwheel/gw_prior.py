"""
Priors for parameter estimation of compact binary mergers.
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import interp1d

import lal
import lalsimulation

from . import cosmology
from . import gw_utils
from . import skyloc_angles
from . import waveform

from .prior import Prior, CombinedPrior, FixedPrior, UniformPriorMixin, \
    IdentityTransformMixin, check_inheritance_order


prior_registry = {}


class GWPriorError(Exception):
    """Base class for all exceptions in this module."""


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
            * par_dic
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
            raise GWPriorError('Only register fully defined priors.')

        prior_registry[cls.__name__] = cls


class ReferenceDetectorMixin:
    """
    Methods for priors that need to know about the reference detector.
    They must have `tgps`, `ref_det_name` attributes.
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
            R = (1+cos^2(iota)) Fp / 2 - i cos(iota) Fc
        that relates a waveform with generic orientation to an overhead
        face-on one for quadrupolar waveforms.
        Note that the amplitude |R| is between 0 and 1.
        """
        fplus, fcross = self.fplus_fcross_refdet(ra, dec, psi)
        return (1 + np.cos(iota)**2) / 2 * fplus - 1j * np.cos(iota) * fcross


# ----------------------------------------------------------------------
# Default modular priors for a subset of the variables, for convenience.
# They can be combined later just by subclassing `CombinedPrior` and
# defining an attribute `prior_classes` that is a list of these.
# Each may consume some arguments in the __init__(), but should forward
# as `**kwargs` any arguments that other priors may need.

class UniformDetectorFrameMassesPrior(Prior):
    """
    Uniform prior for detector frame masses.
    Sampled variables are mchirp, lnq. These are transformed to m1, m2.
    """
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
        """
        Natural logarithm of the prior probability for `mchirp, lnq`
        under a prior flat in detector-frame masses.
        """
        return np.log(mchirp * np.cosh(lnq/2)**.4 / self.prior_norm)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'mchirp_range': self.range_dic['mchirp'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}


class UniformDetectorFrameTotalMassInverseMassRatioPrior(Prior):
    """
    Uniform in detector-frame total mass and inverse mass ratio,
        mtot = m1 + m2
        1 / q = m1 / m2.
    Sampled params are mchirp, lnq, these are transformed to m1, m2.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'mchirp': NotImplemented,
                 'lnq': NotImplemented}

    def __init__(self, *, mchirp_range, q_min, symmetrize_lnq=False,
                 **kwargs):
        if not 0 < q_min <= 1:
            raise ValueError('`q_min` should be between 0 and 1.')

        lnq_min = np.log(q_min)
        self.range_dic = {'mchirp': mchirp_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq)}
        super().__init__(**kwargs)

        self.prior_lognorm = 0
        self.prior_lognorm = np.log(dblquad(
            lambda mchirp, lnq: np.exp(self.lnprior(mchirp, lnq)),
            *self.range_dic['lnq'], *self.range_dic['mchirp'])[0])

    @staticmethod
    def transform(mchirp, lnq):
        """(mchirp, lnq) to (m1, m2)."""
        q = np.exp(-np.abs(lnq))
        return {'m1': mchirp * (1 + q)**.2 / q**.6,
                'm2': mchirp * (1 + 1/q)**.2 * q**.6}

    @staticmethod
    def inverse_transform(m1, m2):
        """(m1, m2) to (mchirp, lnq)."""
        return {'mchirp': (m1 * m2)**.6 / (m1 + m2)**.2,
                'lnq': np.log(m2 / m1)}

    def lnprior(self, mchirp, lnq):
        """
        Uniform in 1/q and mtot ==>
        (using mchirp = eta**(3/5) * mtot, eta = q / (1+q)**2)
        P(lnq, mchirp) = (C/q) * q**(3/5) / (1+q)**(6/5)
                       = C / q**.4 / (1+q)**1.2
                       = C / q / cosh(lnq / 2)**1.2  ==>
        lnP - lnC = -.4*lnq - 1.2*ln(1+q)
                  = -lnq - 1.2*ln(2*cosh(.5*lnq))
        """
        del mchirp
        true_lnq = -np.abs(lnq)
        q = np.exp(true_lnq)
        return -.4*true_lnq - 1.2*np.log(1 + q) - self.prior_lognorm

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'mchirp_range': self.range_dic['mchirp'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}


class UniformSourceFrameTotalMassInverseMassRatioPrior(Prior):
    """
    Uniform in source-frame total mass and inverse mass ratio
        mtot_source = (m1 + m2) / (1 + z),
        1/q = m1/m2.
    Sampled params are mtot_source, lnq, these are transformed to m1, m2
    conditioned on d_luminosity.
    Note: cannot be combined with distance prior that requires mass
    conditioning.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'mtot_source': NotImplemented,
                 'lnq': NotImplemented}
    conditioned_on = ['d_luminosity']

    def __init__(self, *, mtot_source_range, q_min,
                 symmetrize_lnq=False, **kwargs):
        if not 0 < q_min <= 1:
            raise ValueError('`q_min` should be between 0 and 1.')

        lnq_min = np.log(q_min)
        self.range_dic = {'mtot_source': mtot_source_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq)}
        super().__init__(**kwargs)

        self.prior_lognorm = 0
        self.prior_lognorm = np.log(dblquad(
            lambda mtot_source, lnq: np.exp(self.lnprior(mtot_source, lnq)),
            *self.range_dic['lnq'], *self.range_dic['mtot_source'])[0])

    @staticmethod
    def transform(mtot_source, lnq, d_luminosity):
        """(mtot_source, lnq, d_luminosity) to (m1, m2)."""
        q = np.exp(-np.abs(lnq))
        m1 = (1 + cosmology.z_of_DL_Mpc(d_luminosity)) * mtot_source / (1 + q)

        return {'m1': m1,
                'm2': q * m1}

    @staticmethod
    def inverse_transform(m1, m2, d_luminosity):
        """(m1, m2, d_luminosity) to (mtot_source, lnq)."""
        mtot_source = (m1 + m2) / (1 + cosmology.z_of_DL_Mpc(d_luminosity))
        return {'mtot_source': mtot_source,
                'lnq': np.log(m2 / m1)}

    def lnprior(self, mtot_source, lnq):
        """Uniform in 1/q."""
        del mtot_source
        return -np.abs(lnq) - self.prior_lognorm

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'mtot_source_range': self.range_dic['mtot_source'],
                'q_min': np.exp(self.range_dic['lnq'][0]),
                'symmetrize_lnq': self.range_dic['lnq'][1] != 0}


class UniformPhasePrior(UniformPriorMixin, IdentityTransformMixin,
                        Prior):
    """Uniform prior for the orbital phase. No change of coordinates."""
    range_dic = {'vphi': (0, 2*np.pi)}
    periodic_params = ['vphi']


class IsotropicInclinationPrior(UniformPriorMixin, Prior):
    """Uniform-in-cosine prior for the binary's inclination."""
    standard_params = ['iota']
    range_dic = {'cosiota': (-1, 1)}
    folded_params = ['cosiota']

    @staticmethod
    def transform(cosiota):
        """cos(inclination) to inclination."""
        return {'iota': np.arccos(cosiota)}

    @staticmethod
    def inverse_transform(iota):
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


class UniformTimePrior(ReferenceDetectorMixin, UniformPriorMixin,
                       Prior):
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
        """`t_refdet` to `t_geocenter`"""
        return {'t_geocenter': t_refdet - self.time_delay_refdet(ra, dec)}

    def inverse_transform(self, t_geocenter, ra, dec):
        """`t_geocenter` to `t_refdet`"""
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


class UniformPolarizationPrior(ReferenceDetectorMixin,
                               UniformPriorMixin, Prior):
    """
    Prior for the polarization.
    The sampled variable `psi_hat` differs from the standard
    polarization `psi` by an inclination-dependent sign and an additive
    function of `vphi, iota, ra, dec`, such that it describes the well-
    measured phase of the waveform at a reference detector.
    """
    standard_params = ['psi']
    range_dic = {'psi_hat': (-np.pi/2, np.pi/2)}
    periodic_params = ['psi_hat']
    conditioned_on = ['iota', 'ra', 'dec', 'vphi', 't_geocenter']

    def __init__(self, *, tgps, ref_det_name, f_avg, par_dic_0=None, **kwargs):
        """
        Parameters
        ----------
        tgps: float, GPS time of the event, sets Earth orientation.
        ref_det_name: str, reference detector name, e.g. 'H' for
                      Hanford.
        f_avg: float, estimate of the first frequency moment (Hz) of a
               fiducial waveform using the reference detector PSD:
               f_avg = (Integrate[f * |h(f)|^2 / PSD(f)]
                        / Integrate[|h(f)|^2 / PSD(f)])
        par_dic_0: Optional dictionary, must have entries for
                   (vphi, iota, ra, dec, psi, t_geocenter) of a solution
                   with high likelihood (additional keys are ignored).
                   It is not essential but might remove residual
                   correlations between `psi_hat` and other extrinsic
                   parameters. If passed, it will center the measured
                   `psi_hat` near 0.
        """
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, f_avg=f_avg,
                         par_dic_0=par_dic_0, **kwargs)
        self.tgps = tgps
        self.ref_det_name = ref_det_name
        self.f_avg = f_avg
        self.par_dic_0 = par_dic_0

        self._phase_refdet_0 = 0.
        if par_dic_0:
            par_dic_0 = {
                par: par_dic_0[par]
                for par in ('vphi', 'iota', 'ra', 'dec', 'psi', 't_geocenter')}
            self._phase_refdet_0 = self._phase_refdet(**par_dic_0)

    def _phase_refdet(self, iota, ra, dec, psi, vphi, t_geocenter):
        """
        Return the well-measurable overall phase at the reference detector.
        The intuition is that all allowed values of (vphi, iota, ra, dec,
        psi, t_geocenter) would have a consistent value of phase_refdet.
        """
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        return (np.angle(self.geometric_factor_refdet(ra, dec, psi, iota))
                + 2*vphi - 2*np.pi*self.f_avg*t_refdet) % (2*np.pi)

    def _psi_refdet(self, iota, ra, dec, vphi, t_geocenter):
        """
        Return psi that solves
            arg(R) + gamma = 0
        at the reference detector, where
            R = (1+cos^2(iota)) Fplus / 2 - i cos(iota) Fcross,
            gamma = 2 vphi - 2 pi f_avg t_refdet - phase_refdet_0.
        """
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        gamma = 2*vphi - 2*np.pi*self.f_avg*t_refdet - self._phase_refdet_0

        fp0, fc0 = self.fplus_fcross_refdet(ra, dec, psi=0)
        cosiota = np.cos(iota)

        a = 2 * cosiota * np.cos(gamma)
        b = (1+cosiota**2) * np.sin(gamma)
        c = fp0*a + fc0*b
        delta = np.pi * (cosiota * c < 0)  # 0 or pi
        return .5 * (np.arctan((fc0*a - fp0*b) / c) + delta)

    def transform(self, psi_hat, iota, ra, dec, vphi, t_geocenter):
        """psi_hat to psi."""
        psi_refdet = self._psi_refdet(iota, ra, dec, vphi, t_geocenter)
        psi = (psi_hat * np.sign(np.cos(iota)) + psi_refdet) % np.pi
        return {'psi': psi}

    def inverse_transform(self, psi, iota, ra, dec, vphi, t_geocenter):
        """psi to psi_hat"""
        psi_refdet = self._psi_refdet(iota, ra, dec, vphi, t_geocenter)
        psi_hat = ((psi - psi_refdet) * np.sign(np.cos(iota))
                   + np.pi/2) % np.pi - np.pi/2
        return {'psi_hat': psi_hat}

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'tgps': self.tgps,
                'ref_det_name': self.ref_det_name,
                'f_avg': self.f_avg,
                'par_dic_0': self.par_dic_0}


class UniformLuminosityVolumePrior(ReferenceDetectorMixin, Prior):
    """
    Distance prior uniform in luminosity volume and detector-frame time.
    The sampled parameter is
        d_hat := d_effective / mchirp^(5/6)
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
        response = np.abs(self.geometric_factor_refdet(ra, dec, psi, iota))
        return mchirp**(5/6) * response

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
        d_luminosity = d_hat * self._conversion_factor(ra, dec, psi, iota,
                                                       m1, m2)
        z = cosmology.z_of_DL_Mpc(d_luminosity)
        cosmo_weight = (
            (1 - d_luminosity * cosmology.dz_dDL(d_luminosity) / (1+z))
            / (1 + z)**4)
        return np.log(cosmo_weight * d_luminosity**3 / d_hat)


class UniformComovingVolumePriorSampleEffectiveDistance(
        ReferenceDetectorMixin, Prior):
    """
    Distance prior uniform in luminosity volume and detector-frame time.
    The sampled parameter is:
      d_effective = d_luminosity
                    / |self.geometric_factor_refdet(ra, dec, psi, iota)|
    where the effective distance is defined in one "reference" detector
    (see parent class ReferenceDetectorMixin).
    """
    standard_params = ['d_luminosity']
    range_dic = {'d_effective': NotImplemented}
    conditioned_on = ['ra', 'dec', 'psi', 'iota']

    def __init__(self, *, tgps, ref_det_name, d_effective_max=50000,
                 **kwargs):
        self.range_dic = {'d_effective': (0.001, d_effective_max)}
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)
        self.tgps = tgps
        self.ref_det_name = ref_det_name

    def transform(self, d_effective, ra, dec, psi, iota):
        """d_effective to d_luminosity"""
        return {'d_luminosity': d_effective * np.abs(
            self.geometric_factor_refdet(ra, dec, psi, iota))}

    def inverse_transform(self, d_luminosity, ra, dec, psi, iota):
        """d_luminosity to d_effective"""
        return {'d_effective': d_luminosity / np.abs(
            self.geometric_factor_refdet(ra, dec, psi, iota))}

    def lnprior(self, d_effective, ra, dec, psi, iota):
        """
        Natural log of the prior probability density for d_effective.
        """
        d_luminosity = d_effective * np.abs(
            self.geometric_factor_refdet(ra, dec, psi, iota))
        z = cosmology.z_of_DL_Mpc(d_luminosity)
        cosmo_weight = (
            (1 - d_luminosity * cosmology.dz_dDL(d_luminosity) / (1+z))
            / (1 + z)**4)
        return np.log(cosmo_weight * d_luminosity**3 / d_effective)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'tgps': self.tgps,
                'ref_det_name': self.ref_det_name,
                'd_effective_max': self.range_dic['d_effective'][1]}


class FlatChieffPrior(UniformPriorMixin, Prior):
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

    @staticmethod
    def _get_s2z(chieff, q, s1z):
        return ((1+q)*chieff - s1z) / q

    def _s1z_lim(self, chieff, q):
        s1z_min = np.maximum(self._get_s1z(chieff, q, s2z=1), -1)
        s1z_max = np.minimum(self._get_s1z(chieff, q, s2z=-1), 1)
        return s1z_min, s1z_max

    def transform(self, chieff, cumchidiff, m1, m2):
        """(chieff, cumchidiff) to (s1z, s2z)."""
        q = m2 / m1
        s1z_min, s1z_max = self._s1z_lim(chieff, q)
        s1z = s1z_min + cumchidiff * (s1z_max - s1z_min)
        s2z = self._get_s2z(chieff, q, s1z)
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

    def inverse_transform(
            self, s1x, s1y, s2x, s2y, s1z, s2z, vphi, iota):
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
        """(cumsr_sz, sphi_hat) from (sx, sy, sz, vphi, iota)."""
        sz_sq = sz**2
        cumsr_sz = np.log(sz_sq / (sx**2 + sy**2 + sz_sq)) / np.log(sz_sq)
        sphi = np.arctan2(sy, sx)
        sphi_hat = (sphi + vphi + np.pi*(np.cos(iota) > 0)) % (2*np.pi)
        return cumsr_sz, sphi_hat

    def inverse_transform(
            self, s1x, s1y, s2x, s2y, s1z, s2z, vphi, iota):
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


class FixedIntrinsicParametersPrior(FixedPrior):
    """Fix masses, spins and tidal deformabilities."""
    standard_par_dic = {'m1': NotImplemented,
                        'm2': NotImplemented,
                        's1x': NotImplemented,
                        's1y': NotImplemented,
                        's1z': NotImplemented,
                        's2x': NotImplemented,
                        's2y': NotImplemented,
                        's2z': NotImplemented,
                        'l1': NotImplemented,
                        'l2': NotImplemented}

    def __init__(self, *, standard_par_dic, **kwargs):
        """
        Parameters
        ----------
        standard_par_dic:
            dictionary containing entries for
            `m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2`.
            Spins and tidal deformabilities would default to `0.` if not
            passed. Passing a `standard_par_dic` with other additional
            or missing keys will raise a `ValueError`.
        """
        self.standard_par_dic = waveform.DEFAULT_PARS | standard_par_dic

        if missing := (self.__class__.standard_par_dic.keys()
                       - self.standard_par_dic.keys()):
            raise ValueError(f'`standard_par_dic` is missing keys: {missing}')

        if extra := (self.standard_par_dic.keys()
                     - self.__class__.standard_par_dic.keys()):
            raise ValueError(f'`standard_par_dic` has extra keys: {extra}')

        super().__init__(standard_par_dic=self.standard_par_dic, **kwargs)

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'standard_par_dic': self.standard_par_dic}


class FixedReferenceFrequencyPrior(FixedPrior):
    """Fix reference frequency `f_ref`."""
    standard_par_dic = {'f_ref': NotImplemented}

    def __init__(self, *, f_ref, **kwargs):
        super().__init__(**kwargs)
        self.standard_par_dic = {'f_ref': f_ref}

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return self.standard_par_dic


class LogarithmicReferenceFrequencyPrior(UniformPriorMixin, Prior):
    """Promote `f_ref` to a sampled parameter to explore its effect."""
    standard_params = ['f_ref']
    range_dic = {'ln_f_ref': NotImplemented}

    def __init__(self, f_ref_rng=(15, 300), **kwargs):
        """
        Parameters
        ----------
        `f_ref_rng`: minimum and maximum reference frequencies in Hz.
        """
        self.range_dic = {'ln_f_ref': np.log(f_ref_rng)}
        super().__init__(**kwargs)

    def transform(self, ln_f_ref):
        """`ln_f_ref` to `f_ref`."""
        return {'f_ref': np.exp(ln_f_ref)}

    def inverse_transform(self, f_ref):
        """`f_ref` to `ln_f_ref`."""
        return {'ln_f_ref': np.log(f_ref)}

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'f_ref_rng': np.exp(self.range_dic['ln_f_ref'])}


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
    conditioned_on = ['s1z', 's2z', 'vphi', 'm1', 'm2', 'f_ref']

    @staticmethod
    def _spin_transform(cumsr_sz, sz):
        sr = np.sqrt(cumsr_sz * (1 - sz ** 2))
        chi = np.sqrt(sr**2 + sz**2)
        tilt = np.arctan2(sr, sz)
        return chi, tilt

    def transform(self, costheta_jn, phi_jl_hat, phi12, cums1r_s1z,
                  cums2r_s2z, s1z, s2z, vphi, m1, m2, f_ref):
        """Spin prior cumulatives to spin components."""
        chi1, tilt1 = self._spin_transform(cums1r_s1z, s1z)
        chi2, tilt2 = self._spin_transform(cums2r_s2z, s2z)
        theta_jn = np.arccos(costheta_jn)
        phi_jl = (phi_jl_hat + np.pi * (costheta_jn < 0)) % (2*np.pi)

        iota, s1x, s1y, s1z, s2x, s2y, s2z \
            = lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2,
                m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_ref, vphi)

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
                          vphi, m1, m2, f_ref):
        """
        Inclination and spin components to theta_jn, phi_jl, phi12 and
        inplane-spin-magnitude prior cumulatives.
        """
        theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 \
            = lalsimulation.SimInspiralTransformPrecessingWvf2PE(
                iota, s1x, s1y, s1z, s2x, s2y, s2z, m1, m2, f_ref, vphi)

        cums1r_s1z = self._inverse_spin_transform(chi1, tilt1, s1z)
        cums2r_s2z = self._inverse_spin_transform(chi2, tilt2, s2z)

        costheta_jn = np.cos(theta_jn)
        phi_jl_hat = (phi_jl + np.pi * (costheta_jn < 0)) % (2*np.pi)

        return {'costheta_jn': costheta_jn,
                'phi_jl_hat': phi_jl_hat,
                'phi12': phi12,
                'cums1r_s1z': cums1r_s1z,
                'cums2r_s2z': cums2r_s2z}


# ----------------------------------------------------------------------
# Default priors for the full set of variables, for convenience.

class IASPrior(RegisteredPriorMixin, CombinedPrior):
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
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class IASPrior2(RegisteredPriorMixin, CombinedPrior):
    """Precessing, flat in chieff, uniform luminosity volume."""
    prior_classes = [FixedReferenceFrequencyPrior,
                     UniformPhasePrior,
                     UniformDetectorFrameMassesPrior,
                     FlatChieffPrior,
                     IsotropicInclinationUniformDiskInplaneSpinsPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     ZeroTidalDeformabilityPrior]


class AlignedSpinIASPrior(RegisteredPriorMixin, CombinedPrior):
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
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class LVCPrior(RegisteredPriorMixin, CombinedPrior):
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
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class AlignedSpinLVCPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Aligned spin components from isotropic distribution, uniform
    luminosity volume.
    """
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class IASPriorComovingVT(RegisteredPriorMixin, CombinedPrior):
    """Precessing, flat in chieff, uniform comoving VT."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformComovingVolumePrior,
                     FlatChieffPrior,
                     UniformDiskInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class AlignedSpinIASPriorComovingVT(RegisteredPriorMixin,
                                    CombinedPrior):
    """Aligned spin, flat in chieff, uniform comoving VT."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformComovingVolumePrior,
                     FlatChieffPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class LVCPriorComovingVT(RegisteredPriorMixin, CombinedPrior):
    """Precessing, isotropic spins, uniform comoving VT."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformComovingVolumePrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     IsotropicSpinsInplaneComponentsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class AlignedSpinLVCPriorComovingVT(RegisteredPriorMixin,
                                    CombinedPrior):
    """
    Aligned spins from isotropic distribution, uniform comoving VT.
    """
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformComovingVolumePrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class NitzMassIASSpinPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Priors are uniform in source-frame total mass, inverse mass ratio,
    effective spin, and comoving VT.
    Sampling is in mtot_source, lnq, d_effective, and the rest of the
    IAS spin and extrinsic parameters.
    """
    prior_classes = [UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformComovingVolumePriorSampleEffectiveDistance,
                     UniformSourceFrameTotalMassInverseMassRatioPrior,
                     FlatChieffPrior,
                     UniformDiskInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class NitzMassLVCSpinPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Priors have isotropic spins and are uniform in source-frame total
    mass, inverse mass ratio, and comoving VT.
    Sampling is in mtot_source, lnq, d_effective, and the rest of the
    LVC spin and extrinsic parameters.
    """
    prior_classes = [UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformComovingVolumePriorSampleEffectiveDistance,
                     UniformSourceFrameTotalMassInverseMassRatioPrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     IsotropicSpinsInplaneComponentsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]


class ExtrinsicParametersPrior(RegisteredPriorMixin, CombinedPrior):
    """Uniform luminosity volume, fixed intrinsic parameters."""
    prior_classes = [FixedIntrinsicParametersPrior,
                     UniformPhasePrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformLuminosityVolumePrior,
                     FixedReferenceFrequencyPrior]
