"""
Default modular priors for extrinsic parameters, for convenience.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""
import numpy as np

import lal

from cogwheel.cosmology import comoving_to_luminosity_diff_vt_ratio
from cogwheel import gw_utils
from cogwheel import skyloc_angles
from cogwheel.prior import Prior, UniformPriorMixin, IdentityTransformMixin


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


class UniformPhasePrior(UniformPriorMixin, IdentityTransformMixin,
                        Prior):
    """Uniform prior for the orbital phase. No change of coordinates."""
    range_dic = {'phi_ref': (0, 2*np.pi)}
    periodic_params = ['phi_ref']


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
    function of `phi_ref, iota, ra, dec`, such that it describes the well-
    measured phase of the waveform at a reference detector.
    """
    standard_params = ['psi']
    range_dic = {'psi_hat': (-np.pi/2, np.pi/2)}
    periodic_params = ['psi_hat']
    conditioned_on = ['iota', 'ra', 'dec', 'phi_ref', 't_geocenter']

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
                   (phi_ref, iota, ra, dec, psi, t_geocenter) of a solution
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
                for par in ('phi_ref', 'iota', 'ra', 'dec', 'psi',
                            't_geocenter')}
            self._phase_refdet_0 = self._phase_refdet(**par_dic_0)

    def _phase_refdet(self, iota, ra, dec, psi, phi_ref, t_geocenter):
        """
        Return the well-measurable overall phase at the reference detector.
        The intuition is that all allowed values of (phi_ref, iota, ra, dec,
        psi, t_geocenter) would have a consistent value of phase_refdet.
        """
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        return (np.angle(self.geometric_factor_refdet(ra, dec, psi, iota))
                + 2*phi_ref - 2*np.pi*self.f_avg*t_refdet) % (2*np.pi)

    def _psi_refdet(self, iota, ra, dec, phi_ref, t_geocenter):
        """
        Return psi that solves
            arg(R) + gamma = 0
        at the reference detector, where
            R = (1+cos^2(iota)) Fplus / 2 - i cos(iota) Fcross,
            gamma = 2 phi_ref - 2 pi f_avg t_refdet - phase_refdet_0.
        """
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        gamma = 2*phi_ref - 2*np.pi*self.f_avg*t_refdet - self._phase_refdet_0

        fp0, fc0 = self.fplus_fcross_refdet(ra, dec, psi=0)
        cosiota = np.cos(iota)

        a = 2 * cosiota * np.cos(gamma)
        b = (1+cosiota**2) * np.sin(gamma)
        c = fp0*a + fc0*b
        delta = np.pi * (cosiota * c < 0)  # 0 or pi
        return .5 * (np.arctan((fc0*a - fp0*b) / c) + delta)

    def transform(self, psi_hat, iota, ra, dec, phi_ref, t_geocenter):
        """psi_hat to psi."""
        psi_refdet = self._psi_refdet(iota, ra, dec, phi_ref, t_geocenter)
        psi = (psi_hat * np.sign(np.cos(iota)) + psi_refdet) % np.pi
        return {'psi': psi}

    def inverse_transform(self, psi, iota, ra, dec, phi_ref, t_geocenter):
        """psi to psi_hat"""
        psi_refdet = self._psi_refdet(iota, ra, dec, phi_ref, t_geocenter)
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

    def __init__(self, *, tgps, ref_det_name, d_hat_max=500,
                 d_luminosity_max=np.inf, **kwargs):
        """
        Parameters
        ----------
        tgps: float
            GPS time of the event, sets Earth orientation.

        ref_det_name: str
            Reference detector name, e.g. 'H' for Hanford.

        d_hat_max: float
            Upper bound for sampling `d_hat` (Mpc Msun^(-5/6)).

        d_luminosity_max: float
            Maximum luminosity distance allowed by the prior (Msun).
        """
        self.range_dic = {'d_hat': (0, d_hat_max)}
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)

        self.tgps = tgps
        self.ref_det_name = ref_det_name
        self.d_luminosity_max = d_luminosity_max

    def _conversion_factor(self, ra, dec, psi, iota, m1, m2):
        """
        Return conversion factor such that
            d_luminosity = d_hat * conversion_factor.
        """
        mchirp = gw_utils.m1m2_to_mchirp(m1, m2)
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
        d_luminosity = self.transform(
            d_hat, ra, dec, psi, iota, m1, m2)['d_luminosity']

        if d_luminosity > self.d_luminosity_max:
            return -np.inf
        return np.log(d_luminosity**3 / d_hat)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'tgps': self.tgps,
                'ref_det_name': self.ref_det_name,
                'd_hat_max': self.range_dic['d_hat'][1],
                'd_luminosity_max': self.d_luminosity_max}


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

        if d_luminosity > self.d_luminosity_max:
            return -np.inf
        return np.log(d_luminosity**3 / d_hat
                      * comoving_to_luminosity_diff_vt_ratio(d_luminosity))


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
        return np.log(d_luminosity**3 / d_effective
                      * comoving_to_luminosity_diff_vt_ratio(d_luminosity))

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'tgps': self.tgps,
                'ref_det_name': self.ref_det_name,
                'd_effective_max': self.range_dic['d_effective'][1]}
