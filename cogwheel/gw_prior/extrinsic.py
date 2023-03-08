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
from cogwheel import utils
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

    @utils.lru_cache()
    def geometric_factor_refdet(self, ra, dec, psi, iota):
        """
        Return the complex geometric factor
            R = (1+cos^2(iota)) Fp / 2 - i cos(iota) Fc
        that relates a waveform with generic orientation to an overhead
        face-on one for quadrupolar waveforms.
        Note that the amplitude |R| is between 0 and 1.
        """
        fplus, fcross = self.fplus_fcross_refdet(ra, dec, psi)
        cosiota = np.cos(iota)
        return (1 + cosiota**2) / 2 * fplus - 1j * cosiota * fcross


class UniformPhasePrior(ReferenceDetectorMixin, UniformPriorMixin,
                        Prior):
    """
    Deprecated, `LinearFreeTimePhasePrior` is better.

    Uniform prior for the orbital phase.
    The sampled variable `phi_ref_hat` differs from the standard
    coalescence phase `phi_ref` an additive function of
    `psi, iota, ra, dec, time`, such that it describes the well-measured
    phase of the waveform at a reference detector.
    Note: for waveforms with higher modes the posterior will have a
    discontinuity when ``angle(geometric_factor_refdet) = pi``. However
    the folded posterior does not have this discontinuity.
    """
    standard_params = ['phi_ref']
    range_dic = {'phi_ref_hat': (-np.pi/2, 3*np.pi/2)}  # 0, pi away from edges
    conditioned_on = ['iota', 'ra', 'dec', 'psi', 't_geocenter']
    folded_shifted_params = ['phi_ref_hat']

    def __init__(self, *, tgps, ref_det_name, f_avg, par_dic_0=None,
                 **kwargs):
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, f_avg=f_avg,
                         par_dic_0=par_dic_0, **kwargs)
        self.tgps = tgps
        self.ref_det_name = ref_det_name
        self.f_avg = f_avg
        self.par_dic_0 = par_dic_0

        self._phase_refdet_0 = 0.
        if par_dic_0:
            par_dic_0 = {par: par_dic_0[par]
                         for par in ('phi_ref', 'iota', 'ra', 'dec',
                                     'psi', 't_geocenter')}
            self._phase_refdet_0 = self._phase_refdet(**par_dic_0)

    def _phase_refdet(self, iota, ra, dec, psi, t_geocenter, phi_ref):
        """
        Return the well-measurable overall phase at the reference
        detector. The intuition is that all allowed values of
        (phi_ref, iota, ra, dec, psi, t_geocenter) would have a
        consistent value of phase_refdet.
        """
        t_refdet = t_geocenter + self.time_delay_refdet(ra, dec)
        return (np.angle(self.geometric_factor_refdet(ra, dec, psi, iota))
                + 2*phi_ref - 2*np.pi*self.f_avg*t_refdet) % (2*np.pi)

    @utils.lru_cache()
    def transform(self, phi_ref_hat, iota, ra, dec, psi, t_geocenter):
        """phi_ref_hat to phi_ref."""
        phase_refdet = self._phase_refdet(iota, ra, dec, psi, t_geocenter,
                                          phi_ref=0)
        phi_ref = (phi_ref_hat
                   - (phase_refdet - self._phase_refdet_0) / 2) % (2*np.pi)
        return {'phi_ref': phi_ref}

    def inverse_transform(self, psi, iota, ra, dec, phi_ref, t_geocenter):
        """phi_ref to phi_ref_hat"""
        phase_refdet = self._phase_refdet(iota, ra, dec, psi, t_geocenter,
                                          phi_ref=0)
        phi_ref_hat = utils.mod(
            phi_ref + (phase_refdet - self._phase_refdet_0) / 2,
            start=self.range_dic['phi_ref_hat'][0])
        return {'phi_ref_hat': phi_ref_hat}

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        init_dict = {'tgps': self.tgps,
                     'ref_det_name': self.ref_det_name,
                     'f_avg': self.f_avg,
                     'par_dic_0': self.par_dic_0}

        return utils.merge_dictionaries_safely(super().get_init_dict(),
                                               init_dict)


class IsotropicInclinationPrior(UniformPriorMixin, Prior):
    """Uniform-in-cosine prior for the binary's inclination."""
    standard_params = ['iota']
    range_dic = {'cosiota': (-1, 1)}
    folded_reflected_params = ['cosiota']

    @staticmethod
    @utils.lru_cache()
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
    folded_reflected_params = ['phinet_hat']

    def __init__(self, *, detector_pair, tgps, **kwargs):
        super().__init__(detector_pair=detector_pair, tgps=tgps,
                         **kwargs)
        self.skyloc = skyloc_angles.SkyLocAngles(detector_pair, tgps)

    @utils.lru_cache()
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
        init_dict = self.skyloc.get_init_dict()
        return utils.merge_dictionaries_safely(super().get_init_dict(),
                                               init_dict)


class UniformTimePrior(ReferenceDetectorMixin, UniformPriorMixin,
                       Prior):
    """
    Deprecated, `LinearFreeTimePhasePrior` is better.

    Prior for the time of arrival at a reference detector.
    """
    standard_params = ['t_geocenter']
    range_dic = {'t_refdet': None}
    conditioned_on = ['ra', 'dec']

    def __init__(self, *, tgps, ref_det_name, t0_refdet=0, dt0=.07,
                 **kwargs):
        self.range_dic = {'t_refdet': (t0_refdet - dt0, t0_refdet + dt0)}
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)

        self.tgps = tgps
        self.ref_det_name = ref_det_name

    @utils.lru_cache()
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
        init_dict = {'t0_refdet': np.mean(self.range_dic['t_refdet']),
                     'dt0': np.diff(self.range_dic['t_refdet'])[0] / 2,
                     'tgps': self.tgps,
                     'ref_det_name': self.ref_det_name}

        return utils.merge_dictionaries_safely(super().get_init_dict(),
                                               init_dict)


class UniformPolarizationPrior(UniformPriorMixin,
                               IdentityTransformMixin, Prior):
    """Uniform prior for the polarization. No change of coordinates."""
    range_dic = {'psi': (0, np.pi)}
    periodic_params = ['psi']
    folded_shifted_params = ['psi']


class UniformLuminosityVolumePrior(ReferenceDetectorMixin, Prior):
    """
    Distance prior uniform in luminosity volume and detector-frame time.
    The sampled parameter is
        d_hat := d_effective / mchirp^(5/6)
    where the effective distance is defined in one "reference" detector.
    """
    standard_params = ['d_luminosity']
    range_dic = {'d_hat': None}
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

    @utils.lru_cache()
    def transform(self, d_hat, ra, dec, psi, iota, m1, m2):
        """d_hat to d_luminosity"""
        return {'d_luminosity': d_hat * self._conversion_factor(ra, dec, psi,
                                                                iota, m1, m2)}

    def inverse_transform(self, d_luminosity, ra, dec, psi, iota, m1, m2):
        """d_luminosity to d_hat"""
        return {'d_hat': d_luminosity / self._conversion_factor(ra, dec, psi,
                                                                iota, m1, m2)}

    @utils.lru_cache()
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
        d_hat := d_effective / mchirp^(5/6)
    where the effective distance is defined in one "reference" detector.
    """
    @utils.lru_cache()
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
