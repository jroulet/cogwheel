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
from cogwheel import waveform
from cogwheel.likelihood import RelativeBinningLikelihood
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


class LinearFreePhaseTimePrior(UniformPriorMixin, Prior):
    """
    Uniform prior for the orbital phase and time of arrival.
    Combines information from all detectors and applies an
    intrinsic-parameter-dependent phase and time shift to align
    waveforms with different intrinsic parameters with each other.

    The sampled coordinates (phi_linfree, t_linfree) describe orbital
    phase rotation and time shift relative to the orbital phase and time
    that would best align the phase and time of the waveform to the
    reference waveform, as seen at the detectors, with inverse-variance-
    weighted averaging.
    """
    standard_params = ['phi_ref', 't_geocenter']
    range_dic = {'phi_linfree': (-np.pi/2, 3*np.pi/2),  # 0, pi away from edges
                 't_linfree': NotImplemented}
    _intrinsic = ['f_ref', 'l1', 'l2', 'm1', 'm2', 's1x_n', 's1y_n', 's1z',
                  's2x_n', 's2y_n', 's2z']
    conditioned_on = ['iota', 'ra', 'dec', 'psi'] + _intrinsic
    folded_shifted_params = ['phi_linfree']

    def __init__(self, *, approximant, par_dic_0, event_data, dt0=.005,
                 **kwargs):
        # --------------------------------------------------------------
        # 1. Machinery for computing linear-free waveforms

        # Edit eventname so it doesn't clash with the regular
        # event_data's file when saved with ``to_json``:
        event_data = event_data.reinstantiate(
            eventname=event_data.eventname.removesuffix('_aux') + '_aux')

        # Weight frequencies by SNR^2:
        waveform_generator_22 = waveform.WaveformGenerator.from_event_data(
            event_data, approximant=approximant,
            harmonic_modes=None)
        waveform_generator_22.harmonic_modes \
            = waveform_generator_22._harmonic_modes_by_m[2]

        self._likelihood_aux = RelativeBinningLikelihood(
            event_data, waveform_generator_22, par_dic_0, pn_phase_tol=1.)

        self._hplus0_22 = self._get_hplus_22(par_dic_0)

        weights_f = ((np.abs(self._likelihood_aux._get_h_f(par_dic_0))
                      * event_data.wht_filter)**2)  # (ndet, nrfft)
        self._polyfit_weights = np.sqrt(np.clip(
            self._likelihood_aux._get_summary_weights(weights_f.sum(0)).real,
            0, None))  # (nbin,) in very low resolution so it's cheap.

        # --------------------------------------------------------------
        # 2. Machinery for evaluating reference time and phase
        snr_det = np.sqrt(2 * np.maximum(
            self._likelihood_aux.lnlike_detectors_no_asd_drift(par_dic_0),
            1e-4))
        self._f_avg = ((event_data.frequencies * weights_f).sum(axis=1)
                       / weights_f.sum(axis=1))  # (ndet,)
        sigma_f = np.sqrt((event_data.frequencies**2 * weights_f).sum(axis=1)
                          / weights_f.sum(1) - self._f_avg**2)  # (ndet,)
        sigma_phase = 1 / snr_det
        self._phase_weights = sigma_phase**-2 / (sigma_phase**-2).sum()
        sigma_t = 1 / (2 * np.pi * sigma_f * snr_det)
        self._time_weights = sigma_t**-2 / (sigma_t**-2).sum()
        self._det_locations = [gw_utils.DETECTORS[det].location
                               for det in event_data.detector_names]
        self._geometric_phases_0 = self._geometric_phases(
            **{par: par_dic_0[par] for par in ['iota', 'ra', 'dec', 'psi']})
        self.t_detectors_0 = par_dic_0['t_geocenter'] - self._detector_delays(
            **{par: par_dic_0[par] for par in ['ra', 'dec']})

        self._sampled_dic_0 = {'phi_linfree': 0., 't_linfree': 0.}
        self._sampled_dic_0 = self.inverse_transform(**par_dic_0)
        self.range_dic = self.__class__.range_dic | {
            't_linfree': (self._sampled_dic_0['t_linfree'] - dt0,
                          self._sampled_dic_0['t_linfree'] + dt0)}

        super().__init__(approximant=approximant, par_dic_0=par_dic_0,
                         event_data=event_data, **kwargs)

    def transform(self, phi_linfree, t_linfree, iota, ra, dec, psi,
                  *intrinsic, **kw_intrinsic):
        """(phi_linfree, t_linfree) to (phi_ref, t_geocenter)."""
        kw_intrinsic.update(dict(zip(self._intrinsic, intrinsic)))

        total_time_shift = self._get_total_time_shift(ra, dec, kw_intrinsic)
        t_geocenter = t_linfree - total_time_shift
        total_phase_shift = self._get_total_phase_shift(
            t_geocenter, iota, ra, dec, psi, kw_intrinsic)
        phi_ref = (phi_linfree - total_phase_shift) % (2*np.pi)
        return {'phi_ref': phi_ref,
                't_geocenter': t_geocenter}

    def inverse_transform(self, phi_ref, t_geocenter, iota, ra, dec, psi,
                          *intrinsic, **kw_intrinsic):
        """(phi_ref, t_geocenter) to (phi_linfree, t_linfree)."""
        kw_intrinsic.update(dict(zip(self._intrinsic, intrinsic)))
        total_time_shift = self._get_total_time_shift(ra, dec, kw_intrinsic)
        t_linfree = t_geocenter + total_time_shift
        total_phase_shift = self._get_total_phase_shift(
            t_geocenter, iota, ra, dec, psi, kw_intrinsic)
        phi_linfree = utils.mod(phi_ref + total_phase_shift,
                                start=self.range_dic['phi_linfree'][0])
        return {'phi_linfree': phi_linfree,
                't_linfree': t_linfree}

    def _get_total_time_shift(self, ra, dec, intrinsic_dic):
        """
        Return total time shift between t_linfree and t_geocenter.
        The convention is
            t_linfree = t_geocenter + total_time_shift
        """
        _, linfree_time_shift = self._get_linfree_phase_time_shift(
            **intrinsic_dic)

        detector_delays = self._detector_delays(ra, dec)
        return (detector_delays.dot(self._time_weights)
                + linfree_time_shift
                - self._sampled_dic_0['t_linfree'])

    def _get_total_phase_shift(self, t_geocenter, iota, ra, dec, psi,
                               intrinsic_dic):
        """
        Return total phase shift between phi_linfree and phi_ref.
        The convention is
            phi_linfree = phi_ref + total_phase_shift
        """
        linfree_phase_shift, _ = self._get_linfree_phase_time_shift(
            **intrinsic_dic)
        t_detectors = t_geocenter + self._detector_delays(ra, dec)
        geometric_phases = self._geometric_phases(iota, ra, dec, psi)
        # Average angles with \arg(\sum_k(weight_k e^(i angle_k)))
        phasors = np.exp(1j*(geometric_phases - self._geometric_phases_0
                             - 2*np.pi * self._f_avg * (
                                # linfree_time_shift
                                + t_detectors - self.t_detectors_0)))
        return ((np.angle(self._phase_weights.dot(phasors))
                 + linfree_phase_shift) / 2
                - self._sampled_dic_0['phi_linfree'])

    @utils.lru_cache()
    def _get_linfree_phase_time_shift(self, **intrinsic_dic):
        """
        Return phase and time shift that would make the waveform
        hplus(iota=0, phi_ref=0, (l,|m|)=(2,2)) have the same phase and
        time as the ``par_dic_0`` waveform.
        Note, the phase is phase of the gravitational wave, i.e. twice
        that of the orbit.
        """
        hplus_22 = self._get_hplus_22(intrinsic_dic)

        dphase = np.unwrap(np.angle(hplus_22 / self._hplus0_22))
        fit = np.polynomial.Polynomial.fit(self._likelihood_aux.fbin, dphase,
                                           deg=1, w=self._polyfit_weights)

        phase_shift = utils.mod(fit(intrinsic_dic['f_ref']), -np.pi)
        time_shift = - fit.deriv()(intrinsic_dic['f_ref']) / (2*np.pi)
        return phase_shift, time_shift

    def _get_hplus_22(self, par_dic):
        """
        Return plus polarization of the (2, 2) mode of a iota=0,
        phi_ref=0 waveform at low frequency resolution.
        """
        return self._likelihood_aux.waveform_generator.get_hplus_hcross(
            self._likelihood_aux.fbin,
            par_dic | {'iota': 0, 'phi_ref': 0, 'd_luminosity': 1})[0]

    @utils.lru_cache()
    def _detector_delays(self, ra, dec):
        """
        Return array of time delays from Earth center to the detectors.
        """
        return np.array([lal.TimeDelayFromEarthCenter(
            location, ra, dec, self._likelihood_aux.event_data.tgps)
                         for location in self._det_locations])

    @utils.lru_cache()
    def _geometric_phases(self, iota, ra, dec, psi):
        """
        Return array of length n_detectors with argument of complex
        detector responses.
        """
        fplus, fcross = gw_utils.fplus_fcross(
            self._likelihood_aux.event_data.detector_names, ra, dec, psi,
            self._likelihood_aux.event_data.tgps)
        cosiota = np.cos(iota)
        return np.arctan2(-cosiota * fcross, (1 + cosiota**2) / 2 * fplus)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'approximant':
                    self._likelihood_aux.waveform_generator.approximant,
                'par_dic_0': self._likelihood_aux.par_dic_0,
                'event_data': self._likelihood_aux.event_data,
                'dt0': np.diff(self.range_dic['t_linfree'])[0] / 2}


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
    range_dic = {'t_refdet': NotImplemented}
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
