"""
Prior for phase and time parameters using linear-free convention.

They can be combined just by subclassing `CombinedPrior` and defining an
attribute `prior_classes` that is a list of such priors (see
``gw_prior.combined``).
Each may consume some arguments in the __init__(), but should forward
as ``**kwargs`` any arguments that other priors may need.
"""
import numpy as np

import lal

from cogwheel import gw_utils
from cogwheel import utils
from cogwheel import waveform
from cogwheel.likelihood import RelativeBinningLikelihood
from cogwheel.prior import Prior, UniformPriorMixin


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
            event_data, waveform_generator_22, par_dic_0, pn_phase_tol=.1)

        self._hplus0_22 = self._get_hplus_22(par_dic_0)

        weights_f = ((np.abs(self._likelihood_aux._get_h_f(par_dic_0))
                      * event_data.wht_filter)**2)  # (ndet, nrfft)
        self._polyfit_weights = np.sqrt(np.clip(
            self._likelihood_aux._get_summary_weights(weights_f.sum(0)).real,
            0, None))  # (nbin,) in very low resolution so it's cheap.

        # --------------------------------------------------------------
        # 2. Machinery for evaluating reference time and phase
        self._ref = {}  # Summary metadata of reference solution

        snr_det = np.sqrt(2 * np.maximum(
            self._likelihood_aux.lnlike_detectors_no_asd_drift(par_dic_0)
            * self._likelihood_aux.asd_drift**-2,
            1e-4))
        sigma_phases = 1 / snr_det
        self._ref['sigma_phase'] = (sigma_phases**-2).sum()**-.5
        self._ref['phase_weights'] = sigma_phases**-2 / (sigma_phases**-2).sum()

        normalized_weights_f = weights_f / weights_f.sum(axis=1, keepdims=True)
        self._ref['_f_avg'] = normalized_weights_f.dot(event_data.frequencies)
        sigma_f = np.sqrt(normalized_weights_f.dot(event_data.frequencies**2)
                          - self._ref['_f_avg']**2)
        sigma_times = 1 / (2 * np.pi * sigma_f * snr_det)
        self._ref['sigma_time'] = (sigma_times**-2).sum()**-.5
        self._ref['time_weights'] = sigma_times**-2 / (sigma_times**-2).sum()
        self._ref['det_locations'] = [gw_utils.DETECTORS[det].location
                                      for det in event_data.detector_names]
        self._ref['geometric_phases'] = self._geometric_phases(
            **{par: par_dic_0[par] for par in ['iota', 'ra', 'dec', 'psi']})
        self._ref['t_detectors'] = (par_dic_0['t_geocenter']
                                    + self._detector_delays(par_dic_0['ra'],
                                                            par_dic_0['dec']))

        self._ref.update(phi_linfree=0., t_linfree=0.)
        self._ref.update(self.inverse_transform(**par_dic_0))

        self.range_dic = self.__class__.range_dic | {'t_linfree': (-dt0, dt0)}

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
        return (detector_delays.dot(self._ref['time_weights'])
                + linfree_time_shift
                - self._ref['t_linfree'])

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
        phasors = np.exp(1j*(
            geometric_phases - self._ref['geometric_phases']
            - 2*np.pi * self._ref['_f_avg'] * (t_detectors
                                               - self._ref['t_detectors'])))
        return ((np.angle(self._ref['phase_weights'].dot(phasors))
                 + linfree_phase_shift) / 2
                - self._ref['phi_linfree'])

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
                         for location in self._ref['det_locations']])

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
