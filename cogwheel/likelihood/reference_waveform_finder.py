"""
Maximize the likelihood of a compact binary coalescence waveform.

A class ``ReferenceWaveformFinder`` is defined with methods to find
parameters with good likelihood, these can be chosen as a reference
solution for the relative-binning method.
"""
import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.stats import qmc

from cogwheel import data
from cogwheel import gw_utils
from cogwheel import waveform
from cogwheel.gw_prior.extrinsic import UniformTimePrior
from cogwheel.skyloc_angles import SkyLocAngles
from .likelihood import check_bounds
from .relative_binning import RelativeBinningLikelihood


class ReferenceWaveformFinder(RelativeBinningLikelihood):
    """
    Find parameters of a high-likelihood solution.

    Some simplfying restrictions are placed, for speed:
        * (2, 2) mode
        * Aligned, equal spins
        * Inclination = 1 radian
        * Polarization = 0
    """
    # eta_max < .25 to avoid q = 1 solutions that lack harmonics:
    eta_range = (.05, .24)
    chieff_range = (-.999, .999)

    @classmethod
    def from_event(cls, event, mchirp_guess, approximant='IMRPhenomXAS',
                   pn_phase_tol=.02, spline_degree=3,
                   time_range=(-.25, .25), mchirp_range=None, f_ref=None):
        """
        Constructor that finds a reference waveform solution
        automatically by maximizing the likelihood.

        For injections with aligned spin, it will skip the maximization
        and simply use the injected waveform (except that equal-mass and
        face-on configurations are avoided, as they do not have all
        harmonics).

        Parameters
        ----------
        event: Instance of `data.EventData`, or string with event name
               (must correspond to a file in `data.DATADIR`), or path to
               ``npz`` file with `EventData` instance.

        mchirp_guess: float
            Estimate of the detector-frame chirp mass of the signal.

        approximant: str
            Approximant name.

        pn_phase_tol: float
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins.

        spline_degree: int
            Degree of the spline used to interpolate the ratio between
            waveform and reference waveform for relative binning.

        time_range: (float, float)
            Range of arrival times relative to ``tgps`` to explore (s).
            (``tgps`` is given by the ``data.EventData`` instance
            created from `event`.)

        mchirp_range: (float, float), optional.
            Range of chirp mass to explore (Msun). If not provided, an
            automatic choice will be made (see method
            ``set_mchirp_range``).
        """
        if isinstance(event, data.EventData):
            event_data = event
        else:
            try:
                event_data = data.EventData.from_npz(event)
            except FileNotFoundError:
                event_data = data.EventData.from_npz(filename=event)

        waveform_generator = waveform.WaveformGenerator.from_event_data(
            event_data, approximant, harmonic_modes=[(2, 2)],
            lalsimulation_commands=waveform.FORCE_NNLO_ANGLES)

        if event_data.injection:
            par_dic = event_data.injection['par_dic']
            if waveform.ZERO_INPLANE_SPINS.items() < par_dic.items():
                # Injection has aligned spins, use it as reference
                par_dic_0 = cls._get_safe_par_dic(par_dic)

                ref_wf_finder = cls(event_data, waveform_generator,
                                    par_dic_0, pn_phase_tol=pn_phase_tol,
                                    spline_degree=spline_degree,
                                    time_range=time_range,
                                    mchirp_range=mchirp_range)

                # Check that the relative binning is accurate at the injection.
                # If not, will go on to attempt the usual maximization.
                if np.abs(ref_wf_finder.lnlike_fft(par_dic)
                          - ref_wf_finder.lnlike(par_dic)) < 0.1:
                    print('Setting reference from injection.')
                    return ref_wf_finder

            # Allow passing ``mchirp_guess=None`` for injections:
            if mchirp_guess is None:
                mchirp_guess = gw_utils.m1m2_to_mchirp(par_dic['m1'],
                                                       par_dic['m2'])

        # Set initial parameter dictionary. Will get improved by
        # `find_bestfit_pars()`. Serves dual purpose as maximum
        # likelihood result and relative binning reference.
        par_dic_0 = dict.fromkeys(waveform_generator.params, 0.)
        par_dic_0['d_luminosity'] = 1.
        par_dic_0['iota'] = 1.  # So waveform has higher modes
        par_dic_0['f_ref'] = f_ref or 100.
        par_dic_0['m1'], par_dic_0['m2'] = gw_utils.mchirpeta_to_m1m2(
            mchirp_guess, eta=.2)

        ref_wf_finder = cls(event_data, waveform_generator, par_dic_0,
                            pn_phase_tol=pn_phase_tol,
                            spline_degree=spline_degree,
                            time_range=time_range,
                            mchirp_range=mchirp_range)
        ref_wf_finder.find_bestfit_pars(freeze_f_ref=f_ref is not None)

        # If this is an injection, we can "cheat" and check that relative
        # binning is working at the injection, and raise a warning if not:
        if event_data.injection:
            lnl_fft = ref_wf_finder.lnlike_fft(par_dic)
            lnl_rb = ref_wf_finder.lnlike(par_dic)

            if np.abs(lnl_fft - lnl_rb) > .1:
                warnings.warn(
                    'Could not find a good aligned-spin reference waveform:\n'
                    f'Exact (2, 2) mode ln L = {lnl_fft}\n'
                    f'Relative-binning: ln L = {lnl_rb}')

        return ref_wf_finder

    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None, spline_degree=3,
                 time_range=(-.25, .25), mchirp_range=None):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`

        waveform_generator: Instance of `waveform.WaveformGenerator`.

        par_dic_0: dict
            Parameters of the reference waveform, should be close to the
            maximum likelihood waveform.
            Keys should match ``self.waveform_generator.params``.

        fbin: 1-d array or None
            Array with edges of the frequency bins used for relative
            binning [Hz]. Alternatively, pass `pn_phase_tol`.

        pn_phase_tol: float or None
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins. Alternatively, pass `fbin`.

        spline_degree: int
            Degree of the spline used to interpolate the ratio between
            waveform and reference waveform for relative binning.

        time_range: (float, float)
            Minimum and maximum times to search relative to tgps (s).
        """
        self._times = None  # Set by ``.set_summary()``
        self._d_h_timeseries_weights = None  # Set by ``.set_summary()``

        self._time_range = time_range
        self._mchirp_range = mchirp_range

        waveform_generator.n_cached_waveforms = max(
            2, waveform_generator.n_cached_waveforms)  # Will need to flip iota
        super().__init__(event_data, waveform_generator, par_dic_0,
                         fbin, pn_phase_tol, spline_degree)

    @property
    def time_range(self):
        """Minimum and maximum times to search relative to tgps (s)."""
        return self._time_range

    @time_range.setter
    def time_range(self, time_range):
        if time_range[1] - time_range[0] < gw_utils.EARTH_CROSSING_TIME:
            raise ValueError('`time_range` must be broader than the Earth-'
                             f'crossing time {gw_utils.EARTH_CROSSING_TIME} s')

        self._time_range = time_range
        self._set_summary()

    @property
    def mchirp_range(self):
        """
        If `self._mchirp_range` is set return that, otherwise return a
        crude estimate based on the reference waveform's chirp mass.
        See also: `set_mchirp_range`.
        """
        if self._mchirp_range:
            return self._mchirp_range

        mchirp = gw_utils.m1m2_to_mchirp(self.par_dic_0['m1'],
                                         self.par_dic_0['m2'])
        return gw_utils.estimate_mchirp_range(mchirp)

    @staticmethod
    def _get_safe_par_dic(par_dic, eta_max=.24):
        """
        Return copy of `par_dic`, modified to prevent problematic
        behavior if it is used as relative-binning reference waveform
        (e.g. zeros in some modes or frequencies).
        Inplane spins are set to zero, iota is set to 1 or pi-1 and eta
        is capped at `eta_max` at constant mchirp.
        """
        par_dic = par_dic | waveform.ZERO_INPLANE_SPINS

        m1, m2, iota = par_dic['m1'], par_dic['m2'], par_dic['iota']

        par_dic['iota'] = 1 if iota < np.pi / 2 else np.pi - 1

        if gw_utils.q_to_eta(m2 / m1) > eta_max:
            mchirp = gw_utils.m1m2_to_mchirp(m1, m2)
            par_dic['m1'], par_dic['m2'] = gw_utils.mchirpeta_to_m1m2(
                mchirp, eta_max)

        return par_dic

    def find_bestfit_pars(self, seed=0, freeze_f_ref=False):
        """
        Find a good fit solution with restricted parameters (face-on,
        equal aligned spins). Additionally, use that to set
        ``self.mchirp_range`` if it has not been set already.
        Will update `self.par_dic_0` in stages. The relative binning
        summary data (and `asd_drift`) will be updated after maximizing
        over intrinsic parameters.

        First maximize likelihood incoherently w.r.t. intrinsic
        parameters using mchirp, eta, chieff.
        Then maximize likelihood w.r.t. extrinsic parameters, leaving
        the intrinsic fixed.
        Speed is prioritized over quality of the maximization. Use
        `Posterior.refine_reference_waveform()` to refine the solution
        if needed.

        Parameters
        ----------
        seed: To initialize the random state of stochastic maximizers.
        """
        # Optimize intrinsic parameters, update relative binning summary:
        self._optimize_m1m2s1zs2z_incoherently(seed)

        # Use waveform to define reference detector, detector pair and
        # reference frequency:
        kwargs = self.get_coordinate_system_kwargs()
        if not freeze_f_ref:
            self.par_dic_0['f_ref'] = kwargs['f_avg']

        # Optimize time, sky location, orbital phase and distance
        self._optimize_t_refdet(kwargs['ref_det_name'])
        self._optimize_skyloc_and_inclination(kwargs['detector_pair'], seed)
        self._optimize_phase_and_distance()

        if not self._mchirp_range:
            self.set_mchirp_range()

    def _matched_filter_timeseries_rb(self, par_dic):
        """
        Return array of shape ``(n_times, n_det)`` with matched filter
        scores computed with relative binning, where ``n_times`` is
        the number of timeshifts for which summary data were computed.
        """
        h_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)

        # Sum over m and f axes, leave time and detector axis unsummed.
        d_h_timeseries = (self._d_h_timeseries_weights * h_fbin.conj()
                          ).sum(axis=(-3, -1))

        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        h_h = ((self._h_h_weights * h_fbin[m_inds] * h_fbin[mprime_inds].conj()
               ).real.sum(axis=(0, -1)))

        return d_h_timeseries / np.sqrt(h_h)

    @check_bounds
    def lnlike_max_amp_phase_time(self, par_dic,
                                  return_by_detectors=False):
        """
        Return log likelihood maximized over amplitude, phase and time
        incoherently across detectors.
        """
        # We assume that all time shifts are encompassed by `self._times`
        matched_filter_timeseries = self._matched_filter_timeseries_rb(
            par_dic | {'t_geocenter': 0.})
        lnl = np.max(np.abs(matched_filter_timeseries), axis=0)**2 / 2

        if return_by_detectors:
            return lnl
        return np.sum(lnl)

    @check_bounds
    def lnlike_max_amp_phase(self, par_dic, ret_amp_phase_bf=False,
                             det_inds=...):
        """
        Return log likelihood maximized over amplitude and phase
        coherently across detectors (the same amplitude rescaling
        and phase is applied to all detectors).
        """
        h_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)

        det_slice = np.s_[:, det_inds, :]
        d_h = (self._d_h_weights * h_fbin.conj())[det_slice].sum()

        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        h_h = ((self._h_h_weights * h_fbin[m_inds] * h_fbin[mprime_inds].conj()
               ).real[det_slice].sum())

        lnl = np.abs(d_h)**2 / h_h / 2

        if not ret_amp_phase_bf:
            return lnl

        phase_bf = np.angle(d_h)
        amp_bf = np.abs(d_h) / h_h
        return lnl, amp_bf, phase_bf

    def _set_summary(self):
        """Set usual summary data plus ``_d_h_timeseries_weights``."""
        super()._set_summary()
        self._times = np.arange(*self.time_range, 2**-10)
        shifts = np.exp(2j*np.pi * self._times.reshape(-1, 1, 1, 1)
                        * self.event_data.frequencies)
        d_h0_t = self.event_data.blued_strain * self._h0_f.conj() * shifts
        self._d_h_timeseries_weights = (self._get_summary_weights(d_h0_t)
                                        / np.conj(self._h0_fbin))

    def _updated_intrinsic(self, mchirp, eta, chieff):
        """Return `self.par_dic_0` with updated m1, m2, s1z, s2z."""
        m1, m2 = gw_utils.mchirpeta_to_m1m2(mchirp, eta)
        intrinsic = {'m1': m1,
                     'm2': m2,
                     's1z': chieff,
                     's2z': chieff}
        return self.par_dic_0 | intrinsic

    def _lnlike_incoherent(self, mchirp, eta, chieff):
        """
        Log likelihood maximized over amplitude, phase and time
        incoherently across detectors.
        """
        par_dic = self._updated_intrinsic(mchirp, eta, chieff)
        return self.lnlike_max_amp_phase_time(par_dic)

    def _optimize_m1m2s1zs2z_incoherently(self, seed):
        """
        Optimize mchirp, eta and chieff by likelihood maximized over
        amplitude, phase and time incoherently across detectors.
        Modify the entries of `self.par_dic_0` corresponding to
        `m1, m2, s1z, s2z` with the new solution (this will update the
        relative-binning summary data).
        """
        print(f'Searching incoherent solution for {self.event_data.eventname}')

        result = differential_evolution(
            lambda mchirp_eta_chieff:
                -self._lnlike_incoherent(*mchirp_eta_chieff),
            bounds=[self.mchirp_range, self.eta_range, self.chieff_range],
            seed=seed, init='sobol')

        self.par_dic_0 = self._updated_intrinsic(*result.x)
        print(f'Set intrinsic parameters, lnL = {-result.fun}')

    def _optimize_t_refdet(self, ref_det_name):
        """
        Find coalescence time that optimizes SNR maximized over
        amplitude and phase at reference detector.
        Update the 't_geocenter' entry of `par_dic` in-place.
        Note that `t_geocenter` can and should be recomputed if `ra`,
        `dec` change.
        """
        i_refdet = self.event_data.detector_names.index(ref_det_name)

        # Maximize over time on the timeshifts grid
        ind = np.argmax(np.abs(
            self._matched_filter_timeseries_rb(self.par_dic_0)[:, i_refdet]))

        if ind in {0, len(self._times) - 1}:
            raise RuntimeError(
                'Maximum likelihood achieved at edge of timeseries, consider '
                'extending ``time_range`` or providing a more accurate '
                '``event_data.tgps``.')

        self.par_dic_0['t_geocenter'] = self._times[ind]

        super()._set_summary()  # Recompute summary for non-timeshift

        # Refine at subgrid resolution
        def lnlike_refdet(t_geocenter):
            return self.lnlike_max_amp_phase(
                self.par_dic_0 | {'t_geocenter': t_geocenter},
                det_inds=i_refdet)

        result = minimize_scalar(lambda tgeo: -lnlike_refdet(tgeo),
                                 bracket=self._times[ind-1 : ind+2],
                                 bounds=self._times[[ind-1, ind+1]])
        self.par_dic_0['t_geocenter'] = result.x
        print(f'Set time, lnL({ref_det_name}) = {-result.fun}')

    def _optimize_skyloc_and_inclination(self, detector_pair, seed):
        """
        Find right ascension and declination that optimize likelihood
        maximized over amplitude and phase, and choose between
        iota={1, pi-1}.
        t_geocenter is readjusted so as to leave t_refdet unchanged.
        Update 't_geocenter', ra', 'dec' entries of `self.par_dic_0`
        in-place.
        """
        skyloc = SkyLocAngles(detector_pair, self.event_data.tgps)
        time_transformer = UniformTimePrior(tgps=self.event_data.tgps,
                                            ref_det_name=detector_pair[0],
                                            t0_refdet=np.nan, dt0=np.nan)
        t0_refdet = time_transformer.inverse_transform(
            **{key: self.par_dic_0[key]
               for key in ['t_geocenter', 'ra', 'dec']}
            )['t_refdet']  # Will hold t_refdet fixed
        flipped_iota = {'iota': np.pi - self.par_dic_0['iota']}

        def get_updated_par_dic(thetanet, phinet):
            """Return `self.par_dic_0` with updated ra, dec, t_geocenter."""
            ra, dec = skyloc.thetaphinet_to_radec(thetanet, phinet)
            t_geocenter_dic = time_transformer.transform(
                t_refdet=t0_refdet, ra=ra, dec=dec)
            return self.par_dic_0 | {'ra': ra, 'dec': dec} | t_geocenter_dic

        @np.vectorize
        def lnlike_skyloc(thetanet, phinet):
            par_dic = get_updated_par_dic(thetanet, phinet)
            return max(self.lnlike_max_amp_phase(par_dic),
                       self.lnlike_max_amp_phase(par_dic | flipped_iota))

        # Maximize on a Halton sequence, then refine
        u_theta, u_phi = qmc.Halton(2, seed=seed).random(1500).T
        thetanets = np.arccos(2*u_theta - 1)
        phinets = 2 * np.pi * u_phi
        i_max = np.argmax(lnlike_skyloc(thetanets, phinets))

        result = minimize(lambda thetaphinet: -lnlike_skyloc(*thetaphinet),
                          x0=(thetanets[i_max], phinets[i_max]),
                          bounds=[(0, np.pi), (0, 2*np.pi)])

        # Fix sky location and decide whether to flip inclination:
        par_dic = get_updated_par_dic(*result.x)
        if (self.lnlike_max_amp_phase(par_dic)
                < self.lnlike_max_amp_phase(par_dic | flipped_iota)):
            par_dic.update(flipped_iota)

        self._par_dic_0 = par_dic
        print(f'Set sky location, lnL = {-result.fun}')

    def _optimize_phase_and_distance(self):
        """
        Find phase and distance that optimize coherent likelihood.
        Update 'phi_ref', 'd_luminosity' entries of `self.par_dic_0`
        in-place.
        Return log likelihood.
        """
        max_lnl, amp_bf, phase_bf = self.lnlike_max_amp_phase(
            self.par_dic_0, ret_amp_phase_bf=True)
        self.par_dic_0['phi_ref'] += phase_bf / 2
        self.par_dic_0['d_luminosity'] /= amp_bf

        # Decide between phi_ref or (phi_ref + pi) based on higher modes:
        self.waveform_generator.harmonic_modes = None  # Activate all modes
        if self.waveform_generator.harmonic_modes != [(2, 2)]:
            par_dic_1 = self.par_dic_0 | {'phi_ref': self.par_dic_0['phi_ref']
                                                     + np.pi}
            # No relative binning (current weights are for (2, 2) mode)
            if self.lnlike_fft(par_dic_1) > self.lnlike_fft(self.par_dic_0):
                self.par_dic_0['phi_ref'] += np.pi
            self.waveform_generator.harmonic_modes = [(2, 2)]  # Reset

        print(f'Set phase and distance, lnL = {max_lnl}')

    def set_mchirp_range(self, lnl_drop=5., max_doublings=2, seed=0):
        """
        Set self._mchirp_range as `(mchirp_min, mchirp_max)`, bounds for
        the chirp mass that are deemed safe for parameter esimation
        bounds.
        The criterion is that the incoherent likelihood maximized over
        ``(eta, chieff)`` drops by at least `lnl_drop` of its value for
        the reference waveform. Note this can give wrong results if the
        reference waveform is not close to maximum likelihood, or the
        likelihood is multimodal as a function of chirp mass.
        The chirp-mass range is expanded (roughly doubled) a maximum
        number of times given by `max_doublings`, if this is reached a
        warning is issued.
        """
        mchirp_0 = gw_utils.m1m2_to_mchirp(self.par_dic_0['m1'],
                                           self.par_dic_0['m2'])

        # Higher modes and precession can bias the solution. To prevent
        # this, don't shrink the error bars if the SNR is high:
        lnl_0 = self.lnlike_max_amp_phase_time(self.par_dic_0)
        conservative_snr = min(np.sqrt(2*lnl_0), 8)

        mchirp_range = list(
            gw_utils.estimate_mchirp_range(mchirp_0, snr=conservative_snr))

        def has_low_likelihood(mchirp):
            """
            Return boolean, whether the incoherent likelihood maximized
            over ``(eta, chieff)`` drops by at least `lnl_drop` of its
            value for the reference waveform.
            """
            lnl = -differential_evolution(
                lambda eta_chieff:
                    -self._lnlike_incoherent(mchirp, *eta_chieff),
                bounds=[self.eta_range, self.chieff_range],
                seed=seed, init='sobol').fun
            return lnl < lnl_0 - lnl_drop

        # Expand left and right edges of the range as necessary:
        for i in (0, 1):
            n_doublings = 0
            while not has_low_likelihood(mchirp_range[i]):
                if n_doublings >= max_doublings:
                    warnings.warn('Reached maximum `mchirp_range` expansions.')
                    break

                mchirp_range[i] = gw_utils.estimate_mchirp_range.expand_range(
                    mchirp_0, mchirp_range[i])
                n_doublings += 1

        self._mchirp_range = tuple(mchirp_range)
        print(f'Set mchirp_range = {self.mchirp_range}')

        # Issue a warning if the automatic range excludes the injection
        if self.event_data.injection:
            true_mchirp = gw_utils.m1m2_to_mchirp(
                self.event_data.injection['par_dic']['m1'],
                self.event_data.injection['par_dic']['m2'])
            if (self.mchirp_range[0] > true_mchirp
                    or self.mchirp_range[1] < true_mchirp):
                warnings.warn('Proposed `mchirp_range` excludes the injection')

    def get_coordinate_system_kwargs(self):
        """
        Return dictionary with parameters commonly required to set up
        coordinate system for sampling.
        Can be used to instantiate some classes defined in
        ``cogwheel.gw_prior``.

        Return
        ------
        dictionary with entries for:
            * tgps
            * par_dic_0
            * f_avg
            * f_ref
            * ref_det_name
            * detector_pair
            * t0_refdet
            * mchirp_range
            * event_data
            * approximant
        """
        lnl_by_detectors = self.lnlike_max_amp_phase_time(
            self.par_dic_0, return_by_detectors=True)

        sorted_dets = [det for _, det in sorted(zip(
            lnl_by_detectors, self.waveform_generator.detector_names))][::-1]
        ref_det_name = sorted_dets[0]
        detector_pair = ''.join(sorted_dets)[:2]

        f_avg = self.get_average_frequency(self.par_dic_0, ref_det_name)

        delay = gw_utils.time_delay_from_geocenter(
            ref_det_name, self.par_dic_0['ra'], self.par_dic_0['dec'],
            self.event_data.tgps)[0]
        t0_refdet = self.par_dic_0['t_geocenter'] + delay

        return {'tgps': self.event_data.tgps,
                'par_dic_0': self.par_dic_0,
                'f_avg': f_avg,
                'f_ref': self.par_dic_0['f_ref'],
                'ref_det_name': ref_det_name,
                'detector_pair': detector_pair,
                't0_refdet': t0_refdet,
                'mchirp_range': self.mchirp_range,
                'event_data': self.event_data,
                'approximant': self.waveform_generator.approximant}
