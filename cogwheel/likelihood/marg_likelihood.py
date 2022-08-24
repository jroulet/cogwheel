"""Compute likelihood of GW events."""
import numpy as np
from . import RelativeBinningLikelihood
from . import extrinsic_integration as cs
from cogwheel import gw_utils, utils
from scipy.special import i0e


class MarginalizedRelativeBinningLikelihood(RelativeBinningLikelihood):
    """
    Generalization of 'RelativeBinningLikelihood' that implements computation of 
    marginalized likelihood (over distance and phase) with the relative binning method.
    
    It integrates cogwheel and the extrinsic_integration routine.
    Intrinsic parameters are sampled using cogwheel.
    Extrinsic parameters (ra,dec,mu,psi,inclination) are sampled using the
    extrinsic_integration routine.
    Distance and Phase are sampled using analytical distribution functions.
    dist_ref is set to 1Mpc.

    Warning: Assumes detector_names is not something like 'H1L1'
    """
    def __init__(self, event_data, waveform_generator, par_dic_0, fbin=None,
                 pn_phase_tol=None, spline_degree=3,
                 t_rng=(-cs.DEFAULT_DT_MAX/1000, cs.DEFAULT_DT_MAX/1000),
                 dist_ref=1, nsamples=10000, **cs_kwargs):
        """
        :param event_data:
        :param waveform_generator:
        :param par_dic_0:
        :param fbin:
        :param pn_phase_tol:
        :param spline_degree:
        :param t_rng:
        :param dist_ref:
        :param nsamples:
            The number of random extrinsic samples used while evaluating
            the marginalized likelihood
        :param cs_kwargs:
            Dictionary with extra parameters to pass to
            cs.CoherentScore.from_new_samples
        """
        self.t_rng = t_rng
        self.dist_ref = dist_ref
        self.nsamples = nsamples

        # Treat arguments to cs.CoherentScore.from_new_samples
        self.cs_kwargs = cs_kwargs.copy()
        nra = cs_kwargs.pop("nra", 100)
        ndec = cs_kwargs.pop("ndec", 100)
        self.detnames = tuple(event_data.detector_names)
        cs_kwargs["gps_time"] = cs_kwargs.get("gps_time", event_data.tgps)
        cs_kwargs["dt_sinc"] = cs_kwargs.get("dt_sinc", cs.DEFAULT_DT)
        cs_kwargs["nsamples_mupsi"] = cs_kwargs.pop(
            "nsamples_mupsi", 10 * self.nsamples)

        self.cs_obj = cs.CoherentScore.from_new_samples(
            nra, ndec, self.detnames, **cs_kwargs)
        # From milisecond to seconds
        self.dt = self.cs_obj.dt_sinc/1000
        self.timeshifts = np.arange(*t_rng, self.dt)

        self.ref_pardict = {'d_luminosity': self.dist_ref, 'iota': 0.0, 'phi_ref': 0.0}
        
        super().__init__(event_data, waveform_generator, par_dic_0, fbin,
                         pn_phase_tol, spline_degree)
        
    @property
    def params(self):
        return sorted(
            set(self.waveform_generator._waveform_params) -
            self.ref_pardict.keys())

    def _set_summary(self):
        """
        Compute summary data for the fiducial waveform at all detectors.
        `asd_drift` is not applied to the summary data, to not have to
        keep track of it.
        Update `asd_drift` using the reference waveform.
        The summary data `self._d_h_weights` and `self._d_h_weights` are
        such that:
            (d|h) ~= sum(_d_h_weights * conj(h_fbin)) / asd_drift^2
            (h|h) ~= sum(_h_h_weights * abs(h_fbin)^2) / asd_drift^2

        Assuming no higher modes.
        """
        h0_f = self._get_h_f(self.par_dic_0, by_m=False)
        h0_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, self.par_dic_0, by_m=False)  # ndet x len(fbin)

        # (ntimes, ndet, nfft)
        d_h0 = (self.event_data.blued_strain *
                np.conj(h0_f) * np.exp(
                    2j * np.pi * self.event_data.frequencies *
                    self.timeshifts[:, np.newaxis, np.newaxis]))
        self._d_h_weights = self._get_summary_weights(d_h0) / np.conj(h0_fbin)
        # Add a time shift to have the peak at t=0
        # use t_gps(?)
        self._d_h_weights *= np.exp(
            1j * 2 * np.pi * self.fbin *
            (self.waveform_generator.tcoarse + self._par_dic_0['t_geocenter']))

        h0_h0 = h0_f * h0_f.conj() * self.event_data.wht_filter**2
        self._h_h_weights = (self._get_summary_weights(h0_h0)
                             / (h0_fbin * h0_fbin.conj()))  # (ndet, nbin)

        self.asd_drift = self.compute_asd_drift(self.par_dic_0)
        self._lnl_0 = self.lnlike(self.par_dic_0, bypass_tests=True)

        self.timestamps = NotImplemented  # TODO

    def get_z_timeseries(self, par_dic):
        """
        Return (d|h)/sqrt(h|h) timeseries with asd_drift correction
        applied (n_times x n_det), as well as the normfac
        """
        h_fbin, _ = self.waveform_generator.get_hplus_hcross(
            self.fbin, par_dic | self.ref_pardict)
        
        # Sum over f axis, leave det and time axes unsummed.
        d_h = (self._d_h_weights * h_fbin.conj()).sum(axis=-1)
        h_h = (self._h_h_weights * h_fbin * h_fbin.conj()).real.sum(axis=-1)
        norm_h = np.sqrt(h_h)
        
        return d_h / norm_h / self.asd_drift, norm_h

    def query_extrinsic_integrator(self, par_dic, **kwargs):
        """
        Generates timeseries for the intrinsic parameters in par_dic, and
        calls the extrinsic integration routine to compute the marginalized
        likelihood as well as the information needed to reconstruct samples
        of the marginalized parameters
        :param par_dic: Dictionary of intrinsic parameters
        :param kwargs:
            Generic variable to capture extra arguments, in this case,
            we can pass
                fixed_pars = Tuple with the names of the parameters to
                    hold fixed, can be 'tn' (time of the n^th detector), 'mu',
                    'psi', 'radec_ind'
                fixed_vals = Tuple with the values of the parameters we fix
                nsamples = Number of samples to use for the extrinsic parameters
                    while evaluating the marginalized likelihood, defaults to
                    self.nsamples
        :return:
            1. log(marginalized likelihood)
            2. nsamples x 6 array with each row having
               mu,
               psi,
               ra_index,
               dec_index,
               (unnormalized) relative contribution of sample to coherent score
               time corresponding to the sample in the first detector
            3. 2 x nsamples_phys complex array of Z \bar{T}, and \bar{T} T
               (implicit summation over detectors), useful for reconstructing
               the distance and phase samples
            4. z_timeseries (time x det)
            5. normfacs (ndet)
        """
        # 1) get z timeseries: (time x det)
        z_timeseries, norm_h = self.get_z_timeseries(par_dic)

        # Slice data segment that contains the event
        fixed_pars = kwargs.get("fixed_pars", None)
        fixed_vals = kwargs.get("fixed_vals", None)
        fixing = fixed_pars is not None

        t_indices = np.zeros(len(z_timeseries[0]), dtype=np.int32)
        for ind_det in range(len(t_indices)):
            tnstr = 't' + str(ind_det)
            if fixing and (tnstr in fixed_pars):
                tnind = fixed_pars.index(tnstr)
                tn = fixed_vals[tnind]
                t_indices[ind_det] = np.searchsorted(self.timeshifts, tn)
            else:
                t_indices[ind_det] = np.argmax(
                    np.real(z_timeseries[:, ind_det]) ** 2 +
                    np.imag(z_timeseries[:, ind_det]) ** 2)

        # Create a processedclist for the event
        event_phys = np.zeros((len(self.event_data.detector_names), 7))
        # Time, SNR^2, normfac, hole correction, ASD drift, Re(z), Im(z)
        event_phys[:, 0] = self.timeshifts[t_indices]
        event_phys[:, 1] = [np.abs(z_timeseries[tind, i]) ** 2
                            for i, tind in enumerate(t_indices)]
        event_phys[:, 2] = norm_h
        event_phys[:, 3] = 1
        event_phys[:, 4] = self.asd_drift
        event_phys[:, 5] = [np.real(z_timeseries[tind, i])
                            for i, tind in enumerate(t_indices)]
        event_phys[:, 6] = [np.imag(z_timeseries[tind, i])
                            for i, tind in enumerate(t_indices)]

        z_timeseries_cs = np.zeros(
            (len(self.event_data.detector_names), len(self.timeshifts), 3))
        z_timeseries_cs[:, :, 0] = self.timeshifts[:]
        z_timeseries_cs[:, :, 1] = np.transpose(z_timeseries.real)
        z_timeseries_cs[:, :, 2] = np.transpose(z_timeseries.imag)

        # Fix the number of samples
        nsamples = kwargs.get("nsamples", self.nsamples)
        prior_terms, _, samples, UT2samples = \
            self.cs_obj.get_all_prior_terms_with_samp(
                event_phys, timeseries=z_timeseries_cs, nsamples=nsamples,
                fixed_pars=fixed_pars, fixed_vals=fixed_vals)

        # Prior_terms[0] = 2 * log(marginalized likelihood)
        coherent_score = prior_terms[0] / 2

        return coherent_score, samples, UT2samples, z_timeseries, norm_h

    def lnlike(self, par_dic, **kwargs):
        """
        Return likelihood marginalized over distance and phase
        i.e. return coherent score
        """
        return self.query_extrinsic_integrator(par_dic, **kwargs)[0]

    def lnlike_no_marginalization_from_timeseries(
            self, z_timeseries, norm_h, par_dic):
        """Note that par_dic is larger than the one for lnlike as it contains
        all parameters in self.waveform_generator.params"""
        # Find the times in each detector to pick from the timeseries, and
        # compute the z at that time, as well as the predicted z
        zs = np.zeros(len(self.detnames), dtype=np.complex128)
        ts = np.zeros(len(self.detnames), dtype=np.complex128)
        for ind_det, det in enumerate(self.detnames):
            dt_det, = gw_utils.time_delay_from_geocenter(
                det, par_dic['ra'], par_dic['dec'], self.event_data.tgps)
            tdet = \
                par_dic['t_geocenter'] + dt_det - self.par_dic_0['t_geocenter']
            zs[ind_det] = self.sinc_interpolation_bruteforce(
                z_timeseries[:, ind_det], self.timeshifts, np.array([tdet]))[0]
            ts[ind_det] = norm_h[ind_det] / self.asd_drift[ind_det] * \
                cs.gen_sample_amps_from_fplus_fcross(
                    *gw_utils.fplus_fcross_detector(
                        det,
                        par_dic['ra'],
                        par_dic['det'],
                        0.0,
                        self.event_data.tgps).T[0],
                    np.cos(par_dic['iota']),
                    par_dic['psi'])

        U = np.dot(zs, np.conj(ts))
        T2 = np.dot(ts, np.conj(ts))

        Y_pick = \
            (self.dist_ref / par_dic['d_luminosity']) * \
            np.exp(2 * 1j * par_dic['phi_ref'])
        lnl = 0.5 * (np.abs(U) ** 2 / T2 - T2 * np.abs(Y_pick - U / T2) ** 2)

        return lnl

    def lnlike_no_marginalization(self, par_dic):
        """Note that par_dic is larger than the one for lnlike as it contains
        all parameters in self.waveform_generator.params
        """
        # 1) get z timeseries: (time x det)
        z_timeseries, norm_h = self.get_z_timeseries(
            {p: par_dic[p] for p in self.params})

        return self.lnlike_no_marginalization_from_timeseries(
            z_timeseries, norm_h, par_dic)

    def postprocess_samples(
            self, samples, force_update=False, accurate_lnl=False, **kwargs):
        """
        Adds columns 'iota', 'psi', 'ra', 'dec', 'd_luminosity', 'phi_ref',
        't_geocenter', 'lnl' to a DataFrame of samples, with values taken
        randomly from the conditional posteriors
        `samples` needs to have columns for all `self.params`.

        Parameters
        ----------
        samples: Dataframe with self.params
        force_update: bool, whether to force an update if the extrinsic samples
                      already exist
        accurate_lnl: Flag that if true, forces the lnl to be computed without
                      discreteness artifacts
        kwargs:
            Generic variable to capture extra arguments, in this case,
            we can pass
                fixed_pars = Tuple with the names of the parameters to
                    hold fixed, can be 'tn' (time of the n^th detector), 'mu',
                    'psi', 'radec_ind'
                fixed_vals = Tuple with the values of the parameters we fix
                nsamples = Number of samples to use for the extrinsic parameters
                    while evaluating the marginalized likelihood, defaults to
                    self.nsamples
        """
        cols_to_add = ['iota',
                       'psi',
                       'ra',
                       'dec',
                       'd_luminosity',
                       'phi_ref',
                       't_geocenter',
                       'lnl']
        if (not force_update) and (set(cols_to_add) <= set(samples.columns)):
            return

        # Initialize the columns
        for pname in cols_to_add:
            samples[pname] = np.zeros(len(samples))

        for ind, pvals in enumerate(zip(*[samples[p] for p in self.params])):
            par_dic = {key: val for key, val in zip(self.params, pvals)}
            ext_pars = self.generate_extrinsic_params_from_cs(
                par_dic, accurate_lnl=accurate_lnl, **kwargs)
            for key, val in zip(self.params, ext_pars):
                samples.loc[ind, key] = val

    def generate_extrinsic_params_from_cs(
            self, par_dic, accurate_lnl=False, **kwargs):
        """
        Given a set of intrinsic params in par_dic, this function returns
        particular values for 'iota', 'psi', 'ra', 'dec', 'd_luminosity',
        'phi_ref', and 't_geocenter' from their conditional posteriors, and
        additionally computes the lnl (unmarginalized) for these extrinsic
        samples
        The accurate_lnl flag, if true, forces the lnl to be computed without
        discreteness artifacts. Note that the sampled lnl has discreteness
        artifacts, so if reweighting, use accurate_lnl=False
        """
        # Call the extrinsic integrator routine
        coherent_score, samples, UT2samples, z_timeseries, norm_h = \
            self.query_extrinsic_integrator(par_dic, **kwargs)
        
        # Define weights and pick from the samples with these weights
        cweights = np.cumsum(samples[:, 4])
        cweights /= cweights[-1]
        idx_rand = utils.rand_choice_nb(np.arange(len(samples)), cweights, 1)

        # Choose one point for iota, psi, ra, dec, t_geocenter from the
        # distributions
        iota = np.arccos(samples[idx_rand, 0])
        psi = samples[idx_rand, 1]
        # Note: This loses a bit of sky resolution due to discreteness
        ra = self.cs_obj.ra_grid[samples[idx_rand, 2].astype(int)]
        dec = self.cs_obj.dec_grid[samples[idx_rand, 3].astype(int)]

        # Convert time at the first detector to geocentric time
        time_det0 = samples[idx_rand, 5]
        dt_1, = gw_utils.time_delay_from_geocenter(
            self.detnames[0], ra, dec, self.event_data.tgps)
        t_geocenter = time_det0 - dt_1 + self.par_dic_0['t_geocenter']

        U, T2 = UT2samples[:, idx_rand]
        T2 = T2.real

        # Generate distance and phi_ref samples given U and T2
        d_luminosity, phi_ref = \
            self.get_distance_phase_point_for_given_U_T2(U, T2)

        # get corresponding value for U and T2
        if accurate_lnl:
            lnl = self.lnlike_no_marginalization_from_timeseries(
                z_timeseries, norm_h, par_dic | {
                    'iota': iota,
                    'psi': psi,
                    'ra': ra,
                    'dec': dec,
                    'd_luminosity': d_luminosity,
                    'phi_ref': phi_ref,
                    't_geocenter': t_geocenter})
        else:
            # Computes ln(likelihood)
            # Note that this has discreteness errors corresponding to cogwheel
            Y_pick = (self.dist_ref / d_luminosity) * np.exp(2 * 1j * phi_ref)
            lnl = (np.abs(U) ** 2 / T2 - T2 * np.abs(Y_pick - U / T2) ** 2) / 2
        
        return iota, psi, ra, dec, d_luminosity, phi_ref, t_geocenter, lnl

    def get_distance_phase_point_for_given_U_T2(self, U, T2):
        """
        Function to sample distance and phase from their analytical
        distributions given U, T2 and UT2samples from an extrinsic
        marginalization for a single set of intrinsic params
        :return: Single value for distance and phase
        """
        mean = abs(U) / T2  # expected mean for 1/distance in units of dist_ref
        sigma = 1 / np.sqrt(T2) # spread for 1/distance in units of dist_ref
        # Demand that the minimum of u_vals can't be zero or negative because
        # u_vals~1/dist
        u_vals = np.linspace(
            max(1e-5, mean - 7 * sigma), mean + 7 * sigma, 1000)
        like_vals = self.posterior_func_for_dist(u_vals, U, T2)
        # Compute normalized cdf
        cdf = np.cumsum(like_vals)
        cdf /= cdf[-1]
        u_pick = utils.rand_choice_nb(u_vals, cdf, 1)[0]
        dist_pick = self.dist_ref / u_pick

        mean = np.angle(U) / 2  # central value for phase
        sigma = 1 / np.sqrt(abs(U) * u_pick)  # spread for the d_phase
        v_vals = np.linspace(
            max(-7 * sigma, -np.pi / 2), min(7 * sigma, np.pi / 2), 1000)
        like_vals = self.posterior_func_for_phase(v_vals, U, u_pick)
        # compute normalized cdf
        cdf = np.cumsum(like_vals)
        cdf /= cdf[-1]
        # Obtain a d_phase value
        v_pick = utils.rand_choice_nb(v_vals, cdf, 1)[0]
        # Add it to the central value of the phase
        # (np.random.random()>0.5)*np.pi to take care of the pi degeneracy
        # of the 22 mode
        phi_pick = \
            (v_pick + mean + (np.random.random() > 0.5) * np.pi) % (2 * np.pi)

        return dist_pick, phi_pick

    @staticmethod
    def posterior_func_for_dist(u_pr, U, T2):
        """
        Analytical posterior funtion for distance
        Eq. (3.39) in Teja's note
        u_pr is 1/dist
        """
        U_abs = abs(U)
        exponent = - 0.5 * T2 * (u_pr - U_abs / T2) ** 2
        return (1. / u_pr ** 4) * np.exp(exponent) * i0e(u_pr * U_abs)

    @staticmethod
    def posterior_func_for_phase(v_pr, U, y):
        """
        Analytical posterior funtion for phase
        Eq. (3.38) in Teja's note
        v_pr is d_phase
        """
        U_abs = abs(U)
        return np.exp(U_abs * y * np.cos(2 * v_pr))

    @staticmethod
    def sinc_interpolation_bruteforce(x, s, u):
        """
        Interpolates x, sampled at "s" instants
        Output y is sampled at "u" instants ("u" for "upsampled")
        u has to be 1D
        Complexity len(s) x len(u)
        from Matlab:
        http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
        """
        assert len(x) == len(s), "The domain and range must be of the same length"
        # Find the period
        dt = s[1] - s[0]
        sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, None], (1, len(u)))
        y = np.dot(x, np.sinc(sincM / dt))
        return y
