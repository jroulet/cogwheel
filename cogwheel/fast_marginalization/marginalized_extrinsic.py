import itertools
import numpy as np
from scipy.stats import qmc

from cogwheel import gw_prior
from cogwheel import gw_utils
from cogwheel import likelihood
from cogwheel import skyloc_angles
from cogwheel.fast_marginalization.adaptive import ExtrinsicMap


class MarginalizedExtrinsicLikelihood(
        likelihood.RelativeBinningLikelihood):
    """
    # TODO: smooth to_json and fast_parameters_sequence

    Note: comments throughout the code refer to array indices as follows:
        s: sign(cos(theta_jn)) id
        f: fast parameter sample id (each with ra, dec, t_linfree, psi)
        m: harmonic number id
        p: polarization (+ or x) id
        d: detector id
        b: frequency bin id
        r: rfft frequency id
        t: coarse-grained time id
        o: orbital phase id
    """
    # Note: costheta_jn is redundant, but we don't want to compute it
    # here so it's required
    params = ['costheta_jn', 'f_ref', 'iota', 'l1', 'l2',
              'm1', 'm2', 's1x_n', 's1y_n', 's1z', 's2x_n', 's2y_n', 's2z']
    qmc_params = sorted(['ra', 'dec', 'psi', 't_linfree_geocenter'])

    # Remove points with a drop in log-likelihood from the peak bigger
    # than ``DLNL_THRESHOLD`` from the orbital phase integral:
    DLNL_THRESHOLD = 12.

    def __init__(self, fast_parameters_sequence, lookup_table, event_data,
                 waveform_generator, par_dic_0, fbin=None, pn_phase_tol=None,
                 spline_degree=3, nphi=64):

        self.lookup_table = lookup_table
        self.fast_parameters_sequence = {
            k: np.asarray(v) for k, v in fast_parameters_sequence.items()}

        super().__init__(event_data, waveform_generator, par_dic_0,
                         fbin, pn_phase_tol, spline_degree)

        self._dh_phasor = None  # Set by nphi.setter
        self._hh_phasor = None  # Set by nphi.setter
        self._dphi = None  # Set by nphi.setter
        self.nphi = nphi

    @property
    def nphi(self):
        return self._nphi

    @nphi.setter
    def nphi(self, nphi):
        phis, dphi = np.linspace(0, 2*np.pi, nphi, endpoint=False, retstep=True)
        m_arr = np.fromiter(self.waveform_generator._harmonic_modes_by_m, int)
        m_inds, mprime_inds = self._get_m_mprime_inds()
        self._nphi = nphi
        self._dh_phasor = np.exp(-1j * phis * m_arr[:, np.newaxis])  # mo
        self._hh_phasor = np.exp(
            1j * phis * (m_arr[m_inds]-m_arr[mprime_inds])[:, np.newaxis])  # mo
        self._dphi = dphi


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

        Note: all spin components in `self.par_dic_0` are used, even if
        `self.waveform_generator.disable_precession` is set to `True`.
        This is so that the reference waveform remains the same when
        toggling `disable_precession`.
        """
        # Don't zero the in-plane spins for the reference waveform
        disable_precession = self.waveform_generator.disable_precession
        self.waveform_generator.disable_precession = False

        shape = (len(self.waveform_generator._harmonic_modes_by_m),
                 2, len(self.event_data.frequencies))
        self._h0_f = np.zeros(shape, dtype=np.complex_)
        self._h0_f[..., self.event_data.fslice] \
            = self.waveform_generator.get_hplus_hcross(
                self.event_data.frequencies[self.event_data.fslice],
                self.par_dic_0, by_m=True)  # mpr

        self._h0_fbin = self.waveform_generator.get_hplus_hcross(
            self.fbin, self.par_dic_0, by_m=True)  # mpb

        fplus_fcross, time_delays \
            = self._get_fplus_fcross_and_time_delays()  # sfpd, sfd
        self._set_d_h_weights(fplus_fcross, time_delays)
        self._set_h_h_weights(fplus_fcross)

        self.asd_drift = self.compute_asd_drift(self.par_dic_0)

        # Weights to determine the linear-free phase
        ind_m_2 = list(self.waveform_generator._harmonic_modes_by_m).index(2)
        weights_f = np.einsum(
            'dr,dr,d->r',
            np.abs(self._get_h_f(self.par_dic_0, by_m=True)[ind_m_2]),
            self.event_data.wht_filter**2,
            self.asd_drift**-2)  # r

        self._polyfit_weights = np.sqrt(np.clip(
            self._get_summary_weights(weights_f).real,
            0, None))  # b

        # Reset
        self.waveform_generator.disable_precession = disable_precession

    def _set_d_h_weights(self, fplus_fcross, time_delays):
        detector_times = (
            self.fast_parameters_sequence['t_linfree_geocenter'][..., np.newaxis]
            + time_delays)  # sfd

        # For some of the samples the arrival time at a detector might
        # be different enough from that of the reference waveform that
        # new summary data is warranted. Below we get coarse-grained
        # times of arrival at each detector to compute summary data
        # efficiently, because `sf` indices can take many more values
        # than `t`. Avoid arrays with `sfr` indices at any point.
        self._d_h_weights = np.zeros(
            self.fast_parameters_sequence['ra'].shape + self._h0_fbin.shape,
            np.complex_)  # sfmpb

        for i_det in range(len(self.event_data.detector_names)):
            coarse_times, t_inds = self._get_coarse_grained_times_and_inds(
                detector_times[..., i_det], tol=.005)  # t, sf

            coarse_shifts = np.exp(2j * np.pi * np.einsum(
                'r,t->tr', self.event_data.frequencies,
                self.waveform_generator.tcoarse + coarse_times))  # tr

            fine_shifts = np.exp(2j*np.pi * np.einsum(
                'b,sf->sfb',
                self.fbin,
                detector_times[..., i_det] - coarse_times[t_inds]))  # sfb

            d_h0 = np.einsum('r,mpr,tr->tmpr',
                             self.event_data.blued_strain[i_det],
                             self._h0_f.conj(),
                             coarse_shifts)  # tmpr
            aux_weights = (self._get_summary_weights(d_h0)
                           / self.asd_drift[i_det]**2) # tmpb

            self._d_h_weights += np.einsum('sfmpb,sfp,sfb,mpb->sfmpb',
                                           aux_weights[t_inds],
                                           fplus_fcross[..., i_det],
                                           fine_shifts,
                                           1 / self._h0_fbin.conj())  # sfmpb

    def _set_h_h_weights(self, fplus_fcross):
        m_inds, mprime_inds = self._get_m_mprime_inds()
        h0_h0 = np.einsum('mpr,mPr,dr,d->mpPdr',
                          self._h0_f[m_inds],
                          self._h0_f[mprime_inds].conj(),
                          self.event_data.wht_filter ** 2,
                          self.asd_drift ** -2)  # mpPdr
        self._h_h_weights = np.einsum(
            'mpPdb,mpb,mPb->mpPdb',
            self._get_summary_weights(h0_h0),
            1 / self._h0_fbin[m_inds],
            1 / self._h0_fbin[mprime_inds].conj())  # mpPdb

        # Count off-diagonal terms twice:
        self._h_h_weights[~np.equal(m_inds, mprime_inds)] *= 2

        n_s, n_f, n_p, n_d = fplus_fcross.shape
        self._f_f = (np.einsum('sfpd,sfPd->sfpPd', fplus_fcross, fplus_fcross)
                     .reshape(n_s, n_f, n_p * n_p * n_d)
                     .copy())  # sf(pPd)

    def _get_fplus_fcross_and_time_delays(self):
        """Accepts arrays."""
        get_fplus_fcross = np.vectorize(
            gw_utils.fplus_fcross, excluded={0}, signature='(),(),(),()->(2,d)')

        get_time_delay_from_geocenter = np.vectorize(
            gw_utils.time_delay_from_geocenter, excluded={0},
            signature='(),(),()->(n)')

        fplus_fcross = get_fplus_fcross(self.event_data.detector_names,
                                        self.fast_parameters_sequence['ra'],
                                        self.fast_parameters_sequence['dec'],
                                        self.fast_parameters_sequence['psi'],
                                        self.event_data.tgps)

        time_delays = get_time_delay_from_geocenter(
            self.event_data.detector_names,
            self.fast_parameters_sequence['ra'],
            self.fast_parameters_sequence['dec'],
            self.event_data.tgps)

        return fplus_fcross, time_delays

    def _get_hplus_hcross_linear_free(self, par_dic):
        """
        Return strain by mode, polarization and frequency, time-shifted so
        as to remove correlations between time and intrinsic parameters
        arising from the arbitrary time convention.

        Note: the phase is not made linear free, it's set to phi_ref = 0.
        The luminosity distance is set to self.lookup_table.REFERENCE_DISTANCE
        """
        hplus_hcross = self.waveform_generator.get_hplus_hcross(
            self.fbin,
            par_dic | {'d_luminosity': self.lookup_table.REFERENCE_DISTANCE,
                       'phi_ref': 0.},
            by_m=True)  # mpb
        ind_m_2 = list(self.waveform_generator._harmonic_modes_by_m).index(2)

        dphase = np.unwrap(np.angle(hplus_hcross[ind_m_2, 0]
                                    / self._h0_fbin[ind_m_2, 0]))  # b

        fit = np.polynomial.Polynomial.fit(self.fbin, dphase, deg=1,
                                           w=self._polyfit_weights)

        slope = fit.convert().coef[1]
        return np.exp(-1j * slope * self.fbin) * hplus_hcross  # mpb

    def _get_dh_hh(self, par_dic):
        hplus_hcross = self._get_hplus_hcross_linear_free(par_dic)  # mpb
        sign_id = int(par_dic['costheta_jn'] > 0)

        # (d|h): We compute the following quantity in faster way below
        # d_h = np.einsum('fmpb,mpb->fm',
        #                 self._d_h_weights[sign_id],
        #                 hplus_hcross.conj())  # fm

        n_f, n_m, n_p, n_b = self._d_h_weights[sign_id].shape
        d_h = np.empty((n_f, n_m), np.complex_)  # fm
        for i_m in range(n_m):
            d_h[:, i_m] = (
                self._d_h_weights[sign_id, :, i_m].reshape(n_f, n_p*n_b)
                @ hplus_hcross[i_m].conj().reshape(n_p*n_b))  # f

        m_inds, mprime_inds = self._get_m_mprime_inds()
        # (h|h): We compute the following quantity in a faster way below
        # h_h = np.einsum('mpPdb,fpPd,mpb,mPb->fm',
        #                 self._h_h_weights,
        #                 f_f,
        #                 hplus_hcross[m_inds],
        #                 hplus_hcross.conj()[mprime_inds],
        #                 optimize=<precomputed_optimal_path>)  # fm

        h_h_mppb = np.einsum('mpb,mPb->mpPb',
                             hplus_hcross[m_inds],
                             hplus_hcross.conj()[mprime_inds])  # mpPb

        h_h_mppd = np.einsum('mpPb,mpPdb->mpPd',
                             h_h_mppb,
                             self._h_h_weights)  # mpPd

        n_mm, n_p, n_p, n_d, n_b = self._h_h_weights.shape
        n_ppd = np.prod((n_p, n_p, n_d))
        h_h = self._f_f[sign_id] @ h_h_mppd.reshape(n_mm, n_ppd).T  # fm

        return d_h, h_h

    @staticmethod
    def _get_coarse_grained_times_and_inds(times, tol):
        """
        Find a small set of coarse-grained times such that all ``times``
        are closer than ``tol`` to at least one coarse-grained time.
        Also find the corresponding indices.

        Parameters
        ----------
        times: float array

        tol: float
            Maximum distance between a time and a coarse-grained time
            allowed. Larger `tol` values generally result in fewer
            coarse-grained times.

        Return
        ------
        coarse_grained_times: 1-d float array
            Array with sorted coarse-grained times.

        t_inds: int array of the same shape as `times`
            The indices of the closest coarse-grained times
            corresponding to each element of `times`.
        """
        sorted_times = np.sort(times, axis=None)

        coarse_grained_times = [sorted_times[0]]
        for t in sorted_times:
            if t - coarse_grained_times[-1] > tol:
                coarse_grained_times.append(t)
        coarse_grained_times = np.array(coarse_grained_times)

        t_inds = np.searchsorted(
            (coarse_grained_times[1:] + coarse_grained_times[:-1]) / 2,
            times)
        return coarse_grained_times, t_inds

    @classmethod
    def from_aux_posterior(cls, aux_posterior, log2nsamples: int):
        if not isinstance(aux_posterior.likelihood,
                          likelihood.MarginalizedDistanceLikelihood):
            raise ValueError(
                'The posterior should be marginalized over distance.')

        extrinsic_map = ExtrinsicMap.from_posterior(aux_posterior)

        nparams = len(extrinsic_map.sampled_params)
        nsamples = 2 ** log2nsamples
        u_samples = np.reshape(
            qmc.Sobol(nparams).random_base2(log2nsamples + 1).T,
            (nparams, 2, nsamples))  # 2 is for sign(costheta_jn) options
        fast_parameters_sequence = extrinsic_map.transform(*u_samples)
        fast_parameters_sequence['weights'] = np.exp(
            np.vectorize(extrinsic_map.lnprior)(*u_samples))

        # Add ra, dec:
        thetanet = np.arccos(fast_parameters_sequence['costhetanet'])
        phinet = fast_parameters_sequence['phinet_hat'] + [[0], [np.pi]]
        dic = aux_posterior.prior.get_init_dict()
        skyloc = skyloc_angles.SkyLocAngles(dic['detector_pair'], dic['tgps'])
        ra, dec = np.vectorize(skyloc.thetaphinet_to_radec,
                               signature='(),()->(),()')(thetanet, phinet)
        fast_parameters_sequence.update(ra=ra, dec=dec)

        lfp = aux_posterior.prior.subpriors[
            aux_posterior.prior.prior_classes.index(
                gw_prior.linear_free.LinearFreePhaseTimePrior)]

        intrinsic_0 = {par: aux_posterior.likelihood.par_dic_0[par]
                       for par in lfp._intrinsic}
        @np.vectorize
        def get_t_linfree_geocenter(t_linfree, ra, dec):
            return lfp.transform(phi_linfree=0, t_linfree=t_linfree, ra=ra,
                                 dec=dec, psi=0, iota=0, **intrinsic_0
                                 )['t_geocenter']

        fast_parameters_sequence['t_linfree_geocenter'] \
            = get_t_linfree_geocenter(fast_parameters_sequence['t_linfree'],
                                      fast_parameters_sequence['ra'],
                                      fast_parameters_sequence['dec'])

        return cls(fast_parameters_sequence,
                   **aux_posterior.likelihood.get_init_dict())

    def _lnlike_marginalized_over_distance(self, d_h, h_h):
        """
        Return log of the distance-marginalized likelihood.
        Note, d_h and h_h are real numbers (already summed over modes,
        polarizations, detectors).

        Parameters
        ----------
        d_h: float
            Inner product of data and model strain.

        h_h: float
            Inner product of strain with itself.
        """
        return self.lookup_table(d_h, h_h) + d_h**2 / h_h / 2

    def lnlike(self, par_dic):
        d_h_0, h_h_0 = self._get_dh_hh(par_dic)  # fm, fm

        d_h = (d_h_0 @ self._dh_phasor).real  # fo
        h_h = (h_h_0 @ self._hh_phasor).real  # fo

        max_over_distance_lnl = d_h * np.abs(d_h) / h_h / 2  # fo
        important = np.where(
            max_over_distance_lnl
            > np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD)
        lnl_marg_dist = self._lnlike_marginalized_over_distance(
            d_h[important], h_h[important])

        lnl_max = lnl_marg_dist.max()
        marg_like = np.exp(lnl_marg_dist - lnl_max)
        sign_id = int(par_dic['costheta_jn'] > 0)
        weights = self.fast_parameters_sequence['weights'][sign_id,
                                                           important[0]]
        return lnl_max + np.log(marg_like.dot(weights) * self._dphi)
