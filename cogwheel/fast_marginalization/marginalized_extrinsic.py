from collections import namedtuple
import scipy
from scipy.stats import qmc
import numpy as np
import pandas as pd

from cogwheel import gw_prior
from cogwheel import gw_utils
from cogwheel import likelihood
from cogwheel import skyloc_angles
from cogwheel import utils
from cogwheel.fast_marginalization.adaptive import ExtrinsicMap


_MarginalizationProducts = namedtuple(
    '_MarginalizationProducts',
    ['lnl_max', 'marg_like', 'weights', 'd_h', 'h_h', 'sign_id',
     'important', 'dt_linfree'])
_MarginalizationProducts.__doc__ = """
    Attributes
    ----------
    lnl_max: float
        log(likelihood marginalized over distance) and maximized over
        time, sky location, polarization and orbital phase.

    marg_like: 1d float array (i)
        log(likelihood marginalized over distance) of important samples.

    weights: 1d float array (i)
        Prior weight of important samples.

    d_h: 2d float array (fm)
        Inner product <d|h> for QMC samples, mode by mode.

    h_h: 2d float array (fm)
        Inner product <h|h> for QMC samples, mode by mode.

    sign_id: 0 or 1
        Whether costheta_jn > 0

    important: [tuple of int, tuple of int]
        Indices to QMC samples and orbital phases, of configurations
        with incoherent likelihood sufficiently high to be included in
        the marginalization integral.

    dt_linfree: float
        Time convention difference between LAL and the linear-free
        waveform, the convention is t_lal = t_linfree + dt_linfree.
    """

class MarginalizedExtrinsicLikelihood(
        likelihood.RelativeBinningLikelihood):
    """
    Class to evaluate the likelihood marginalized over sky location,
    time of arrival, polarization, distance and orbital phase for
    quasicircular waveforms with generic harmonic modes and spins, and
    to resample these parameters from the conditional posterior for
    demarginalization in postprocessing.

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
        i: important (i.e. with high enough likelihood) sample id
    """
    # Note: costheta_jn is redundant, but we don't want to compute it
    # here so it's required
    params = ['costheta_jn', 'f_ref', 'iota', 'l1', 'l2',
              'm1', 'm2', 's1x_n', 's1y_n', 's1z', 's2x_n', 's2y_n', 's2z']
    qmc_params = sorted(['ra', 'dec', 'psi', 't_linfree_geocenter'])

    # Remove from the orbital phase integral any sample with a drop in
    # log-likelihood from the peak bigger than ``DLNL_THRESHOLD``:
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
        self.fast_parameter_slice = slice(None)

    @property
    def nphi(self):
        return self._nphi

    @nphi.setter
    def nphi(self, nphi):
        phi_ref, dphi = np.linspace(0, 2*np.pi, nphi,
                                    endpoint=False, retstep=True)
        m_arr = np.fromiter(self.waveform_generator._harmonic_modes_by_m, int)
        m_inds, mprime_inds = self._get_m_mprime_inds()
        self._nphi = nphi
        self._dh_phasor = np.exp(-1j * np.outer(m_arr, phi_ref))  # mo
        self._hh_phasor = np.exp(
            1j * np.outer(m_arr[m_inds] - m_arr[mprime_inds], phi_ref))  # mo
        self._phi_ref = phi_ref
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

        self._set_fplus_fcross_and_detector_times()
        self._coarse_times, self._t_inds, self._d_h_weights = zip(
            *[self._get_d_h_weights_det(i_det)
              for i_det in range(len(self.event_data.detector_names))])
        self._set_h_h_weights()

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

    def _get_d_h_weights_det(self, i_det, time_resolution=5e-4):
        coarse_times_det, t_inds_det = self._get_coarse_grained_times_and_inds(
            self._detector_times[i_det], tol=time_resolution)  # t, sf
        t_inds_det = [tuple(inds) for inds in t_inds_det]

        shifts = np.exp(2j*np.pi * np.einsum(
            'r,t->tr',
            self.event_data.frequencies,
            self.waveform_generator.tcoarse + coarse_times_det))
        d_h_integrand = np.einsum('r,mpr,tr->tmpr',
                                  self.event_data.blued_strain[i_det],
                                  self._h0_f.conj(),
                                  shifts)  # tmpr
        d_h_weights = (self._get_summary_weights(d_h_integrand)
                       / self._h0_fbin.conj()
                       / self.asd_drift[i_det]**2)  # tmpb
        return coarse_times_det, t_inds_det, d_h_weights

    def _set_h_h_weights(self):
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

    def _set_fplus_fcross_and_detector_times(self):
        get_fplus_fcross = np.vectorize(gw_utils.fplus_fcross, excluded={0},
                                        signature='(),(),(),()->(2,d)')

        get_time_delay_from_geocenter = np.vectorize(
            gw_utils.time_delay_from_geocenter, excluded={0},
            signature='(),(),()->(n)')

        self._fplus_fcross = get_fplus_fcross(
            self.event_data.detector_names,
            self.fast_parameters_sequence['ra'],
            self.fast_parameters_sequence['dec'],
            self.fast_parameters_sequence['psi'],
            self.event_data.tgps)

        n_s, n_f, n_p, n_d = self._fplus_fcross.shape
        self._f_f = (np.einsum('sfpd,sfPd->sfpPd',
                               self._fplus_fcross,
                               self._fplus_fcross)
                     .reshape(n_s, n_f, n_p * n_p * n_d)
                     .copy())  # sf(pPd)

        time_delays = get_time_delay_from_geocenter(
            self.event_data.detector_names,
            self.fast_parameters_sequence['ra'],
            self.fast_parameters_sequence['dec'],
            self.event_data.tgps).transpose(2, 0, 1)  # dsf

        self._detector_times = (
            self.fast_parameters_sequence['t_linfree_geocenter']
            + time_delays)  # dsf

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
        hplus_hcross_lf = np.exp(-1j * slope * self.fbin) * hplus_hcross  # mpb
        dt_linfree = slope / (2*np.pi)  # t_lal = t_linfree + dt_linfree
        return hplus_hcross_lf, dt_linfree

    def _get_dh_hh(self, par_dic):
        hplus_hcross, dt_linfree = self._get_hplus_hcross_linear_free(
            par_dic)  # mpb
        sign_id = int(par_dic['costheta_jn'] > 0)

        # (d|h):
        # Compute sparse (d|h) timeseries with hplus, hcross at detectors
        n_detectors = len(self.event_data.detector_names)
        d_h_timeseries = [np.einsum('tmpb,mpb->mpt',
                                    self._d_h_weights[i_det],
                                    hplus_hcross.conj())
                          for i_det in range(n_detectors)]

        # Spline-interpolate to the desired times
        detector_times = self._detector_times[:,
                                              sign_id,
                                              self.fast_parameter_slice]
        d_h_pluscross = np.array(
            [scipy.interpolate.interp1d(self._coarse_times[i_det],
                                        d_h_timeseries[i_det],
                                        assume_sorted=True, kind=3,
                                        bounds_error=False, fill_value=0.
                                       )(detector_times[i_det])
             for i_det in range(n_detectors)])  # dmpf

        # Apply antenna factors:
        d_h = np.einsum('dmpf,fpd->fm',
                        d_h_pluscross,
                        self._fplus_fcross[sign_id, self.fast_parameter_slice]
                        )  # fm

        # (h|h): We compute the following quantity in a faster way below
        # h_h = np.einsum('mpPdb,fpPd,mpb,mPb->fm',
        #                 self._h_h_weights,
        #                 f_f,
        #                 hplus_hcross[m_inds],
        #                 hplus_hcross.conj()[mprime_inds],
        #                 optimize=<precomputed_optimal_path>)  # fm

        m_inds, mprime_inds = self._get_m_mprime_inds()
        h_h_mppb = np.einsum('mpb,mPb->mpPb',
                             hplus_hcross[m_inds],
                             hplus_hcross.conj()[mprime_inds])  # mpPb

        h_h_mppd = np.einsum('mpPb,mpPdb->mpPd',
                             h_h_mppb,
                             self._h_h_weights)  # mpPd

        n_mm, n_p, _, n_d = h_h_mppd.shape
        h_h = (self._f_f[sign_id, self.fast_parameter_slice]
               @ h_h_mppd.reshape(n_mm, n_p*n_p*n_d).T)  # fm

        return d_h, h_h, dt_linfree

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
        """
        Constructor that automatically chooses a set of fast parameter
        samples (as a quasi Monte Carlo sequence following a
        distribution matched to slices of the posterior around the
        maximum likelihood solution).

        Parameters
        ----------
        aux_posterior: posterior.Posterior
            A Posterior object, should be marginalized over distance
            only.

        log2nsamples: int
            Base-2 logarithm of the length of the QMC sequence.
        """
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
        polarizations, detectors). The strain must correspond to the
        reference distance ``self.lookup_table.REFERENCE_DISTANCE``.

        Parameters
        ----------
        d_h: float
            Inner product of data and model strain.

        h_h: float
            Inner product of strain with itself.
        """
        return self.lookup_table(d_h, h_h) + d_h**2 / h_h / 2

    def _get_marginalization_products(self, par_dic):
        d_h_0, h_h_0, dt_linfree = self._get_dh_hh(par_dic)  # fm, fm, scalar

        d_h = (d_h_0 @ self._dh_phasor).real  # fo
        h_h = (h_h_0 @ self._hh_phasor).real  # fo

        max_over_distance_lnl = d_h * np.abs(d_h) / h_h / 2  # fo
        important = np.where(
            max_over_distance_lnl
            > np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD)
        lnl_marg_dist = self._lnlike_marginalized_over_distance(
            d_h[important], h_h[important])  # i

        lnl_max = lnl_marg_dist.max()
        marg_like = np.exp(lnl_marg_dist - lnl_max)  # i
        sign_id = int(par_dic['costheta_jn'] > 0)
        weights = self.fast_parameters_sequence['weights'][
            sign_id, important[0]]  # i
        return _MarginalizationProducts(lnl_max=lnl_max,
                                        marg_like=marg_like,
                                        weights=weights,
                                        d_h=d_h,
                                        h_h=h_h,
                                        sign_id=sign_id,
                                        important=important,
                                        dt_linfree=dt_linfree)

    def lnlike(self, par_dic):
        """
        Return natural logarithm of the marginalized likelihood.
        The likelihood is marginalized over arrival time, polarization,
        sky location, orbital phase and distance.
        """
        mar = self._get_marginalization_products(par_dic)
        return mar.lnl_max + np.log(mar.marg_like.dot(mar.weights)
                                    * self._dphi / len(mar.d_h))

    def _sample_fast_parameters(self, **par_dic):
        """
        Return a dict with a sample of fast parameters drawn from the
        posterior conditional on given slow parameters.
        """
        mar = self._get_marginalization_products(par_dic)
        prob = mar.marg_like * mar.weights
        prob /= prob.sum()
        ind = np.random.choice(len(prob), p=prob)
        f_ind = mar.important[0][ind]
        o_ind = mar.important[1][ind]

        fast_sample = {
            par: self.fast_parameters_sequence[par][mar.sign_id, f_ind]
            for par in self.qmc_params}

        fast_sample['d_luminosity'] = self.lookup_table.sample_distance(
            mar.d_h[f_ind, o_ind], mar.h_h[f_ind, o_ind])

        fast_sample['phi_ref'] = self._phi_ref[o_ind]

        fast_sample['t_geocenter'] = (
            fast_sample['t_linfree_geocenter'] + mar.dt_linfree)
        return fast_sample

    def postprocess_samples(self, samples: pd.DataFrame):
        """
        Add columns for fast parameters (self.qmc_parameters,
        'd_luminosity' and 'phi_ref') to a DataFrame of slow-parameter
        samples, with values taken randomly from the conditional
        posterior.
        `samples` needs to have columns for all `self.params`.
        """
        slow = samples[self.params]
        fast = pd.DataFrame(
            list(np.vectorize(self._sample_fast_parameters)(**slow)))
        utils.update_dataframe(samples, fast)
