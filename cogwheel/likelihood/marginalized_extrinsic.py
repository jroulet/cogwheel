"""
Define class ``MarginalizedExtrinsicLikelihood``, to use with
``IntrinsicIASPrior`` (or similar).
"""
from abc import abstractmethod
import numpy as np
import pandas as pd

import lal

from cogwheel import skyloc_angles
from cogwheel import utils

from .relative_binning import BaseRelativeBinning
from .marginalization import SkyDictionary, CoherentScoreHM


class BaseMarginalizedExtrinsicLikelihood(BaseRelativeBinning):
    """
    Base class for computation of marginalized likelihood over extrinsic
    parameters.

    Subclassed by MarginalizedExtrinsicLikelihood and
    MarginalizedExtrinsicLikelihoodQAS.
    """
    @abstractmethod
    def _create_coherent_score(self, sky_dict, m_arr):
        """Return a coherent score instance of the appropriate type."""

    @abstractmethod
    def _get_dh_hh(self, par_dic):
        """
        Return (d|h) and (h|h) as required by
        ``self._coherent_score_cls.get_marginalization_info()``.
        """

    @abstractmethod
    def _get_many_dh_hh(self, samples):
        """Similar to `_get_dh_hh` except for an input dataframe."""

    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None, spline_degree=3,
                 t_range=(-.07, .07), coherent_score=None):
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

        t_range: 2-tuple of floats
            Bounds of a time range (s) over which to compute
            matched-filtering series, relative to
            ``event_data.tgps + par_dic_0['t_geocenter']``.

        coherent_score: CoherentScoreHM, CoherentScoreQAS as appropriate
            Instance of coherent score, optional. One with default
            settings will be created by default.
        """
        if coherent_score is None:
            # Ensure sky_dict's and event_data's sampling frequencies
            # are commensurate:
            f_sampling = SkyDictionary.choose_f_sampling(
                event_data.frequencies[-1])

            coherent_score = self._create_coherent_score(
                sky_dict=SkyDictionary(event_data.detector_names,
                                       f_sampling=f_sampling),
                m_arr=waveform_generator.m_arr)
        elif not np.array_equal(coherent_score.m_arr,
                                waveform_generator.m_arr):
            raise ValueError('`coherent_score` and `waveform_generator` use '
                             'different harmonic modes.')
        self.coherent_score = coherent_score

        self.t_range = t_range
        self._times = (np.arange(*t_range, 1 / (2*event_data.frequencies[-1]))
                       + par_dic_0.get('t_geocenter', 0))

        self._d_h_weights = None  # Set by ``._set_summary()``
        self._h_h_weights = None  # Set by ``._set_summary()``

        super().__init__(event_data, waveform_generator, par_dic_0,
                         fbin, pn_phase_tol, spline_degree)

        self._ref_dic = {
            'd_luminosity':
                self.coherent_score.lookup_table.REFERENCE_DISTANCE,
            'phi_ref':0.}

        self._log = []

        self.optimize_beta_temperature(self.par_dic_0)

    def lnlike(self, par_dic):
        """
        Natural log of the likelihood marginalized over extrinsic
        parameters (sky location, time of arrival, polarization,
        distance and orbital phase). This quantity is estimated via
        Quasi Monte Carlo integration so it is not deterministic
        (calling the method twice with the same parameters does not
        produce the same answer).

        Parameters
        ----------
        par_dic: dict
            Must contain keys for all ``.params``.

        Return
        ------
        lnlike: float
            Log of the marginalized likelihood.
        """
        marg_info = self.coherent_score.get_marginalization_info(
            *self._get_dh_hh(par_dic), self._times)

        self._log.append({'lnl_marginalized': marg_info.lnl_marginalized,
                          'n_effective': marg_info.n_effective,
                          'n_qmc': marg_info.n_qmc})

        # Reject samples with large variance to avoid artifacts. If they
        # should contribute to the posterior, by now we are in trouble
        # anyways.
        if marg_info.n_effective < self.coherent_score.min_n_effective:
            return -np.inf

        return marg_info.lnl_marginalized

    def postprocess_samples(self, samples: pd.DataFrame, num=None):
        """
        Add extrinsic parameter samples to given intrinsic parameters,
        with values taken randomly from the conditional posterior.
        `samples` is edited in-place.

        Parameters
        ----------
        samples: pd.DataFrame
            Dataframe of intrinsic parameter samples, needs to have
            columns for all `self.params`.

        num: None (default) or int
            How many extrinsic parameters to draw for every intrinsic.
            If None, columns for extrinsic parameters are added to
            `samples`.
            If an int, rows are added so that the total is
            `num * len(samples)`. Each intrinsic parameter value will be
            repeated `num` times. The index will not be preserved.
        """
        all_dh, all_hh = self._get_many_dh_hh(samples)
        extrinsic = [self.coherent_score.gen_samples(*dh_hh, self._times, num)
                     for dh_hh in zip(all_dh, all_hh)]

        gmst = lal.GreenwichMeanSiderealTime(self.event_data.tgps)
        for ext in extrinsic:
            ext['ra'] = skyloc_angles.lon_to_ra(ext['lon'], gmst)

        if isinstance(num, int):
            samples.reset_index(drop=True, inplace=True)
            fullsamples = samples.loc[samples.index.repeat(num)].reset_index(
                drop=True)
            for i, sample in fullsamples.iterrows():
                samples.loc[i] = sample

            extrinsic_df = pd.concat((pd.DataFrame(ext) for ext in extrinsic),
                                     ignore_index=True)
        else:
            extrinsic_df = pd.DataFrame.from_records(extrinsic)
        utils.update_dataframe(samples, extrinsic_df)

    def optimize_beta_temperature(self, par_dic):
        """
        Set `.coherent_score.beta_temperature` to a value that optimizes
        the importance sampling efficiency (for the intrinsic parameter
        values in `par_dic`).
        """
        optimal_beta = self.coherent_score.get_optimal_beta_temperature(
            *self._get_dh_hh(par_dic), self._times)
        self.coherent_score.beta_temperature = optimal_beta


class MarginalizedExtrinsicLikelihood(
        BaseMarginalizedExtrinsicLikelihood):
    """
    Class to evaluate the likelihood marginalized over sky location,
    time of arrival, polarization, distance and orbital phase for
    quasicircular waveforms with generic harmonic modes and spins, and
    to resample these parameters from the conditional posterior for
    demarginalization in postprocessing.

    Note: comments throughout the code refer to array indices per:
        q: QMC sample id
        m: harmonic m number id
        p: polarization (+ or x) id
        d: detector id
        b: frequency bin id
        r: rfft frequency id
        t: detector time id
        o: orbital phase id
        i: important (i.e. with high enough likelihood) sample id
    """
    params = ['f_ref', 'iota', 'l1', 'l2', 'm1', 'm2', 's1x_n', 's1y_n', 's1z',
              's2x_n', 's2y_n', 's2z']

    def _create_coherent_score(self, sky_dict, m_arr):
        return CoherentScoreHM(sky_dict, m_arr=m_arr)

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

        shape = (self.waveform_generator.m_arr.shape
                 + self.event_data.frequencies.shape)

        h0_f = np.zeros(shape, dtype=np.complex_)
        h0_f[:, self.event_data.fslice] \
            = (1, 1j) @ self.waveform_generator.get_hplus_hcross(
                self.event_data.frequencies[self.event_data.fslice],
                self.par_dic_0, by_m=True)  # mr

        h0_fbin = (1, 1j) @ self.waveform_generator.get_hplus_hcross(
            self.fbin, self.par_dic_0, by_m=True)  # mb

        self._stall_ringdown(h0_f, h0_fbin)

        self.asd_drift = self.compute_asd_drift(self.par_dic_0)

        self._set_d_h_weights(h0_f, h0_fbin)
        self._set_h_h_weights(h0_f, h0_fbin)

        # Reset
        self.waveform_generator.disable_precession = disable_precession

    def _set_d_h_weights(self, h0_f, h0_fbin):
        shifts = np.exp(2j*np.pi * np.outer(self.event_data.frequencies,
                                            self.waveform_generator.tcoarse
                                            + self._times))  # rt
        d_h_no_shift = np.einsum('dr,mr->mdr',
                                 self.event_data.blued_strain,
                                 h0_f.conj())  # mdr
        d_h_summary = np.array(
            [self._get_summary_weights(d_h_no_shift * shift)  # mdb
             for shift in shifts.T])  # tmdb  # Comprehension saves memory

        self._d_h_weights = np.einsum('tmdb,mb,d->mtdb',
                                      d_h_summary,
                                      1 / h0_fbin.conj(),
                                      1 / self.asd_drift**2
                                     ).astype(np.complex64)  # mtdb

    def _set_h_h_weights(self, h0_f, h0_fbin):
        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        h0_h0 = np.einsum('mr,mr,dr,d->mdr',
                          h0_f[m_inds],
                          h0_f[mprime_inds].conj(),
                          self.event_data.wht_filter ** 2,
                          self.asd_drift ** -2)  # mdr
        self._h_h_weights = np.einsum('mdb,mb,mb->mdb',
                                      self._get_summary_weights(h0_h0),
                                      1 / h0_fbin[m_inds],
                                      1 / h0_fbin[mprime_inds].conj()
                                     )  # mdb

        # Count off-diagonal terms twice:
        self._h_h_weights[~np.equal(m_inds, mprime_inds)] *= 2

    def _get_dh_hh(self, par_dic):
        h_mpb = self.waveform_generator.get_hplus_hcross(
            self.fbin, dict(par_dic) | self._ref_dic, by_m=True
            ).astype(np.complex64)  # mpb

        # Same but faster:
        # dh_mptd = np.einsum('mtdb,mpb->mptd',
        #                     self._d_h_weights, h_mpb.conj())
        dh_mptd = (self._d_h_weights[:, np.newaxis]
                   @ h_mpb.conj()[:, :, np.newaxis, :, np.newaxis])[..., 0]

        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        hh_mppd = np.einsum('mdb,mpb,mPb->mpPd',
                            self._h_h_weights,
                            h_mpb[m_inds],
                            h_mpb.conj()[mprime_inds]).astype(np.complex64)
        return dh_mptd, hh_mppd

    def _get_many_dh_hh(self, samples: pd.DataFrame):
        """
        Faster than a for loop over `_get_dh_hh` thanks to Strassen
        matrix multiplication to get (d|h) timeseries.
        """
        h_mpbn = np.moveaxis(
            [self.waveform_generator.get_hplus_hcross(
                self.fbin, dict(sample) | self._ref_dic,
                by_m=True)
             for _, sample in samples[self.params].iterrows()],
            0, -1).astype(np.complex64)  # mpbn

        n_m, n_t, n_d, n_b = self._d_h_weights.shape
        n_n  = len(samples)
        n_p = 2
        d_h_weights = self._d_h_weights.reshape(
            (n_m, n_t*n_d, n_b))  # m(td)b

        # Loop instead of broadcasting, to save memory:
        dh_mptdn = np.zeros((n_m, n_p, n_t*n_d, n_n), np.complex64)
        for i_m, i_p in np.ndindex(n_m, n_p):
            dh_mptdn[i_m, i_p] = d_h_weights[i_m] @ h_mpbn[i_m, i_p].conj()

        dh_nmptd = np.moveaxis(dh_mptdn, -1, 0).reshape(
            n_n, n_m, n_p, n_t, n_d)

        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        hh_nmppd = np.einsum('mdb,mpbn,mPbn->nmpPd',
                             self._h_h_weights,
                             h_mpbn[m_inds],
                             h_mpbn.conj()[mprime_inds])
        return dh_nmptd, hh_nmppd
