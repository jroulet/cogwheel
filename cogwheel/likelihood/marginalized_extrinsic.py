"""
Define class ``MarginalizedExtrinsicLikelihood``, to use with
``IntrinsicIASPrior`` (or similar).
"""
import numpy as np
import pandas as pd

import lal

from cogwheel import skyloc_angles
from cogwheel import utils

from .likelihood import check_bounds
from .relative_binning import BaseRelativeBinning
from .marginalization import SkyDictionary, CoherentScoreHM


class MarginalizedExtrinsicLikelihood(BaseRelativeBinning):
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

        coherent_score: cogwheel.likelihood.CoherentScoreHM
            Instance of coherent score, optional. One with default
            settings will be created by default.
        """
        if coherent_score is None:
            coherent_score = CoherentScoreHM(
                sky_dict=SkyDictionary(event_data.detector_names),
                m_arr=list(waveform_generator._harmonic_modes_by_m))
        self.coherent_score = coherent_score

        self.t_range = t_range
        self._times = (np.arange(*t_range, 1 / (2*event_data.frequencies[-1]))
                       + par_dic_0.get('t_geocenter', 0))
        self._ref_dic = dict(
            d_luminosity=self.coherent_score.lookup_table.REFERENCE_DISTANCE,
            phi_ref=0.)

        self._d_h_weights = None  # Set by ``._set_summary()``
        self._h_h_weights = None  # Set by ``._set_summary()``

        super().__init__(event_data, waveform_generator, par_dic_0,
                         fbin, pn_phase_tol, spline_degree)

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
                 len(self.event_data.frequencies))

        h0_f = np.zeros(shape, dtype=np.complex_)
        h0_f[:, self.event_data.fslice] \
            = (1, 1j) @ self.waveform_generator.get_hplus_hcross(
                self.event_data.frequencies[self.event_data.fslice],
                self.par_dic_0, by_m=True)  # mpr

        h0_fbin = (1, 1j) @ self.waveform_generator.get_hplus_hcross(
            self.fbin, self.par_dic_0, by_m=True)  # mpb

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
                                 h0_f.conj())  # mpdr
        d_h_summary = np.array(
            [self._get_summary_weights(d_h_no_shift * shift)  # mdb
             for shift in shifts.T])  # tmdb  # Comprehension saves memory

        self._d_h_weights = np.einsum('tmdb,mb,d->mtdb',
                                      d_h_summary,
                                      1 / h0_fbin.conj(),
                                      1 / self.asd_drift**2)  # mtdb

    def _set_h_h_weights(self, h0_f, h0_fbin):
        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        h0_h0 = np.einsum('mr,mr,dr,d->mdr',
                          h0_f[m_inds],
                          h0_f[mprime_inds].conj(),
                          self.event_data.wht_filter ** 2,
                          self.asd_drift ** -2)  # mpPdr
        self._h_h_weights = np.einsum('mdb,mb,mb->mdb',
                                      self._get_summary_weights(h0_h0),
                                      1 / h0_fbin[m_inds],
                                      1 / h0_fbin[mprime_inds].conj())  # mdb

        # Count off-diagonal terms twice:
        self._h_h_weights[~np.equal(m_inds, mprime_inds)] *= 2

    def _get_dh_hh(self, par_dic):
        h_mpb = self.waveform_generator.get_hplus_hcross(
            self.fbin, dict(par_dic) | self._ref_dic, by_m=True)  # mpb

        # Same but faster:
        # dh_mptd = np.einsum('mtdb,mpb->mptd',
        #                     self._d_h_weights, h_mpb.conj())
        dh_mptd = (self._d_h_weights[:, np.newaxis]
                   @ h_mpb.conj()[:, :, np.newaxis, :, np.newaxis])[..., 0]

        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        hh_mppd = np.einsum('mdb,mpb,mPb->mpPd',
                            self._h_h_weights,
                            h_mpb[m_inds],
                            h_mpb.conj()[mprime_inds])
        return dh_mptd, hh_mppd

    @check_bounds
    def lnlike(self, par_dic):
        """
        Natural log of the likelihood marginalized over extrinsic
        parameters (sky location, time of arrival, polarization,
        distance and orbital phase).

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

        # Reject samples with large variance to avoid artifacts. If they
        # should contribute to the posterior, by now we are in trouble
        # anyways.
        if marg_info.n_effective < self.coherent_score.min_n_effective:
            return -np.inf

        return marg_info.lnl_marginalized

    def postprocess_samples(self, samples: pd.DataFrame, num=None):
        """
        Generate extrinsic parameter samples given intrinsic parameters,
        with values taken randomly from the conditional posterior.

        Parameters
        ----------
        samples: pd.DataFrame
            Dataframe of intrinsic parameter samples, needs to have
            columns for all `self.params`.

        num: int or None
            How many extrinsic parameters to draw for every intrinsic.
            If None, columns for extrinsic parameters are added to
            `samples` in-place.
            If an int, a new DataFrame of length `num * len(samples)` is
            returned. Each intrinsic parameter value will be repeated
            `num` times.

        Return
        ------
        ``pd.DataFrame`` with postprocessed samples (if `num` is an
        integer) or ``None`` (if `num` is ``None``).
        """
        dh_nmptd, hh_nmppd = self._get_many_dh_hh(samples)

        extrinsic = [self.coherent_score.gen_samples(dh_mptd, hh_mppd,
                                                     self._times, num)
                     for dh_mptd, hh_mppd in zip(dh_nmptd, hh_nmppd)]

        for ext in extrinsic:
            ext['ra'] = skyloc_angles.lon_to_ra(
                ext['lon'],
                lal.GreenwichMeanSiderealTime(self.event_data.tgps))

        if isinstance(num, int):
            fullsamples = samples.loc[samples.index.repeat(num)].reset_index(
                drop=True)
            extrinsic_df = pd.concat((pd.DataFrame(ext) for ext in extrinsic),
                                     ignore_index=True)
            utils.update_dataframe(fullsamples, extrinsic_df)
            return fullsamples

        utils.update_dataframe(samples, pd.DataFrame.from_records(extrinsic))
        return None

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
            0, -1)  # mpbn

        n_m, n_t, n_d, n_b = self._d_h_weights.shape
        n_n  = len(samples)
        n_p = 2
        d_h_weights = self._d_h_weights.reshape(
            n_m, n_t*n_d, n_b)  # m(td)b

        # Loop instead of broadcasting, to save memory:
        dh_mptdn = np.zeros((n_m, n_p, n_t*n_d, n_n), np.complex_)
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
