"""
Define class ``MarginalizedExtrinsicLikelihoodQAS``, to use with
``IntrinsicAlignedSpinIASPrior`` (or similar).
"""
import numpy as np
import pandas as pd

import lal

from cogwheel import skyloc_angles
from cogwheel import utils

from .marginalization import SkyDictionary, CoherentScoreQAS
from .relative_binning import BaseRelativeBinning


class MarginalizedExtrinsicLikelihoodQAS(BaseRelativeBinning):
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
    params = ['f_ref', 'l1', 'l2', 'm1', 'm2', 's1z', 's2z']

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

        coherent_score: cogwheel.likelihood.CoherentScoreQAS
            Instance of coherent score, optional. One with default
            settings will be created by default.
        """
        if waveform_generator.harmonic_modes != [(2, 2)]:
            raise ValueError('``CoherentScoreLikelihoodQAS`` only works with '
                             'quadrupolar waveform models.')

        if coherent_score is None:
            coherent_score = CoherentScoreQAS(
                SkyDictionary(event_data.detector_names))
        self.coherent_score = coherent_score

        self.t_range = t_range
        self._times = (np.arange(*t_range, 1 / (2*event_data.fbounds[1]))
                       + par_dic_0.get('t_geocenter', 0))
        self._ref_dic = dict(
            d_luminosity=self.coherent_score.lookup_table.REFERENCE_DISTANCE,
            phi_ref=0.,
            iota=0.,
            s1x_n=0.,
            s1y_n=0.,
            s2x_n=0.,
            s2y_n=0.)

        self._h0_f = None  # Set by ``._set_summary()``
        self._h0_fbin = None  # Set by ``._set_summary()``
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
        super()._set_summary()

        # Don't zero the in-plane spins for the reference waveform
        disable_precession = self.waveform_generator.disable_precession
        self.waveform_generator.disable_precession = False

        self._h0_f = np.zeros(len(self.event_data.frequencies),
                              dtype=np.complex_)
        self._h0_f[..., self.event_data.fslice] \
            = self.waveform_generator.get_hplus_hcross(
                self.event_data.frequencies[self.event_data.fslice],
                self.par_dic_0)[0]  # r

        self._h0_fbin = self.waveform_generator.get_hplus_hcross(
            self.fbin, self.par_dic_0)[0]  # b

        self._set_d_h_weights()
        self._set_h_h_weights()

        # Reset
        self.waveform_generator.disable_precession = disable_precession

    def _set_d_h_weights(self):
        shifts = np.exp(2j*np.pi * np.outer(self.event_data.frequencies,
                                            self.waveform_generator.tcoarse
                                            + self._times))  # rt

        d_h_no_shift = self.event_data.blued_strain * self._h0_f.conj()  # dr
        d_h_summary = np.array(
            [self._get_summary_weights(d_h_no_shift * shift)  # db
             for shift in shifts.T])  # tdb  # Comprehension saves memory

        self._d_h_weights = np.einsum('tdb,b,d->tdb',
                                      d_h_summary,
                                      1 / self._h0_fbin.conj(),
                                      1 / self.asd_drift**2)  # mptdb

    def _set_h_h_weights(self):
        h0_h0 = np.einsum('r,dr,d->dr',
                          utils.abs_sq(self._h0_f),
                          self.event_data.wht_filter ** 2,
                          self.asd_drift ** -2)  # dr
        self._h_h_weights = (self._get_summary_weights(h0_h0).real
                             / utils.abs_sq(self._h0_fbin))  # db

    def _get_dh_hh(self, par_dic):
        h_b = self.waveform_generator.get_hplus_hcross(
            self.fbin, dict(par_dic) | self._ref_dic)[0]  # b
        dh_td = self._d_h_weights @ h_b.conj()  # td
        hh_d = self._h_h_weights @ utils.abs_sq(h_b)  # d
        return dh_td, hh_d

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
        return self.coherent_score.get_marginalization_info(
            *self._get_dh_hh(par_dic), self._times).lnl_marginalized

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
        """
        dh_ntd, hh_nd = self._get_many_dh_hh(samples)

        extrinsic = [self.coherent_score.gen_samples(dh_td, hh_d, self._times,
                                                     num)
                     for dh_td, hh_d in zip(dh_ntd, hh_nd)]

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

    def _get_many_dh_hh(self, samples: pd.DataFrame):
        """
        Faster than a for loop over `_get_dh_hh` thanks to Strassen
        matrix multiplication to get (d|h) timeseries.
        """
        h_bn = np.transpose(
            [self.waveform_generator.get_hplus_hcross(
                self.fbin, dict(sample) | self._ref_dic)[0]
             for _, sample in samples[self.params].iterrows()])  # bn

        n_t, n_d, n_b = self._d_h_weights.shape
        n_n  = len(samples)
        d_h_weights = self._d_h_weights.reshape(n_t*n_d, n_b)  # (td)b
        dh_tdn = d_h_weights @ h_bn.conj()
        dh_ntd = np.moveaxis(dh_tdn, -1, 0).reshape(n_n, n_t, n_d)

        hh_nd = np.transpose(self._h_h_weights @ utils.abs_sq(h_bn))  # nd
        return dh_ntd, hh_nd
