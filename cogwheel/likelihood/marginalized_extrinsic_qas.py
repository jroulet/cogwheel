"""
Define class ``MarginalizedExtrinsicLikelihoodQAS``, to use with
``IntrinsicAlignedSpinIASPrior`` (or similar).
"""
import functools
import numpy as np

from cogwheel import utils

from .marginalization import CoherentScoreQAS
from .marginalized_extrinsic import BaseMarginalizedExtrinsicLikelihood


class MarginalizedExtrinsicLikelihoodQAS(
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
    _coherent_score_cls = CoherentScoreQAS
    params = ['f_ref', 'l1', 'l2', 'm1', 'm2', 's1z', 's2z']

    @functools.wraps(BaseMarginalizedExtrinsicLikelihood.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ref_dic.update(iota=0.,
                             s1x_n=0.,
                             s1y_n=0.,
                             s2x_n=0.,
                             s2y_n=0.)

    def _create_coherent_score(self, sky_dict, m_arr, **kwargs):
        if list(m_arr) != [2]:
            raise ValueError(f'{self.__class__.__name__} only works with '
                             '(l, |m|) = (2, 2) waveforms.')
        return CoherentScoreQAS(sky_dict=sky_dict, **kwargs)

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
        with utils.temporarily_change_attributes(self.waveform_generator,
                                                 disable_precession=False):
            h0_f = np.zeros(len(self.event_data.frequencies),
                            dtype=np.complex128)
            h0_f[..., self.event_data.fslice] \
                = self.waveform_generator.get_hplus_hcross(
                    self.event_data.frequencies[self.event_data.fslice],
                    self.par_dic_0)[0]  # r

            h0_fbin = self.waveform_generator.get_hplus_hcross(
                self.fbin, self.par_dic_0)[0]  # b

            self._stall_ringdown(h0_f, h0_fbin)

            self._set_d_h_weights(h0_f, h0_fbin)
            self._set_h_h_weights(h0_f, h0_fbin)

    def _set_d_h_weights(self, h0_f, h0_fbin):
        shifts = np.exp(2j*np.pi * np.outer(self.event_data.frequencies,
                                            self.waveform_generator.tcoarse
                                            + self._times))  # rt

        d_h_no_shift = self.event_data.blued_strain * h0_f.conj()  # dr
        d_h_summary = np.array(
            [self._get_summary_weights(d_h_no_shift * shift)  # db
             for shift in shifts.T])  # tdb  # Comprehension saves memory

        self._d_h_weights = np.einsum('tdb,b->tdb',
                                      d_h_summary,
                                      1 / h0_fbin.conj())  # mptdb

    def _set_h_h_weights(self, h0_f, h0_fbin):
        h0_h0 = np.einsum('r,dr->dr',
                          utils.abs_sq(h0_f),
                          self.event_data.wht_filter ** 2)  # dr
        self._h_h_weights = (self._get_summary_weights(h0_h0).real
                             / utils.abs_sq(h0_fbin))  # db

    def _get_dh_hh_timeshift(self, par_dic):
        (h_b, _), timeshift = self._get_linearfree_hplus_hcross_dt(
            dict(par_dic) | self._ref_dic)
        asd_drift_correction = self.asd_drift ** -2  # d
        dh_td = self._d_h_weights @ h_b.conj() * asd_drift_correction  # td
        hh_d = (self._h_h_weights @ utils.abs_sq(h_b)
                * asd_drift_correction)  # d
        return dh_td, hh_d, timeshift
