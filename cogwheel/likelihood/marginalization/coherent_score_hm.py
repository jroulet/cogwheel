"""
Define class ``CoherentScoreHM`` to marginalize and demarginalize the
likelihood over extrinsic parameters from matched-filtering timeseries.

``CoherentScoreHM`` works for quasi-circular waveforms with precession
and higher modes. The inclination can't be marginalized over and is
treated as an intrinsic parameter.
"""
from collections import namedtuple
import numpy as np
from scipy.interpolate import make_interp_spline

from .base import BaseCoherentScoreHM


class CoherentScoreHM(BaseCoherentScoreHM):
    """
    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters
    (``.get_marginalization_info()``). Extrinsic parameters samples can
    be generated as well (``.gen_samples()``).

    Works for quasi-circular waveforms with generic spins and higher
    modes.

    Inherits from ``BaseCoherentScoreHM``.
    """
    _MarginalizationInfo = namedtuple('_MarginalizationInfo',
                                      ['physical_mask',
                                       't_first_det',
                                       'dh_qo',
                                       'hh_qo',
                                       'sky_inds',
                                       'weights',
                                       'lnl_marginalized',
                                       'important'])
    _MarginalizationInfo.__doc__ = """
        Contains likelihood marginalized over extrinsic parameters, and
        intermediate products that can be used to generate extrinsic
        parameter samples or compute other auxiliary quantities like
        partial marginalizations.

        Fields
        ------
        physical_mask: boolean array of length n_qmc
            Some choices of time of arrival at detectors may not
            correspond to any physical sky location, these are flagged
            ``False`` in this array. Unphysical samples are discarded.
            ``n_physical`` below means ``count_nonzero(physical_mask)``.

        t_first_det: float array of length n_physical
            Time of arrival at the first detector.

        dh_qo: float array of shape (n_physical, n_phi)
            Real inner product ⟨d|h⟩, indexed by (physical) QMC sample
            `q` and orbital phase `o`.

        hh_qo: float array of shape (n_physical, n_phi)
            Real inner product ⟨h|h⟩, indexed by (physical) QMC sample
            `q` and orbital phase `o`.

        sky_inds: tuple of ints, of length n_physical
            Indices to sky_dict.sky_samples corresponding to the
            (physical) QMC samples.

        weights: float array of length n_important
            Positive weights of the QMC samples, including the
            likelihood and the importance-sampling correction.

        lnl_marginalized: float
            log of the marginalized likelihood over extrinsic parameters
            excluding inclination (i.e.: time of arrival, sky location,
            polarization, distance, orbital phase).

        important: (tuple of ints, tuple of ints) of lengths n_important
            The first tuple contains indices between 0 and n_physical-1
            corresponding to (physical) QMC samples.
            The second tuple contains indices between 0 and n_phi-1
            corresponding to orbital phases.
            They correspond to samples with sufficiently high maximum
            likelihood over distance to be included in the integral.
        """

    def get_marginalization_info(self, dh_mptd, hh_mppd, times):
        """
        Evaluate inner products (d|h) and (h|h) at QMC integration
        points over extrinsic parameters, given timeseries of (d|h) and
        value of (h|h) by mode `m`, polarization `p` and detector `d`.

        Parameters
        ----------
        dh_mptd: (n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        hh_mppd: (n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        Return
        ------
        Instance of ``._MarginalizationInfo`` with several fields
        (physical_mask, t_first_det, dh_qo, hh_qo, sky_inds, weights,
        lnl_marginalized, important), see its documentation.
        """
        self._switch_qmc_sequence()

        # Resample to match sky_dict's dt:
        dh_mptd, times = self.sky_dict.resample_timeseries(dh_mptd, times,
                                                           axis=2)

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(dh_mptd,
                                                             hh_mppd)  # td
        t_first_det, delays, importance_sampling_weight \
            = self._draw_single_det_times(t_arrival_lnprob, times)

        sky_inds, sky_prior, physical_mask \
            = self.sky_dict.get_sky_inds_and_prior(delays)  # q, q, q

        if not any(physical_mask):
            return self._MarginalizationInfo(physical_mask=physical_mask,
                                             t_first_det=np.array([]),
                                             dh_q=np.array([]),
                                             hh_q=np.array([]),
                                             sky_inds=(),
                                             weights=np.array([]),
                                             lnl_marginalized=-np.inf)

        t_first_det = t_first_det[physical_mask]
        importance_sampling_weight = importance_sampling_weight[physical_mask]


        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, physical_mask, t_first_det,
                                          times, dh_mptd, hh_mppd)  # qo, qo

        lnl_marginalized, weights, important \
            = self._get_lnl_marginalized_and_weights(
                dh_qo, hh_qo, importance_sampling_weight * sky_prior)

        return self._MarginalizationInfo(physical_mask=physical_mask,
                                         t_first_det=t_first_det,
                                         dh_qo=dh_qo,
                                         hh_qo=hh_qo,
                                         sky_inds=sky_inds,
                                         weights=weights,
                                         lnl_marginalized=lnl_marginalized,
                                         important=important)

    def gen_samples(self, dh_mptd, hh_mppd, times, num=None):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        dh_mptd: (n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        hh_mppd: (n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        num: int, optional
            Number of samples to generate, defaults to a single sample.

        Return
        ------
        samples: dict
            Values are scalar if `num` is ``None``, else numpy arrays.
            If ``marg_info`` corresponds to an unphysical sample (i.e.,
            a realization of matched-filtering timeseries in the
            detectors incompatible with a real signal) the values will
            be NaN.
        """
        marg_info = self.get_marginalization_info(dh_mptd, hh_mppd, times)
        return self._gen_samples_from_marg_info(marg_info, num)

    def _gen_samples_from_marg_info(self, marg_info, num):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        marg_info: CoherentScoreHM._MarginalizationInfo
            Output of ``.get_marginalization_info``.

        num: int, optional
            Number of samples to generate, defaults to a single sample.

        Return
        ------
        samples: dict
            Values are scalar if `num` is ``None``, else numpy arrays.
            If ``marg_info`` correspond to an unphysical sample (i.e.,
            a realization of matched-filtering timeseries in the
            detectors incompatible with a real signal) the values will
            be NaN.
        """
        if not any(marg_info.physical_mask):
            unphysical_value = np.nan if num is None else np.full(num, np.nan)
            return dict.fromkeys(
                ['d_luminosity', 'dec', 'lon', 'phi_ref', 'psi', 't_geocenter',
                 'lnl_marginalized', 'lnl'],
                unphysical_value)

        i_ids = self._rng.choice(len(marg_info.weights),
                                 p=marg_info.weights / marg_info.weights.sum(),
                                 size=num)

        q_ids = marg_info.important[0][i_ids]
        o_ids = marg_info.important[1][i_ids]
        sky_ids = np.array(marg_info.sky_inds)[q_ids]
        t_geocenter = (marg_info.t_first_det[q_ids]
                       - self.sky_dict.geocenter_delay_first_det[sky_ids])

        d_h = marg_info.dh_qo[q_ids, o_ids]
        h_h = marg_info.hh_qo[q_ids, o_ids]
        d_luminosity = self._sample_distance(d_h, h_h)
        distance_ratio = d_luminosity / self.lookup_table.REFERENCE_DISTANCE
        return {
            'd_luminosity': d_luminosity,
            'dec': self.sky_dict.sky_samples['lat'][sky_ids],
            'lon': self.sky_dict.sky_samples['lon'][sky_ids],
            'phi_ref': self._phi_ref[o_ids],
            'psi': self._qmc_sequence['psi'][marg_info.physical_mask][q_ids],
            't_geocenter': t_geocenter,
            'lnl_marginalized': marg_info.lnl_marginalized,
            'lnl': d_h / distance_ratio - h_h / distance_ratio**2 / 2,
            'h_h': h_h / distance_ratio**2}

    def _get_dh_hh_qo(self, sky_inds, physical_mask, t_first_det, times,
                      dh_mptd, hh_mppd):
        """
        Apply antenna factors and orbital phase to the polarizations, to
        obtain (d|h) and (h|h) by extrinsic sample 'q' and orbital phase
        'o'.
        """
        fplus_fcross = self._get_fplus_fcross(sky_inds, physical_mask)

        # (d|h):
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmpq = np.array(
            [make_interp_spline(times, dh_mptd[..., i_det], k=3,
                                check_finite=False, axis=-1)(t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        dh_qm = np.einsum('dmpq,qdp->qm', dh_dmpq, fplus_fcross)  # qm

        # (h|h):
        f_f = np.einsum('qdp,qdP->qpPd', fplus_fcross, fplus_fcross)
        hh_qm = (f_f.reshape(f_f.shape[0], -1)
                 @ hh_mppd.reshape(hh_mppd.shape[0], -1).T)  # qm

        dh_qo = (dh_qm @ self._dh_phasor).real  # qo
        hh_qo = (hh_qm @ self._hh_phasor).real  # qo
        return dh_qo, hh_qo

    def _incoherent_t_arrival_lnprob(self, dh_mptd, hh_mppd):
        """
        Simple chi-squared approximating that different modes and
        polarizations are all orthogonal.
        """
        hh_mpdiagonal = hh_mppd[np.equal(self.m_inds, self.mprime_inds)
                               ][:, (0, 1), (0, 1)].real  # mpd
        chi_squared = np.einsum('mptd,mpd->td',
                                np.abs(dh_mptd)**2,
                                1 / hh_mpdiagonal)  # td
        return self.beta_temperature * chi_squared / 2 # td
