"""
Define class ``CoherentScoreQAS`` to marginalize and demarginalize the
likelihood over extrinsic parameters from matched-filtering timeseries,
for quasi-circular waveforms with quadrupole radiation (l, |m|) = (2, 2)
and aligned spins.
"""
from collections import namedtuple
import numpy as np
import scipy.interpolate

from cogwheel import utils
from cogwheel.likelihood.marginalized_distance import (
    LookupTableMarginalizedPhase22)
from cogwheel.coherent_score.base import BaseCoherentScore


class CoherentScoreQAS(BaseCoherentScore):
    """
    Coherent score for quadrupole, aligned-spin waveforms.

    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters
    (``.get_marginalization_info()``). Extrinsic parameters samples can
    be generated as well (``.gen_samples()``).
    Works for quasi-circular waveforms with quadrupole radiation, i.e.
    (l,|m|) = (2, 2), and aligned spins.

    Inherits from ``BaseCoherentScore``.
    """
    _lookup_table_marginalized_params = {'d_luminosity', 'phi_ref'}

    _MarginalizationInfo = namedtuple('_MarginalizationInfo',
                                      ['physical_mask',
                                       't_first_det',
                                       'dh_q',
                                       'hh_q',
                                       'sky_inds',
                                       'weights',
                                       'lnl_marginalized'])
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

        dh_q: complex array of length n_physical
            Complex inner product (d|h), indexed by (physical) QMC
            sample `q`.

        hh_q: float array of length n_physical
            Real inner product ⟨h|h⟩, indexed by (physical) QMC sample
            `q`.

        sky_inds: tuple of ints, of length n_physical
            Indices to sky_dict.sky_samples corresponding to the
            (physical) QMC samples.

        weights: float array of length n_important
            Positive weights of the QMC samples, including the
            likelihood and the importance-sampling correction.

        lnl_marginalized: float
            log of the marginalized likelihood over extrinsic parameters
            (i.e.: time of arrival, sky location, polarization, distance,
            orbital phase, inclination).
        """

    def __init__(self, sky_dict, lookup_table=None,
                 log2n_qmc: int = 11, seed=0, beta_temperature=.5):
        """
        Parameters
        ----------
        sky_dict:
            Instance of cogwheel.coherent_score_hm.skydict.SkyDictionary

        lookup_table:
            Instance of cogwheel.likelihood.marginalized_distance\
            .LookupTableMarginalizedPhase22

        log2n_qmc: int
            Base-2 logarithm of the number of requested extrinsic
            parameter samples.

        seed: {int, None, np.random.RandomState}
            For reproducibility of the extrinsic parameter samples.

        beta_temperature: float
            Inverse temperature, tempers the arrival time probability at
            each detector.
        """
        if lookup_table is None:
            lookup_table = LookupTableMarginalizedPhase22()

        super().__init__(sky_dict=sky_dict,
                         lookup_table=lookup_table,
                         log2n_qmc=log2n_qmc,
                         seed=seed,
                         beta_temperature=beta_temperature)

        self._sample_phase = utils.handle_scalars(
            np.vectorize(self.lookup_table.sample_phase, otypes=[float]))

    def get_marginalization_info(self, dh_td, hh_d, times):
        """
        Evaluate inner products (d|h) and ⟨h|h⟩ at QMC integration
        points over extrinsic parameters, given timeseries of (d|h) and
        value of ⟨h|h⟩ by detector `d`.

        Parameters
        ----------
        dh_td: (n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by time, detector.

        hh_d: (n_d,) float array
            Positive ⟨h|h⟩ inner product of a waveform with itself,
            decomposed by detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        Return
        ------
        Instance of ``._MarginalizationInfo`` with several fields
        (physical_mask, t_first_det, dh_q, hh_q, sky_inds, weights,
        lnl_marginalized), see its documentation.
        """
        # Resample to match sky_dict's dt:
        dh_td, times = self.sky_dict.resample_timeseries(dh_td, times,
                                                         axis=0)

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(dh_td,
                                                             hh_d)  # td
        t_first_det, delays, physical_mask, importance_sampling_weight \
            = self._draw_single_det_times(t_arrival_lnprob, times)

        if not any(physical_mask):
            return self._MarginalizationInfo(physical_mask=physical_mask,
                                             t_first_det=np.array([]),
                                             dh_q=np.array([]),
                                             hh_q=np.array([]),
                                             sky_inds=(),
                                             weights=np.array([]),
                                             lnl_marginalized=-np.inf)

        sky_inds, sky_prior = self.sky_dict.get_sky_inds_and_prior(
            delays)  # q, q

        dh_q, hh_q = self._get_dh_hh_q(sky_inds, physical_mask, t_first_det,
                                       times, dh_td, hh_d)  # q, q

        lnl_marg_dist_phase = self.lookup_table.lnlike_marginalized(
            np.abs(dh_q), hh_q)  # q

        lnl_max = lnl_marg_dist_phase.max()
        like_marg_dist_phase = np.exp(lnl_marg_dist_phase - lnl_max)  # q

        weights = (importance_sampling_weight * sky_prior
                   * like_marg_dist_phase)  # q

        lnl_marginalized = lnl_max + np.log(weights.sum() / 2**self.log2n_qmc)

        return self._MarginalizationInfo(physical_mask=physical_mask,
                                         t_first_det=t_first_det,
                                         dh_q=dh_q,
                                         hh_q=hh_q,
                                         sky_inds=sky_inds,
                                         weights=weights,
                                         lnl_marginalized=lnl_marginalized)

    def _get_dh_hh_q(self, sky_inds, physical_mask, t_first_det, times,
                     dh_td, hh_d):
        """
        Apply antenna factors to the waveform, to obtain (d|h) and ⟨h|h⟩
        by extrinsic sample 'q'.
        """
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dq = np.array(
            [scipy.interpolate.interp1d(times, dh_td[:, i_det], kind=3,
                                        copy=False, assume_sorted=True,
                                        fill_value=0., bounds_error=False
                                        )(t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        response_qd = np.einsum('qp,qdp->qd',
                                self._qmc_sequence['response'][physical_mask],
                                self.sky_dict.fplus_fcross_0[sky_inds,])  # qd

        dh_q = np.einsum('dq,qd->q', dh_dq, response_qd)
        hh_q = utils.abs_sq(response_qd) @ hh_d

        return dh_q, hh_q

    @property
    def _qmc_range_dic(self):
        """
        Parameter ranges for the QMC sequence.
        The sequence explores the cumulatives of the single-detector
        (incoherent) likelihood of arrival times, the polarization, the
        fine (subpixel) time of arrival and the cosine inclination.
        """
        return super()._qmc_range_dic | {'cosiota': (-1, 1)}

    def _create_qmc_sequence(self):
        """
        Return a dictionary whose values are arrays corresponding to a
        Quasi Monte Carlo sequence that explores parameters per
        ``._qmc_range_dic``.
        The arrival time cumulatives are packed in a single entry
        'u_tdet'. An entry 'rot_psi' has the rotation matrices to
        transform the antenna factors between psi=0 and psi=psi_qmc.
        Also entries for 'response' are provided. The response is defined
        so that:
          total_response :=
            := (1+cosiota**2)/2*fplus - 1j*cosiota*fcross
            = ((1+cosiota**2)/2, - 1j*cosiota) @ (fplus, fcross)
            = ((1+cosiota**2)/2, - 1j*cosiota) @ rot @ (fplus0, fcross0)
            = response @ (fplus0, fcross0)
        for the (2, 2) mode.
        """
        qmc_sequence = super()._create_qmc_sequence()
        qmc_sequence['response'] = np.einsum(
            'Pq,qPp->qp',
            ((1 + qmc_sequence['cosiota']**2) / 2,
             - 1j * qmc_sequence['cosiota']),
            qmc_sequence['rot_psi'])  # qp
        return qmc_sequence

    def _incoherent_t_arrival_lnprob(self, dh_td, hh_d):
        """Return tempered chi-squared timeseries at each detector."""
        return self.beta_temperature * utils.abs_sq(dh_td) / hh_d / 2 # td

    def gen_samples(self, dh_td, hh_d, times, num=None):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        dh_td: (n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by time, detector.

        hh_d: (n_d,) float array
            Positive ⟨h|h⟩ inner product of a waveform with itself,
            decomposed by detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

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
        marg_info = self.get_marginalization_info(dh_td, hh_d, times)
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

        q_ids = self._rng.choice(len(marg_info.weights),
                                 p=marg_info.weights / marg_info.weights.sum(),
                                 size=num)

        sky_ids = np.array(marg_info.sky_inds)[q_ids]
        t_geocenter = (marg_info.t_first_det[q_ids]
                       - self.sky_dict.geocenter_delay_first_det[sky_ids])

        d_h = marg_info.dh_q[q_ids]
        h_h = marg_info.hh_q[q_ids]
        d_luminosity = self._sample_distance(np.abs(d_h), h_h)
        phi_ref = self._sample_phase(d_luminosity, d_h)
        real_dh = np.real(d_h * np.exp(-2j*phi_ref))
        distance_ratio = d_luminosity / self.lookup_table.REFERENCE_DISTANCE
        return {
            'd_luminosity': d_luminosity,
            'dec': self.sky_dict.sky_samples['lat'][sky_ids],
            'lon': self.sky_dict.sky_samples['lon'][sky_ids],
            'phi_ref': phi_ref,
            'psi': self._qmc_sequence['psi'][marg_info.physical_mask][q_ids],
            't_geocenter': t_geocenter,
            'lnl_marginalized': marg_info.lnl_marginalized,
            'lnl': real_dh / distance_ratio - h_h / distance_ratio**2 / 2}
