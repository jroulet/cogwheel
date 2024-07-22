"""
Define class ``CoherentScoreQAS`` to marginalize and demarginalize the
likelihood over extrinsic parameters from matched-filtering timeseries,
for quasi-circular waveforms with quadrupole radiation (l, |m|) = (2, 2)
and aligned spins.
"""
import warnings
import numpy as np

from cogwheel import utils
from .base import ProposingCoherentScore, MarginalizationInfo


class CoherentScoreQAS(ProposingCoherentScore):
    """
    Coherent score for quadrupole, aligned-spin waveforms.

    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters
    (``.get_marginalization_info()``). Extrinsic parameters samples can
    be generated as well (``.gen_samples()``).
    Works for quasi-circular waveforms with quadrupole radiation, i.e.
    (l,|m|) = (2, 2), and aligned spins.

    Inherits from ``ProposingCoherentScore``.
    """
    m_arr = np.array([2])
    _lookup_table_marginalized_params = {'d_luminosity', 'phi_ref'}

    def _get_marginalization_info_chunk(self, d_h_timeseries, h_h,
                                        times, t_arrival_prob, i_chunk):
        """
        Evaluate inner products (d|h) and (h|h) at integration points
        over a chunk of a QMC sequence of extrinsic parameters, given
        timeseries of (d|h) and value of (h|h) by detector `d`.

        Parameters
        ----------
        d_h_timeseries: (n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by time, detector.

        h_h: (n_d,) float array
            Positive ⟨h|h⟩ inner product of a waveform with itself,
            decomposed by detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        t_arrival_prob: (n_d, n_t) float array
            Proposal probability of time of arrival at each detector,
            normalized to sum to 1 along the time axis.

        i_chunk: int
            Index to ``._qmc_ind_chunks``.

        Return
        ------
        Instance of ``MarginalizationInfo`` with several fields, see its
        documentation.
        """
        q_inds = self._qmc_ind_chunks[i_chunk]  # Will update along the way
        n_qmc = len(q_inds)
        tdet_inds = self._get_tdet_inds(t_arrival_prob, q_inds)

        sky_inds, sky_prior, physical_mask \
            = self.sky_dict.get_sky_inds_and_prior(
                tdet_inds[1:] - tdet_inds[0])  # q, q, q

        # Apply physical mask (sensible time delays):
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]

        if not any(physical_mask):
            return MarginalizationInfo(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals=[t_arrival_prob])

        t_first_det = (times[tdet_inds[0]]
                       + self._qmc_sequence['t_fine'][q_inds])

        dh_q, hh_q = self._get_dh_hh_q(sky_inds, q_inds, t_first_det,
                                       times, d_h_timeseries, h_h)  # q, q

        ln_numerators = (
            self.lookup_table.lnlike_marginalized(np.abs(dh_q), hh_q)
            + np.log(sky_prior))  # q

        return MarginalizationInfo(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds,
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_q,
            h_h=hh_q,
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals=[t_arrival_prob])

    def _get_dh_hh_q(self, sky_inds, q_inds, t_first_det, times,
                     dh_td, hh_d):
        """
        Apply antenna factors to the waveform, to obtain (d|h) and ⟨h|h⟩
        by extrinsic sample 'q'.
        """
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dq = np.array(
            [self._interp_locally(times, dh_td[:, i_det], t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        response_qd = np.einsum('qp,qdp->qd',
                                self._qmc_sequence['response'][q_inds],
                                self.sky_dict.fplus_fcross_0[sky_inds,])  # qd

        dh_q = np.einsum('dq,qd->q', dh_dq, response_qd.conj())
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

    def _incoherent_t_arrival_lnprob(self, d_h_timeseries, h_h):
        """
        Return tempered chi-squared timeseries at each detector.

        Parameters
        ----------
        d_h_timeseries: (n_d, n_t) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by time, detector.

        h_h: (n_d,) float array
            Positive ⟨h|h⟩ inner product of a waveform with itself,
            decomposed by detector.

        Return
        ------
        t_arrival_lnprob: (n_d, n_t) float array
        """
        return np.transpose(self.beta_temperature / 2
                            * utils.abs_sq(d_h_timeseries) / h_h)  # dt

    def gen_samples(self, dh_td, hh_d, times, num=(),
                    lnl_marginalized_threshold=-np.inf):
        """Deprecated, use ``.gen_samples_from_marg_info``"""
        warnings.warn('Use ``gen_samples_from_marg_info``', DeprecationWarning)
        marg_info = self.get_marginalization_info(dh_td, hh_d, times,
                                                  lnl_marginalized_threshold)
        return self.gen_samples_from_marg_info(marg_info, num)

    def gen_samples_from_marg_info(self, marg_info, num=()):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        marg_info: CoherentScoreQAS._MarginalizationInfo
            Output of ``.get_marginalization_info``.

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
        self._switch_qmc_sequence(marg_info.qmc_sequence_id)

        if marg_info.q_inds.size == 0:
            return dict.fromkeys(['d_luminosity', 'dec', 'lon', 'phi_ref',
                                  'psi', 'iota', 't_geocenter',
                                  'lnl_marginalized', 'lnl', 'h_h',
                                  'n_effective', 'n_qmc'],
                np.full(num, np.nan)[()])

        random_ids = self._rng.choice(len(marg_info.q_inds), size=num,
                                      p=marg_info.weights)[()]

        q_ids = marg_info.q_inds[random_ids]
        sky_ids = marg_info.sky_inds[random_ids]
        t_geocenter = (marg_info.t_first_det[random_ids]
                       - self.sky_dict.geocenter_delay_first_det[sky_ids])
        d_h = marg_info.d_h[random_ids]
        h_h = marg_info.h_h[random_ids]

        d_luminosity = self._sample_distance(np.abs(d_h), h_h)
        phi_ref = self.lookup_table.sample_phase(d_luminosity, d_h)
        real_dh = np.real(d_h * np.exp(-2j*phi_ref))
        distance_ratio = d_luminosity / self.lookup_table.REFERENCE_DISTANCE
        cosiota = self._qmc_sequence['cosiota'][q_ids]
        return {
            'd_luminosity': d_luminosity,
            'dec': self.sky_dict.sky_samples['lat'][sky_ids],
            'lon': self.sky_dict.sky_samples['lon'][sky_ids],
            'phi_ref': phi_ref,
            'psi': self._qmc_sequence['psi'][q_ids],
            'iota': np.arccos(cosiota),
            't_geocenter': t_geocenter,
            'lnl_marginalized': np.full(num, marg_info.lnl_marginalized)[()],
            'lnl': real_dh / distance_ratio - h_h / distance_ratio**2 / 2,
            'h_h': h_h / distance_ratio**2,
            'n_effective': np.full(num, marg_info.n_effective)[()],
            'n_qmc': np.full(num, marg_info.n_qmc)[()]}
