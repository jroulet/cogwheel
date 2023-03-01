"""
Define class ``CoherentScoreHM`` to marginalize and demarginalize the
likelihood over extrinsic parameters from matched-filtering timeseries.

``CoherentScoreHM`` works for quasi-circular waveforms with precession
and higher modes. The inclination can't be marginalized over and is
treated as an intrinsic parameter.
"""
import numpy as np

from cogwheel import utils
from .base import BaseCoherentScoreHM, MarginalizationInfoHM


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
        Instance of ``MarginalizationInfoHM`` with several fields, see
        its documentation.
        """
        return super()._get_marginalization_info(dh_mptd, hh_mppd, times)

    def _get_marginalization_info_chunk(self, dh_mptd, hh_mppd, times,
                                        i_chunk):
        """
        Like ``.get_marginalization_info`` but integrates over a
        specific chunk of the QMC sequence (without checking
        convergence).

        i_chunk: int
            Index to ``._qmc_ind_chunks``.
        """
        q_inds = self._qmc_ind_chunks[i_chunk]  # Will update along the way
        n_qmc = len(q_inds)

        # Resample to match sky_dict's dt:
        dh_mptd, times = self.sky_dict.resample_timeseries(dh_mptd, times,
                                                           axis=2)

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(dh_mptd,
                                                             hh_mppd)  # td
        t_first_det, delays, importance_sampling_weight \
            = self._draw_single_det_times(t_arrival_lnprob, times, q_inds)

        sky_inds, sky_prior, physical_mask \
            = self.sky_dict.get_sky_inds_and_prior(delays)  # q, q, q

        if not any(physical_mask):
            return MarginalizationInfoHM(ln_weights=np.array([]),
                                         n_qmc=n_qmc,
                                         q_inds=np.array([], int),
                                         o_inds=np.array([], int),
                                         sky_inds=np.array([], int),
                                         t_first_det=np.array([]),
                                         d_h=np.array([]),
                                         h_h=np.array([]))

        # Apply physical mask (sensible time delays):
        q_inds = q_inds[physical_mask]
        t_first_det = t_first_det[physical_mask]
        importance_sampling_weight = importance_sampling_weight[physical_mask]

        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, q_inds, t_first_det,
                                          times, dh_mptd, hh_mppd)  # qo, qo

        ln_weights, important = self._get_lnweights_important(
            dh_qo, hh_qo, importance_sampling_weight * sky_prior)

        # Keep important samples (lnl above threshold):
        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]

        return MarginalizationInfoHM(ln_weights=ln_weights,
                                     n_qmc=n_qmc,
                                     q_inds=q_inds,
                                     o_inds=important[1],
                                     sky_inds=sky_inds,
                                     t_first_det=t_first_det,
                                     d_h=dh_qo[important],
                                     h_h=hh_qo[important])

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
        marg_info: MarginalizationInfoHM
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
        if marg_info.q_inds.size == 0:
            return dict.fromkeys(['d_luminosity', 'dec', 'lon', 'phi_ref',
                                  'psi', 't_geocenter', 'lnl_marginalized',
                                  'lnl'],
                                 np.full(num, np.nan)[()])

        random_ids = self._rng.choice(len(marg_info.q_inds), size=num,
                                      p=marg_info.weights)

        q_ids = marg_info.q_inds[random_ids]
        o_ids = marg_info.o_inds[random_ids]
        sky_ids = marg_info.sky_inds[random_ids]
        t_geocenter = (marg_info.t_first_det[random_ids]
                       - self.sky_dict.geocenter_delay_first_det[sky_ids])
        d_h = marg_info.d_h[random_ids]
        h_h = marg_info.h_h[random_ids]

        d_luminosity = self._sample_distance(d_h, h_h)
        distance_ratio = d_luminosity / self.lookup_table.REFERENCE_DISTANCE
        return {'d_luminosity': d_luminosity,
                'dec': self.sky_dict.sky_samples['lat'][sky_ids],
                'lon': self.sky_dict.sky_samples['lon'][sky_ids],
                'phi_ref': self._phi_ref[o_ids],
                'psi': self._qmc_sequence['psi'][q_ids],
                't_geocenter': t_geocenter,
                'lnl_marginalized': marg_info.lnl_marginalized,
                'lnl': d_h / distance_ratio - h_h / distance_ratio**2 / 2,
                'h_h': h_h / distance_ratio**2}

    def _get_dh_hh_qo(self, sky_inds, q_inds, t_first_det, times,
                      dh_mptd, hh_mppd):
        """
        Apply antenna factors and orbital phase to the polarizations, to
        obtain (d|h) and (h|h) by extrinsic sample 'q' and orbital phase
        'o'.
        """
        fplus_fcross = self._get_fplus_fcross(sky_inds, q_inds)

        # (d|h):
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmpq = np.array(
            [self._interp_locally(times, dh_mptd[..., i_det], t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        # Same but faster:
        # dh_qm = np.einsum('dmpq,qdp->qm', dh_dmpq, fplus_fcross)  # qm
        n_d, n_m, n_p, n_q = dh_dmpq.shape
        dh_qm = (np.moveaxis(dh_dmpq, (3, 1), (0, 1)
                            ).reshape(n_q, n_m, n_d*n_p)  # qm(dp)
                 @ fplus_fcross.reshape(n_q, n_d*n_p, 1)  # q(dp)_
                ).reshape(n_q, n_m)  # qm

        # (h|h):
        f_f = np.einsum('qdp,qdP->qpPd', fplus_fcross, fplus_fcross)
        hh_qm = (f_f.reshape(f_f.shape[0], -1)
                 @ hh_mppd.reshape(hh_mppd.shape[0], -1).T)  # qm

        dh_qo = utils.real_matmul(dh_qm, self._dh_phasor)  # qo
        hh_qo = utils.real_matmul(hh_qm, self._hh_phasor)  # qo
        return dh_qo, hh_qo

    def _incoherent_t_arrival_lnprob(self, dh_mptd, hh_mppd):
        """
        Log likelihood maximized over distance and phase, approximating
        that different modes and polarizations are all orthogonal and
        have independent phases.
        """
        hh_mpdiagonal = hh_mppd[np.equal(self.m_inds, self.mprime_inds)
                               ][:, (0, 1), (0, 1)].real  # mpd
        chi_squared = (np.einsum('mptd->td', np.abs(dh_mptd))**2
                       / np.einsum('mpd->d', hh_mpdiagonal))

        return self.beta_temperature * chi_squared / 2  # td
