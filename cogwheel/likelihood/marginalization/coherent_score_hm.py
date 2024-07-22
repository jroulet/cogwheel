"""
Define class ``CoherentScoreHM`` to marginalize and demarginalize the
likelihood over extrinsic parameters from matched-filtering timeseries.

``CoherentScoreHM`` works for quasi-circular waveforms with precession
and higher modes. The inclination can't be marginalized over and is
treated as an intrinsic parameter.
"""
import warnings
import numpy as np

from cogwheel import utils
from .base import (BaseCoherentScoreHM,
                   MarginalizationInfoHM,
                   ProposingCoherentScore)


def _flip_psi(psi, d_h, flip_psi):
    if flip_psi:
        return (psi + np.pi/2) % np.pi, -d_h
    return psi, d_h


_flip_psi = utils.handle_scalars(np.vectorize(_flip_psi,
                                              otypes=[float, float]))


class CoherentScoreHM(ProposingCoherentScore, BaseCoherentScoreHM):
    """
    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters
    (``.get_marginalization_info()``). Extrinsic parameters samples can
    be generated as well (``.gen_samples()``).

    Works for quasi-circular waveforms with generic spins and higher
    modes.

    Inherits from ``BaseCoherentScoreHM``.
    """
    def __init__(self, sky_dict, m_arr, lookup_table=None,
                 log2n_qmc: int = 11, nphi=128, seed=0,
                 n_qmc_sequences=128, min_n_effective=50,
                 max_log2n_qmc: int = 15,
                 beta_temperature=.5, learning_rate=1e-2):
        super().__init__(sky_dict=sky_dict,
                         m_arr=m_arr,
                         lookup_table=lookup_table,
                         log2n_qmc=log2n_qmc,
                         nphi=nphi,
                         seed=seed,
                         n_qmc_sequences=n_qmc_sequences,
                         min_n_effective=min_n_effective,
                         max_log2n_qmc=max_log2n_qmc,
                         beta_temperature=beta_temperature,
                         learning_rate=learning_rate)

    def _get_marginalization_info_chunk(self, d_h_timeseries, h_h,
                                        times, t_arrival_prob, i_chunk):
        """
        Evaluate inner products (d|h) and (h|h) at integration points
        over a chunk of a QMC sequence of extrinsic parameters, given
        timeseries of (d|h) and value of (h|h) by mode `m`, polarization
        `p` and detector `d`.

        Parameters
        ----------
        d_h_timeseries: (n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        h_h: (n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        t_arrival_prob: (n_d, n_t) float array
            Proposal probability of time of arrival at each detector,
            normalized to sum to 1 along the time axis.

        i_chunk: int
            Index to ``._qmc_ind_chunks``.

        Return
        ------
        Instance of ``MarginalizationInfoHM`` with several fields, see
        its documentation.
        """
        if d_h_timeseries.shape[0] != self.m_arr.size:
            raise ValueError('Incorrect number of harmonic modes.')

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
            return MarginalizationInfoHM(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals=[t_arrival_prob],
                flip_psi=np.array([], bool)
                )

        t_first_det = (times[tdet_inds[0]]
                       + self._qmc_sequence['t_fine'][q_inds])

        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, q_inds, t_first_det,
                                          times, d_h_timeseries, h_h)  # qo, qo

        ln_numerators, important, flip_psi \
            = self._get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)

        # Keep important samples (lnl above threshold):
        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]
        tdet_inds = tdet_inds[:, important[0]]

        return MarginalizationInfoHM(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds,
            o_inds=important[1],
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals=[t_arrival_prob],
            flip_psi=flip_psi,
            )

    def gen_samples(self, dh_mptd, hh_mppd, times, num=(),
                    lnl_marginalized_threshold=-np.inf):
        """Deprecated, use ``.gen_samples_from_marg_info``"""
        warnings.warn('Use ``gen_samples_from_marg_info``', DeprecationWarning)
        marg_info = self.get_marginalization_info(dh_mptd, hh_mppd, times,
                                                  lnl_marginalized_threshold)
        return self.gen_samples_from_marg_info(marg_info, num)

    def gen_samples_from_marg_info(self, marg_info, num=()):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        marg_info: MarginalizationInfoHM or None
            Normally, output of ``.get_marginalization_info``.
            If ``None``, assume that the sampled parameters were unphysical
            and return samples full of nans.

        num: int, optional
            Number of samples to generate, ``None`` makes a single sample.

        Return
        ------
        samples: dict
            Values are scalar if `num` is ``None``, else numpy arrays.
            If ``marg_info`` correspond to an unphysical sample (i.e.,
            a realization of matched-filtering timeseries in the
            detectors incompatible with a real signal) the values will
            be NaN.
        """
        if marg_info is None or marg_info.q_inds.size == 0:
            # Order and dtype must match that of regular output
            out = dict.fromkeys(['d_luminosity', 'dec', 'lon', 'phi_ref',
                                 'psi', 't_geocenter', 'lnl_marginalized',
                                 'lnl', 'h_h', 'n_effective', 'n_qmc'],
                                np.full(num, np.nan)[()])
            if marg_info is None:
                out['n_qmc'] = np.full(num, 0)[()]
            else:
                out['lnl_marginalized'] = np.full(
                    num, marg_info.lnl_marginalized)[()]
                out['n_effective'] = np.full(num, marg_info.n_effective)[()]
                out['n_qmc'] = np.full(num, marg_info.n_qmc)[()]
            return out

        self._switch_qmc_sequence(marg_info.qmc_sequence_id)
        random_ids = self._rng.choice(len(marg_info.q_inds), size=num,
                                      p=marg_info.weights)[()]

        q_ids = marg_info.q_inds[random_ids]
        o_ids = marg_info.o_inds[random_ids]
        sky_ids = marg_info.sky_inds[random_ids]
        t_geocenter = (marg_info.t_first_det[random_ids]
                       - self.sky_dict.geocenter_delay_first_det[sky_ids])
        h_h = marg_info.h_h[random_ids]

        psi, d_h = _flip_psi(self._qmc_sequence['psi'][q_ids],
                             marg_info.d_h[random_ids],
                             marg_info.flip_psi[random_ids])

        d_luminosity = self._sample_distance(d_h, h_h)
        distance_ratio = d_luminosity / self.lookup_table.REFERENCE_DISTANCE

        return {
            'd_luminosity': d_luminosity,
            'dec': self.sky_dict.sky_samples['lat'][sky_ids],
            'lon': self.sky_dict.sky_samples['lon'][sky_ids],
            'phi_ref': self._phi_ref[o_ids],
            'psi': psi,
            't_geocenter': t_geocenter,
            'lnl_marginalized': np.full(num, marg_info.lnl_marginalized)[()],
            'lnl': d_h / distance_ratio - h_h / distance_ratio**2 / 2,
            'h_h': h_h / distance_ratio**2,
            'n_effective': np.full(num, marg_info.n_effective)[()],
            'n_qmc': np.full(num, marg_info.n_qmc)[()]}

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
             for i_det in range(len(self.sky_dict.detector_names))],
            dtype=np.complex64)

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

    def _incoherent_t_arrival_lnprob(self, d_h_timeseries, h_h):
        """
        Log likelihood maximized over distance and phase, approximating
        that different modes and polarizations are all orthogonal and
        have independent phases.

        Parameters
        ----------
        d_h_timeseries: (n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        h_h: (n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.
        """
        hh_mpdiagonal = h_h[np.equal(self.m_inds, self.mprime_inds)
                           ][:, (0, 1), (0, 1)].real  # mpd
        chi_squared = (np.einsum('mptd->td', np.abs(d_h_timeseries))**2
                       / np.einsum('mpd->d', hh_mpdiagonal))

        return (self.beta_temperature / 2 * chi_squared).T  # dt
