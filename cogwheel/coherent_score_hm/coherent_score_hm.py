"""
Define classes ``BaseCoherentScore`` and ``CoherentScoreHM`` to
marginalize the likelihood over extrinsic parameters from
matched-filtering timeseries.

``BaseCoherentScore`` is meant for subclassing differently depending on
the waveform physics included (precession and/or higher modes).

``CoherentScoreHM`` works for quasi-circular waveforms with precession
and higher modes. The inclination can't be marginalized over and is
treated as an intrinsic parameter.
"""

import itertools
from collections import namedtuple
import numba
import numpy as np
from scipy.stats import qmc
import scipy.signal

from cogwheel import likelihood
from cogwheel import utils


class BaseCoherentScore(utils.JSONMixin):
    """
    Base class for computing coherent scores (i.e., marginalized
    likelihoods over extrinsic parameters) from matched-filtering
    timeseries.
    Meant to be subclassed differently depending on the waveform physics
    (precession and/or higher modes) that may require different
    algorithms.
    This class provides methods to initialize and perform some of the
    generic steps that the coherent score computation normally requires.
    """
    # Remove from the orbital phase integral any sample with a drop in
    # log-likelihood from the peak bigger than ``DLNL_THRESHOLD``:
    DLNL_THRESHOLD = 12.

    def __init__(self, sky_dict, m_arr, lookup_table=None,
                 log2n_qmc: int = 11, nphi=128, seed=0,
                 beta_temperature=.1):
        """
        Parameters
        ----------
        sky_dict:
            Instance of cogwheel.coherent_score_hm.skydict.SkyDictionary

        m_arr: int array
            m number of the harmonic modes considered.

        lookup_table:
            Instance of cogwheel.likelihood.marginalized_distance.LookupTable

        log2n_qmc: int
            Base-2 logarithm of the number of requested extrinsic
            parameter samples.

        nphi: int
            Number of orbital phases over which to perform
            marginalization with trapezoid quadrature rule.

        seed: {int, None, np.random.RandomState}
            For reproducibility of the extrinsic parameter samples.

        beta_temperature: float
            Inverse temperature, tempers the arrival time probability at
            each detector.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if lookup_table is None:
            lookup_table = likelihood.LookupTable()
        self.lookup_table = lookup_table

        self.log2n_qmc = log2n_qmc
        self.sky_dict = sky_dict

        self.m_arr = np.asarray(m_arr)
        self.m_inds, self.mprime_inds = (
            zip(*itertools.combinations_with_replacement(
                range(len(self.m_arr)), 2)))

        self._dh_phasor = None  # Set by nphi.setter
        self._hh_phasor = None  # Set by nphi.setter
        self._dphi = None  # Set by nphi.setter
        self.nphi = nphi

        self.beta_temperature = beta_temperature

        self._qmc_sequence = self._create_qmc_sequence()

        self._sample_distance = utils.handle_scalars(
            np.vectorize(self.lookup_table.sample_distance, otypes=[float]))

    @property
    def nphi(self):
        """
        Number of orbital phases to integrate over with the trapezoid
        rule.
        """
        return self._nphi

    @nphi.setter
    def nphi(self, nphi):
        phi_ref, dphi = np.linspace(0, 2*np.pi, nphi,
                                    endpoint=False, retstep=True)
        self._nphi = nphi
        self._dh_phasor = np.exp(-1j * np.outer(self.m_arr, phi_ref))  # mo
        self._hh_phasor = np.exp(1j * np.outer(
            self.m_arr[self.m_inds,] - self.m_arr[self.mprime_inds,],
            phi_ref))  # mo
        self._phi_ref = phi_ref
        self._dphi = dphi

    def _create_qmc_sequence(self):
        """
        Return a dictionary whose values are arrays corresponding to a
        Quasi Monte Carlo sequence that explores parameters per
        ``._qmc_range_dic``.
        The arrival time cumulatives are packed in a single entry
        'u_tdet'. An entry 'rot_psi' has the rotation matrices to
        transform the antenna factors between psi=0 and psi=psi_qmc.
        """
        sequence_values = qmc.scale(
            qmc.Sobol(len(self._qmc_range_dic), seed=self._rng
                     ).random_base2(self.log2n_qmc),
            *zip(*self._qmc_range_dic.values())).T

        n_det = len(self.sky_dict.detector_names)

        qmc_sequence = dict(zip(list(self._qmc_range_dic)[n_det:],
                                sequence_values[n_det:]),
                            u_tdet=sequence_values[:n_det])

        sintwopsi = np.sin(2 * qmc_sequence['psi'])
        costwopsi = np.cos(2 * qmc_sequence['psi'])
        qmc_sequence['rot_psi'] = np.moveaxis([[costwopsi, sintwopsi],
                                               [-sintwopsi, costwopsi]],
                                              -1, 0)  # qpp'
        return qmc_sequence

    @property
    def _qmc_range_dic(self):
        """
        Parameter ranges for the QMC sequence.
        The sequence explores the cumulatives of the single-detector
        (incoherent) likelihood of arrival times, the polarization, and
        the fine (subpixel) time of arrival.
        """
        dt = 1 / self.sky_dict.f_sampling
        return dict.fromkeys(self.sky_dict.detector_names, (0, 1)
                            ) | {'t_fine': (-dt/2, dt/2),
                                 'psi': (0, np.pi)}

    def _get_lnl_marginalized_and_weights(self, dh_qo, hh_qo,
                                          prior_weights_q):
        max_over_distance_lnl = dh_qo * np.abs(dh_qo) / hh_qo / 2  # qo
        important = np.where(
            max_over_distance_lnl
            > np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD)
        lnl_marg_dist = self._lnlike_marginalized_over_distance(
            dh_qo[important], hh_qo[important])  # i

        lnl_max = lnl_marg_dist.max()
        like_marg_dist = np.exp(lnl_marg_dist - lnl_max)  # i

        full_weights_i = like_marg_dist * prior_weights_q[important[0]]  # i
        lnl_marginalized = lnl_max + np.log(full_weights_i.sum() * self._dphi
                                            / 2**self.log2n_qmc)
        return lnl_marginalized, full_weights_i, important

    def _get_fplus_fcross(self, sky_inds, physical_mask):
        """
        Parameters
        ----------
        sky_inds: tuple of ints, of length n_physical
            Indices to sky_dict.sky_samples corresponding to the
            (physical) QMC samples.

        physical_mask: boolean array of length n_qmc
            Some choices of time of arrival at detectors may not
            correspond to any physical sky location, these are flagged
            ``False`` in this array.

        Return
        ------
        fplus_fcross: float array of shape (n_physical, n_detectors, 2)
            Antenna factors.
        """
        fplus_fcross_0 = self.sky_dict.fplus_fcross_0[sky_inds,]  # qdp
        rot_psi = self._qmc_sequence['rot_psi'][physical_mask]  # qpp'
        fplus_fcross = np.einsum('qpP,qdP->qdp', rot_psi, fplus_fcross_0)
        return fplus_fcross  # qdp

    def _draw_single_det_times(self, t_arrival_lnprob, times):
        """
        Choose time of arrivals independently at each detector according
        to the QMC sequence, according to a proposal distribution based
        on the matched-filtering timeseries.

        Parameters
        ----------
        t_arrival_lnprob: (n_times, n_det) float array
            Incoherent proposal for log probability of arrival times at
            each detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        Return
        ------
        t_first_det: float array of length n_physical
            Time of arrival at the first detector.

        delays: int array of shape (n_det-1, n_physical)
            Time delay between the first detector and the other
            detectors, in units of 1/.skydict.f_sampling

        physical_mask: boolean array of length n_qmc
            Some choices of time of arrival at detectors may not
            correspond to any physical sky location, these are flagged
            ``False`` in this array. Unphysical samples are discarded.

        importance_sampling_weight: array
            Density ratio between the astrophysical prior and the
            proposal distribution of arrival times.
        """
        tdet_inds, tdet_weights = _draw_indices(
            t_arrival_lnprob.T, self._qmc_sequence['u_tdet'])  # dq, dq

        delays = tdet_inds[1:] - tdet_inds[0]  # dq  # In units of dt
        physical_mask = np.array([delays_key in self.sky_dict.delays2genind_map
                                  for delays_key in zip(*delays)])
        delays = delays[:, physical_mask]

        importance_sampling_weight = np.prod(tdet_weights[:, physical_mask]
                                             / self.sky_dict.f_sampling,
                                             axis=0)  # q

        t_first_det = (times[tdet_inds[0, physical_mask]]
                       + self._qmc_sequence['t_fine'][physical_mask])  # q

        return t_first_det, delays, physical_mask, importance_sampling_weight

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


class CoherentScoreHM(BaseCoherentScore):
    """
    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters
    (``.get_marginalization_info()``). Extrinsic parameters samples can
    be generated as well (``.gen_samples()``).

    Works for quasi-circular waveforms with generic spins and higher
    modes.

    Inherits from ``BaseCoherentScore``.
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
        # Resample to match sky_dict's dt:
        dh_mptd, times = self.sky_dict.resample_timeseries(dh_mptd, times,
                                                           axis=2)

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(dh_mptd,
                                                             hh_mppd)  # td
        t_first_det, delays, physical_mask, importance_sampling_weight \
            = self._draw_single_det_times(t_arrival_lnprob, times)

        if not any(physical_mask):
            return self._MarginalizationInfo(physical_mask=physical_mask,
                                             t_first_det=np.array([]),
                                             dh_qo=np.empty((0, self.nphi)),
                                             hh_qo=np.empty((0, self.nphi)),
                                             sky_inds=(),
                                             weights=np.array([]),
                                             lnl_marginalized=-np.inf,
                                             important=[(), ()])

        sky_inds, sky_prior = zip(
            *(next(self.sky_dict.delays2genind_map[delays_key])
              for delays_key in zip(*delays)))  # q, q

        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, physical_mask, t_first_det,
                                          times, dh_mptd, hh_mppd)  # qo, qo

        max_over_distance_lnl = dh_qo * np.abs(dh_qo) / hh_qo / 2  # qo
        important = np.where(
            max_over_distance_lnl
            > np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD)
        lnl_marg_dist = self._lnlike_marginalized_over_distance(
            dh_qo[important], hh_qo[important])  # i

        lnl_max = lnl_marg_dist.max()
        like_marg_dist = np.exp(lnl_marg_dist - lnl_max)  # i

        weights_i = (np.array(sky_prior)[important[0]]
                     * importance_sampling_weight[important[0]])  # i

        full_weights = like_marg_dist * weights_i
        lnl_marginalized = lnl_max + np.log(full_weights.sum() * self._dphi
                                            / 2**self.log2n_qmc)

        return self._MarginalizationInfo(physical_mask=physical_mask,
                                         t_first_det=t_first_det,
                                         dh_qo=dh_qo,
                                         hh_qo=hh_qo,
                                         sky_inds=sky_inds,
                                         weights=full_weights,
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
            If ``marg_info`` correspond to an unphysical sample (i.e.,
            a realization of matched-filtering timeseries in the
            detectors incompatible with a real signal) the values will
            be NaN.
        """
        marg_info = self.get_marginalization_info(
            dh_mptd, hh_mppd, times)
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

        i_ids = np.random.choice(len(marg_info.weights),
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
        return {'d_luminosity': d_luminosity,
                'dec': self.sky_dict.sky_samples['lat'][sky_ids],
                'lon': self.sky_dict.sky_samples['lon'][sky_ids],
                'phi_ref': self._phi_ref[o_ids],
                'psi': self._qmc_sequence['psi'][marg_info.physical_mask][q_ids],
                't_geocenter': t_geocenter,
                'lnl_marginalized': marg_info.lnl_marginalized,
                'lnl': d_h / distance_ratio - h_h / distance_ratio**2 / 2}

    def _get_dh_hh_qo(self, sky_inds, physical_mask, t_first_det, times,
                      dh_mptd, hh_mppd):
        """
        Apply antenna factors and orbital phase to the polarizations, to
        obtain (d|h) and (h|h) by extrinsic sample 'q' and orbital phase
        'o'.
        """
        fplus_fcross_0 = self.sky_dict.fplus_fcross_0[sky_inds,]  # qdp
        rot_psi = self._qmc_sequence['rot_psi'][physical_mask]  # qpp'
        fplus_fcross = np.einsum('qpP,qdP->qdp', rot_psi, fplus_fcross_0)

        # # (d|h):
        # select = (...,  # mp stay the same
        #           tdet_inds.T[physical_mask],  # t -> q depending on d
        #           np.arange(len(self.sky_dict.detector_names))  # d
        #           )
        # dh_mpqd = dh_mptd[select]  # mpqd
        # dh_dmpq = np.moveaxis(dh_mpqd, -1, 0)

        # Alternative computation of dh_dmpq above, more accurate
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmpq = np.array(
            [scipy.interpolate.interp1d(times, dh_mptd[..., i_det], kind=3,
                                        copy=False, assume_sorted=True,
                                        fill_value=0., bounds_error=False
                                        )(t_det[i_det])
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


@numba.guvectorize([(numba.float64[:], numba.float64[:],
                     numba.int64[:], numba.float64[:])],
                   '(n),(m)->(m),(m)')
def _draw_indices(unnormalized_lnprob, quantiles, indices, weights):
    """
    Sample desired quantiles from a distribution using the inverse of
    its cumulative.

    Parameters
    ----------
    unnormalized_lnprob, quantiles

    Return
    ------
    indices, weights
    """
    prob = np.exp(unnormalized_lnprob - unnormalized_lnprob.max())
    cumprob = np.cumsum(prob)
    prob /= cumprob[-1]  # Unit sum
    cumprob /= cumprob[-1]
    indices[:] = np.searchsorted(cumprob, quantiles)
    weights[:] = 1 / prob[indices]
