"""
Define classes ``BaseCoherentScore`` and ``BaseCoherentScoreHM`` to
marginalize the likelihood over extrinsic parameters from
matched-filtering timeseries.

``BaseCoherentScore`` is meant for subclassing differently depending on
the waveform physics included (precession and/or higher modes).
``BaseCoherentScoreHM`` is an abstract subclass that implements phase
marginalization for waveforms with higher modes.
"""
from abc import abstractmethod, ABC
import itertools
from scipy.stats import qmc
import numba
import numpy as np

from cogwheel import utils
from .lookup_table import LookupTable, LookupTableMarginalizedPhase22


class BaseCoherentScore(utils.JSONMixin, ABC):
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
    def __init__(self, sky_dict, lookup_table=None, log2n_qmc: int = 11,
                 seed=0, beta_temperature=.1):
        """
        Parameters
        ----------
        sky_dict:
            Instance of .skydict.SkyDictionary

        lookup_table:
            Instance of .lookup_table.LookupTable

        log2n_qmc: int
            Base-2 logarithm of the number of requested extrinsic
            parameter samples.

        seed: {int, None, np.random.RandomState}
            For reproducibility of the extrinsic parameter samples.

        beta_temperature: float
            Inverse temperature, tempers the arrival time probability at
            each detector.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Set up and check lookup_table with correct marginalized_params:
        if lookup_table is None:
            if self._lookup_table_marginalized_params == {'d_luminosity'}:
                lookup_table = LookupTable()
            elif self._lookup_table_marginalized_params == {'d_luminosity',
                                                            'phi_ref'}:
                lookup_table = LookupTableMarginalizedPhase22()
            else:
                raise ValueError('Unable to initialize `lookup_table`.')
        if (lookup_table.marginalized_params
                != self._lookup_table_marginalized_params):
            raise ValueError('Incompatible lookup_table.marginalized_params')
        self.lookup_table = lookup_table

        self.log2n_qmc = log2n_qmc
        self.sky_dict = sky_dict
        self.beta_temperature = beta_temperature
        self._qmc_sequence = self._create_qmc_sequence()

        self._sample_distance = utils.handle_scalars(
            np.vectorize(self.lookup_table.sample_distance, otypes=[float]))

    @staticmethod
    @property
    @abstractmethod
    def _lookup_table_marginalized_params():
        """
        ``{'d_luminosity'}`` or ``{'d_luminosity', 'phi_ref'}``,
        allows to verify that the ``LookupTable`` is of the correct
        type.
        """

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


class BaseCoherentScoreHM(BaseCoherentScore):
    """
    With higher order modes it is not possible to marginalize the
    orbital phase analytically so we use trapezoid quadrature.
    ``BaseCoherentScoreHM`` provides attributes and methods for doing
    that.

    Attributes
    ----------
    m_arr
    m_inds
    mprime_inds
    nphi
    """
    # Remove from the orbital phase integral any sample with a drop in
    # log-likelihood from the peak bigger than ``DLNL_THRESHOLD``:
    DLNL_THRESHOLD = 12.

    _lookup_table_marginalized_params = {'d_luminosity'}

    def __init__(self, sky_dict, m_arr, lookup_table=None,
                 log2n_qmc: int = 11, nphi=128, seed=0,
                 beta_temperature=.1):
        """
        Parameters
        ----------
        sky_dict:
            Instance of .skydict.SkyDictionary

        m_arr: int array
            m number of the harmonic modes considered.

        lookup_table:
            Instance of lookup_table.LookupTable

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
        super().__init__(sky_dict=sky_dict,
                         lookup_table=lookup_table,
                         log2n_qmc=log2n_qmc,
                         seed=seed,
                         beta_temperature=beta_temperature)

        self.m_arr = np.asarray(m_arr)
        self.m_inds, self.mprime_inds = (
            zip(*itertools.combinations_with_replacement(
                range(len(self.m_arr)), 2)))

        self._dh_phasor = None  # Set by nphi.setter
        self._hh_phasor = None  # Set by nphi.setter
        self._dphi = None  # Set by nphi.setter
        self.nphi = nphi

    @property
    def nphi(self):
        """
        Number of orbital phase values to integrate over using the
        trapezoid rule. Setting this attribute also defines:
            ._dh_phasor
            ._hh_phasor
            ._phi_ref
            ._dphi
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

    def _get_lnl_marginalized_and_weights(self, dh_qo, hh_qo,
                                          prior_weights_q):
        max_over_distance_lnl = dh_qo * np.abs(dh_qo) / hh_qo / 2  # qo
        important = np.where(
            max_over_distance_lnl
            > np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD)
        lnl_marg_dist = self.lookup_table.lnlike_marginalized(
            dh_qo[important], hh_qo[important])  # i

        lnl_max = lnl_marg_dist.max()
        like_marg_dist = np.exp(lnl_marg_dist - lnl_max)  # i

        full_weights_i = like_marg_dist * prior_weights_q[important[0]]  # i
        lnl_marginalized = lnl_max + np.log(full_weights_i.sum() * self._dphi
                                            / 2**self.log2n_qmc)
        return lnl_marginalized, full_weights_i, important



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
