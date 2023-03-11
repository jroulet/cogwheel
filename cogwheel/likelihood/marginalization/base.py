"""
Define classes ``BaseCoherentScore`` and ``BaseCoherentScoreHM`` to
marginalize the likelihood over extrinsic parameters from
matched-filtering timeseries.
Define classes ``MarginalizationInfo`` and ``MarginalizationInfoHM`` to
store auxiliary results, typical users would not directly use these.

``BaseCoherentScore`` is meant for subclassing differently depending on
the waveform physics included (precession and/or higher modes).
``BaseCoherentScoreHM`` is an abstract subclass that implements phase
marginalization for waveforms with higher modes.
"""
from abc import abstractmethod, ABC
import itertools
from dataclasses import dataclass
import warnings
import numba
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import krogh_interpolate, make_interp_spline
from scipy.special import logsumexp
from scipy import sparse

from cogwheel import gw_utils
from cogwheel import utils
from .lookup_table import LookupTable, LookupTableMarginalizedPhase22


@dataclass
class MarginalizationInfo:
    """
    Contains likelihood marginalized over extrinsic parameters, and
    intermediate products that can be used to generate extrinsic
    parameter samples or compute other auxiliary quantities like
    partial marginalizations.
    Below we refer to samples with sufficiently high maximum likelihood
    over distance to be included in the integral as "important" samples.

    Attributes
    ------
    ln_weights: float array of length n_important
        Natural log of the weights of the QMC samples, including the
        likelihood and the importance-sampling correction. Normalized so
        that the expectation value of all ``exp(ln_weights)`` (i.e.
        before selecting the important ones) is ``lnl_marginalized``.

    n_qmc: int
        Total number of QMC samples used in the integral, including
        unphysical and unimportant ones. Should be a power of 2.

    q_inds: int array of length n_important
        Indices to the QMC sequence.

    sky_inds: int array of length n_important
        Indices to sky_dict.sky_samples.

    t_first_det: float array of length n_important
        Time of arrival at the first detector.

    d_h: complex array of length n_important
        Inner product ⟨d|h⟩, where `h` is the waveform at a reference
        distance.

    h_h: float array of length n_important
        Real inner product ⟨h|h⟩.

    Properties
    ----------
    weights: float array of length n_important
        Weights of the QMC samples normalized to have unit sum.
        Proportional to ``exp(ln_weights)`` but with a different
        normalization.

    n_effective: float
        Effective number of samples achieved for the marginalization
        integral.

    lnl_marginalized: float
        log of the marginalized likelihood over extrinsic parameters
        excluding inclination (i.e.: time of arrival, sky location,
        polarization, distance, orbital phase).
    """
    ln_weights: np.ndarray
    n_qmc: int
    q_inds: np.ndarray
    sky_inds: np.ndarray
    t_first_det: np.ndarray
    d_h: np.ndarray
    h_h: np.ndarray

    def update(self, other):
        """
        Update entries of this instance of MarginalizationInfo to
        include information from another instance. The intended use is
        to extend the QMC sequence if it has too low ``.n_effective``.

        other: MarginalizationInfo
            Typically ``self`` will be the first half of the extended QMC
            sequence and ``other`` would be the second half.
        """
        self.n_qmc += other.n_qmc

        for attr in ('ln_weights', 'q_inds', 'sky_inds', 't_first_det', 'd_h',
                     'h_h'):
            updated = np.concatenate([getattr(self, attr),
                                      getattr(other, attr)])
            setattr(self, attr, updated)

    @property
    def n_effective(self):
        """
        A proxy of the effective number of samples contributing to the
        QMC integral. Note that the variance decreases faster than
        1 / n_effective because the QMC samples are not independent.
        This function uses the formula for independent samples, which
        may not be well justified.
        """
        if self.q_inds.size == 0:
            return 0.

        weights_q = sparse.coo_array(
            (self.weights, (np.zeros_like(self.q_inds), self.q_inds))
            ).toarray()[0]  # Repeated q_inds get summed
        return utils.n_effective(weights_q)

    @property
    def weights(self):
        """Weights of the QMC samples normalized to have unit sum."""
        return np.exp(self.ln_weights - logsumexp(self.ln_weights))

    @property
    def lnl_marginalized(self):
        if self.q_inds.size == 0:
            return -np.inf
        return logsumexp(self.ln_weights) - np.log(self.n_qmc)


@dataclass
class MarginalizationInfoHM(MarginalizationInfo):
    """
    Like ``MarginalizationInfo`` except:
      * it additionally contains ``o_inds``
      * ``d_h`` has dtype float, not complex.

    o_inds: int array of length n_important
        Indices to the orbital phase.

    d_h: float array of length n_important
        Real inner product ⟨d|h⟩.
    """
    o_inds: np.ndarray

    def update(self, other):
        super().update(other)
        self.o_inds = np.concatenate([self.o_inds, other.o_inds])


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
                 seed=0, beta_temperature=.5, n_qmc_sequences=128,
                 min_n_effective=50, max_log2n_qmc: int = 15):
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

        n_qmc_sequences: int
            The coherent score instance will generate `n_qmc_sequences`
            QMC sequences and cycle through them in subsequent calls.
            This alleviates the problem of repeated extrinsic parameter
            samples, without increasing the computational cost. It can
            also be used to estimate the uncertainty of the marginalized
            likelihood computation.

        min_n_effective: int
            Minimum effective sample size to use as convergence
            criterion. The program will try doubling the number of
            samples from `log2n_qmc` until a the effective sample size
            reaches `min_n_effective` or the number of extrinsic samples
            reaches ``2**max_log2n_qmc``.

        max_log2n_qmc: int
            Base-2 logarithm of the maximum number of extrinsic
            parameter samples to request. The program will try doubling
            the number of samples from `log2n_qmc` until a the effective
            sample size reaches `min_n_effective` or the number of
            extrinsic samples reaches ``2**max_log2n_qmc``.
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

        if max_log2n_qmc < log2n_qmc:
            raise ValueError('max_log2n_qmc < log2n_qmc')

        self.log2n_qmc = log2n_qmc
        self.sky_dict = sky_dict
        self.beta_temperature = beta_temperature
        self.min_n_effective = min_n_effective
        self.max_log2n_qmc = max_log2n_qmc

        self._current_qmc_sequence_id = 0
        self._qmc_sequences = [self._create_qmc_sequence()
                               for _ in range(n_qmc_sequences)]

        self._qmc_ind_chunks = self._create_qmc_ind_chunks()

        self._sample_distance = utils.handle_scalars(
            np.vectorize(self.lookup_table.sample_distance, otypes=[float]))

    @staticmethod
    @property
    @abstractmethod
    def _lookup_table_marginalized_params():
        """
        ``{'d_luminosity'}`` or ``{'d_luminosity', 'phi_ref'}``,
        allows to verify that the lookup table is of the correct type.
        """

    def _get_marginalization_info(self, *args, **kwargs):
        """
        Return a MarginalizationInfo object with extrinsic parameter
        integration results, ensuring that one of three conditions
        regarding the effective sample size holds:
            * n_effective >= .min_n_effective; or
            * n_qmc == 2 ** .max_log2n_qmc; or
            * n_effective is so low that even after extending the QMC
              sequence it is not expected to reach .min_n_effective

        The inputs are the same as ``._get_marginalization_info_chunks``
        except they do not include ``i_chunk``. Subclasses can override
        signature and docstring.
        """
        self._switch_qmc_sequence()

        i_chunk = 0
        marginalization_info = self._get_marginalization_info_chunk(
            *args, **kwargs, i_chunk=i_chunk)

        while self._worth_refining(marginalization_info):
            i_chunk += 1

            if i_chunk == len(self._qmc_ind_chunks):
                warnings.warn('Maximum QMC resolution reached.')
                break

            marginalization_info.update(self._get_marginalization_info_chunk(
                *args, **kwargs, i_chunk=i_chunk))

        return marginalization_info

    def _worth_refining(self, marginalization_info) -> bool:
        """
        Return ``True`` if the ``n_effective`` is lower than the minimum
        required, but high enough that extending the QMC sequence up to
        the maximum length would likely make it higher than that.
        """
        n_effective = marginalization_info.n_effective

        if n_effective >= self.min_n_effective:  # Has converged
            return False

        # Is there hope to achieve the desired n_effective?
        expected_increase = 2**self.max_log2n_qmc / marginalization_info.n_qmc
        return n_effective * expected_increase >= self.min_n_effective

    @abstractmethod
    def _get_marginalization_info_chunk(self, *args, **kwargs):
        """
        Return a MarginalizationInfo object using a specific chunk of
        the QMC sequence (without checking convergence).
        Provided by the subclass.
        """

    def _switch_qmc_sequence(self, qmc_sequence_id=None):
        if qmc_sequence_id is None:
            qmc_sequence_id = ((self._current_qmc_sequence_id + 1)
                               % self.n_qmc_sequences)
        self._current_qmc_sequence_id = qmc_sequence_id

    @property
    def _qmc_sequence(self):
        return self._qmc_sequences[self._current_qmc_sequence_id]

    @property
    def n_qmc_sequences(self):
        """Number of QMC sequences to alternate between."""
        return len(self._qmc_sequences)

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
                     ).random_base2(self.max_log2n_qmc),
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

    def _create_qmc_ind_chunks(self):
        qmc_inds = np.arange(2 ** self.max_log2n_qmc)
        breaks = 2 ** np.arange(self.log2n_qmc, self.max_log2n_qmc)
        return np.split(qmc_inds, breaks)

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

    def _get_fplus_fcross(self, sky_inds, q_inds):
        """
        Parameters
        ----------
        sky_inds: tuple of ints, of length n_physical
            Indices to sky_dict.sky_samples corresponding to the
            (physical) QMC samples.

        q_inds: int array of length n_physical
            Indices to the QMC sequence.

        Return
        ------
        fplus_fcross: float array of shape (n_physical, n_detectors, 2)
            Antenna factors.
        """
        fplus_fcross_0 = self.sky_dict.fplus_fcross_0[sky_inds,]  # qdp
        rot_psi = self._qmc_sequence['rot_psi'][q_inds]  # qpp'

        # Same but faster:
        # fplus_fcross = np.einsum('qpP,qdP->qdp', rot_psi, fplus_fcross_0)
        fplus_fcross = (rot_psi[:, np.newaxis]
                        @ fplus_fcross_0[..., np.newaxis]
                        )[..., 0]
        return fplus_fcross  # qdp

    def _draw_single_det_times(self, t_arrival_lnprob, times, q_inds):
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

        q_inds: (n_qmc,) int array
            Indices to the QMC sequence.

        Return
        ------
        t_first_det: float array of length n_qmc
            Time of arrival at the first detector (s) relative to tgps.

        delays: int array of shape (n_det-1, n_qmc)
            Time delay between the first detector and the other
            detectors, in units of 1/.skydict.f_sampling

        importance_sampling_weight: float array of length n_qmc
            Density ratio between the astrophysical prior and the
            proposal distribution of arrival times.
        """
        _, n_det = t_arrival_lnprob.shape
        n_qmc, = q_inds.shape

        # Sort detectors by SNR
        det_order = np.argsort(t_arrival_lnprob.max(axis=0))[::-1]

        tdet_inds = np.empty((n_det, n_qmc), int)  # dq
        tdet_weights = np.empty((n_det, n_qmc), float)  # dq
        dt = times[1] - times[0]

        for i, det_id in enumerate(det_order):
            # Rule out arrival times at current detector that are
            # already unphysical given arrival times at previous
            # detectors:
            for previous_det_id in det_order[:i]:
                max_delay = gw_utils.detector_travel_times(
                    self.sky_dict.detector_names[det_id],
                    self.sky_dict.detector_names[previous_det_id])

                t_previous_det = times[tdet_inds[previous_det_id]]
                unphysical = (
                    (times < t_previous_det.min() - max_delay - 2*dt)
                    | (times > t_previous_det.max() + max_delay + 2*dt))

                t_arrival_lnprob[unphysical, det_id] = -np.inf

            tdet_inds[det_id], tdet_weights[det_id] = _draw_indices(
                t_arrival_lnprob[:, det_id],
                self._qmc_sequence['u_tdet'][det_id, q_inds])

        delays = tdet_inds[1:] - tdet_inds[0]  # dq  # In units of dt
        importance_sampling_weight = np.prod(
            tdet_weights / self.sky_dict.f_sampling, axis=0)  # q

        t_first_det = (times[tdet_inds[0]]
                       + self._qmc_sequence['t_fine'][q_inds])  # q

        return t_first_det, delays, importance_sampling_weight

    @staticmethod
    def _interp_locally(times, timeseries, new_times, spline_degree=3):
        """
        Spline interpolation along last axis of ``timeseries`` excluding
        data outside the range spanned by `new_times`. If there is too
        little data inside to make a spline, use an interpolating
        polynomial.
        """
        t_rng = new_times.min(), new_times.max()
        i_min, i_max = np.clip(np.searchsorted(times, t_rng) + (-1, 1),
                               0, len(times) - 1)

        if i_max - i_min > spline_degree:  # If possible make a spline
            return make_interp_spline(
                times[i_min : i_max], timeseries[..., i_min : i_max],
                k=spline_degree, check_finite=False, axis=-1)(new_times)

        return krogh_interpolate(times[i_min : i_max],
                                 timeseries[..., i_min : i_max], new_times,
                                 axis=-1)


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
                 beta_temperature=.5, n_qmc_sequences=128,
                 min_n_effective=50, max_log2n_qmc: int = 15):
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

        n_qmc_sequences: int
            The coherent score instance will generate `n_qmc_sequences`
            QMC sequences and cycle through them in subsequent calls.
            This alleviates the problem of repeated extrinsic parameter
            samples, without increasing the computational cost. It can
            also be used to estimate the uncertainty of the marginalized
            likelihood computation.

        min_n_effective: int
            Minimum effective sample size to use as convergence
            criterion. The program will try doubling the number of
            samples from `log2n_qmc` until a the effective sample size
            reaches `min_n_effective` or the number of extrinsic samples
            reaches ``2**max_log2n_qmc``.

        max_log2n_qmc: int
            Base-2 logarithm of the maximum number of extrinsic
            parameter samples to request. The program will try doubling
            the number of samples from `log2n_qmc` until a the effective
            sample size reaches `min_n_effective` or the number of
            extrinsic samples reaches ``2**max_log2n_qmc``.
        """
        super().__init__(sky_dict=sky_dict,
                         lookup_table=lookup_table,
                         log2n_qmc=log2n_qmc,
                         seed=seed,
                         beta_temperature=beta_temperature,
                         n_qmc_sequences=n_qmc_sequences,
                         min_n_effective=min_n_effective,
                         max_log2n_qmc=max_log2n_qmc)

        self.m_arr = np.asarray(m_arr)
        self.m_inds, self.mprime_inds = (
            zip(*itertools.combinations_with_replacement(
                range(len(self.m_arr)), 2)))

        self._dh_phasor = None  # Set by nphi.setter
        self._hh_phasor = None  # Set by nphi.setter
        self.nphi = nphi

    @property
    def nphi(self):
        """
        Number of orbital phase values to integrate over using the
        trapezoid rule. Setting this attribute also defines:
            ._dh_phasor
            ._hh_phasor
            ._phi_ref
        """
        return self._nphi

    @nphi.setter
    def nphi(self, nphi):
        phi_ref = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        self._nphi = nphi
        self._dh_phasor = np.exp(-1j * np.outer(self.m_arr, phi_ref))  # mo
        self._hh_phasor = np.exp(1j * np.outer(
            self.m_arr[self.m_inds,] - self.m_arr[self.mprime_inds,],
            phi_ref))  # mo
        self._phi_ref = phi_ref

    def _get_lnweights_important(self, dh_qo, hh_qo, prior_weights_q):
        """
        Parameters
        ----------
        dh_qo: (n_physical, n_phi) float array
            ⟨d|h⟩ real inner product between data and waveform at
            ``self.lookup_table.REFERENCE_DISTANCE``.

        hh_qo: (n_physical, n_phi) float array
            ⟨h|h⟩ real inner product of a waveform at
            ``self.lookup_table.REFERENCE_DISTANCE`` with itself.

        prior_weights_q: (n_physical,) float array
            Positive importance-sampling weights of the QMC sequence.

        Return
        ------
        ln_weights: float array of length n_important
            Natural log of the weights of the QMC samples, including the
            likelihood and the importance-sampling correction.
            Normalized so that the expectation value of all
            ``exp(ln_weights)`` (i.e. not just the important ones kept
            here) is ``lnl_marginalized``.

        important: (array of ints, array of ints) of lengths n_important
            The first array contains indices between 0 and n_physical-1
            corresponding to (physical) QMC samples.
            The second array contains indices between 0 and n_phi-1
            corresponding to orbital phases.
            They correspond to samples with sufficiently high maximum
            likelihood over distance to be included in the integral.
        """
        max_over_distance_lnl = dh_qo * np.abs(dh_qo) / hh_qo / 2  # qo
        threshold = np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD
        important = np.where(max_over_distance_lnl > threshold)

        ln_weights = (self.lookup_table.lnlike_marginalized(dh_qo[important],
                                                            hh_qo[important])
                      + np.log(prior_weights_q)[important[0]]
                      - np.log(self._nphi))  # i

        return ln_weights, important


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
    prob /= prob.sum()
    cumprob = np.cumsum(prob)
    indices[:] = np.searchsorted(cumprob, quantiles)
    weights[:] = 1 / prob[indices]
