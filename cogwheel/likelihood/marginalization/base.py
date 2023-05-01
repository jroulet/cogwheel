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
from dataclasses import dataclass, field
from functools import wraps
import warnings
import numpy as np
from scipy.interpolate import krogh_interpolate, make_interp_spline
from scipy.special import logsumexp
from scipy import signal
from scipy import sparse
from scipy import stats

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
    ln_numerators: np.ndarray
    q_inds: np.ndarray
    sky_inds: np.ndarray
    t_first_det: np.ndarray
    d_h: np.ndarray
    h_h: np.ndarray
    tdet_inds: np.ndarray
    proposals_n_qmc: list
    proposals: list

    weights: np.ndarray = field(init=False)
    weights_q: np.ndarray = field(init=False)
    n_qmc: int = field(init=False)
    n_effective: float = field(init=False)
    lnl_marginalized: float = field(init=False)

    def __post_init__(self):
        """Set derived attributes."""
        denominators = np.zeros(len(self.q_inds))
        for n_qmc, proposal in zip(self.proposals_n_qmc, self.proposals):
            denominators += n_qmc * np.prod(
                np.take_along_axis(proposal, self.tdet_inds, axis=1),
                axis=0)  # q

        ln_weights = self.ln_numerators - np.log(denominators)
        self.weights = utils.exp_normalize(ln_weights)

        weights_q = sparse.coo_array(
            (self.weights, (np.zeros_like(self.q_inds), self.q_inds))
            ).toarray()[0]  # Repeated q_inds get summed
        self.weights_q = weights_q[weights_q > 0]
        self.n_effective = utils.n_effective(self.weights_q)

        self.n_qmc = sum(self.proposals_n_qmc)

        if self.q_inds.size == 0:
            self.lnl_marginalized = -np.inf
        else:
            self.lnl_marginalized = logsumexp(ln_weights)

    def update(self, other):
        """
        Update entries of this instance of MarginalizationInfo to
        include information from another instance. The intended use is
        to extend the QMC sequence if it has too low ``.n_effective``.

        Parameters
        ----------
        other: MarginalizationInfo
            Typically ``self`` will be the first half of the extended
            QMC sequence and ``other`` would be the second half.
        """
        self.proposals_n_qmc += other.proposals_n_qmc
        self.proposals += other.proposals

        for attr in ('ln_numerators', 'q_inds', 'sky_inds', 't_first_det',
                     'd_h', 'h_h', 'tdet_inds'):
            updated = np.concatenate([getattr(self, attr),
                                      getattr(other, attr)], axis=-1)
            setattr(self, attr, updated)

        self.__post_init__()


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

    @wraps(MarginalizationInfo.update)
    def update(self, other):
        self.o_inds = np.concatenate([self.o_inds, other.o_inds])
        super().update(other)


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
    _searchsorted = np.vectorize(np.searchsorted, signature='(n),(m)->(m)')
    _temperature_factor = 2

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

        beta_temperature: float or float array of shape (n_detectors,)
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

        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.sky_dict = sky_dict
        self.beta_temperature = np.asarray(beta_temperature)
        self.min_n_effective = min_n_effective
        self.log2n_qmc = log2n_qmc
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

    def get_marginalization_info(self, d_h_timeseries, h_h, times):
        """
        Return a MarginalizationInfo object with extrinsic parameter
        integration results, ensuring that one of three conditions
        regarding the effective sample size holds:
            * n_effective >= .min_n_effective; or
            * n_qmc == 2 ** .max_log2n_qmc; or
            * n_effective = 0 (if the first proposal only gave
                               unphysical samples)
        """
        self._switch_qmc_sequence()

        # Resample to match sky_dict's dt:
        d_h_timeseries, times = self.sky_dict.resample_timeseries(
            d_h_timeseries, times, axis=-2)

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(d_h_timeseries,
                                                             h_h)  # dt
        self.sky_dict.apply_tdet_prior(t_arrival_lnprob)
        t_arrival_prob = utils.exp_normalize(t_arrival_lnprob, axis=1)

        i_chunk = 0
        marginalization_info = self._get_marginalization_info_chunk(
            d_h_timeseries, h_h, times, t_arrival_prob, i_chunk)

        while marginalization_info.n_effective < self.min_n_effective:
            # Perform adaptive mixture importance sampling:
            i_chunk += 1
            if i_chunk == len(self._qmc_ind_chunks):
                warnings.warn('Maximum QMC resolution reached.')
                break

            if marginalization_info.n_effective > 5:
                # Use a KDE of the weighted samples as next proposal:
                t_arrival_prob = self._kde_t_arrival_prob(
                    marginalization_info, times)
            else:
                # Increase temperature as next proposal:
                t_arrival_prob **= 1 / self._temperature_factor
                t_arrival_prob /= t_arrival_prob.sum(axis=1, keepdims=True)

            marginalization_info.update(self._get_marginalization_info_chunk(
                d_h_timeseries, h_h, times, t_arrival_prob, i_chunk))

        return marginalization_info

    def _kde_t_arrival_prob(self, marginalization_info, times):
        """
        Return array of shape (n_det, n_times) with a time-of-arrival
        proposal probability that is based on a kernel density
        estimation of the previous iterations of importance sampling.
        Intended for adaptive multiple importance sampling.
        """
        delays = self.sky_dict.delays[:, marginalization_info.sky_inds]
        t_det = np.vstack((marginalization_info.t_first_det,
                           marginalization_info.t_first_det + delays))  # dt

        scotts_factor = marginalization_info.n_effective ** -.2
        dt = times[1] - times[0]
        bins = np.concatenate([[times[0] - dt/2], times + dt/2])

        prob = []
        for t_samples in t_det:  # Loop over detectors
            hist = np.histogram(t_samples, weights=marginalization_info.weights,
                                bins=bins)[0]

            scale = scotts_factor * utils.weighted_avg_and_std(
                t_samples, marginalization_info.weights)[1]
            kernel = stats.cauchy.pdf(times, loc=times[len(times)//2 - 1],
                                      scale=scale)
            prob.append(signal.convolve(hist, kernel, mode='same'))
        prob = np.array(prob)
        prob /= prob.sum(axis=1, keepdims=True)

        return prob

    @abstractmethod
    def _get_marginalization_info_chunk(self, d_h_timeseries, h_h,
                                        times, t_arrival_prob, i_chunk):
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
        sequence_values = stats.qmc.scale(
            stats.qmc.Sobol(len(self._qmc_range_dic), seed=self._rng
                     ).random_base2(self.max_log2n_qmc) % 1,  # %1 sends 1→0
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

    def _get_tdet_inds(self, t_arrival_prob, q_inds):
        cumprob = np.cumsum(t_arrival_prob, axis=1)
        np.testing.assert_allclose(cumprob[:, -1], 1)
        return self._searchsorted(cumprob,
                                  self._qmc_sequence['u_tdet'][:, q_inds])

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
                                 timeseries[..., i_min : i_max],
                                 new_times, axis=-1)

    def get_optimal_beta_temperature(self, dh_timeseries, h_h, times,
                                     beta_rng=(.1, 1)):
        """
        Return array of shape (n_det,) with per-detector inverse-
        temperature values (used to temper the proposal distribution)
        tuned so as to maximize the efficiency of importance sampling.
        """
        beta_temperature = np.copy(np.broadcast_to(
            self.beta_temperature, len(self.sky_dict.detector_names)))

        betas = np.geomspace(*beta_rng, 100)

        for _ in range(2):  # Cycle twice over detectors for convergence
            for i_det in range(len(self.sky_dict.detector_names)):
                cost = []  # sample_size / effective_sample_size, to minimize
                for beta in betas:
                    beta_temperature[i_det] = beta
                    with utils.temporarily_change_attributes(
                            self,
                            beta_temperature=beta_temperature,
                            min_n_effective=0):
                        marg_info = self.get_marginalization_info(
                            dh_timeseries, h_h, times)
                        cost.append(marg_info.n_qmc / marg_info.n_effective)

                cost_smooth = signal.savgol_filter(
                    cost, len(cost) // 10, 1, mode='nearest')
                beta_temperature[i_det] = betas[np.argmin(cost_smooth)]

        return beta_temperature


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

    def _get_lnnumerators_important(self, dh_qo, hh_qo, sky_prior):
        """
        Parameters
        ----------
        dh_qo: (n_physical, n_phi) float array
            ⟨d|h⟩ real inner product between data and waveform at
            ``self.lookup_table.REFERENCE_DISTANCE``.

        hh_qo: (n_physical, n_phi) float array
            ⟨h|h⟩ real inner product of a waveform at
            ``self.lookup_table.REFERENCE_DISTANCE`` with itself.

        sky_prior: (n_physical,) float array
            Prior weights of the QMC sequence.

        Return
        ------
        ln_numerators: float array of length n_important
            Natural log of the weights of the QMC samples, including the
            likelihood and prior but excluding the importance sampling
            weights.

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

        ln_numerators = (
            self.lookup_table.lnlike_marginalized(dh_qo[important],
                                                  hh_qo[important])
            + np.log(sky_prior)[important[0]]
            - np.log(self._nphi))  # i

        return ln_numerators, important
