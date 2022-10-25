import itertools
import numba
import numpy as np
from scipy.stats.qmc import Sobol
import scipy.signal

from cogwheel import likelihood
from cogwheel import utils


class CoherentScoreHM(utils.JSONMixin):
    """
    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters.
    """
    marginalize_pars = sorted(['d_luminosity', 'lat', 'lon', 'phi_ref', 'psi',
                               't_fine'])

    # Remove from the orbital phase integral any sample with a drop in
    # log-likelihood from the peak bigger than ``DLNL_THRESHOLD``:
    DLNL_THRESHOLD = 12.

    def __init__(self, sky_dict, m_arr,
                 lookup_table=None,
                 log2n_qmc: int = 11, nphi=128, seed=0,
                 beta_temperature=.1):
        """
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

        self._u_tdet = None  # Set by _create_qmc_sequence
        self._t_fine = None  # Set by _create_qmc_sequence
        self._psi = None  # Set by _create_qmc_sequence
        self._rot_psi = None  # Set by _create_qmc_sequence
        self._create_qmc_sequence()

        self._sample_distance = utils.handle_scalars(
            np.vectorize(self.lookup_table.sample_distance, otypes=[float]))

    @property
    def nphi(self):
        return self._nphi

    @nphi.setter
    def nphi(self, nphi):
        phi_ref, dphi = np.linspace(0, 2*np.pi, nphi,
                                    endpoint=False, retstep=True)
        self._nphi = nphi
        self._dh_phasor = np.exp(-1j * np.outer(self.m_arr, phi_ref))  # mo
        self._hh_phasor = np.exp(
            1j * np.outer(self.m_arr[self.m_inds,]
                          - self.m_arr[self.mprime_inds,],
                          phi_ref))  # mo
        self._phi_ref = phi_ref
        self._dphi = dphi

    def _create_qmc_sequence(self):
        """
        Generate QMC sequence of (n_det+1, n_qmc) points.
        The sequence explores the cumulatives of the single-detector
        (incoherent) likelihood of arrival times, and polarization.
        Return the sequence of arrival time cumulatives (n_det, n_qmc)
        and the polarization rotation matrices (2, 2, n_qmc).
        """
        ndim = len(self.sky_dict.detector_names) + len(['psi', 't_fine'])
        sequence = Sobol(ndim, seed=self._rng).random_base2(self.log2n_qmc).T
        self._u_tdet = sequence[:-2]

        u_tfine = sequence[-2]
        self._t_fine = (u_tfine - .5) / self.sky_dict.f_sampling  # [s]

        self._psi = np.pi * sequence[-1]
        sintwopsi = np.sin(2 * self._psi)
        costwopsi = np.cos(2 * self._psi)
        self._rot_psi = np.moveaxis(np.array([[costwopsi, sintwopsi],
                                              [-sintwopsi, costwopsi]]),
                                    -1, 0)  # qpp'

    def marginalize(self, dh_mptd, hh_mppd, times,
                    return_samples=False):
        """
        Evaluate inner products (d|h) and (h|h) at QMC integration
        points over extrinsic parameters, given timeseries of (d|h) and
        value of (h|h) by mode `m`, polarization `p` and detector `d`.
        """
        # Resample to match sky_dict's dt:
        if (fs_ratio := self.sky_dict.f_sampling * (times[1]-times[0])) != 1:
            dh_mptd, times = scipy.signal.resample(dh_mptd,
                                                   int(len(times) * fs_ratio),
                                                   times,
                                                   axis=2)
            if not np.isclose(1/self.sky_dict.f_sampling, times[1] - times[0]):
                raise ValueError(
                    '`times` is incommensurate with `sky_dict.f_sampling`.')

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(dh_mptd,
                                                             hh_mppd)  # td

        tdet_inds, tdet_weights = _draw_indices(t_arrival_lnprob.T,
                                                self._u_tdet)  # dq, dq

        delays = tdet_inds[1:] - tdet_inds[0]  # dq  # In units of dt
        physical_mask = np.array([delays_key in self.sky_dict.delays2genind_map
                                  for delays_key in zip(*delays)])

        importance_sampling_weight = np.prod(tdet_weights[:, physical_mask]
                                             / self.sky_dict.f_sampling,
                                             axis=0)  # q

        if not any(physical_mask):
            if not return_samples:
                return - np.inf
            return - np.inf, dict.fromkeys(self.marginalize_pars,
                                           np.full(return_samples, np.nan))

        sky_inds, sky_prior = zip(
            *(next(self.sky_dict.delays2genind_map[delays_key])
              for delays_key in zip(*delays[:, physical_mask])))  # q, q


        fplus_fcross_0 = self.sky_dict.fplus_fcross_0[sky_inds,]  # qdp
        rot_psi = self._rot_psi[physical_mask]  # qpp'
        fplus_fcross = np.einsum('qpP,qdP->qdp', rot_psi, fplus_fcross_0)

        # # (d|h)
        # select = (...,  # mp stay the same
        #           tdet_inds.T[physical_mask],  # t -> q depending on d
        #           np.arange(len(self.sky_dict.detector_names))  # d
        #           )
        # dh_mpqd = dh_mptd[select]  # mpqd
        # dh_dmpq = np.moveaxis(dh_mpqd, -1, 0)

        # Alternative computation of dh_dmpq above, more accurate
        t_first_det = (times[tdet_inds[0, physical_mask]]
                       + self._t_fine[physical_mask])
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmpq = np.array(
            [scipy.interpolate.interp1d(times, dh_mptd[..., i_det], kind=3,
                                        copy=False, assume_sorted=True,
                                        fill_value=0., bounds_error=False
                                        )(t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        dh_qm = np.einsum('dmpq,qdp->qm', dh_dmpq, fplus_fcross)  # qm

        # (h|h)
        f_f = np.einsum('qdp,qdP->qpPd', fplus_fcross, fplus_fcross)
        hh_qm = (f_f.reshape(f_f.shape[0], -1)
                 @ hh_mppd.reshape(hh_mppd.shape[0], -1).T)  # qm

        dh_qo = (dh_qm @ self._dh_phasor).real  # qo
        hh_qo = (hh_qm @ self._hh_phasor).real  # qo

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
        sum_full_weights = full_weights.sum()

        lnl = lnl_max + np.log(sum_full_weights * self._dphi
                               / 2**self.log2n_qmc)

        self.status = locals()
        if not return_samples:
            return lnl

        # Generate requested number of extrinsic samples:

        if return_samples is True:  # Distinguish from int
            return_samples = None

        i_ids = np.random.choice(len(full_weights),
                                 p=full_weights/sum_full_weights,
                                 size=return_samples)

        q_ids = important[0][i_ids]
        o_ids = important[1][i_ids]
        sky_ids = np.array(sky_inds)[q_ids]

        d_h = dh_qo[q_ids, o_ids]
        h_h = hh_qo[q_ids, o_ids]

        samples = {
            'd_luminosity': self._sample_distance(d_h, h_h),
            'dec': self.sky_dict.sky_samples['lat'][sky_ids],
            'lon': self.sky_dict.sky_samples['lon'][sky_ids],
            'phi_ref': self._phi_ref[o_ids],
            'psi': self._psi[q_ids],
            't_geocenter': (t_first_det[q_ids]
                            - self.sky_dict.geocenter_delay_first_det[sky_ids]),
            'lnl_marginalized': lnl
            }

        return samples

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
