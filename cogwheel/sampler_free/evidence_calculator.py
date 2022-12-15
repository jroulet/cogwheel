import time
import itertools
import numpy as np
from numba import njit

from cogwheel import likelihood


@njit()
def sum_logs(x):
    x = np.sort(x)
    y = x[0]
    for i in range(1, len(x)):  # replace with "if" for a<b and b<a cases
        a = np.minimum(y, x[i])
        b = np.maximum(y, x[i])
        y = b + np.log(1 + np.exp(a - b))
    return y


class Evidence:
    """
    A class that receives as input the components of intrinsic and
    extrinsic factorizations and calculates the likelihood at each
    (int., ext., phi) combination (distance marginalized).
    Indexing and size convention:
        * i: intrinsic parameter
        * m: modes, or combinations of modes
        * p: polarization, plus (0) or cross (1)
        * b: frequency bin (as in relative binning)
        * e: extrinsic parameters
        * d: detector
    """
    def __init__(self, n_phi, m_arr, lookup_table=None):
        """
        n_phi : number of points to evaluate phi on,
        m_arr : modes
        lookup_table = lookup table for distance marginalization
        """
        self.n_phi = n_phi
        self.m_arr = m_arr
        self.phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.m_arr = np.asarray(m_arr)
        # Used for terms with 2 modes indices
        self.m_inds, self.mprime_inds = zip(
            *itertools.combinations_with_replacement(range(len(self.m_arr)), 2))

        self.lookup_table = lookup_table or likelihood.LookupTable()

    @staticmethod
    def get_dh_by_mode(dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe):
        # TODO: figure ahead of time what is the optimal method, and pass it as
        # an argument, instead of using optimize=True
        dh_iem = np.einsum('dmpb, impb, dpe, dbe -> iem',
                           dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe,
                           optimize=True)
        return dh_iem

    @staticmethod
    def get_hh_by_mode(h_impb, response_dpe, hh_weights_dmppb):
        # modes here means all unique modes combinations, ((2,2), (2,0), ..., (3,3))
        hh = np.einsum('impb, impb -> imp', h_impb, h_impb.conj())  # imp
        ff = np.einsum('dpe, dPe -> dpPe', response_dpe, response_dpe)  # dppe
        hh_iem = np.einsum('dmpPb, imp, dpPe -> iem', hh_weights_dmppb, hh, ff)  # iem
        return hh_iem

    def get_dh_hh_phi_grid(self, dh_iem, hh_iem):
        # change the orbital phase of each mode by exp(1j*phi*m), and keep
        # the real part assumes that factor of two from off-diagonal modes
        # (e.g. (2,0) and not e.g. (2,2)) is already in the hh_weights_dmppb

        phi_grid = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)  # o
        dh_phasor = np.exp(1j * np.outer(self.m_arr, phi_grid))  # mo
        hh_phasor = np.exp(1j * np.outer(
            self.m_arr[self.m_inds,] - self.m_arr[self.mprime_inds,],
            phi_grid))  # mo
        dh_ieo = (dh_iem[..., np.newaxis] * dh_phasor).real.sum(axis=2)  # ieo
        hh_ieo = (hh_iem[..., np.newaxis] * hh_phasor).real.sum(axis=2)  # ieo

        return dh_ieo, hh_ieo

    @staticmethod
    def select_ieo_by_approx_lnlike_dist_marginalized(
            dh_ieo, hh_ieo, weights_i, weights_e, cut_threshold=20.):
        """
        Return three arrays with intrinsic, extrinsic and phi sample indices.
        """
        h_norm = np.sqrt(hh_ieo)
        z = dh_ieo / h_norm
        i_inds, e_inds, o_inds = np.where(z > 0)
        lnl_approx = (z[i_inds, e_inds, o_inds] ** 2 / 2
                      + 3 * np.log(h_norm[i_inds, e_inds, o_inds])
                      -4 * np.log(z[i_inds, e_inds, o_inds])
                      + np.log(weights_i[i_inds]) + np.log(weights_e[e_inds]))

        flattened_inds = np.where(
            lnl_approx >= lnl_approx.max() - cut_threshold)[0]

        return (i_inds[flattened_inds],
                e_inds[flattened_inds],
                o_inds[flattened_inds])

    @staticmethod
    def evaluate_evidence(lnlike, weights, n_samples):
        """
        Evaluate the logarithm of the evidence (ratio) integral
        """
        return sum_logs(lnlike + np.log(weights)) - np.log(n_samples)

    def calculate_lnlike_and_evidence(
            self, dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe,
            hh_weights_dmppb, weights_i, weights_e, cut_threshold=20.):
        """
        Use stored samples to compute lnl (distance marginalized) on
        grid of intrinsic x extrinsic x phi
        phi will later be marginalized
        :param print_runtime:
        :return:
        """

        event_log = []
        t_list = []
        t_list.append(time.time())

        dh_iem = self.get_dh_by_mode(dh_weights_dmpb, h_impb, response_dpe,
                                     timeshift_dbe)
        t_list.append(time.time())
        event_log.append(
            f'calculated dh_iem in {t_list[-1] - t_list[-2]:.2E} seconds')

        hh_iem = self.get_hh_by_mode(h_impb, response_dpe, hh_weights_dmppb)
        t_list.append(time.time())
        event_log.append(
            f'calculated hh_iem in {t_list[-1] - t_list[-2]:.2E} seconds')

        dh_ieo, hh_ieo = self.get_dh_hh_phi_grid(dh_iem, hh_iem)
        n_samples = dh_ieo.size
        t_list.append(time.time())
        event_log.append(
            f'applied phase in {t_list[-1] - t_list[-2]:.2E} seconds')

        inds_i, inds_e, inds_o \
            = self.select_ieo_by_approx_lnlike_dist_marginalized(
                dh_ieo, hh_ieo, weights_i, weights_e, cut_threshold)

        t_list.append(time.time())
        event_log.append('pre-selected high lnlike combinations in '
                         f'{t_list[-1] - t_list[-2]:.2E} seconds')
        event_log.append(f'{len(inds_i)} selected out of {dh_ieo.size} '
                         f'(frac. of {len(inds_i) / n_samples.size:.2E})')

        lnlike_distance_marginalized = (
            self.lookup_table(dh_ieo[inds_i, inds_e, inds_o],
                              hh_ieo[inds_i, inds_e, inds_o])
            + dh_ieo[inds_i, inds_e, inds_o] ** 2
            / hh_ieo[inds_i, inds_e, inds_o] / 2)

        t_list.append(time.time())
        event_log.append(
            f'lnlike evaluated in {t_list[-1] - t_list[-2]:.2E} seconds')

        ln_evidence = self.evaluate_evidence(
            lnlike_distance_marginalized,
            weights_i[inds_i] * weights_e[inds_e],
            n_samples)

        t_list.append(time.time())
        event_log.append('evidence integral evaluated in '
                         f'{t_list[-1] - t_list[-2]:.2E} seconds')

        return lnlike_distance_marginalized, ln_evidence, inds_i, inds_e, inds_o
