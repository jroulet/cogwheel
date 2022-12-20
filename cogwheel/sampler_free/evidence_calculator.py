import time
import itertools
import numpy as np
from scipy.special import logsumexp
from cogwheel import likelihood
from cogwheel.gw_utils import get_fplus_fcross_0, get_geocenter_delays
from cogwheel.waveform import FORCE_NNLO_ANGLES, compute_hplus_hcross
from cogwheel.gw_prior import linear_free
from lal import CreateDict
import lalsimulation as lalsim

lalsimulation_commands = FORCE_NNLO_ANGLES


def create_lal_dict():
    """Return a LAL dict object per ``self.lalsimulation_commands``."""
    lal_dic = CreateDict()
    for function_name, value in lalsimulation_commands:
        getattr(lalsim, function_name)(lal_dic, value)
    return lal_dic

def get_shift(f, t):
    """
    unhsifted h is centered at event_data.tcoarse
    further shifts are due to relative times (t_geonceter + time_delays) = t_refdet
    """
    return np.exp(-2j * np.pi * f * t)

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
    def get_dh_by_mode(dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d):
        # TODO: figure ahead of time what is the optimal method, and pass it as
        # an argument, instead of using optimize=True
        dh_iem = np.einsum('dmpb, impb, dpe, dbe, d -> iem',
                           dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d,
                           optimize=True)
        return dh_iem

    @staticmethod
    def get_hh_by_mode(h_impb, response_dpe, hh_weights_dmppb, asd_drift_d, m_inds, mprime_inds):
        # modes here means all unique modes combinations, ((2,2), (2,0), ..., (3,3))
        hh = np.einsum('impb, impb -> imp', h_impb[:, m_inds, ...], h_impb.conj()[:, mprime_inds, ...],
                       optimize=True)  # imp
        ff = np.einsum('dpe, dPe, d -> dpPe', response_dpe, response_dpe, asd_drift_d,
                       optimize=True)  # dppe
        hh_iem = np.einsum('dmpPb, imp, dpPe -> iem', hh_weights_dmppb, hh, ff, optimize=True)  # iem
        return hh_iem

    def get_dh_hh_phi_grid(self, dh_iem, hh_iem):
        # change the orbital phase of each mode by exp(1j*phi*m), and keep
        # the real part assumes that factor of two from off-diagonal modes
        # (e.g. (2,0) and not e.g. (2,2)) is already in the hh_weights_dmppb

        phi_grid = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)  # o
        dh_phasor = np.exp(1j * np.outer(self.m_arr, phi_grid))  # mo
        hh_phasor = np.exp(1j * np.outer(
            self.m_arr[self.m_inds, ] - self.m_arr[self.mprime_inds, ],
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
                      - 4 * np.log(z[i_inds, e_inds, o_inds])
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
        return logsumexp(lnlike + np.log(weights)) - np.log(n_samples)

    def calculate_lnlike_and_evidence(
            self, dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe,
            hh_weights_dmppb, asd_drift_d, weights_i, weights_e, cut_threshold=20.):
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
                                     timeshift_dbe, asd_drift_d)
        t_list.append(time.time())
        event_log.append(
            f'calculated dh_iem in {t_list[-1] - t_list[-2]:.2E} seconds')

        hh_iem = self.get_hh_by_mode(h_impb, response_dpe, hh_weights_dmppb,
                                     asd_drift_d, self.m_inds, self.mprime_inds)
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
                         f'(frac. of {len(inds_i) / n_samples:.2E})')

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

class SampleProcessing:
    """
    A class to process intrinsic and extirnsic samples and creates the relevant high-dimensional arrays to pass
    to an Evidence class.
    """

    def __init__(self, intrinsic_samples, extrinsic_samples, likelihood):

        self.n_polarizations = 2
        self.likelihood = likelihood
        self.intrinsic_samples = intrinsic_samples
        self.extrinsic_samples = extrinsic_samples
        self.linear_free_prior = linear_free.LinearFreePhaseTimePrior(approximant='IMRPhenomXPHM',
                                                                      par_dic_0=likelihood.par_dic_0,
                                                                      event_data=likelihood.event_data)
        self.m_arr = np.fromiter(likelihood.waveform_generator._harmonic_modes_by_m, int)
        self.m_inds, self.mprime_inds = zip(
            *itertools.combinations_with_replacement(range(len(self.m_arr)), 2))

    @property
    def n_intrinsic(self):
        shape = getattr(getattr(self, 'intrinsic_samples', None), 'shape', None)
        return shape[0] if shape else 0

    @property
    def n_extrinsic(self):
        shape = getattr(getattr(self, 'extrinsic_samples', None), 'shape', None)
        return shape[0] if shape else 0

    @property
    def n_fbin(self):
        return len(self.likelihood.fbin)

    @property
    def n_det(self):
        return len(self.likelihood.event_data.detector_names)

    @property
    def n_modes(self):
        return len(self.likelihood.waveform_generator._harmonic_modes_by_m.values())

    @property
    def detector_names(self):
        return self.likelihood.event_data.detector_names

    @property
    def tgps(self):
        return self.likelihood.event_data.tgps

    @staticmethod
    def compute_detector_responses(detector_names, lat, lon, psi):
        """ Compute detector response at specific lat, lon and psi """
        fplus_fcross_0 = get_fplus_fcross_0(detector_names, lat, lon) # edP
        psi_rot = np.array([[np.cos(2*psi), np.sin(2*psi)],[-np.sin(2*psi), np.cos(2*psi)]]) # pPe
        return np.einsum('edP, pPe-> edp', fplus_fcross_0, psi_rot, optimize=True)  # edp

    def get_hplus_hcross_0(self, par_dic, f=None, force_fslice=False, fslice=None):
        """
        create (n modes x 2 polarizations x n frequencies) array
        using d_luminosity = 1Mpc and phi_ref = 0
        shifted to center for lk.event_data.times, without t_refdet shifts or linear-free time shifts
        """
        if f is None:
            f = self.likelihood.fbin
        if force_fslice and (fslice is None):
            fslice = self.likelihood.event_data.fslice
        elif not force_fslice:
            fslice = slice(0, len(f), 1)

        slow_par_vals = np.array([par_dic[par]
                                  for par in self.likelihood.waveform_generator.slow_params])
        # Compute the waveform mode by mode
        lal_dic = create_lal_dict()
        # force d_luminosity=1, phi_ref=0
        waveform_par_dic_0 = dict(zip(self.likelihood.waveform_generator.slow_params, slow_par_vals),
                                  d_luminosity=1., phi_ref=0.)

        # hplus_hcross_0 is a (n_m x 2 x n_frequencies) array with
        # sum_l (hlm+, hlmx), at phi_ref=0, d_luminosity=1Mpc.

        n_freq = len(f)
        shape = (self.n_modes, self.n_polarizations, n_freq)
        hplus_hcross_0_mpf = np.zeros(shape, np.complex_)
        hplus_hcross_0_mpf[..., fslice] = np.array(
            [compute_hplus_hcross(f[fslice], waveform_par_dic_0, self.likelihood.waveform_generator.approximant,
                                  modes, lal_dic)
             for modes in self.likelihood.waveform_generator._harmonic_modes_by_m.values()])
        # tcoarse_time_shift
        shift = get_shift(f, self.likelihood.event_data.tcoarse)
        return hplus_hcross_0_mpf * shift

    def get_intrinsic_linfree_time_shift_exp(self, par_dic):
        """
        return the timeshift to convert from linear free to reference_detector times
        since t_refdet  = t_linear_free -  linear_free_time_shift,
        the return exponent argument  "+2j*pi*f* delta_t" instead of "-2j*pi..."

        :param par_dic: parameter dictionary
        :return: linfree_timeshift_exp
        """
        f = self.likelihood.fbin
        linfree_phase_shifts, linfree_time_shifts = \
            self.linear_free_prior._get_linfree_phase_time_shift(
                **{k: par_dic.get(k, self.likelihood.par_dic_0[k])
                   for k in self.linear_free_prior._intrinsic})
        linfree_timeshift_exp = np.exp(+2j * np.pi *
                                       (linfree_time_shifts - self.linear_free_prior._ref['t_linfree']) * f)
        return linfree_timeshift_exp

    def compute_intrinsic_array(self, intrinsic_samples):
        """
        compute the intrinsic ndim array needed for computing inner products
        waveform are timeshifted to according to the linear free convention.

        :params intrinsc_samples: pandas DataFrame with intrinsic samples in rows

        """

        h_impb = np.zeros((self.n_intrinsic, self.n_modes, self.n_polarizations, self.n_fbin), np.complex_)
        timeshift_intrinsic_ib = np.zeros((self.n_intrinsic, self.n_fbin), np.complex_)
        for i, sampler in intrinsic_samples.iterrows():
            par_dic = {k: sampler.get(k, self.likelihood.par_dic_0[k])
                       for k in self.likelihood.waveform_generator.slow_params}
            h_impb[i] = self.get_hplus_hcross_0(par_dic)
            timeshift_intrinsic_ib[i] = self.get_intrinsic_linfree_time_shift_exp(par_dic)

        return h_impb, timeshift_intrinsic_ib

    def compute_extrinsic_timeshift(self, extrinsic_samples):

        geocentric_delays = get_geocenter_delays(self.detector_names, extrinsic_samples['lat'].values,
                                                 extrinsic_samples['lon'].values).T  # ed

        total_delays = (geocentric_delays + extrinsic_samples['t_geocenter'].values[:, np.newaxis])  # ed

        timeshift_extrinsic = np.exp(-2j * np.pi * total_delays[..., np.newaxis] *
                                     self.likelihood.fbin[np.newaxis, np.newaxis, :])  # edb

        return timeshift_extrinsic

    def get_summary(self):
        """
        get dh_weights_dmpb and hh_weights_dmppb
        """
        # impose zero orbital phase and distance 1Mpc
        par_dic = self.likelihood.par_dic_0 | self.likelihood._FIDUCIAL_CONFIGURATION

        h0_mpf = self.get_hplus_hcross_0(par_dic, f=self.likelihood.event_data.frequencies, force_fslice=True)
        h0_mpb = self.get_hplus_hcross_0(par_dic, f=self.likelihood.fbin)

        d_h0_dmpf = (self.likelihood.event_data.blued_strain[:, np.newaxis, np.newaxis, :]
                     * h0_mpf.conj())

        dh_weights_dmpb = self.likelihood._get_summary_weights(d_h0_dmpf) / h0_mpb.conj()

        whitened_h0_dmpf = (h0_mpf[np.newaxis, ...]
                            * self.likelihood.event_data.wht_filter[:, np.newaxis, np.newaxis, :])
        h0_h0_dmppf = np.einsum('dmpf, dmPf-> dmpPf', whitened_h0_dmpf[:, self.m_inds, ...],
                                whitened_h0_dmpf.conj()[:, self.mprime_inds, ...], optimize=True)

        hh_weights_dmppb = self.likelihood._get_summary_weights(h0_h0_dmppf)

        hh_weights_dmppb = np.einsum('mpb, mPb, dmpPb -> dmpPb',
                                     h0_mpb[self.m_inds, ...] ** (-1),
                                     h0_mpb[self.mprime_inds, ...].conj() ** (-1),
                                     hh_weights_dmppb, optimize=True)

        hh_weights_dmppb[:, ~np.equal(self.m_inds, self.mprime_inds)] *= 2
        return dh_weights_dmpb, hh_weights_dmppb

