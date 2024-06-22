"""
Prior for masses and aligned spin components using PN-inspired
coordinates by [Lee, Morisaki & Tagoshi 2203.05216].
"""
import itertools
import scipy.optimize
import numpy as np
import pandas as pd

import lal

from cogwheel import gw_utils
from cogwheel.prior import Prior

# pylint: disable=arguments-differ


class PNCoordinatesPrior(Prior):
    """
    Implement the coordinates for intrinsic parameters of [Lee, Morisaki
    & Tagoshi 2203.05216]. These are quite similar except we normalize
    the eigenvectors of the Fisher matrix by the square root of their
    eigenvalue, so the Fisher errorbars in (mu1, mu2) are 1/snr.
    """
    standard_params = ['m1', 'm2', 's1z', 's2z']
    range_dic = {'mu1': None,
                 'mu2': None,
                 'lnq': None,
                 's2z': (-1, 1)}

    def __init__(self, eigvecs, f_ref, par_dic_0, mchirp_range,
                 dmu=2., q_min=.05, **kwargs):
        """
        Parameters
        ----------
        eigvecs: float array of shape (3, 2)
            Fisher matrix eigenvectors, see
            ``.eigvecs_from_reference_waveform_finder()``.

        f_ref: float
            Reference frequency (Hz).

        q_min: float
            Minimum mass ratio

        **kwargs: Passed to super().__init__()
        """
        eigvecs = np.asarray(eigvecs)
        if eigvecs.shape != (3, 2):
            raise ValueError('Expecting 2 column-vectors of size 3.')

        self.eigvecs = eigvecs
        self.f_ref = f_ref
        self.mchirp_range = mchirp_range
        self.dmu = dmu
        self.par_dic_0 = par_dic_0

        lnq_range = np.log(q_min), 0

        sampled_dic_0 = self.inverse_transform(
            **{par: par_dic_0[par] for par in self.standard_params})
        mu1_range = np.add(sampled_dic_0['mu1'], (-dmu, dmu))
        mu2_range = np.add(sampled_dic_0['mu2'], (-dmu, dmu))

        # Attempt to restrict parameter space further than +/- dmu since
        # some (mu1, mu2) are unphysical.
        # Make a box in mu1, mu2 space that encloses the desired mchirp range
        corners = pd.DataFrame(
            itertools.product(mchirp_range, lnq_range, self.range_dic['s2z'],
                              self.range_dic['s2z']),
            columns=['mchirp', 'lnq', 's1z', 's2z'])
        corners['eta'] = gw_utils.q_to_eta(np.exp(corners['lnq']))
        corners['m1'], corners['m2'] = gw_utils.mchirpeta_to_m1m2(
            **corners[['mchirp', 'eta']])
        self.inverse_transform_samples(corners)
        mu1_range = np.clip(
            mu1_range, corners['mu1'].min(), corners['mu1'].max())
        mu2_range = np.clip(
            mu2_range, corners['mu2'].min(), corners['mu2'].max())

        self.range_dic = self.range_dic | {'mu1': mu1_range,
                                           'mu2': mu2_range,
                                           'lnq': lnq_range}

        super().__init__(f_ref=f_ref, mchirp_range=mchirp_range, q_min=q_min,
                         par_dic_0=par_dic_0, **kwargs)

    @staticmethod
    def eigvecs_from_reference_waveform_finder(
            reference_waveform_finder):
        """
        Return a float array of shape (3, 2) with the two main
        eigenvectors of the Fisher matrix in the space of the first 3
        coefficients of the post-Newtonian expansion.
        These are the first 2 columns of ``U.T`` in the notation of
        [2203.05216], except that we normalize each eigenvector to have
        norm ``sqrt(eigenvalue)``.
        This can be used as input to ``.__init__()``.
        """
        f_ref = reference_waveform_finder.par_dic_0['f_ref']
        fmin = reference_waveform_finder.event_data.fbounds[0]
        fmax = max(f_ref, gw_utils.isco_frequency(
            reference_waveform_finder.par_dic_0['m1']
            + reference_waveform_finder.par_dic_0['m2']))
        fslice = slice(*np.searchsorted(
            reference_waveform_finder.event_data.frequencies, (fmin, fmax)))
        frequencies = reference_waveform_finder.event_data.frequencies[fslice]

        h_f = reference_waveform_finder.waveform_generator \
            .get_strain_at_detectors(frequencies,
                                     reference_waveform_finder.par_dic_0)
        whitened_amplitude = np.linalg.norm(
            h_f * reference_waveform_finder.event_data.wht_filter[:, fslice],
            axis=0)  # Quadrature sum over detectors
        whitened_amplitude /= np.linalg.norm(whitened_amplitude)
        pn_exponents = 0, 1, -5/3, -3/3, -2/3  # Phase, time, 1PN, 2PN, 2.5PN
        pn_functions = np.power.outer(frequencies / f_ref, pn_exponents)
        weighted_functions = pn_functions * whitened_amplitude[:, np.newaxis]
        full_fisher_mat = weighted_functions.T @ weighted_functions

        marginalize_inds = {0, 1}  # Phase, time
        keep_inds = sorted(set(range(len(pn_exponents))) - marginalize_inds)
        keep_mat = np.eye(len(full_fisher_mat))[:, keep_inds]
        fisher_mat = np.linalg.inv(keep_mat.T
                                   @ np.linalg.inv(full_fisher_mat)
                                   @ keep_mat)
        eigvals, eigvecs = np.linalg.eig(fisher_mat)
        inds = np.argsort(-eigvals)[:2]  # Keep 2 main eigenvectors
        return eigvecs[:, inds] * np.sqrt(eigvals[inds])

    def _mchirp_lnq_s1z_s2z_lnprior(self, mchirp, lnq, s1z, s2z):
        """
        Return ``ln(prior(mchirp, lnq, s1z, s2z))`` corresponding to
        uniform density in detector-frame component masses and the
        aligned component of a "volumetric" spin prior, i.e. where the
        spin components are independent, isotropic and p(|s|) ~ |s|^2,
        with s being the dimensionless spin.
        """
        return np.log(mchirp
                      * np.cosh(lnq/2)**.4
                      * .75 * (1 - s1z**2)
                      * .75 * (1 - s2z**2))

    def inverse_transform(self, m1, m2, s1z, s2z):
        """Standard parameters to sampled parameters."""
        eta, beta, v_ref = self._eta_beta_vref(m1, m2, s1z, s2z)
        mu1, mu2 = self.eigvecs.T @ (self._pn1(v_ref, eta),
                                     self._pn2(v_ref, eta),
                                     self._pn2_5(v_ref, eta, beta))
        return {'mu1': mu1,
                'mu2': mu2,
                'lnq': np.log(m2/m1),
                's2z': s2z}

    def transform(self, mu1, mu2, lnq, s2z):
        """Sampled parameters to standard parameters."""
        q = np.exp(lnq)
        eta = gw_utils.q_to_eta(q)
        delta = (1-q) / (1+q)

        # Must find mass and s1z from mu1, mu2; encoded in v_ref, beta.
        # Solve for v_ref, eliminate the 2.5 PN term that contains beta:
        coeffs = np.r_[self.eigvecs[(0, 1),], ((-mu1, -mu2),)
                      ] @ (self.eigvecs[2, 1], -self.eigvecs[2, 0])

        def objective(v):
            """Function whose root is `v_ref`."""
            # v**3 smoothens the function without changing the root.
            return coeffs @ (self._pn1(v, eta), self._pn2(v, eta), 1) * v**3

        try:
            v_ref = scipy.optimize.brentq(objective, 1e-3, 1)
        except ValueError:  # Unphysical (mu1, mu2) given lnq
            return dict.fromkeys(self.standard_params, np.nan)

        mtot = v_ref**3 / (np.pi * lal.MTSUN_SI * self.f_ref)
        m1 = mtot / (1+q)

        # Now solve for beta:
        pn_2_5 = (mu1 - self.eigvecs[(0, 1), 0] @ (self._pn1(v_ref, eta),
                                                   self._pn2(v_ref, eta))
                  ) / self.eigvecs[2, 0]

        beta = 32/3 * eta * v_ref**2 * pn_2_5 + 4*np.pi
        s1z = ((24/113*beta - (1 - delta - 76/113*eta)*s2z)
               / (1 + delta - 76/113*eta))

        if np.abs(s1z) > 1:  # Unphysical (mu1, mu2) given (lnq, s2z)
            return dict.fromkeys(self.standard_params, np.nan)

        return {'m1': m1,
                'm2': q * m1,
                's1z': s1z,
                's2z': s2z}

    def lnprior(self, mu1, mu2, lnq, s2z):
        """
        Natural logarithm of the prior probability for (mu1, mu2, lnq,
        s2z) under a prior flat in detector-frame masses and volumetric
        in component spins.
        """
        standard_par_dic = self.transform(mu1, mu2, lnq, s2z)

        if any(np.isnan(value) for value in standard_par_dic.values()):
            return -np.inf  # Unphysical sampled-parameter values

        mchirp = gw_utils.m1m2_to_mchirp(standard_par_dic['m1'],
                                         standard_par_dic['m2'])
        if mchirp < self.mchirp_range[0] or mchirp > self.mchirp_range[1]:
            return -np.inf

        eta, beta0, v_ref = self._eta_beta_vref(
            **standard_par_dic | {'s2z': 0})

        dbeta_ds1z = beta0 / standard_par_dic['s1z']  # Note s2z=0
        dpn1_dmchirp = -5/3 * self._pn1(v_ref, eta) / mchirp
        dpn2_dmchirp = - self._pn2(v_ref, eta) / mchirp
        dpn3_ds1z = 3/32 * v_ref**-2 / eta * dbeta_ds1z

        jacobian_determinant = np.abs(
            (np.linalg.det(self.eigvecs[(0, 2),]) * dpn1_dmchirp
             + np.linalg.det(self.eigvecs[(1, 2),]) * dpn2_dmchirp)
            * dpn3_ds1z)  # |d(mu1, mu2) / d(mchirp, s1z)|

        return (self._mchirp_lnq_s1z_s2z_lnprior(mchirp, lnq,
                                                 standard_par_dic['s1z'],
                                                 standard_par_dic['s2z'])
                - np.log(jacobian_determinant))

    def _eta_beta_vref(self, m1, m2, s1z, s2z):
        """Return auxiliary PN quantities eta, beta and v(f_ref)."""
        mtot = m1 + m2
        eta = m1 * m2 / mtot**2
        chis = (s1z + s2z) / 2
        chia = (s1z - s2z) / 2
        delta = (m1 - m2) / mtot
        beta = 113/12 * (chis + delta*chia - 76/113*eta*chis)
        v_ref = np.cbrt(np.pi * lal.MTSUN_SI * mtot * self.f_ref)
        return eta, beta, v_ref

    @staticmethod
    def _pn1(v, eta):
        """1PN term of the phase"""
        return 3/128 / eta * v**-5

    @staticmethod
    def _pn2(v, eta):
        """2PN term of the phase."""
        return 3/128 * (55/9 + 3715/756/eta) * v**-3

    @staticmethod
    def _pn2_5(v, eta, beta):
        """2.5 PN term of the phase."""
        return 3/128 * (4*beta - 16*np.pi) / eta * v**-2

    def get_init_dict(self):
        """Return keyword arguments to reproduce the class instance."""
        return {'eigvecs': self.eigvecs,
                'f_ref': self.f_ref,
                'par_dic_0': self.par_dic_0,
                'mchirp_range': self.mchirp_range,
                'dmu': self.dmu,
                'q_min': np.exp(self.range_dic['lnq'][0])}
