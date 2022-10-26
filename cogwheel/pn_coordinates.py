"""
Module to generate intrinsic parameter samples with Quasi Monte Carlo
from an approximation of the posterior.
This uses an approximation of the likelihood in terms of (mchirp, lnq,
chieff) that accounts for the post-Newtonian inspiral and the cutoff
frequency.

Classes ``PNCoordinates``, ``PNMap`` and ``IntrinsicMap`` are defined.
``IntrinsicMap`` is the top-level class that most users would use.

Example usage:
```
# ``post`` is an instance of ``posterior.Posterior``
pn_map = PNMap.from_posterior(post)
imap = IntrinsicMap(pn_map)
qmc_samples = imap.generate_intrinsic_samples(14)
```
"""
import scipy.interpolate
import scipy.linalg
import scipy.stats
import scipy.special
import scipy.signal

import numpy as np
import pandas as pd

import lal

from cogwheel import gw_utils
from cogwheel import utils


def unique_qr(mat):
    """
    QR decomposition ensuring R has positive diagonal elements.
    """
    qmat, rmat = scipy.linalg.qr(mat, mode='economic')
    signs = np.diagflat(np.sign(np.diag(rmat)))
    return qmat @ signs, signs @ rmat


class UnphysicalMtotFmerger(ValueError):
    """
    Raise when no solution with -1 < chieff < 1 satisfies
        f_merger = get_f_merger(mtot, chieff).
    """


def get_f_merger(mtot, chieff):
    """
    Estimate the merger frequency (Hz) [Ajith+ arxiv.org/abs/0909.2867].
    """
    return (1 - .63 * (1-chieff)**.3) / (2 * np.pi * mtot * lal.MTSUN_SI)


def chieff_from_mtot_fmerger(mtot, f_merger):
    """
    Return chieff that satisfies get_f_merger(mtot, chieff) = f_merger.
    """
    if (np.any(f_merger < get_f_merger(mtot, -1))
            or np.any(f_merger > get_f_merger(mtot, 1))):
        raise UnphysicalMtotFmerger(
            'No solution for chieff given mtot, f_merger.')

    return 1 - ((1 - 2 * np.pi * f_merger * mtot * lal.MTSUN_SI) / .63)**(1/.3)


class TruncCauchy(scipy.stats.rv_continuous):
    """Truncated Cauchy (Lorentz) distribution."""
    def _cdf_cauchy(self, x):
        return np.arctan(x) / np.pi + .5

    def _cdf(self, x, a, b):
        u_a, u_b, u_x = self._cdf_cauchy((a, b, x))
        return (u_x - u_a) / (u_b - u_a)

    def _ppf(self, q, a, b):
        u_a, u_b = self._cdf_cauchy((a, b))
        u_q = u_a + (u_b - u_a) * q
        return np.tan(np.pi * (u_q - .5))

    def _pdf(self, x, a, b):
        u_a, u_b = self._cdf_cauchy((a, b))
        return 1 / (np.pi * (1 + x**2) * (u_b - u_a))

    def _argcheck(self, a, b):
        return a < b


trunccauchy = TruncCauchy()


def inverse_cdf_and_jacobian(cdf_val, points, pdf_points):
    """
    Find the value of x such that
        int_{-inf}^x P(x') dx' = cdf
    where P(x) is a linear spline interpolating {points: pdf_points},
    then normalized to integrate to 1.
    Also return the jacobian dc/dx.
    Note: we interpolate P(x) with a linear spline because that's the
    highest order that preserves P >= 0.
    """
    if np.any(points[:-1] >= points[1:]):
        raise ValueError('`points` should be sorted.')

    if np.any(pdf_points < 0):
        raise ValueError('`pdf_points` should be >= 0.')

    cdf_points = scipy.integrate.cumulative_trapezoid(pdf_points, points,
                                                      initial=0)
    pdf_points /= cdf_points[-1]
    cdf_points /= cdf_points[-1]
    x_guess = np.interp(cdf_val, cdf_points, points)

    pdf = scipy.interpolate.InterpolatedUnivariateSpline(points, pdf_points,
                                                         k=1, ext='raise')
    cdf = pdf.antiderivative()
    x_val = scipy.optimize.newton(lambda x: cdf(x) - cdf_val, x_guess,
                                  fprime=pdf,
                                  # fprime2=pdf.derivative()
                                  )

    dc_dx = pdf(x_val)[()]
    return x_val, dc_dx


class PNCoordinates:
    """
    Provide an analytic transform from (mchirp, eta, chieff) to a system
    of coordinates where the Euclidean distance is the mismatch distance
    for PN waveforms.
    In other words, the log likelihood (inspiral only) is
        lnl = - (SNR * |c - c_peak|)^2 / 2
    near the peak.

    Note: the third PN term is approximate since in reality it depends
    on beta, not eta and chieff.
    """
    # phase, time, 2.5PN, 2PN, 1PN. Ordered this way we ensure that
    # intrinsic parameters are orthogonalized to phase and time and
    # chi_eff only changes one of the coordinates, moreover, linearly.
    _PN_EXPONENTS = 0, 1, -2/3, -1, -5/3

    def __init__(self, frequencies, whitened_amplitude, smooth=True):
        """
        Parameters
        ----------
        frequencies: 1d float array or None
            Frequencies in Hz. If `None`, `whitened_amplitude` also must
            be `None`.

        whitened_amplitude: array like frequencies or None
            |h| / sqrt(PSD), with arbitrary normalization, evaluated at
            `frequencies`. If there are multiple detectors, pass the
            root mean square over detectors.

        smooth: bool
            If True, will smooth the whitened amplitude as a function of
            frequency so it can be interpolated meaningfully (useful for
            estimating the amplitude at merger frequency).
        """
        delta_f = frequencies[1] - frequencies[0]
        if not np.allclose(delta_f, np.diff(frequencies)):
            raise ValueError('`frequencies` must be regularly spaced.')

        self.frequencies = frequencies

        if smooth:
            whitened_amplitude = scipy.signal.savgol_filter(
                whitened_amplitude, len(whitened_amplitude) // 100, 1)

        whitened_amplitude /= np.sqrt((whitened_amplitude**2).sum() * delta_f)

        self.interp_wht_amplitude = scipy.interpolate.interp1d(
            self.frequencies, whitened_amplitude,
            assume_sorted=True, bounds_error=False, fill_value=0)

        pn_funcs = 3 / 128 * np.power.outer(lal.MTSUN_SI * np.pi * frequencies,
                                            self._PN_EXPONENTS)
        weights = whitened_amplitude / np.linalg.norm(whitened_amplitude)
        _, fullrmat = unique_qr(pn_funcs * weights[:, np.newaxis])

        # Ignore phase & time:
        self._rmat = fullrmat[2:, 2:]  # Transformation matrix
        self._inv_rmat = np.linalg.inv(self._rmat)


        # x := 1PN^(3/2) / 2PN^(5/2)
        eta_pts = np.linspace(0, .25, 512)
        x_pts = eta_pts * (3715/756 + 55/9*eta_pts)**-2.5
        self._eta_of_x = utils.handle_scalars(
            scipy.interpolate.interp1d(x_pts, eta_pts))

    @classmethod
    def from_likelihood(cls, likelihood):
        """
        Instantiate from a RelativeBinningLikelihood object, used to
        estimate the whitened_amplitude from the whitening filter and reference
        waveform.
        """
        fslice = likelihood.event_data.fslice
        frequencies = likelihood.event_data.frequencies[fslice]

        h_f = likelihood._get_h_f_interpolated(likelihood.par_dic_0)
        whitened_amplitude = np.linalg.norm(
            h_f * likelihood.event_data.wht_filter, axis=0)[fslice]
        return cls(frequencies, whitened_amplitude)

    def transform(self, mchirp, eta, chieff):
        """
        Return array of coordinates whose Euclidean distance is the
        mismatch distance for PN waveforms.
        """
        pn_coeffs = self._get_pn_coeffs(mchirp, eta, chieff)
        # phase = pn_funcs @ pn_coeffs
        #       = qmat @ rmat @ pn_coeffs
        #       = qmat @ coords
        return np.einsum('ij,...j->...i', self._rmat, pn_coeffs)

    def inverse_transform(self, coords):
        """
        Inverse of ``transform``: return (mchirp, eta, chieff) given
        metric coordinates.
        """
        pn_coeffs = np.einsum('ij,...j->...i', self._inv_rmat, coords)
        return self._invert_pn_coeffs(pn_coeffs)

    @staticmethod
    def _get_pn_coeffs(mchirp, eta, chieff):
        """
        Return PN coefficients such that the waveform phase is
            phase = pn_funcs @ pn_coeffs.
        This makes the approximation chieff = chis, strictly valid for
        equal mass. The maximum error is ~5% in the 2.5PN coefficient.
        """
        mchirp, eta, chieff = np.broadcast_arrays(mchirp, eta, chieff)
        mtot = gw_utils.mchirpeta_to_mtot(mchirp, eta)
        # Approximation here so we can use chieff instead of beta.
        chis_approx = chieff
        beta_approx = 113/12*(chieff - 76/113 * eta * chis_approx)
        return np.moveaxis([(4*beta_approx - 16*np.pi) / eta * mtot**(-2/3),
                            (55/9 + 3715/756/eta) / mtot,
                            mchirp**(-5/3)],
                           0, -1)

    def _invert_pn_coeffs(self, pn_coeffs):
        """
        Given the first 3 PN coefficients, return mchirp, eta, chieff.
        This makes the approximation chieff = chis, strictly valid for
        equal mass. It is the same approximation we make in
        _get_pn_coeffs so these functions are exact inverses.
        It is assumed that the *last* axis of pn_coeffs has length 3 and
        corresponds to the 2.5PN, 2PN and 1PN coefficients.
        """
        pn2, pn1, pn0 = np.moveaxis(pn_coeffs, -1, 0)
        mchirp = pn0 ** -.6
        eta = self._eta_of_x(pn0**1.5 * pn1**-2.5)
        beta = .25 * pn2 * eta**.6 * mchirp**(2/3) + 4*np.pi
        chieff_approx = 12 * beta / (113 - 76*eta)  # Use chis ~ chieff
        return mchirp, eta, chieff_approx


class PNMap:
    """
    Construct cumulatives of a fiducial posterior distribution for
    mchirp, eta, chieff.
    These cumulatives can be used as coordinates where the posterior
    should look uniform in the unit cube, desirable for quasi Monte
    Carlo integration.
    The fiducial posterior is based on the prior (flat in component
    masses and effective spin) times a Gaussian likelihood in the
    space of PN coefficients with a covariance matrix given by
    Fisher analysis from `par_dic_0` and `snr`.
    To increase robustness, the log posterior is scaled by
    `beta_temperature`.

    Provides a constructor `from_posterior`, it is the recommended
    instantiation method for simple cases.

    Provides a method `transform_and_weights` to draw (mchirp, lnq,
    chieff) samples following the approximate posterior along with their
    importance-sampling weights from uniform values in the unit cube.
    """
    def __init__(self, par_dic_0, snr, pn_coordinates, mchirp_range,
                 q_min=.05, resolution=128, beta_temperature=.1):
        self.pn_coordinates = pn_coordinates

        self.snr = snr
        self._coords_0 = None  # Set by par_dic_0.setter
        self._fmerger_0 = None  # Set by par_dic_0.setter
        self.par_dic_0 = par_dic_0

        self.beta_temperature = beta_temperature

        self.mchirp_range = np.asarray(mchirp_range)
        self.q_min = q_min

        self._likelihood_chieff_given_mchirp_lnq = scipy.stats.cauchy
        self._p_chieff_given_mchirp_lnq = trunccauchy

        self._mchirp_grid = None  # Set by resolution.setter
        self._mchirp_pdf = None  # Set by resolution.setter
        self._lnq_grid = None  # Set by resolution.setter
        self.resolution = resolution

        self.transform_and_weights = np.vectorize(self._transform_and_weights,
                                                  otypes=[object])

    @classmethod
    def from_posterior(cls, posterior, resolution=128,
                       beta_temperature=.1):
        harmonic_modes = posterior.likelihood.waveform_generator.harmonic_modes
        posterior.likelihood.waveform_generator.harmonic_modes = [(2, 2)]

        par_dic_0 = posterior.likelihood.par_dic_0
        snr = np.sqrt(2 * posterior.likelihood.lnlike(par_dic_0))
        pn_coordinates = PNCoordinates.from_likelihood(posterior.likelihood)
        init_dict = posterior.prior.get_init_dict()
        mchirp_range = init_dict['mchirp_range']
        q_min = init_dict['q_min']

        posterior.likelihood.waveform_generator.harmonic_modes = harmonic_modes
        return cls(par_dic_0, snr, pn_coordinates, mchirp_range,
                   q_min=q_min, resolution=resolution,
                   beta_temperature=beta_temperature)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution
        self._mchirp_grid = np.linspace(*self.mchirp_range, resolution)
        self._lnq_grid = np.linspace(np.log(self.q_min), 0, resolution)
        self._mchirp_pdf =(
            self._mchirp_prior(self._mchirp_grid)
            * np.vectorize(self._evidence_lnq_chieff_given_mchirp)(
                self._mchirp_grid))

    @property
    def par_dic_0(self):
        return self._par_dic_0

    @par_dic_0.setter
    def par_dic_0(self, par_dic_0):
        self._par_dic_0 = par_dic_0
        mchirp0 = gw_utils.m1m2_to_mchirp(par_dic_0['m1'],
                                          par_dic_0['m2'])
        eta0 = gw_utils.q_to_eta(par_dic_0['m2'] / par_dic_0['m1'])
        chieff0 = gw_utils.chieff(par_dic_0['m1'],
                                  par_dic_0['m2'],
                                  par_dic_0['s1z'],
                                  par_dic_0['s2z'])
        mtot0 = gw_utils.mchirpeta_to_mtot(mchirp0, eta0)

        self._coords_0 = self.pn_coordinates.transform(mchirp0, eta0, chieff0)
        self._fmerger_0 = get_f_merger(mtot0, chieff0)
        self._fmerger_scale_0 = 2 * (
            self.snr
            * self.pn_coordinates.interp_wht_amplitude(self._fmerger_0))**-2

    @staticmethod
    def _lnq_prior(lnq):
        # return np.cosh(lnq/2)**.4
        return 1

    @staticmethod
    def _mchirp_prior(mchirp):
        return mchirp

    def _evidence_lnq_chieff_given_mchirp(self, mchirp):
        return scipy.integrate.trapezoid(
            self._lnq_prior(self._lnq_grid)
            * self._evidence_chieff_given_mchirp_lnq(mchirp, self._lnq_grid),
            self._lnq_grid)

    def _evidence_chieff_given_mchirp_lnq(self, mchirp, lnq):
        """
        Return
            int_{-1}^1 dchieff (prior(chieff)
                                * likelihood(mchirp, lnq, chieff))
        with
            prior(chieff) = uniform
            likelihood = an analytical approximation to the likelihood
                         that accounts for the inspiral phase at 2.5PN
                         and an estimate of the merger frequency.
        """
        chieff_loc, chieff_scale, weight = self._chieff_loc_scale_and_weight(
            mchirp, lnq)
        chieff_min = -1
        chieff_max = 1

        return weight * (
            self._likelihood_chieff_given_mchirp_lnq.cdf(
                chieff_max, loc=chieff_loc, scale=chieff_scale)
            - self._likelihood_chieff_given_mchirp_lnq.cdf(
                chieff_min, loc=chieff_loc, scale=chieff_scale))

    def _chieff_loc_and_scale_due_to_fmerger(self, mchirp, lnq):
        """
        Return estimate of chieff and its uncertainty, conditioned on
        (mchirp, lnq), just accounting for the cutoff frequency.

        Return
        ------
        chieff_loc: float
            Expected peak of the likelihood. Need not be in (-1, 1).

        chieff_scale: float > 0
            Expected width of the likelihood.
        """
        mtot = gw_utils.mchirpeta_to_mtot(mchirp,
                                          gw_utils.q_to_eta(np.exp(lnq)))
        fmerger_min = get_f_merger(mtot, -1)
        fmerger_max = get_f_merger(mtot, 1)

        # TODO maybe change trunccauchy to trunclaplace
        kwargs = dict(
            a=(fmerger_min - self._fmerger_0) / self._fmerger_scale_0,
            b=(fmerger_max - self._fmerger_0) / self._fmerger_scale_0,
            loc=self._fmerger_0,
            scale=self._fmerger_scale_0)
        fmerger5 = trunccauchy.ppf(.05, **kwargs)
        fmerger95 = trunccauchy.ppf(.95, **kwargs)

        # Linear approximation to chieff(fmerger) near region with support
        chieff5 = chieff_from_mtot_fmerger(mtot, fmerger5)
        chieff95 = chieff_from_mtot_fmerger(mtot, fmerger95)

        dchieff_dfmerger = (chieff95 - chieff5) / (fmerger95 - fmerger5)

        chieff_loc = chieff5 + (self._fmerger_0-fmerger5) * dchieff_dfmerger
        chieff_scale = self._fmerger_scale_0 * dchieff_dfmerger

        return chieff_loc, chieff_scale

    def _chieff_loc_scale_and_weight_due_to_inspiral(self, mchirp, lnq):
        """
        Return estimate of chieff and its uncertainty, conditioned on
        (mchirp, lnq), just accounting for the inspiral.

        Return
        ------
        chieff_loc: float
            Expected peak of the likelihood. Need not be in (-1, 1).

        chieff_scale: float > 0
            Expected width of the likelihood.

        weight: float between 0 and 1
            Penalty in case the mismatch-metric-coordinates orthogonal
            to chieff are inconsistent with the reference solution.
        """
        chieff_min = -1
        chieff_max = 1
        eta = gw_utils.q_to_eta(np.exp(lnq))

        coords_start = self.pn_coordinates.transform(mchirp, eta, chieff_min)
        c0_min = coords_start[..., 0]
        c0_max = self.pn_coordinates.transform(mchirp, eta, chieff_max)[..., 0]

        dchieff_dc0 = (chieff_max - chieff_min) / (c0_max - c0_min)
        chieff_loc = chieff_min + (self._coords_0[0] - c0_min) * dchieff_dc0
        chieff_scale = dchieff_dc0 / self.snr

        # Penalize if the coordinates orthogonal to chieff are
        # inconsistent with reference solution:
        perpendicular_distance = np.linalg.norm(
            coords_start[..., 1:] - self._coords_0[1:], axis=-1)
        weight = np.exp(- self.beta_temperature * self.snr**2
                        * perpendicular_distance**2 / 2)

        return chieff_loc, chieff_scale, weight

    def _chieff_loc_scale_and_weight(self, mchirp, lnq):
        """
        Return
        ------
        chieff_loc, chieff_scale: float
            Estimates of the location and width of the likelihood as a
            function of chieff. Note, `loc` needs not be in (-1, 1).

        weight: float between 0 and 1
            Multiplicative normalization of the likelihood, due to the
            mismatch distance orthogonal to the chieff direction.
        """
        loc1, scale1 = self._chieff_loc_and_scale_due_to_fmerger(mchirp, lnq)
        loc2, scale2, weight \
            = self._chieff_loc_scale_and_weight_due_to_inspiral(mchirp, lnq)

        weights = scale1**-2, scale2**-2
        chieff_loc = np.average((loc1, loc2), weights=weights, axis=0)
        chieff_scale = sum(weights)**-.5 / self.beta_temperature

        # Penalize if chieff prediction from merger frequency is
        # inconsistent with chieff prediction from inspiral:
        weight *= (self._likelihood_chieff_given_mchirp_lnq.pdf(
                       (loc1 - chieff_loc) / chieff_scale)
                   * self._likelihood_chieff_given_mchirp_lnq.pdf(
                       (loc2 - chieff_loc) / chieff_scale))
        return chieff_loc, chieff_scale, weight

    def _draw_chieff_given_mchirp_lnq(self, u_chieff, mchirp, lnq):
        """
        Draw a value of chieff from its conditional distribution given
        mchirp, lnq.

        Parameters
        ----------
        u_chieff: float
            Quantile of P(chieff | mchirp, lnq) to draw.

        mchirp: float
            Detector-frame chirp mass (Msun).

        lnq: float
            Natural logarithm of mass ratio.

        Return
        ------
        chieff: float
            Value of chieff corresponding to the requested quantile of
            the conditional distribution.

        du_dchieff: float > 0
            P(chieff | mchirp, lnq), i.e., the Jacobian of the CDF.
        """
        chieff_loc, chieff_scale, _ = self._chieff_loc_scale_and_weight(mchirp,
                                                                        lnq)
        chieff_min = -1
        chieff_max = 1
        kwargs = dict(a=(chieff_min - chieff_loc) / chieff_scale,
                      b=(chieff_max - chieff_loc) / chieff_scale,
                      loc=chieff_loc,
                      scale=chieff_scale)

        chieff = self._p_chieff_given_mchirp_lnq.ppf(u_chieff, **kwargs)
        du_dchieff = self._p_chieff_given_mchirp_lnq.pdf(chieff, **kwargs)
        return chieff, du_dchieff

    def _transform_and_weights(self, u_mchirp, u_lnq, u_chieff):
        """
        Return mchirp, lnq, chieff from their cumulatives.
            u_mchirp := C(mchirp)
            u_lnq := C(lnq|mchirp)
            u_chieff := C(chieff|mchirp, lnq)
        where C denotes the cumulative of the fiducial posterior.
        Also return weights.
        """
        mchirp, du_dmchirp = inverse_cdf_and_jacobian(
            u_mchirp, self._mchirp_grid, self._mchirp_pdf)

        lnq_post = (self._lnq_prior(self._lnq_grid)
                    * self._evidence_chieff_given_mchirp_lnq(mchirp,
                                                             self._lnq_grid))

        lnq, du_dlnq = inverse_cdf_and_jacobian(u_lnq, self._lnq_grid,
                                                lnq_post)

        chieff, du_dchieff = self._draw_chieff_given_mchirp_lnq(u_chieff,
                                                                mchirp, lnq)
        self.status = locals()
        return {'mchirp': mchirp,
                'lnq': lnq,
                'chieff': chieff,
                'weights': 1 / (du_dmchirp * du_dlnq * du_dchieff)}


class IntrinsicMap:
    """
    Use the cumulative of the prior on theta_jn as inclination
    coordinate.
    The formula for (2, 2) aligned spin is used.
    """
    _costheta_jn_grid = np.linspace(-1, 1, 256)
    _costheta_jn_prior = (((1+_costheta_jn_grid**2)/2)**2
                          + _costheta_jn_grid**2) ** 1.5

    params = ['mchirp', 'lnq', 'chieff', 'cumchidiff', 'cums1r_s1z',
              'cums2r_s2z', 'phi_jl_hat', 'phi12', 'costheta_jn']

    def __init__(self, pn_map: PNMap):
        super().__init__()
        self.pn_map = pn_map

    def generate_intrinsic_samples(self, log2n_qmc: int):
        """
        Return pd.DataFrame with `2**log2n_qmc` Quasi Monte Carlo
        samples of ``params``. A Sobol sequence is used.
        """
        qmc_sequence = pd.DataFrame(
            scipy.stats.qmc.Sobol(len(self.params)).random_base2(log2n_qmc),
            columns=[f'u_{par}' for par in self.params])

        samples = pd.DataFrame.from_records(
            self.pn_map.transform_and_weights(
                **qmc_sequence[['u_mchirp', 'u_lnq', 'u_chieff']]))

        samples['cumchidiff'] = qmc_sequence['u_cumchidiff']
        samples['cums1r_s1z'] = qmc_sequence['u_cums1r_s1z']
        samples['cums2r_s2z'] = qmc_sequence['u_cums2r_s2z']
        samples['phi_jl_hat'] = qmc_sequence['u_phi_jl_hat'] * 2 * np.pi
        samples['phi12'] = qmc_sequence['u_phi12'] * 2 * np.pi


        vec_inverse_cdf_and_jacobian = np.vectorize(inverse_cdf_and_jacobian,
                                                    otypes=[float, float],
                                                    excluded=[1, 2])

        samples['costheta_jn'], du_dcostheta_jn = vec_inverse_cdf_and_jacobian(
            qmc_sequence['u_costheta_jn'],
            self._costheta_jn_grid,
            self._costheta_jn_prior)

        samples['weights'] /= du_dcostheta_jn
        return samples
