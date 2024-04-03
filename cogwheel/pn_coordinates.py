"""
Module to generate intrinsic parameter samples with Quasi Monte Carlo
from an approximation of the posterior.
This uses an approximation of the likelihood that accounts for the
post-Newtonian inspiral and the cutoff frequency.

Classes ``IntrinsicParameterProposal``, ``_InspiralAnalysis`` and
``_MergerAnalysis`` are defined. ``IntrinsicParameterProposal`` is the
top-level class that most users would use.

Example usage:
```
# ``post`` is an instance of ``posterior.Posterior``
intrinsic_proposal = IntrinsicParameterProposal.from_posterior(post)
qmc_samples = intrinsic_proposal.generate_intrinsic_samples(14)

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


class TruncatedDistribution(scipy.stats.rv_continuous):
    """Truncate a distribution."""
    def __init__(self, non_truncated_distribution, **kwargs):
        super().__init__(**kwargs)
        self.non_truncated_distribution = non_truncated_distribution

    def _cdf(self, x, a, b):
        u_a, u_b, u_x = self.non_truncated_distribution.cdf((a, b, x))
        return (u_x - u_a) / (u_b - u_a)

    def _ppf(self, q, a, b):
        u_a, u_b = self.non_truncated_distribution.cdf((a, b))
        u_q = u_a + (u_b - u_a) * q
        return self.non_truncated_distribution.ppf(u_q)

    def _pdf(self, x, a, b):
        u_a, u_b = self.non_truncated_distribution.cdf((a, b))
        return np.where((x > a) & (x < b),
                        self.non_truncated_distribution.pdf(x) / (u_b - u_a),
                        0.)

    def _argcheck(self, a, b):
        return a < b


trunclaplace = TruncatedDistribution(scipy.stats.laplace)


def _inverse_cdf_and_jacobian(cdf_val, points, pdf_points):
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


inverse_cdf_and_jacobian = np.vectorize(_inverse_cdf_and_jacobian,
                                        otypes=[float, float],
                                        excluded=[1, 2])


class _InspiralAnalysis:
    """
    Provide a method `s1z_loc_scale_and_weight_due_to_inspiral` that
    estimates the best fit value and 1-sigma errorbar of s1z given other
    intrinsic parameters. The estimate uses the phase evolution to
    2.5 post-Newtonian order plus the leading correction from precession
    in the phase of the zeroth precession harmonic.

    """
    def __init__(self, frequencies, whitened_amplitude, par_dic_0, snr):
        """
        Parameters
        ----------
        frequencies: 1d float array
            Frequencies in Hz, must be regularly spaced.

        whitened_amplitude: 1d float array
            Whitened amplitude of the best-fit template evaluated at
            `frequencies`: A(f) / sqrt(PSD(f)). If there are multiple
            detectors, use the root-mean-square over detectors. The
            overall normalization is unimportant.

        par_dic_0: dict
            Parameters of the best fit solution, must contain keys for
            m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n, s2y_n.

        snr: float
            Signal-to-noise ratio of the best-fit template.
        """
        if not np.allclose(np.diff(frequencies, 2), 0):
            raise ValueError('`frequencies` must be regularly spaced.')

        # (phase, time, 1PN, 2PN, 2.5PN, precession). Ordered this way
        # we ensure that intrinsic parameters are orthogonalized to
        # phase and time.
        pn_exponents = 0, 1, -5/3, -3/3, -2/3, -1/3
        pn_functions = np.power.outer(lal.MTSUN_SI * np.pi * frequencies,
                                      pn_exponents)
        weights = whitened_amplitude / np.linalg.norm(whitened_amplitude)
        _, fullrmat = unique_qr(pn_functions * weights[:, np.newaxis])

        # Ignore phase & time:
        self._rmat = fullrmat[2:, 2:]  # Transformation matrix

        self._coords_0 = None  # Set by par_dic_0.setter
        self.par_dic_0 = par_dic_0

        self.snr = snr

    @classmethod
    def from_likelihood(cls, likelihood):
        """
        Instantiate from a RelativeBinningLikelihood object, used to
        estimate the whitened_amplitude from the whitening filter and
        reference waveform.
        """
        mtot = likelihood.par_dic_0['m1'] + likelihood.par_dic_0['m2']
        chieff = gw_utils.chieff(**{par: likelihood.par_dic_0[par]
                                    for par in ['m1', 'm2', 's1z', 's2z']})
        fmin = likelihood.event_data.fbounds[0]
        fmax = get_f_merger(mtot, chieff)
        fslice = slice(*np.searchsorted(likelihood.event_data.frequencies,
                                        (fmin, fmax)))
        frequencies = likelihood.event_data.frequencies[fslice]

        h_f = likelihood._get_h_f(likelihood.par_dic_0)
        whitened_amplitude = np.linalg.norm(
            h_f * likelihood.event_data.wht_filter, axis=0)[fslice]
        d_h = likelihood._compute_d_h(h_f)
        h_h = likelihood._compute_h_h(h_f)
        snr = np.sqrt(np.sum(np.abs(d_h)**2 / h_h))
        return cls(frequencies, whitened_amplitude, likelihood.par_dic_0, snr)

    @property
    def par_dic_0(self):
        """Dictionary of best-fit parameters."""
        return self._par_dic_0

    @par_dic_0.setter
    def par_dic_0(self, par_dic_0):
        self._coords_0 = self._get_coordinates(
            **{par: par_dic_0[par]
               for par in ['m1', 'm2', 's1z', 's2z', 's1x_n', 's1y_n', 's2x_n',
                           's2y_n']})
        self._par_dic_0 = par_dic_0

    @staticmethod
    def _get_pn_coefficients(m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n,
                             s2y_n):
        """
        Return PN coefficients such that the frequency-domain waveform
        phase is
            phase = pn_functions @ pn_coefficients.
        """
        m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n, s2y_n = np.broadcast_arrays(
            m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n, s2y_n)

        mtot = m1 + m2
        eta = m1 * m2 / mtot**2
        q = m2 / m1
        chis = (s1z + s2z) / 2
        chia = (s1z - s2z) / 2
        delta = (m1 - m2) / mtot
        beta = 113/12 * (chis + delta*chia - 76/113*eta*chis)
        sx = m1**2 * s1x_n + m2**2 * s2x_n
        sy = m1**2 * s1y_n + m2**2 * s2y_n

        pn_coefficients = (
            3/128 / eta * mtot**(-5/3),
            3/128 * (55/9 + 3715/756/eta) / mtot,
            3/128 * (4*beta - 16*np.pi) / eta * mtot**(-2/3),
            15/128 * (1+q)**4*(4/3+q)*(sx**2+sy**2)/mtot**4/q**2 * mtot**(-1/3)
            )

        return np.moveaxis(pn_coefficients, 0, -1)  # Last axis is PN order

    def _get_coordinates(self, m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n,
                         s2y_n):
        """
        Return array of coordinates whose Euclidean distance is the
        mismatch distance for PN waveforms.
        """
        pn_coefficients = self._get_pn_coefficients(
            m1, m2, s1z, s2z, s1x_n, s1y_n, s2x_n, s2y_n)

        # Note:
        #     weights * phase = (weights * pn_funcs) @ pn_coefficients
        #                     = qmat @ rmat @ pn_coefficients
        #                     = qmat @ coords
        # I.e., coords = rmat @ pn_coefficients
        return np.einsum('ij,...j->...i', self._rmat, pn_coefficients)

    def s1z_loc_scale_and_weight_due_to_inspiral(
            self, m1, m2, s2z, s1x_n, s1y_n, s2x_n, s2y_n):
        """
        Return estimate of s1z and its uncertainty, conditioned on
        (m1, m2, s2z, s1x_n, s1y_n, s2x_n, s2y_n), just accounting
        for the inspiral.

        Return
        ------
        s1z_loc: float
            Expected peak of the likelihood. Need not be in (-1, 1).

        s1z_scale: float > 0
            Expected width of the likelihood.

        weight: float between 0 and 1
            Penalty in case the mismatch-metric-coordinates orthogonal
            to s1z are inconsistent with the reference solution.
        """
        # Use the fact that coordinates depend linearly on s1z:
        coords_s1z_0 = self._get_coordinates(m1, m2, 0, s2z, s1x_n, s1y_n,
                                             s2x_n, s2y_n)
        coords_s1z_1 = self._get_coordinates(m1, m2, 1, s2z, s1x_n, s1y_n,
                                             s2x_n, s2y_n)
        dcoords = coords_s1z_1 - coords_s1z_0
        direction = dcoords / np.linalg.norm(dcoords, axis=-1, keepdims=True)
        dcoords_projection = np.einsum('...i,...i->...',
                                       self._coords_0 - coords_s1z_0,
                                       direction)
        ds1z_dcoords = 1 / np.linalg.norm(dcoords, axis=-1)

        s1z_loc = ds1z_dcoords * dcoords_projection
        s1z_scale = ds1z_dcoords / self.snr

        # Penalize if the coordinates orthogonal to s1z are inconsistent
        # with reference solution:
        perpendicular_distance = np.linalg.norm(
            self._coords_0 - coords_s1z_0
            - dcoords_projection[..., np.newaxis] * direction, axis=-1)
        weight = np.exp(-self.snr**2 * perpendicular_distance**2 / 2)

        return s1z_loc, s1z_scale, weight


class _MergerAnalysis:
    """
    Class that allows to estimate a best-fit value and standard
    deviation of s1z given other intrinsic parameters and knowledge of
    the best-fit merger frequency.
    """
    def __init__(self, frequencies, whitened_amplitude, par_dic_0, snr,
                 smooth=True):
        """
        Parameters
        ----------
        frequencies: 1d float array
            Frequencies in Hz.

        whitened_amplitude: array like frequencies
            |h| / sqrt(PSD), with arbitrary normalization, evaluated at
            `frequencies`. If there are multiple detectors, pass the
            root mean square over detectors.

        smooth: bool
            If True, will smooth the whitened amplitude as a function of
            frequency so it can be interpolated meaningfully (useful for
            estimating the amplitude at merger frequency).
        """
        if smooth:
            whitened_amplitude = scipy.signal.savgol_filter(
                whitened_amplitude, len(whitened_amplitude) // 100, 1)

        whitened_amplitude /= np.sqrt(
            scipy.integrate.trapezoid(whitened_amplitude**2, frequencies))

        mtot0 = par_dic_0['m1'] + par_dic_0['m2']
        chieff0 = gw_utils.chieff(par_dic_0['m1'],  par_dic_0['m2'],
                                   par_dic_0['s1z'],  par_dic_0['s2z'])

        self._fmerger_0 = get_f_merger(mtot0, chieff0)
        wht_amp_f_merger = np.interp(self._fmerger_0, frequencies,
                                     whitened_amplitude)

        self._fmerger_scale_0 = 2 * (snr * wht_amp_f_merger)**-2

    @classmethod
    def from_likelihood(cls, likelihood):
        """
        Instantiate from a RelativeBinningLikelihood object, used to
        estimate the whitened_amplitude from the whitening filter and
        reference waveform.
        """
        fslice = likelihood.event_data.fslice
        frequencies = likelihood.event_data.frequencies[fslice]

        h_f = likelihood._get_h_f(likelihood.par_dic_0)
        d_h = likelihood._compute_d_h(h_f)
        h_h = likelihood._compute_h_h(h_f)
        snr = np.sqrt(np.sum(np.abs(d_h)**2 / h_h))
        whitened_amplitude = np.linalg.norm(
            h_f * likelihood.event_data.wht_filter, axis=0)[fslice]

        return cls(frequencies, whitened_amplitude, likelihood.par_dic_0, snr)

    def s1z_loc_and_scale_due_to_fmerger(self, m1, m2, s2z):
        """
        Return estimate of chieff and its uncertainty, conditioned on
        (m1, m2, s2z), just accounting for the cutoff frequency.

        Return
        ------
        chieff_loc: float
            Expected peak of the likelihood. Need not be in (-1, 1).

        chieff_scale: float > 0
            Expected width of the likelihood.
        """
        fmerger_min = self._get_fmerger(m1, m2, -1, s2z)
        fmerger_max = self._get_fmerger(m1, m2, 1, s2z)

        kwargs = {
            'a': (fmerger_min - self._fmerger_0) / self._fmerger_scale_0,
            'b': (fmerger_max - self._fmerger_0) / self._fmerger_scale_0,
            'loc': self._fmerger_0,
            'scale': self._fmerger_scale_0}
        fmerger2 = trunclaplace.ppf(.2, **kwargs)
        fmerger8 = trunclaplace.ppf(.8, **kwargs)

        # Linear approximation to s1z(fmerger) near region with support
        s1z2 = self._get_s1z(fmerger2, m1, m2, s2z)
        s1z8 = self._get_s1z(fmerger8, m1, m2, s2z)

        ds1z_dfmerger = (s1z8 - s1z2) / (fmerger8 - fmerger2)

        s1z_loc = s1z2 + (self._fmerger_0 - fmerger2) * ds1z_dfmerger
        s1z_scale = self._fmerger_scale_0 * ds1z_dfmerger

        return s1z_loc, s1z_scale

    @staticmethod
    def _get_s1z(fmerger, m1, m2, s2z):
        chieff = chieff_from_mtot_fmerger(m1 + m2, fmerger)
        return ((m1 + m2) * chieff - m2 * s2z) / m1

    @staticmethod
    def _get_fmerger(m1, m2, s1z, s2z):
        chieff = gw_utils.chieff(m1, m2, s1z, s2z)
        return get_f_merger(m1 + m2, chieff)


class IntrinsicParameterProposal:
    """
    Provide a method `generate_intrinsic_samples` that generates samples
    of intrinsic parameters (per `.params`) from an importance-sampling
    proposal using Quasi Monte Carlo. The importance-sampling proposal
    is informed by the inspiral and the merger frequency via Fisher
    analysis using post-Newtonian models.
    Provide a constructor `from_posterior`.
    """
    params = ['m1', 'm2', 's1z', 's2z', 's1x_n', 's1y_n', 's2x_n', 's2y_n',
              'iota']
    _likelihood_s1z = scipy.stats.cauchy
    _p_s1z = TruncatedDistribution(_likelihood_s1z)

    _cosiota_grid = np.linspace(-1, 1, 256)
    _cosiota_prior = (
        ((1 + _cosiota_grid**2) / 2)**2 + _cosiota_grid**2) ** (3/2)

    def __init__(self, inspiral_analysis, merger_analysis, mchirp_range,
                 q_min=.05, resolution=128, beta_temperature=.1):
        self.inspiral_analysis = inspiral_analysis
        self.merger_analysis = merger_analysis
        self.beta_temperature = beta_temperature

        # Choice: sample mchirp and lnq independently. Revise as needed.
        self._mchirp_grid = np.linspace(*mchirp_range, resolution)
        self._lnq_grid = np.linspace(np.log(q_min), 0, resolution)
        mchirp, lnq = np.meshgrid(self._mchirp_grid, self._lnq_grid,
                                  indexing='ij')
        m1, m2 = gw_utils.mchirpeta_to_m1m2(mchirp,
                                            gw_utils.q_to_eta(np.exp(lnq)))
        evidence_s1z = self._evidence_s1z(m1, m2, s2z=0, s1x_n=0, s1y_n=0,
                                          s2x_n=0, s2y_n=0)
        self._mchirp_pdf = evidence_s1z.sum(axis=1)
        self._lnq_pdf = evidence_s1z.sum(axis=0)

    @classmethod
    def from_posterior(cls, posterior, **kwargs):
        """
        Parameters
        ----------
        posterior: cogwheel.posterior.Posterior
            Posterior instance from which to take best-fit parameters,
            parameter ranges, and detector PSDs.
        **kwargs
        """
        inspiral_analysis = _InspiralAnalysis.from_likelihood(
            posterior.likelihood)
        merger_analysis = _MergerAnalysis.from_likelihood(
            posterior.likelihood)

        dic = posterior.prior.get_init_dict()
        if 'q_min' not in kwargs:
            kwargs['q_min'] = dic['q_min']

        if 'mchirp_range' not in kwargs:
            kwargs['mchirp_range'] = dic['mchirp_range']

        return cls(inspiral_analysis, merger_analysis, **kwargs)

    def generate_intrinsic_samples(self, log2n_qmc: int):
        """
        Return pd.DataFrame with `2**log2n_qmc` Quasi Monte Carlo
        samples of ``params``. A Sobol sequence is used.
        """
        qmc_sequence = pd.DataFrame(
            scipy.stats.qmc.Sobol(len(self.params)).random_base2(log2n_qmc),
            columns=['cdf_mchirp', 'cdf_lnq', 'cdf_s1r', 'cdf_s1z_conditioned',
                     'cdf_phi1', 'cdf_s2r', 'cdf_s2z_s2r', 'cdf_phi2',
                     'cdf_cosiota'])

        samples = pd.DataFrame()

        samples['s1x_n'], samples['s1y_n'] = self._get_inplane_spins(
            qmc_sequence['cdf_s1r'], qmc_sequence['cdf_phi1'])

        samples['s2x_n'], samples['s2y_n'] = self._get_inplane_spins(
            qmc_sequence['cdf_s2r'], qmc_sequence['cdf_phi2'])

        s2z_max = np.sqrt(1 - samples['s2x_n']**2 - samples['s2y_n']**2)
        samples['s2z'] = 2 * (qmc_sequence['cdf_s2z_s2r'] - .5) * s2z_max

        mchirp, pdf_mchirp = inverse_cdf_and_jacobian(
            qmc_sequence['cdf_mchirp'], self._mchirp_grid, self._mchirp_pdf)
        lnq, pdf_lnq = inverse_cdf_and_jacobian(
            qmc_sequence['cdf_lnq'], self._lnq_grid, self._lnq_pdf)
        samples['m1'], samples['m2'] = gw_utils.mchirpeta_to_m1m2(
            mchirp, gw_utils.q_to_eta(np.exp(lnq)))

        samples['s1z'], weight_s1z = self._get_s1z_and_weight(
            **qmc_sequence[['cdf_s1z_conditioned']], **samples)

        cosiota, pdf_cosiota = inverse_cdf_and_jacobian(
            qmc_sequence['cdf_cosiota'],
            self._cosiota_grid,
            self._cosiota_prior)
        samples['iota'] = np.arccos(cosiota)
        samples['weights'] = weight_s1z / pdf_cosiota / pdf_mchirp / pdf_lnq

        return samples

    def _evidence_s1z(self, m1, m2, s2z, s1x_n, s1y_n, s2x_n, s2y_n):
        """
        Return
            int_{s1z_min}^{s1z_max} ds1z (prior(s1z | m1, ...)
                                          * likelihood)
        with
            prior(s1z) = uniform (within Kerr bound given inplane s1).
            likelihood = an analytical approximation to the likelihood
                         that accounts for the inspiral phase at 2.5PN,
                         a precession correction and an estimate of the
                         merger frequency.
        """
        s1z_loc, s1z_scale, weight = self._s1z_loc_scale_and_weight(
            m1, m2, s2z, s1x_n, s1y_n, s2x_n, s2y_n)
        s1z_max = np.sqrt(1 - s1x_n**2 - s1y_n**2)
        s1z_min = -s1z_max
        parameter_min = (s1z_min - s1z_loc) / s1z_scale
        parameter_max = (s1z_max - s1z_loc) / s1z_scale
        evidence_s1z = weight * (self._likelihood_s1z.cdf(parameter_max)
                                 - self._likelihood_s1z.cdf(parameter_min)
                                ) / (parameter_max - parameter_min)
        return evidence_s1z

    def _s1z_loc_scale_and_weight(self, m1, m2, s2z, s1x_n, s1y_n,
                                  s2x_n, s2y_n):
        """
        Return
        ------
        s1z_loc, s1z_scale: float
            Estimates of the location and width of the likelihood as a
            function of s1z. Note, `loc` needs not be in (-1, 1).

        weight: float between 0 and 1
            Multiplicative normalization of the likelihood, penalizes
            mismatch distance orthogonal to the s1z direction, and
            inconsistency between s1z predictions from inspiral and
            merger.
        """
        loc1, scale1, inspiral_weight \
            = self.inspiral_analysis.s1z_loc_scale_and_weight_due_to_inspiral(
                m1, m2, s2z, s1x_n, s1y_n, s2x_n, s2y_n)

        loc2, scale2 = self.merger_analysis.s1z_loc_and_scale_due_to_fmerger(
            m1, m2, s2z)

        inv_variances = np.array((scale1, scale2))**-2
        s1z_loc = np.average((loc1, loc2), weights=inv_variances, axis=0)
        s1z_scale = sum(inv_variances * self.beta_temperature)**-.5

        # Penalize if s1z prediction from merger frequency is
        # inconsistent with s1z prediction from inspiral:
        weight = inspiral_weight ** self.beta_temperature * (
            self._likelihood_s1z.pdf((loc1 - s1z_loc) / scale1)
            * self._likelihood_s1z.pdf((loc2 - s1z_loc) / scale2))
        return s1z_loc, s1z_scale, weight

    def _get_s1z_and_weight(self, cdf_s1z_conditioned, m1, m2, s2z,
                            s1x_n, s1y_n, s2x_n, s2y_n):
        """
        Return the value of s1z from its CDF, given other intrinsic
        parameters. Also return the importance sampling weight w.r.t.
        a conditional prior uniform in s1z within the Kerr bound.
        """
        s1z_loc, s1z_scale, _ = self._s1z_loc_scale_and_weight(
            m1, m2, s2z, s1x_n, s1y_n, s2x_n, s2y_n)

        s1z_max = np.sqrt(1 - s1x_n**2 - s1y_n**2)
        s1z_min = -s1z_max
        kwargs = {'a': (s1z_min - s1z_loc) / s1z_scale,
                  'b': (s1z_max - s1z_loc) / s1z_scale,
                  'loc': s1z_loc,
                  'scale': s1z_scale}

        s1z = self._p_s1z.ppf(cdf_s1z_conditioned, **kwargs)
        pdf_s1z = self._p_s1z.pdf(s1z, **kwargs)
        prior_s1z = 1 / (s1z_max - s1z_min)
        return s1z, prior_s1z / pdf_s1z

    @staticmethod
    def _get_inplane_spins(cdf_sr, cdf_phi):
        """
        Inverse of the cumulative distribution of the inplane spin
        magnitude and azimuth under a "volumetric" prior (i.e. density
        uniform in the ball |s| < 1).
        """
        sr = np.sqrt(1 - (1 - cdf_sr)**(2/3))
        phi = 2*np.pi * cdf_phi
        sx = sr * np.cos(phi)
        sy = sr * np.sin(phi)
        return sx, sy
