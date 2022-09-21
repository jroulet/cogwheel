import copy
import scipy
import numpy as np

import lal

from cogwheel import gw_prior
from cogwheel import gw_utils
from cogwheel import prior
from cogwheel import utils

class LinearFreeIASPrior(gw_prior.RegisteredPriorMixin, prior.CombinedPrior):
    prior_classes = [
        gw_prior.miscellaneous.FixedReferenceFrequencyPrior,
        gw_prior.mass.UniformDetectorFrameMassesPrior,
        gw_prior.spin.UniformEffectiveSpinPrior,
        gw_prior.spin.UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior,
        gw_prior.extrinsic.UniformPolarizationPrior,
        gw_prior.miscellaneous.ZeroTidalDeformabilityPrior,
        gw_prior.linear_free.LinearFreePhaseTimePrior]


class SlicePosteriorMixin:
    @classmethod
    def get_loc_scale(cls, posterior, n_grid=10**3):
        if len(cls.standard_params) != 1:
            raise ValueError('I expect that the class samples one parameter')

        par = cls.standard_params[0]
        i_par = posterior.prior.sampled_params.index(par)
        grid = posterior.prior.cubemin[i_par] + np.linspace(
            0, posterior.prior.folded_cubesize[i_par], n_grid)
        sampled_dic_0 = posterior.prior.inverse_transform(
            **posterior.likelihood.par_dic_0)

        get_lnposts = posterior.prior.unfold_apply(posterior.lnposterior)
        log_weights = np.array(
            [scipy.special.logsumexp(get_lnposts(**sampled_dic_0 | {par: val}))
             for val in grid])
        weights = np.exp(log_weights - log_weights.max())
        loc, scale = utils.weighted_avg_and_std(grid, weights)
        return loc, scale


class PsiMap(prior.UniformPriorMixin, prior.Prior):
    """Uniform proposal for the polarization.."""
    range_dic = {'u_psi': (0, 1)}
    standard_params = ['psi']
    periodic_params = ['u_psi']

    def transform(self, u_psi):
        return {'psi': np.pi * u_psi}

    def inverse_transform(self, psi):
        return {'u_psi': psi / np.pi}


class TimeMap(SlicePosteriorMixin, prior.Prior):
    """Lorentzian proposal for the linear-free time."""
    range_dic = {'u_t_linfree': (0, 1)}
    standard_params = ['t_linfree']

    def __init__(self, *, time_scale, t_linfree_0=0., **kwargs):
        self._init_dict =  {'time_scale': time_scale,
                            't_linfree_0': t_linfree_0}
        self._dist = scipy.stats.cauchy(scale=time_scale)
        super().__init__(**kwargs)

    def transform(self, u_t_linfree):
        return {'t_linfree': self._dist.ppf(u_t_linfree)}

    def inverse_transform(self, t_linfree):
        return {'u_t_linfree': self._dist.cdf(t_linfree)}

    def get_init_dict(self):
        return {'time_scale': self._dist.kwds['scale']}

    def lnprior(self, u_t_linfree):
        return -self._dist.logpdf(self._dist.ppf(u_t_linfree))

    @classmethod
    def kwargs_from_posterior(cls, posterior, time_fudge=2., **_):
        loc, scale = cls.get_loc_scale(posterior)
        return {'time_scale': scale * time_fudge,
                't_linfree_0': loc}


# class PhaseMap(SlicePosteriorMixin, prior.Prior):
#     """Double Lorentzian prior for the linear-free phase."""
#     range_dic = {'u_phi_linfree': (0, 1)}
#     standard_params = ['phi_linfree']
#     periodic_params = ['u_phi_linfree']

#     def __init__(self, *, phase_scale, **kwargs):
#         self._init_dict =  {'phase_scale': phase_scale}
#         self._dist = wrapcauchy(c=np.exp(-phase_scale))
#         super().__init__(**kwargs)

#     def transform(self, u_phi_linfree):
#         return {'phi_linfree': utils.mod((self._dist.ppf(2*u_phi_linfree % 1)
#                                          + (self._dist.b - self._dist.a)*(u_phi_linfree>.5)) / 2,
#                                          -np.pi/2)}

#     def inverse_transform(self, phi_linfree):
#         phi_linfree =  phi_linfree % (2*np.pi)
#         return {'u_phi_linfree': (self._dist.cdf(2*phi_linfree)
#                                   + self._dist.cdf(2*phi_linfree - self._dist.b + self._dist.a)
#                                  ) / 2}

#     def lnprior(self, u_phi_linfree):
#         return -self._dist.logpdf((self.transform(u_phi_linfree)['phi_linfree']*2)%(2*np.pi))

#     def get_init_dict(self):
#         return self._init_dict

#     @classmethod
#     def kwargs_from_posterior(cls, posterior, phase_fudge=3., **_):
#         loc, scale = cls.get_loc_scale(posterior)
#         if np.abs(loc) > scale:
#             warnings.warn('Phase loc greater than scale! Suspicious')
#         return {'phase_scale': scale * phase_fudge,}

def refine(points, factor: int):
    """
    Given a 1d array of values, return another 1d array with more finely
    spaced points. The returned array contains the points in the
    original array plus more points in between that interpolate
    linearly. The endpoints are the same.

    Parameters
    ----------
    points: 1-d array-like
        Points to interpolate

    factor: int
        By how much to refine. Modulo edge effects, the return array
        will have `factor` points for each input point.
    """
    num = factor * (len(points) - 1) + 1
    return np.interp(np.linspace(0, 1, num),
                     np.linspace(0, 1, len(points)),
                     points)


class BivariateAdaptiveMap:
    """
    Coordinate transformation between (x, y) and (u_x, u_y),
    defined based on P(x, y) with the properties that
        * P(u_x, u_y) = 1
        * 0 < (u_x, u_y) < 1
        * `u_x` is a function of `x` only
    """
    def __init__(self, x_points, y_points, p_xy_mesh, refine_factor=8):
        # Refine P(x, y) with linear interpolation
        p_xy = scipy.interpolate.RectBivariateSpline(x_points, y_points,
                                                     p_xy_mesh, kx=1, ky=1)
        x_points = refine(x_points, refine_factor)
        y_points = refine(y_points, refine_factor)
        p_xy_mesh = p_xy(x_points, y_points) / p_xy.integral(
            *x_points[[0, -1]], *y_points[[0, -1]])

        self.x_points = x_points
        # p_xy_mesh /= scipy.interpolate.RectBivariateSpline(
        #     x_points, y_points, p_xy_mesh, kx=1, ky=1).integral(
        #     *x_points[[0, -1]], *y_points[[0, -1]])

        # P(x)
        p_x_points = scipy.integrate.trapezoid(p_xy_mesh, y_points, axis=1)
        self._x_to_ux, self._ux_to_x = self._get_cdf_and_ppf(
            x_points, p_x_points)
        self._dux_dx = self._x_to_ux.derivative()

        # P(y | x)
        p_y_given_x_mesh = p_xy_mesh / p_x_points[:, np.newaxis]
        p_y_given_x = scipy.interpolate.RectBivariateSpline(
            x_points, y_points, p_y_given_x_mesh, kx=1, ky=1)
        x_midpoints = (x_points[:-1] + x_points[1:]) / 2
        self._y_to_uy_splines, self._uy_to_y_splines = zip(
            *[self._get_cdf_and_ppf(y_points,
                                    p_y_given_x.ev(x_midpoint, y_points))
              for x_midpoint in x_midpoints])
        self._duy_dy = [y_to_uy.derivative()
                        for y_to_uy in self._y_to_uy_splines]

        @np.vectorize
        def _get_uy(x, y):
            return self._y_to_uy_splines[self._spline_index(x)](y)

        @np.vectorize
        def _get_y(x, u_y):
            return self._uy_to_y_splines[self._spline_index(x)](u_y)

        @np.vectorize
        def jacobian(u_x, u_y):
            """"|d(u_x, u_y)/d(x, y)|"""
            x, y = self.uxy_to_xy(u_x, u_y)
            dux_dx = self._dux_dx(x)
            duy_dy = self._duy_dy[self._spline_index(x)](y)
            return dux_dx * duy_dy

        self._get_uy = _get_uy
        self._get_y = _get_y
        self.jacobian = jacobian

    def uxy_to_xy(self, u_x, u_y):
        x = self._ux_to_x(u_x)[()]
        y = self._get_y(x, u_y)[()]
        return x, y

    def xy_to_uxy(self, x, y):
        u_x = self._x_to_ux(x)[()]
        u_y = self._get_uy(x, y)[()]
        return u_x, u_y

    def _spline_index(self, x):
        return np.digitize(x, self.x_points) - 1

    @staticmethod
    def _get_cdf_and_ppf(points, pdf_points):
        if np.any(points[:-1] > points[1:]):
            raise ValueError('`points` should be sorted.')

        if np.any(pdf_points < 0):
            raise ValueError('`pdf_points` should be >= 0.')

        pdf_points /= scipy.integrate.trapezoid(pdf_points, points)

        cdf_points = scipy.interpolate.InterpolatedUnivariateSpline(
            points, pdf_points, k=1, ext='raise').antiderivative()(points)
        cdf = scipy.interpolate.InterpolatedUnivariateSpline(
            points, cdf_points, k=1, ext='raise')
        ppf = scipy.interpolate.InterpolatedUnivariateSpline(
            cdf_points, points, k=1, ext='raise')
        return cdf, ppf


class SkyLocationMap(prior.Prior):
    range_dic = dict.fromkeys(('u_costhetanet', 'u_phinet_hat'), (0, 1))
    standard_params = ['costhetanet', 'phinet_hat']
    def __init__(self, *, posterior, costhetanet_sigmas=8., resolution=128,
                 beta_temperature=.5, min_prob_ratio=1e-4, **kwargs):
        super().__init__(**kwargs)
        posterior = copy.deepcopy(posterior)
        self._init_dict = {'posterior': posterior,
                           'costhetanet_sigmas': costhetanet_sigmas,
                           'resolution': resolution,
                           'beta_temperature': beta_temperature}

        points = {'costheta_jn': np.linspace(-1, 1, 4),
                  'costhetanet': self._get_costhetanet_points(
                      posterior, costhetanet_sigmas, resolution),
                  'phinet_hat': np.linspace(0, 2*np.pi, resolution),
                  'psi': np.linspace(0, np.pi, 5, endpoint=False)
                 }  # Important that fast parameters come later

        mesh_dic = dict(zip(points.keys(),
                            np.meshgrid(*points.values(), indexing='ij')))

        sampled_dic_0 = posterior.prior.inverse_transform(
            **posterior.likelihood.par_dic_0)

        @np.vectorize
        def lnpost(**dic):
            return posterior.lnposterior(**sampled_dic_0 | dic)

        lnpost_mesh = lnpost(**mesh_dic)

        marginalize_axes = tuple(i for i, par in enumerate(points)
                                 if par not in ('costhetanet', 'phinet_hat'))
        prob_mesh = np.maximum(
            np.exp(beta_temperature * (lnpost_mesh - lnpost_mesh.max())
                  ).sum(axis=marginalize_axes),
            min_prob_ratio)
        self._bam = BivariateAdaptiveMap(
            points['costhetanet'], points['phinet_hat'], prob_mesh)

    @staticmethod
    def _get_costhetanet_points(posterior, costhetanet_sigmas, resolution):
        detector_pair = posterior.prior.get_init_dict()['detector_pair']
        if len(detector_pair) == 1:
            return np.linspace(-1, 1, resolution)

        # There are at least two detectors. We can make a better guess
        # for the sky location as a ring in the sky, and add more points
        # around the expected peak. This is just to resolve the peak
        # well when constructing the adaptive map:
        pair_inds = [
            posterior.likelihood.event_data.detector_names.index(det)
            for det in detector_pair]
        lfp = posterior.prior.subpriors[
            posterior.prior.prior_classes.index(
                gw_prior.linear_free.LinearFreePhaseTimePrior)]

        sigma_dt = (np.sqrt(np.sum(1 / lfp._ref['time_weights'][pair_inds]))
                    * lfp._ref['sigma_time'])

        t_delay = np.linalg.norm(
            gw_utils.DETECTORS[detector_pair[0]].location
            - gw_utils.DETECTORS[detector_pair[1]].location) / lal.C_SI

        costhetanet_0 = posterior.prior.inverse_transform(
            **posterior.likelihood.par_dic_0)['costhetanet']
        delta_costhetanet = costhetanet_sigmas * sigma_dt / t_delay

        ring_bounds = np.clip((costhetanet_0 - delta_costhetanet,
                               costhetanet_0 + delta_costhetanet),
                              -1, 1)

        # Instersperse uniform points and points in the ring:
        costhetanet_points = np.unique(np.sort(
            np.r_[np.linspace(-1, 1, resolution//2),
                  np.linspace(*ring_bounds, resolution//2)]))

        return costhetanet_points

    def lnprior(self, u_costhetanet, u_phinet_hat):
        return - np.log(self._bam.jacobian(u_costhetanet, u_phinet_hat))

    def transform(self, u_costhetanet, u_phinet_hat):
        return dict(zip(self.standard_params,
                        self._bam.uxy_to_xy(u_costhetanet, u_phinet_hat)))

    def inverse_transform(self, costhetanet, phinet_hat):
        return dict(zip(self.sampled_params,
                        self._bam.xy_to_uxy(costhetanet, phinet_hat)))

    def get_init_dict(self):
        return self._init_dict


class ExtrinsicMap(prior.CombinedPrior):
    prior_classes = [PsiMap,
                     TimeMap,
                     # PhaseMap,
                     SkyLocationMap]

    @classmethod
    def from_posterior(cls, posterior, **kwargs):
        subprior_kwargs = {'posterior': posterior}
        for prior_class in cls.prior_classes:
            if hasattr(prior_class, 'kwargs_from_posterior'):
                subprior_kwargs = utils.merge_dictionaries_safely(
                    subprior_kwargs,
                    prior_class.kwargs_from_posterior(posterior, **kwargs))
        return cls(**subprior_kwargs | kwargs)




# # --------------------------------------------------------
# # Phase integral

# h_fbin = self.waveform_generator.get_strain_at_detectors(
#     self.fbin, par_dic | {'d_luminosity': 1.}, by_m=True)

# d_h = (self._d_h_weights * h_fbin.conj()).sum(axis=(2))  # Sum m & f

# m_inds, mprime_inds = self._get_m_mprime_inds()
# h_h = ((self._h_h_weights * h_fbin[m_inds] * h_fbin[mprime_inds].conj()
#        ).real.sum(axis=(2)))