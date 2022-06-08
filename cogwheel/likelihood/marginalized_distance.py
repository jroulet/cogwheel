"""
Provide classes ``LookupTable`` and ``MarginalizedDistanceLikelihood``
to sample using a likelihood marginalized over distance.
"""

from scipy.integrate import quad
from scipy.interpolate import (RectBivariateSpline,
                               InterpolatedUnivariateSpline)
import numpy as np

from cogwheel.likelihood import RelativeBinningLikelihood
from cogwheel import utils


D_LUMINOSITY_MAX = 1.5e4  # Default distance integration limit (Mpc)


def euclidean_distance_prior(d_luminosity):
    """
    Distance prior uniform in luminosity volume, normalized so that
    its integral is the luminosity volume in Mpc^3.
    Note: no maximum is enforced here.
    """
    return d_luminosity**2 / 3


# Dictionary of luminosity distance priors. Its values are functions of
# the luminosity distance in Mpc. Its keys are strings that can be
# passed to `LookupTable` instances.
d_luminosity_priors = {'euclidean': euclidean_distance_prior}


class LookupTable(utils.JSONMixin):
    """
    Auxiliary class to marginalize the likelihood over distance.
    The instances are callable, and use interpolation to compute
    ``log(evidence) - d_h**2 / h_h / 2``, where``evidence`` is the value
    of the likelihood marginalized over distance.
    The interpolation is done in some coordinates `x`, `y` in which the
    function is smooth (see ``_get_x_y``, ``get_dh_hh``).
    """

    REFERENCE_DISTANCE = 1.  # Luminosity distance at which h is defined (Mpc).
    _Z0 = 10.  # Typical SNR of events.
    _SIGMAS = 10.  # How far out the tail of the distribution to tabulate.

    def __init__(self, d_luminosity_prior_name: str = 'euclidean',
                 d_luminosity_max=D_LUMINOSITY_MAX,
                 shape=(256, 128), table=None):
        """
        Construct the interpolation table.

        Parameters
        ----------
        d_luminosity_prior: string, a key in `d_luminosity_priors`.
        d_luminosity_max: Maximum luminosity distance (Mpc).
        shape: (int, int), number of interpolating points in x and y.
        """
        self.d_luminosity_prior_name = d_luminosity_prior_name
        self.d_luminosity_prior = d_luminosity_priors[
            d_luminosity_prior_name]
        self.d_luminosity_max = d_luminosity_max
        self.shape = shape

        x_arr = np.linspace(-self._SIGMAS, 0, shape[0])
        y_arr = np.linspace(self._compactify(- self._SIGMAS / self._Z0),
                            1 - 1e-8, shape[1])
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

        dh_grid, hh_grid = self._get_dh_hh(x_grid, y_grid)

        self.table = table if table else np.vectorize(self._function)(dh_grid, hh_grid)

        self._interpolated_table = RectBivariateSpline(x_arr, y_arr, self.table)
        self.tabulated = {'x': x_grid,
                          'y': y_grid,
                          'd_h': dh_grid,
                          'h_h': hh_grid,
                          'function': self.table}  # Bookkeeping, not used.

    def __call__(self, d_h, h_h):
        """
        Return ``log(evidence) - d_h**2 / h_h / 2``, where``evidence``
        is the value of the likelihood marginalized over distance.
        This uses interpolation from a precomputed table.

        Parameters
        ----------
        d_h, h_h: float
            Inner products (d|h), (h|h) where `d` is data and `h` is the
            model strain at a fiducial distance _REFERENCE_DISTANCE.
            These are scalars (detectors are summed over). A real part
            is taken in (d|h), not an absolute value (phase is not
            marginalized over so the computation is robust to higher
            modes).
        """
        return self._interpolated_table(*self._get_x_y(d_h, h_h),
                                        grid=False)[()]

    def _get_distance_bounds(self, d_h, h_h, sigmas=5.):
        """
        Return ``(d_min, d_max)`` pair of luminosity distance bounds to
        the distribution at the ``sigmas`` level.
        Let ``u = REFERENCE_DISTANCE / d_luminosity``, the likelihood is
        Gaussian in ``u``. This function returns the luminosity
        distances corresponding to ``u`` +/- `sigmas` deviations away
        from the maximum.
        Note: this can return negative values for the distance. This
        behavior is intentional. These may be interpreted as infinity.
        """
        u_peak = d_h / (self.REFERENCE_DISTANCE * h_h)
        delta_u = sigmas / np.sqrt(h_h)
        return np.array([self.REFERENCE_DISTANCE / (u_peak + delta_u),
                         self.REFERENCE_DISTANCE / (u_peak - delta_u)])

    def sample_distance(self, d_h, h_h, num=None, resolution=256):
        """
        Return samples from the luminosity distance distribution given
        the inner products (d|h), (h|h) of a waveform at distance
        ``REFERENCE_DISTANCE``.

        Parameters
        ----------
        d_h: float
            Inner product (summed over detectors) between data and
            waveform at ``self.REFERENCE_DISTANCE``.

        h_h: float
            Inner product (summed over detectors) of waveform at
            ``self.REFERENCE_DISTANCE`` with itself.

        num: int or None
            How many samples to generate. ``None`` (default) generates a
            single (scalar) sample.

        resolution: int
            How finely to interpolate the distance distribution when
            generating samples.
        """
        u_bounds = 1 / self._get_distance_bounds(d_h, h_h, sigmas=7.)
        focused_grid = 1 / np.linspace(*u_bounds, resolution)
        focused_grid = focused_grid[(focused_grid > 0)
                                    & (focused_grid < self.d_luminosity_max)]
        broad_grid = np.linspace(0, self.d_luminosity_max, resolution)[1:]
        distances = np.sort(np.concatenate([broad_grid, focused_grid]))
        posterior = self._function_integrand(distances, d_h, h_h)
        cumulative = InterpolatedUnivariateSpline(distances, posterior, k=1
                                                 ).antiderivative()(distances)
        return np.interp(np.random.uniform(0, cumulative[-1], num),
                         cumulative, distances)

    def _function(self, d_h, h_h):
        """
        Function to interpolate with the aid of a lookup table.
        Return ``log(evidence) - overlap**2 / 2``, where ``evidence``
        is the value of the likelihood marginalized over distance.
        """
        return np.log(quad(self._function_integrand, 0, self.d_luminosity_max,
                           args=(d_h, h_h),
                           points=self._get_distance_bounds(d_h, h_h)
                           )[0]
                      + 1e-100)

    def _function_integrand(self, d_luminosity, d_h, h_h):
        """
        Proportional to the distance posterior. The log of the integral
        of this function is stored in the lookup table.
        """
        norm_h = np.sqrt(h_h)
        return (self.d_luminosity_prior(d_luminosity)
                * np.exp(-(norm_h * self.REFERENCE_DISTANCE / d_luminosity
                           - d_h / norm_h)**2 / 2))

    def _get_x_y(self, d_h, h_h):
        """
        Interpolation coordinates (x, y) in which the function to
        interpolate is smooth, as a function of the inner products
        (d|h), (h|h).
        Inverse of ``_get_dh_hh``.
        """
        norm_h = np.sqrt(h_h)
        overlap = d_h / norm_h
        x = np.log(norm_h / (self.d_luminosity_max
                              * (self._SIGMAS + np.abs(overlap))))
        y = self._compactify(overlap / self._Z0)
        return x, y

    def _get_dh_hh(self, x, y):
        """
        Inner products (d|h), (h|h), as a function of the interpolation
        coordinates (x, y) in which the function to interpolate is
        smooth.
        Inverse of ``_get_x_y``.
        """
        overlap = self._uncompactify(y) * self._Z0
        norm_h = (np.exp(x) * self.d_luminosity_max
                  * (self._SIGMAS + np.abs(overlap)))
        d_h = overlap * norm_h
        h_h = norm_h**2
        return d_h, h_h

    @staticmethod
    def _compactify(value):
        """Monotonic function from (-inf, inf) to (-1, 1)."""
        return value / (1 + np.abs(value))

    @staticmethod
    def _uncompactify(value):
        """
        Inverse of _compactify. Monotonic function from (-1, 1) to (-inf, inf).
        """
        return value / (1 - np.abs(value))


class MarginalizedDistanceLikelihood(RelativeBinningLikelihood):
    """
    Modified `RelativeBinningLikelihood` that uses a likelihood
    function marginalized semi-analytically over distance.
    Thus, it removes one dimension from the parameter space.
    """
    def __init__(self, lookup_table: LookupTable, event_data,
                 waveform_generator, par_dic_0, fbin=None,
                 pn_phase_tol=None, spline_degree=3):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`
        waveform_generator: Instance of `waveform.WaveformGenerator`.
        par_dic_0: dictionary with parameters of the reference waveform,
                   should be close to the maximum likelihood waveform.
        fbin: Array with edges of the frequency bins used for relative
              binning [Hz]. Alternatively, pass `pn_phase_tol`.
        pn_phase_tol: Tolerance in the post-Newtonian phase [rad] used
                      for defining frequency bins. Alternatively, pass
                      `fbin`.
        spline_degree: int, degree of the spline used to interpolate the
                       ratio between waveform and reference waveform for
                       relative binning.
        lookup_table: Instance of ``LookupTable`` to compute the
                      marginalized likelihood.
        """

        super().__init__(event_data, waveform_generator, par_dic_0, fbin,
                         pn_phase_tol, spline_degree)

        self.lookup_table = lookup_table

    @property
    def params(self):
        """
        Parameters expected in `par_dic` for likelihood evaluations.
        """
        return sorted(set(self.waveform_generator.params) - {'d_luminosity'})

    def lnlike(self, par_dic):
        """
        Return log likelihood, marginalized over distance, using
        relative binning.

        Note: the marginalized likelihood depends on two parameters,
        ``(d|h)`` and ``(h|h)``. The dependence is recast in terms of
        ``(d|h)**2 / (h|h) := overlap**2`` and ``(h|h)``
        """
        dh_hh = self._get_dh_hh_no_asd_drift(
            dict(par_dic) | {'d_luminosity': self.lookup_table.REFERENCE_DISTANCE})

        d_h, h_h = np.matmul(dh_hh, self.asd_drift**-2)

        return self.lookup_table(d_h, h_h) + d_h**2 / h_h / 2

    def postprocess_samples(self, samples):
        """
        Add a column 'd_luminosity' to a DataFrame of samples, with values taken
        randomly from the conditional posterior.
        `samples` needs to have columns for all `self.params`.
        """
        @np.vectorize
        def sample_distance(**par_dic):
            dh_hh = self._get_dh_hh_no_asd_drift(
                par_dic | {'d_luminosity': self.lookup_table.REFERENCE_DISTANCE})

            d_h, h_h = np.matmul(dh_hh, self.asd_drift**-2)
            return self.lookup_table.sample_distance(d_h, h_h)
        samples['d_luminosity'] = sample_distance(**samples[self.params])
