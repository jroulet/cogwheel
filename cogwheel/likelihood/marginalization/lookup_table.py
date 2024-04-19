"""
Provide class ``LookupTable`` to marginalize the likelihood over
distance; and ``LookupTableMarginalizedPhase22`` to marginalize the
likelihood over both distance and phase for (l, |m|) = (2, 2) waveforms.
"""
from pathlib import Path
from functools import wraps
import textwrap
import warnings
import numpy as np
import scipy.special
import scipy.stats
from scipy.integrate import quad
from scipy.interpolate import (RectBivariateSpline,
                               InterpolatedUnivariateSpline)

from cogwheel import utils
from cogwheel import cosmology

LOOKUP_TABLES_FNAME = Path(__file__).parent/'lookup_tables.npz'
D_LUMINOSITY_MAX = 1.5e4  # Default distance integration limit (Mpc)

# Refactored code, ``LOOKUP_TABLES_FNAME`` moved to a new location:
_OLD_LOOKUP_TABLES_FNAME = Path(__file__).parent.parent/'lookup_tables.npz'
if _OLD_LOOKUP_TABLES_FNAME.exists():
    _OLD_LOOKUP_TABLES_FNAME.rename(LOOKUP_TABLES_FNAME)


_VERSION = 1
_VERSION_KEY = 'version'
_VERSION_WARNING = textwrap.dedent(f"""
    Clearing cache to switch to new version {_VERSION} of marginalized
    likelihood over distance. New values might not reproduce old results.

    Changelog
    ---------
    version 1:
        Due to a bug and a change in convention, log marginalized likelihoods
        previously computed had a constant offset with respect to the new
        behavior.
        This is not a problem for sampling the posterior. Unless you are using
        the absolute value of the marginalized log likelihood (as opposed to
        log-likelihood differences) it is safe to ignore this.
        Concretely, the euclidean_distance_prior was d**2/3 and it should have
        been 4*pi*d**2. Also, now the table returns the dimensionless evidence
        relative to Gaussian noise, while previously it returned this number
        times the volume up to `D_LUMINOSITY_MAX` in Mpc^3.
    """)


def clear_cache_if_outdated():
    """
    If this file's ``_VERSION`` value is different than it was when the
    lookup tables were cached, then delete the file with the cache, this
    will force recomputing future ``LookupTable``s.
    """
    if LOOKUP_TABLES_FNAME.exists():
        cache = np.load(LOOKUP_TABLES_FNAME)
        if cache.get(_VERSION_KEY, 0) != _VERSION:
            warnings.warn(_VERSION_WARNING)
            LOOKUP_TABLES_FNAME.unlink()


clear_cache_if_outdated()


def euclidean_distance_prior(d_luminosity):
    """
    Distance prior uniform in luminosity volume, normalized so that
    its integral is the luminosity volume in Mpc^3.
    Note: no maximum is enforced here.
    """
    return 4 * np.pi * d_luminosity**2


def comoving_distance_prior(d_luminosity):
    """
    Distance prior uniform in comoving volume-time, normalized so that
    its integral is the comoving volume-time per unit time, in comoving
    Mpc^3.
    Note: no maximum is enforced here.
    """
    return (euclidean_distance_prior(d_luminosity)
            * cosmology.comoving_to_luminosity_diff_vt_ratio(d_luminosity))


# Dictionary of luminosity distance priors. Its values are functions of
# the luminosity distance in Mpc. Its keys are strings that can be
# passed to `LookupTable` instances.
d_luminosity_priors = {'euclidean': euclidean_distance_prior,
                       'comoving': comoving_distance_prior,}


class LookupTable(utils.JSONMixin):
    """
    Auxiliary class to marginalize the likelihood over distance.
    The instances are callable, and use interpolation to compute
    ``log(evidence) - d_h**2 / h_h / 2``, where``evidence`` is the value
    of the likelihood marginalized over distance.
    The interpolation is done in some coordinates `x`, `y` in which the
    function is smooth (see ``_get_x_y``, ``get_dh_hh``).
    """
    marginalized_params = {'d_luminosity'}

    REFERENCE_DISTANCE = 1.  # Luminosity distance at which h is defined (Mpc).
    _Z0 = 10.  # Typical SNR of events.
    _SIGMAS = 10.  # How far out the tail of the distribution to tabulate.
    _rng = np.random.default_rng()

    def __init__(self, d_luminosity_prior_name: str = 'euclidean',
                 d_luminosity_max=D_LUMINOSITY_MAX, shape=(256, 128)):
        """
        Construct the interpolation table.
        If a table with the same settings is found in the file in
        ``LOOKUP_TABLES_FNAME``, it will be loaded for faster
        instantiation. If not, the table will be computed and saved.

        Parameters
        ----------
        d_luminosity_prior: string
            Key in `d_luminosity_priors`.

        d_luminosity_max: float
            Maximum luminosity distance (Mpc).

        shape: (int, int)
            Number of interpolating points in x and y.
        """
        self.d_luminosity_prior_name = d_luminosity_prior_name
        self.d_luminosity_prior = d_luminosity_priors[d_luminosity_prior_name]
        self.d_luminosity_max = d_luminosity_max
        self.shape = shape

        self._inverse_volume = 1 / quad(self.d_luminosity_prior,
                                        0, self.d_luminosity_max)[0]

        x_arr = np.linspace(-self._SIGMAS, 0, shape[0])
        y_arr = np.linspace(self._compactify(- self._SIGMAS / self._Z0),
                            1 - 1e-8, shape[1])
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

        dh_grid, hh_grid = self._get_dh_hh(x_grid, y_grid)

        table = self._get_table(dh_grid, hh_grid)
        self._interpolated_table = RectBivariateSpline(x_arr, y_arr, table)
        self.tabulated = {'x': x_grid,
                          'y': y_grid,
                          'd_h': dh_grid,
                          'h_h': hh_grid,
                          'function': table}  # Bookkeeping, not used.

    def _get_table(self, dh_grid, hh_grid):
        """
        Attempt to load a previously computed table with the requested
        settings. If this is not possible, compute the table and save it
        for faster access in the future.
        Note: if at some point the ``LOOKUP_TABLES_FNAME`` file gets too
        large you are free to delete it.
        """
        load = LOOKUP_TABLES_FNAME.exists()
        lookup_tables = np.load(LOOKUP_TABLES_FNAME) if load else {}

        key = repr(self)
        if key in lookup_tables:
            table = lookup_tables.get(key)
        else:
            table = np.vectorize(self._function)(dh_grid, hh_grid)
            np.savez(LOOKUP_TABLES_FNAME,
                     **{**lookup_tables, key: table, _VERSION_KEY: _VERSION})

        return table

    def __call__(self, d_h, h_h):
        """
        Return ``log(evidence) - d_h**2 / h_h / 2``, where``evidence``
        is the value of the likelihood marginalized over distance.
        This uses interpolation from a precomputed table.

        Parameters
        ----------
        d_h, h_h: float
            Inner products (d|h), (h|h) where `d` is data and `h` is the
            model strain at a fiducial distance REFERENCE_DISTANCE.
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

    def lnlike_marginalized(self, d_h, h_h):
        """
        Parameters
        ----------
        d_h, h_h: float
            Inner products (d|h), (h|h) where `d` is data and `h` is the
            model strain at a fiducial distance REFERENCE_DISTANCE.
            These are scalars (detectors are summed over).
        """
        return self(d_h, h_h) + d_h**2 / h_h / 2

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
        u_bounds = 1 / self._get_distance_bounds(d_h, h_h, sigmas=10.)
        focused_grid = 1 / np.linspace(*u_bounds, resolution)
        focused_grid = focused_grid[(focused_grid > 0)
                                    & (focused_grid < self.d_luminosity_max)]
        broad_grid = np.linspace(0, self.d_luminosity_max, resolution)[1:]
        distances = np.sort(np.concatenate([broad_grid, focused_grid]))
        posterior = self._function_integrand(distances, d_h, h_h)
        cumulative = InterpolatedUnivariateSpline(
            distances, posterior, k=1).antiderivative()(distances)[()]
        return np.interp(self._rng.uniform(0, cumulative[-1], num),
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
        return (self.d_luminosity_prior(d_luminosity) * self._inverse_volume
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
        Inverse of _compactify. Monotonic function from (-1, 1) to
        (-inf, inf).
        """
        return value / (1 - np.abs(value))

    def __repr__(self):
        """
        Return a string of the form `LookupTable(key1=val1, ...)`
        """
        kwargs_str = ", ".join(f'{key}={val!r}'
                               for key, val in self.get_init_dict().items())
        return f'{self.__class__.__name__}({kwargs_str})'


class LookupTableMarginalizedPhase22(LookupTable):
    """
    Similar to ``LookupTable`` except the likelihood is marginalized
    over both distance and phase, assuming quadrupolar radiation
    (actually, just |m|=2 is required, no restriction on l).

    ``d_h`` is now assumed to be the absolute value of the complex (d|h)
    throughout, except in ``sample_phase`` it is the complex (d|h).
    """
    marginalized_params = {'d_luminosity', 'phi_ref'}

    @wraps(LookupTable.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_phase = utils.handle_scalars(
            np.vectorize(self._sample_phase, otypes=[float]))

    def _function_integrand(self, d_luminosity, d_h, h_h):
        """
        Proportional to the distance posterior. The log of the integral
        of this function is stored in the lookup table.
        """
        return (super()._function_integrand(d_luminosity, d_h, h_h)
                * scipy.special.i0e(d_h * self.REFERENCE_DISTANCE
                                    / d_luminosity))

    def _sample_phase(self, d_luminosity, d_h, num=None):
        """
        Return a random value for the orbital phase according to the posterior
        conditioned on all other parameters.

        Parameters
        ----------
        d_luminosity: float
            Luminosity distance of the sample (Mpc).

        d_h: complex
            Complex inner product (d|h) between data and waveform at
            ``self.REFERENCE_DISTANCE``.

        num: int, optional
            How many samples to return, defaults to one.
        """
        if np.isrealobj(d_h):
            raise ValueError('`d_h` expects the complex inner product.')

        waveform_phase = self._rng.vonmises(
            np.angle(d_h),
            np.abs(d_h) * self.REFERENCE_DISTANCE / d_luminosity,
            size=num)
        phi_ref = waveform_phase / 2 + self._rng.choice((0, np.pi), size=num)
        return phi_ref % (2*np.pi)
