"""
Implementation of the relative binning algorithm for fast likelihood
evaluation.

Valid for frequency domain waveforms, if these have higher harmonics
these must be given in the coprecessing frame.
Splines of arbitrary degree are used to interpolate the ratio between an
arbitrary waveform and a reference waveform.

Two classes are provided: ``BaseRelativeBinningLikelihood`` and
``RelativeBinningLikelihood``:
``BaseRelativeBinningLikelihood`` is an abstract class, intended for
subclassing. It provides the infrastructure to set up the
relative-binning frequency bins and basis splines but does not make
assumptions about which waveform is used to heterodyne the data or how.
This is done by the method ``_set_summary`` which the subclass must
provide.
``RelativeBinningLikelihood`` is one such concrete subclass. Its method
``lnlike`` computes the log likelihood using relative binning.
"""
import warnings
from functools import wraps
from abc import ABC, abstractmethod
import numpy as np
import scipy.interpolate
import scipy.sparse

from cogwheel import gw_utils
from cogwheel import utils
from cogwheel import waveform
from .likelihood import CBCLikelihood, check_bounds


class BaseRelativeBinning(CBCLikelihood, ABC):
    """
    Abstract class that implements general relative binning methods.
    Concrete classes need to specify how they construct their summary
    data.
    """
    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None, spline_degree=3):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`

        waveform_generator: Instance of `waveform.WaveformGenerator`.

        par_dic_0: dict
            Parameters of the reference waveform, should be close to the
            maximum likelihood waveform.
            Keys should match ``self.waveform_generator.params``.

        fbin: 1-d array or None
            Array with edges of the frequency bins used for relative
            binning [Hz]. Alternatively, pass `pn_phase_tol`.

        pn_phase_tol: float or None
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins. Alternatively, pass `fbin`.

        spline_degree: int
            Degree of the spline used to interpolate the ratio between
            waveform and reference waveform for relative binning.
        """
        if (fbin is None) == (pn_phase_tol is None):
            raise ValueError('Pass exactly one of `fbin` or `pn_phase_tol`.')

        super().__init__(event_data, waveform_generator)

        self._coefficients = None  # Set by ``._set_splines``
        self._basis_splines = None  # Set by ``._set_splines``

        self._spline_degree = spline_degree

        # Backward compatibility fix, shouldn't happen in new code:
        if ({'s1x_n', 's1y_n', 's2x_n', 's2y_n'}.isdisjoint(par_dic_0.keys())
                and {'s1x_n', 's1y_n', 's2x_n', 's2y_n'} <= set(self.params)
                and ({'s1x', 's1y', 's2x', 's2y', 'phi_ref'}
                     <= par_dic_0.keys())):
            waveform.inplane_spins_xy_to_xy_n(par_dic_0)
            for key in 's1x', 's1y', 's2x', 's2y':
                del par_dic_0[key]

        self._par_dic_0 = par_dic_0

        if pn_phase_tol:
            self.pn_phase_tol = pn_phase_tol
        else:
            self.fbin = fbin

    @abstractmethod
    def _set_summary(self):
        """
        Compute summary data, and update ``.asd_drift`` using the
        reference waveform ``.par_dic_0``.
        This template computes ``.asd_drift``, subclasses should expand
        it to compute the summary data.
        """
        self.asd_drift = self.compute_asd_drift(self.par_dic_0)

    @property
    def pn_phase_tol(self):
        """
        Tolerance in the post-Newtonian phase [rad] used for defining
        frequency bins.
        Setting this will recompute frequency bins such that across each
        frequency bin the change in the post-Newtonian waveform phase
        with respect to the fiducial waveform is bounded by
        `pn_phase_tol` [rad].
        """
        return self._pn_phase_tol

    @pn_phase_tol.setter
    def pn_phase_tol(self, pn_phase_tol):
        pn_exponents = [-5/3, -2/3, 1]
        if waveform.APPROXIMANTS[self.waveform_generator.approximant].tides:
            pn_exponents.append(5/3)
        pn_exponents = np.array(pn_exponents)

        pn_coeff_rng = 2*np.pi / np.abs(np.subtract(
            *self.event_data.fbounds[:, np.newaxis] ** pn_exponents))

        f_arr = np.linspace(*self.event_data.fbounds, 10000)

        diff_phase = np.sum([np.sign(exp) * rng * f_arr**exp
                             for rng, exp in zip(pn_coeff_rng, pn_exponents)],
                            axis=0)
        diff_phase -= diff_phase[0]  # Worst case scenario differential phase

        # Construct frequency bins
        nbin = np.ceil(diff_phase[-1] / pn_phase_tol).astype(int)
        diff_phase_arr = np.linspace(0, diff_phase[-1], nbin + 1)
        self.fbin = np.interp(diff_phase_arr, diff_phase, f_arr)
        self._pn_phase_tol = pn_phase_tol

    @property
    def fbin(self):
        """
        Edges of the frequency bins for relative binning [Hz].
        Setting this will automatically round them to fall in the FFT
        array, recompute the splines and summary data, and set
        ``._pn_phase_tol`` to ``None`` to keep logs clean.
        """
        return self._fbin

    @fbin.setter
    def fbin(self, fbin):
        fbin_ind = np.unique(np.searchsorted(self.event_data.frequencies,
                                             fbin - self.event_data.df/2))
        self._fbin = self.event_data.frequencies[fbin_ind]  # Bin edges

        self._set_splines()
        self._set_summary()
        self._pn_phase_tol = None  # Erase potentially outdated information

    @property
    def par_dic_0(self):
        """Dictionary with reference waveform parameters."""
        return self._par_dic_0

    @par_dic_0.setter
    def par_dic_0(self, par_dic_0):
        self._par_dic_0 = par_dic_0
        self._set_summary()

    @property
    def spline_degree(self):
        """
        Integer between 1 and 5, degree of the spline used to
        interpolate waveform ratios. Editing it will automatically
        recompute the splines and summary data.
        """
        return self._spline_degree

    @spline_degree.setter
    def spline_degree(self, spline_degree):
        self._spline_degree = spline_degree
        self._set_splines()
        self._set_summary()

    def _set_splines(self):
        """
        Set attributes `_basis_splines` and `_coefficients`.
        `_basis_splines` is a sparse array of shape `(nbin, nrfft)`
        whose rows are the B-spline basis elements for `fbin` evaluated
        on the FFT grid.
        `_coefficients` is an array of shape `(nbin, nbin)` whose i-th
        row are the B-spline coefficients for a spline that interpolates
        an array of zeros with a one in the i-th place, on `fbin`.
        In other words, `_coefficients @ _basis_splines` is an array of
        shape `(nbin, nrfft)` whose i-th row is a spline that
        interpolates on `fbin` an array of zeros with a one in the i-th
        place; this spline is evaluated on the RFFT grid.
        """
        nbin = len(self.fbin)
        coefficients = np.empty((nbin, nbin))
        for i_bin, y_points in enumerate(np.eye(nbin)):
            # Note knots depend on fbin only, they're always the same
            knots, coeffs, _ = scipy.interpolate.splrep(
                self.fbin, y_points, s=0, k=self.spline_degree)
            coefficients[i_bin] = coeffs[:nbin]
        self._coefficients = coefficients

        nrfft = len(self.event_data.frequencies)
        basis_splines = scipy.sparse.lil_matrix((nbin, nrfft))

        frequencies = self.event_data.frequencies.copy()
        frequencies[-1] -= 1e-10  # Or last basis_element evaluates to 0

        for i_bin in range(nbin):
            element_knots = knots[i_bin : i_bin + self.spline_degree + 2]
            basis_element = scipy.interpolate.BSpline.basis_element(
                element_knots)
            i_start, i_end = np.searchsorted(frequencies,
                                             element_knots[[0, -1]], 'right')
            basis_splines[i_bin, i_start : i_end] = basis_element(
                frequencies[i_start : i_end])

        self._basis_splines = basis_splines.tocsr()

    def _get_summary_weights(self, integrand):
        """
        Return summary data to compute efficiently integrals of the form
            4 integral g(f) r(f) df,
        where r(f) is a smooth function.
        The above integral is approximated by
            summary_weights * r(fbin)
        which is the exact result of replacing `r(f)` by a spline that
        interpolates it at `fbin`.

        Parameters
        ----------
        integrand: array of shape (..., nrfft)
            g(f) in the above notation (the oscillatory part of the
            integrand), array whose last axis corresponds to the FFT
            frequency grid.

        Return
        ------
        summary_weights: array of shape (..., nbin)
            array shaped like `integrand` except the last axis now
            correponds to the frequency bins.
        """
        # Broadcast manually
        *pre_shape, nrfft = integrand.shape
        shape = pre_shape + [len(self.fbin)]
        projected_integrand = np.zeros(shape, dtype=np.complex128)
        for i, arr_f in enumerate(integrand.reshape(-1, nrfft)):
            projected_integrand[np.unravel_index(i, pre_shape)] \
                = self._basis_splines @ arr_f

        return (4 * self.event_data.df
                * projected_integrand.dot(self._coefficients.T))

    def _stall_ringdown(self, h0_f, h0_fbin):
        """
        Identify a cutoff frequency and set `h0_f` and `h0_fbin` to a
        nonzero constant above that frequency, inplace.
        Useful for preventing the denominator from going to zero in
        relative binning.

        `h0_f` and `h0_fbin` must have frequencies in their last axis,
        and their shape must match in the remaining dimensions.
        """
        assert h0_f.shape[:-1] == h0_fbin.shape[:-1]
        assert h0_f.shape[-1] == len(self.event_data.frequencies)
        assert h0_fbin.shape[-1] == len(self.fbin)

        wht_filter = np.linalg.norm(self.event_data.wht_filter, axis=0)

        h0_f = np.atleast_2d(h0_f)
        h0_fbin = np.atleast_2d(h0_fbin)
        for i in np.ndindex(h0_f.shape[:-1]):  # Work with 1-d arrays
            cumsnr2 = np.cumsum((np.abs(h0_f[i]) * wht_filter)**2)
            cumsnr2 /= cumsnr2[-1]
            i_99 = np.searchsorted(cumsnr2, .99)
            f_99 = self.event_data.frequencies[i_99]
            ibin_99 = np.searchsorted(self.fbin, f_99)

            f_start = self.fbin[ibin_99 - self.spline_degree]  # Start fade
            f_end = self.fbin[ibin_99]  # End fade

            def fadeout(f, f_start=f_start, f_end=f_end):
                return np.cos(np.clip((f-f_start) / (f_end-f_start), 0, 1)
                              * np.pi / 2) ** 2

            # Replace h by a constant above f_end
            fadeout_f = fadeout(self.event_data.frequencies)
            fadeout_fbin = fadeout(self.fbin)

            h0_f[i][:] = (h0_f[i] * fadeout_f
                          + h0_fbin[i][ibin_99] * (1-fadeout_f))
            h0_fbin[i][:] = (h0_fbin[i] * fadeout_fbin
                             + h0_fbin[i][ibin_99] * (1-fadeout_fbin))

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return super().get_init_dict() | ({'fbin': None} if self.pn_phase_tol
                                          else {})

    @classmethod
    def from_reference_waveform_finder(
            cls, reference_waveform_finder, approximant,
            fbin=None, pn_phase_tol=.05, spline_degree=3, **kwargs):
        """
        Instantiate with help from a `ReferenceWaveformFinder` instance,
        which provides `waveform_generator`, `event_data` and
        `par_dic_0` objects.

        Parameters
        ----------
        reference_waveform_finder: Instance of
                ``cogwheel.likelihood.ReferenceWaveformFinder``.

        approximant: str
            Approximant name.

        fbin: 1-d array or None
            Array with edges of the frequency bins used for relative
            binning [Hz]. Alternatively, pass `pn_phase_tol`.

        pn_phase_tol: float or None
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins. Alternatively, pass `fbin`.

        spline_degree: int
            Degree of the spline used to interpolate the ratio between
            waveform and reference waveform for relative binning.

        **kwargs:
            Keyword arguments, in case a subclass needs them.

        Return
        ------
        Instance of ``cls``.
        """
        waveform_generator = reference_waveform_finder.waveform_generator \
            .reinstantiate(approximant=approximant, harmonic_modes=None,
                           lalsimulation_commands=())

        return cls(event_data=reference_waveform_finder.event_data,
                   waveform_generator=waveform_generator,
                   par_dic_0=reference_waveform_finder.par_dic_0,
                   fbin=fbin,
                   pn_phase_tol=pn_phase_tol,
                   spline_degree=spline_degree,
                   **kwargs)

    def lnlike(self, par_dic):
        """
        Return log-likelihood (float).
        Mainly for backwards compatibility.
        """
        return self.lnlike_and_metadata(par_dic)[0]

    @abstractmethod
    def lnlike_and_metadata(self, par_dic) -> tuple[float, object]:
        """
        Return
        ------
        lnl: float
            Log likelihood.

        metadata: object
            Arbitrary object that stores information to postprocess the
            posterior samples. See ``.get_blob``.
        """

    def get_blob(self, metadata) -> dict:
        """
        Return dictionary of ancillary information ("blob").
        The sampler will use this to create extra columns for the
        posterior samples.

        metadata: object
            Arbitrary object that stores information to postprocess the
            posterior samples. Second output of ``lnlike_and_metadata``.
        """
        del metadata
        return {}


class BaseLinearFree(BaseRelativeBinning):
    """
    Define a method ``._get_linearfree_hplus_hcross_dt`` to
    compute linear-free waveforms and the timeshift required
    to go back to the approximant's convention.
    Experimental class. For now the convention is that only
    a timeshift is applied, no phase shift, but perhaps this
    might change.
    """
    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None, spline_degree=3):
        self._h2plus0_fbin = None  # Set by ._set_summary
        self._polyfit_weights = None  # Set by ._set_summary
        super().__init__(event_data, waveform_generator, par_dic_0,
                         fbin, pn_phase_tol, spline_degree)

    def _set_summary(self):
        super()._set_summary()

        with utils.temporarily_change_attributes(self.waveform_generator,
                                                 disable_precession=False):
            m2_index = list(self.waveform_generator.m_arr).index(2)

            h2plus0_f = np.zeros(len(self.event_data.frequencies),
                                 dtype=np.complex128)
            h2plus0_f[self.event_data.fslice] \
                = self.waveform_generator.get_hplus_hcross(
                    self.event_data.frequencies[self.event_data.fslice],
                    self.par_dic_0, by_m=True)[m2_index, 0]

            h2plus0_fbin = self.waveform_generator.get_hplus_hcross(
                self.fbin, self.par_dic_0, by_m=True)[m2_index, 0]
        self._h2plus0_fbin = h2plus0_fbin.copy()

        self._stall_ringdown(h2plus0_f, self._h2plus0_fbin)

        weights_f = (np.abs(h2plus0_f) * self.event_data.wht_filter)**2
        self._polyfit_weights = np.sqrt(np.clip(
            self._get_summary_weights(weights_f.sum(0)).real, 0, None))

    def _get_linearfree_hplus_hcross_dt(self, waveform_par_dic, by_m=False):
        """
        Return a linear-free waveform at relative-binning (coarse)
        frequency resolution, and the timeshift to go back to the
        original convention.
        Linear-free here means that the (2, 2) hplus has been time-
        aligned to the reference waveform.

        Parameters
        ----------
        waveform_par_dic: dict
            Parameters per ``.waveform_generator.waveform_params``.
            Any extra keys would be silently ignored.

        by_m: bool
            Wheter to return the waveform mode-by-mode.

        Return
        ------
        h_mpb: array of shape (m?, 2, len(.fbin))
            + and x polarizations at the relative-binning frequencies,
            with a timeshift applied so that waveforms with different
            parameters would be aligned in time (Hz^-1).

        dt_linearfree: float
            Time shift to go back to the approximant's convention (s).
        """
        h_mpb = self.waveform_generator.get_hplus_hcross(
            self.fbin, waveform_par_dic, by_m=True)
        m2_index = list(self.waveform_generator.m_arr).index(2)
        h2plus_fbin = h_mpb[m2_index, 0]
        hplus_ratio = h2plus_fbin / self._h2plus0_fbin
        dphase = np.unwrap(np.angle(hplus_ratio))
        fit = np.polynomial.Polynomial.fit(
            self.fbin, dphase, deg=1,
            w=self._polyfit_weights * np.sqrt(np.abs(hplus_ratio)))

        timeshift = - fit.deriv()(0) / (2*np.pi)
        h_mpb *= np.exp(2j*np.pi * timeshift * self.fbin)

        if by_m:
            return h_mpb, timeshift
        return h_mpb.sum(axis=0), timeshift


class RelativeBinningLikelihood(BaseRelativeBinning):
    """
    Generalization of ``CBCLikelihood`` that implements computation of
    likelihood with the relative binning method (fast).

    Subclassed by ``ReferenceWaveformFinder``.
    """
    _FIDUCIAL_CONFIGURATION = {'d_luminosity': 1.,
                               'phi_ref': 0.}

    @wraps(BaseRelativeBinning.__init__)
    def __init__(self, *args, **kwargs):
        self._h0_f = None  # Set by _set_summary()
        self._h0_fbin = None  # Set by _set_summary()
        self._d_h_weights = None  # Set by _set_summary()
        self._h_h_weights = None  # Set by _set_summary()

        super().__init__(*args, **kwargs)

    @check_bounds
    def lnlike_and_metadata(self, par_dic):
        """Return log likelihood using relative binning."""
        lnl = self.lnlike_detectors_no_asd_drift(par_dic) @ self.asd_drift**-2
        return lnl, {'lnl': lnl}

    def get_blob(self, metadata):
        """
        Return dictionary of ancillary information ("blob"). This will
        be appended to the posterior samples as extra columns.
        """
        return metadata

    def lnlike_detectors_no_asd_drift(self, par_dic):
        """
        Return an array of length n_detectors with the values of
        `(d|h) - (h|h)/2`, no ASD-drift correction applied, using
        relative binning.

        Parameters
        ----------
        par_dic: dict
            Waveform parameters, keys should match ``self.params``.
        """
        d_h, h_h = self._get_dh_hh_no_asd_drift(par_dic)
        return d_h - h_h/2

    def _get_dh_hh_no_asd_drift(self, par_dic):
        """
        Return two arrays of length n_detectors with the values of
        ``(d|h)``, ``(h|h)``, no ASD-drift correction applied, using
        relative binning.

        Parameters
        ----------
        par_dic: dict
            Waveform parameters, keys should match ``self.params``.
        """
        # Pass fiducial configuration to hit cache often:
        d_h, h_h = self._get_dh_hh_complex_no_asd_drift(par_dic) # mpd, mpPd
        return np.sum(d_h.real, axis=(0, 1)), np.sum(h_h.real, axis=(0, 1, 2))

    def _get_dh_hh_complex_no_asd_drift(self, par_dic):
        """
        Return two arrays complex with the values of
        ``(d|h)`` (modes, polarizations, detectors), and
        ``(h|h)`` (mode pairs, polarizations, polarizations, detectors),
        no ASD-drift correction applied, using relative binning.

        Parameters
        ----------
        par_dic: dict
            Waveform parameters, keys should match ``self.params``.
        """
        # Pass fiducial configuration to hit cache often:
        dphi = par_dic['phi_ref'] - self._FIDUCIAL_CONFIGURATION['phi_ref']
        amp_ratio = (self._FIDUCIAL_CONFIGURATION['d_luminosity']
                     / par_dic['d_luminosity'])
        par_dic_fiducial = (
            {par: par_dic[par]
             for par in self.waveform_generator.polarization_params}
            | self._FIDUCIAL_CONFIGURATION)
        d_h_mpd, h_h_mpd = self._get_dh_hh_by_m_polarization_detector(
            tuple(par_dic_fiducial.items()))

        m_arr = self.waveform_generator.m_arr
        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        dh_phasor = np.exp(-1j * dphi * m_arr)
        hh_phasor = np.exp(1j * dphi * (m_arr[m_inds] - m_arr[mprime_inds]))

        # fplus_fcross shape: (2, n_detectors)
        fplus_fcross = gw_utils.fplus_fcross(
            self.waveform_generator.detector_names,
            par_dic['ra'], par_dic['dec'], par_dic['psi'],
            self.waveform_generator.tgps)

        d_h = amp_ratio * np.einsum('mpd, pd, m -> mpd',
                                    d_h_mpd, fplus_fcross, dh_phasor)
        h_h = amp_ratio**2 * np.einsum('mpPd, pd, Pd, m -> mpPd',
                                       h_h_mpd, fplus_fcross,
                                       fplus_fcross, hh_phasor)
        return d_h, h_h

    @utils.lru_cache(maxsize=16)
    def _get_dh_hh_by_m_polarization_detector(self, par_dic_items):
        """
        Return ``d_h_0`` and ``h_h_0``, complex inner products for a
        waveform ``h`` by azimuthal mode ``m`` and polarization (plus
        and cross). No ASD-drift correction is applied.
        Useful for reusing computations when only ``psi``, ``phi_ref``
        and/or ``d_luminosity`` change, as these parameters only affect
        ``h`` by a scalar factor dependent on m and polarization.

        Parameters
        ----------
        par_dic_items: tuple of (key, value) tuples
            Contents of ``par_dic``, in tuple format so it's hashable.

        Return
        ------
        d_h: (n_m, 2) array
            ``(d|h_mp)`` complex inner product, where ``d`` is data and
            ``h_mp`` is the waveform with co-precessing azimuthal mode
            ``m`` and polarization ``p`` (plus or cross).

        h_h: (n_m*(n_m+1)/2, 2, 2, n_detectors) array
            ``(h_mp|h_m'p')`` complex inner product.

        """
        par_dic = dict(par_dic_items)
        hplus_hcross_at_detectors \
            = self.waveform_generator.get_hplus_hcross_at_detectors(
                self.fbin, par_dic, by_m=True)

        if len(hplus_hcross_at_detectors) != len(self._d_h_weights):
            warnings.warn('Summary data and waveform_generator have '
                          'incompatible number of harmonic modes, recomputing '
                          'the summary data.')
            self.par_dic_0 = self.par_dic_0  # Recomputes summary

        # Shape (n_m, 2, n_detectors), complex
        d_h = np.einsum('mdf, mpdf -> mpd',
                        self._d_h_weights, hplus_hcross_at_detectors.conj())

        # Shape (n_m, 2, 2, n_detectors), complex
        m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
        h_h = np.einsum('mdf, mpdf, mPdf -> mpPd', self._h_h_weights,
                        hplus_hcross_at_detectors[m_inds],
                        hplus_hcross_at_detectors.conj()[mprime_inds])
        return d_h, h_h

    def _set_summary(self):
        """
        Compute summary data for the fiducial waveform at all detectors.
        `asd_drift` is not applied to the summary data, to not have to
        keep track of it.
        Update `asd_drift` using the reference waveform.
        The summary data `self._d_h_weights` and `self._d_h_weights` are
        such that:
            (d|h) ~= sum(_d_h_weights * conj(h_fbin)) / asd_drift^2
            (h|h) ~= sum(_h_h_weights * abs(h_fbin)^2) / asd_drift^2

        Note: all spin components in `self.par_dic_0` are used, even if
        `self.waveform_generator.disable_precession` is set to `True`.
        This is so that the reference waveform remains the same when
        toggling `disable_precession`.
        """
        super()._set_summary()
        # Don't zero the in-plane spins for the reference waveform
        with utils.temporarily_change_attributes(self.waveform_generator,
                                                 disable_precession=False):

            self._h0_f = self._get_h_f(self.par_dic_0, by_m=True)
            self._h0_fbin = self.waveform_generator.get_strain_at_detectors(
                self.fbin, self.par_dic_0, by_m=True)  # n_m x ndet x len(fbin)

            # Temporarily undo big time shift so waveform is smooth at
            # high frequencies:
            shift_f = np.exp(-2j*np.pi * self.event_data.frequencies
                             * self.event_data.tcoarse)
            shift_fbin = np.exp(-2j*np.pi * self.fbin
                                * self.event_data.tcoarse)
            self._h0_f *= shift_f.conj()
            self._h0_fbin *= shift_fbin.conj()
            # Ensure reference waveform is smooth and !=0 at high frequency:
            self._stall_ringdown(self._h0_f, self._h0_fbin)
            # Reapply big time shift:
            self._h0_f *= shift_f
            self._h0_fbin *= shift_fbin

            d_h0 = self.event_data.blued_strain * self._h0_f.conj()
            self._d_h_weights = (self._get_summary_weights(d_h0)
                                 / np.conj(self._h0_fbin))

            m_inds, mprime_inds = self.waveform_generator.get_m_mprime_inds()
            h0m_h0mprime = (self._h0_f[m_inds] * self._h0_f[mprime_inds].conj()
                            * self.event_data.wht_filter ** 2)
            self._h_h_weights = (self._get_summary_weights(h0m_h0mprime)
                                 / (self._h0_fbin[m_inds]
                                    * self._h0_fbin[mprime_inds].conj()))
            # Count off-diagonal terms twice:
            self._h_h_weights[~np.equal(m_inds, mprime_inds)] *= 2

    def _get_h_f_interpolated(self, par_dic, *, normalize=False,
                              by_m=False):
        """
        Fast approximation to `_get_h_f`.
        Return (ndet, nfreq) array with waveform strain at detectors
        evaluated on the FFT frequency grid and zeroized outside
        `(fmin, fmax)`, computed using relative binning from a low
        frequency resolution waveform.
        """
        h_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)

        ratio = scipy.interpolate.interp1d(
            self.fbin, h_fbin / self._h0_fbin, assume_sorted=True,
            kind=self.spline_degree, bounds_error=False, fill_value=0.
            )(self.event_data.frequencies)

        h_f = ratio * self._h0_f

        if normalize:
            h_f /= np.sqrt(self._compute_h_h(h_f))[..., np.newaxis]

        if by_m:
            return h_f
        return np.sum(h_f, axis=0)
