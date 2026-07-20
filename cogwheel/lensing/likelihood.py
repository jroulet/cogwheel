"""
Multi-component relative-binning likelihood for microlensed CBC signals.

WHAT
----
`LensedRelativeBinningLikelihood` evaluates the Gaussian log-likelihood of
a Chang--Refsdal microlensed compact-binary signal with the relative
binning (heterodyning) speed-up generalized to the multi-component lensed
waveform.  The engine (`chang_refsdal.ChangRefsdalChannels`) decomposes
the amplification into four topology-stable channels,

    F(w) = sum_a exp(1j * w * tau_a) * K_a(w),

so the candidate lensed detector strain is the ``a``-sum of time-shifted,
kernel-weighted copies of the unlensed strain,

    h_L(f) = sum_a exp(2*pi*1j*f*dt_a) * K_a(w(f)) * h(f),

with ``dt_a = xi * tau_a / (2*pi)`` a frequency-independent relative image
delay (``w = xi*f`` is linear in frequency; `lensing.waveform`).  See
`.claude/spec/lensing_paper` Eqs. (fiducial-component)--(slow-component-
ratio) for the derivation this module implements.

HOW (the three design decisions, locked)
-----------------------------------------
1. Delay-continuous summaries, never a product of summaries.  The rapid
   image-delay phase ``exp(2*pi*1j*f*dt_a)`` multiplies the rapid fiducial
   waveform *inside* the frequency sum.  We keep it analytic: within each
   coarse bin the delay phase is evaluated exactly at the bin center and
   expanded to first order across the bin.  Concretely the summaries are
   the delay-free frequency moments

       A^(p)_{m,d,b} = 4 df sum_{f in b} d(f) conj(h0_{m,d}(f)) (f-f_b)^p
       B^(p)_{mm',d,b} = 4 df sum_{f in b}
                         h0_{m,d}(f) conj(h0_{m',d}(f)) wht_d(f)^2 (f-f_b)^p

   (data moments ``p <= 2``; norm moments ``p <= 3`` -- the ``(h|h)`` term
   costs one extra moment because two component ratios multiply).  The
   candidate delay enters analytically as ``exp(-2*pi*1j*f_b*dt_a)`` per
   bin plus a linear in-bin correction folded into the higher moments.

2. Sequential mode-then-image contraction, additive ``M^2 + n_img^2``.
   Because the amplification is a common factor across harmonic modes and
   the unlensed ratio is common across images, the contraction factorizes:
   the mode structure is reduced first (``M`` or ``M^2`` mode pairs), then
   the image structure (``n_img`` or ``n_img^2`` channel pairs) is
   contracted against the mode-reduced tensors.  The cost is additive, not
   multiplicative, and there are no FFTs and no Python loops over frequency
   on the hot path.

3. Lens-aware bins with a guard.  Bins must be fine enough that the linear
   in-bin delay expansion holds for the largest candidate relative image
   delay ``delta_t_max``: ``pi * Delta_f_bin * delta_t_max < bin_delay_tol``.
   `LensedBinningError` is raised (a named error, never a bare ``assert``
   that ``-O`` would strip) when the criterion is violated at construction
   or when a candidate presents delays beyond ``delta_t_max``.

The overall/common time shift reuses `BaseLinearFree`'s
``_get_linearfree_hplus_hcross_dt`` idiom (a single time alignment of the
candidate to the fiducial waveform); the resulting shift is applied as a
common delay in the data term and cancels in the norm term.

SCOPE / ERRORS THAT PROPAGATE
-----------------------------
Both macro-image parities are served (positive parity and the macro
saddle); the amplification is a parity-blind common factor ``F(w)``.  The
engine's named refusals are *not* caught -- they propagate unswallowed so
the certified-or-refuse contract is preserved end to end:
`geometry.LensDomainError` (over-critical / Type III ``1 - kappa <= 0`` and
the exact ``det A = 0`` parity boundary), `operator.CancellationError`
(the wave-branch contraction cannot certify its accuracy), and
`SchwingerCertificationError` (the saddle / strong-shear wave branch above
its certified ceiling).

Conventions
-----------
Frequencies in Hz, times in GPS seconds, delays in seconds; lens mass
``m_lens_msun`` in solar masses.  Inner products follow `CBCLikelihood`
(ASD-drift applied at evaluation, not baked into the summaries).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
from scipy.interpolate import CubicSpline

from cogwheel import utils
from cogwheel.likelihood.relative_binning import BaseLinearFree
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.channels import (_channel_switch,
                                                     _physical_kernels,
                                                     reconstruct_from_envelope)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal.operator import CancellationError
from cogwheel.lensing.waveform import (LensedWaveformGenerator,
                                       dimensionless_frequency)

__all__ = ['LensedRelativeBinningLikelihood', 'LensedBinningError']

#: Highest frequency moment retained for the data term ``(d|h)``.  The
#: candidate component ratio is expanded to first order across a bin and
#: the delay phase to first order, whose product is quadratic.
_DATA_MAX_MOMENT = 2

#: Highest frequency moment retained for the norm term ``(h|h)``.  Two
#: component ratios and one delay phase multiply, giving a cubic in-bin
#: polynomial -- one moment more than the data term.
_NORM_MAX_MOMENT = 3

#: Default tolerance [rad] for the lens-aware bin criterion
#: ``pi * Delta_f_bin * delta_t_max < bin_delay_tol``.
_DEFAULT_BIN_DELAY_TOL = 0.5

#: Default post-Newtonian phase tolerance [rad] for bin selection when
#: neither ``fbin`` nor ``pn_phase_tol`` is given.  Kept small so the
#: per-bin linear component-ratio expansion is accurate.
_DEFAULT_PN_PHASE_TOL = 0.02

#: Default number of frequency sub-samples per coarse bin used to reduce
#: the candidate channel kernels to their per-bin (value, slope)
#: coefficients.  With bounded channel kernels the plain bin-edge secant
#: (the value 2) is accurate for the gated configurations, so 2 is the
#: default.  The earlier large-amplitude kernel "blow-up" was not an
#: edge-secant aliasing failure but the channel-switch neighbourhood bug
#: in the engine (fixed in the Chang--Refsdal channels; see FINDINGS on
#: the switch-neighbourhood fix); once the kernels are O(1) the secant no
#: longer aliases.
#: A larger value fits a least-squares line over interior sub-samples and
#: is retained as a robustness margin against pure geometric phase
#: oscillation near a caustic, not as a correctness requirement.  Must be
#: >= 2; the value 2 reproduces the plain edge secant.
_DEFAULT_KERNEL_SUBSAMPLES = 2

#: Seed size of the coarse ``w`` node grid on which the engine evaluates
#: the single smooth SACR-C transition envelope ``E(w)`` before
#: leave-one-out refinement.  A small log-spaced base spanning the in-band
#: ``w`` range; the adaptive loop (`_envelope_loo_nodes`) then adds nodes
#: where the held-out interpolation error is largest.
_LOO_SEED_NODES = 8

#: Leave-one-out stop tolerance for the envelope node grid in the
#: certified fast region (mass-sheet-reduced shear ``gamma' < 0.5``).
#: Refinement halts once the worst held-out node error (relative to the
#: peak amplification magnitude ``max|F|``, the same currency as the
#: reconstruction gate ``max|dF|/max|F|``) drops below this value.  The
#: held-out estimate overestimates the true global-spline reconstruction
#: error, so this stop is deliberately conservative and drives the true
#: reconstruction error well inside the ``1e-3`` gate.  It is a fixed
#: physical property of the certified interpolant, NOT a constructor
#: argument or configuration key: the node count it produces is
#: self-certifying and config-independent by construction (and, being
#: keyed only on the candidate's lens geometry, preserves the memoized-
#: fiducial contract exactly).
_LOO_STOP_FAST = 4e-3

#: Tighter leave-one-out stop tolerance for the strong-shear/saddle region
#: (``gamma' >= _STRONG_SHEAR_STOP_THRESHOLD``).  Its justification is the
#: research's saddle-side SACR-C gate (envelope reconstruction error
#: ``< 1e-3``, enforced by ``test_lensing_saddle_channels``): the fast
#: stop cannot guarantee that on saddle windows.  It does NOT close the
#: measured ~0.7-1.4-nat direct/ratio-vs-brute-force gap at rescued
#: strong-shear configs (``gamma' ~ 0.94``): that gap is RB-binning /
#: data-noise-limited, insensitive to the stop (1e-3 -> 1e-5 leaves it
#: unchanged) and seed-dependent, so it is gated at the standard RB
#: tolerance, not an envelope tolerance.  The certified fast region (and
#: the crown fixture, ``gamma' = 0.20``) stays on `_LOO_STOP_FAST` --
#: byte-identical node count and warm cost.  The `_LOO_MAX_NODES` ceiling
#: still bounds worst-case cost.
_LOO_STOP_STRONG = 1e-3

#: Threshold on the mass-sheet-reduced shear ``gamma' = gamma / (1 - kappa)``
#: separating the certified fast region (``gamma' < 0.5`` -> `_LOO_STOP_FAST`)
#: from the strong-shear/saddle region (``gamma' >= 0.5`` -> `_LOO_STOP_STRONG`).
#: The key is ``gamma'`` (NOT ``abs(gamma)``): the rescued cancellation
#: family ``gamma = 0.405, kappa = 0.57`` has ``gamma' = 0.94`` -- an
#: ``abs(gamma) >= 0.5`` key would wrongly leave it on the fast stop and
#: fail the accuracy gate.  In the ``kappa = 0`` sampled space ``gamma' ==
#: gamma``, so the crown fixture stays on the fast stop unchanged.
_STRONG_SHEAR_STOP_THRESHOLD = 0.5

#: Hard ceiling on the number of coarse envelope nodes.  The SACR-C
#: envelope is beat-free by construction, so a certified reconstruction
#: needs far fewer nodes than the old kernel grid (report: greedy-oracle
#: 19-26, production LOO 30-44); this ceiling bounds the per-eval engine
#: cost and is never expected to bind on the gated configurations.  Must
#: be ``>= 4`` so the node grid always supports a not-a-knot cubic spline.
_LOO_MAX_NODES = 48

#: Floor on the peak amplification magnitude ``max|F|`` used to normalize
#: the leave-one-out node error.  Guards the ``|F| -> 1`` regime against a
#: spuriously large relative error should ``max|F|`` momentarily read near
#: zero; ``|F|`` is order unity or larger for a positive-parity lens, so
#: this floor never binds in practice.
_ENVELOPE_SCALE_FLOOR = 1e-12

#: Fiducial-lattice spacings for the ratio-layer fiducial key
#: (`_fiducial_key`).  The candidate lens parameters are snapped to this
#: lattice so that lens configurations within one cell share a single
#: fiducial envelope; the mass and redshift (which set ``w = xi*f``) are
#: shared EXACTLY (unsnapped) so the candidate and fiducial ``w`` grids
#: coincide.  Spacings are fixed physical properties of the ratio layer,
#: NOT constructor arguments or configuration keys (Professor authority).
_FID_GAMMA_SPACING = 0.03
_FID_BETA_SPACING = np.pi / 16
_FID_KAPPA_SPACING = 0.02
_FID_Y_SPACING = 0.05

#: Floor on the fiducial envelope magnitude below which the ratio layer
#: treats the fiducial as unhealthy and rebuilds it.  Guards the ratio
#: ``E_candidate / E_fiducial`` against division by a near-zero fiducial.
_ENVELOPE_HEALTH_FLOOR = 0.01

#: Maximum number of macro images a Chang--Refsdal lens produces (four
#: inside the caustic, two outside).  Matches the engine's channel count
#: ``channels._N_CHANNELS``: when fewer than this many real images exist
#: the remaining cluster labels are parked at the nearest critical point
#: as virtual images, whose delay enters the full-cluster node placement
#: (F008).
_MAX_LENS_IMAGES = 4

#: Lens parameters expected in ``par_dic`` (in addition to the waveform
#: parameters) to evaluate the amplification decomposition.
_LENS_PARAMS = ('m_lens_msun', 'z_lens', 'y1', 'y2', 'gamma', 'beta',
                'kappa')

_TWO_PI_I = 2j * np.pi


def _snap(x, dx):
    """
    Snap ``x`` to the nearest multiple of the lattice spacing ``dx``.

    Pure and deterministic: ``round(x / dx) * dx``.  Used to build the
    ratio-layer fiducial key so that nearby lens parameters collapse onto
    a shared fiducial cell.

    Parameters
    ----------
    x : float
        Value to snap.
    dx : float
        Lattice spacing (positive).

    Returns
    -------
    float
        The nearest lattice point ``round(x / dx) * dx``.
    """
    return round(x / dx) * dx


def _fiducial_key(lens):
    """
    Ratio-layer fiducial cell key for a lens sub-dictionary.

    Returns the 7-tuple identifying the fiducial cell that ``lens`` falls
    into: the shear ``gamma``, orientation ``beta``, convergence
    ``kappa`` and impact-parameter components ``y1``/``y2`` are snapped to
    their respective lattices (`_snap`), while the lens mass
    ``m_lens_msun`` and redshift ``z_lens`` are shared EXACTLY (unsnapped)
    so that the candidate and fiducial dimensionless-frequency grids
    ``w = xi*f`` coincide.

    Parameters
    ----------
    lens : dict
        Lens sub-dictionary with keys ``'m_lens_msun'``, ``'z_lens'``,
        ``'y1'``, ``'y2'``, ``'gamma'``, ``'beta'``, ``'kappa'``.

    Returns
    -------
    tuple
        ``(snap(gamma), snap(beta), snap(kappa), snap(y1), snap(y2),
        m_lens_msun, z_lens)``.
    """
    return (_snap(lens['gamma'], _FID_GAMMA_SPACING),
            _snap(lens['beta'], _FID_BETA_SPACING),
            _snap(lens['kappa'], _FID_KAPPA_SPACING),
            _snap(lens['y1'], _FID_Y_SPACING),
            _snap(lens['y2'], _FID_Y_SPACING),
            lens['m_lens_msun'],
            lens['z_lens'])


def _lens_from_key(key):
    """
    Reconstruct the fiducial lens sub-dictionary from a `_fiducial_key`.

    The inverse of `_fiducial_key`: a fiducial cell is fully described by
    its (snapped) key, so the fiducial lens parameters are recovered from
    the key alone -- never from the raw candidate parameters.  This keeps
    the fiducial deterministic in the cell key and independent of which
    candidate first populated the cell.

    Parameters
    ----------
    key : tuple
        A `_fiducial_key` 7-tuple ``(gamma, beta, kappa, y1, y2,
        m_lens_msun, z_lens)`` (the first five already snapped).

    Returns
    -------
    dict
        Lens sub-dictionary with keys ``'m_lens_msun'``, ``'z_lens'``,
        ``'y1'``, ``'y2'``, ``'gamma'``, ``'beta'``, ``'kappa'``.
    """
    gamma, beta, kappa, y1, y2, m_lens_msun, z_lens = key
    return {'m_lens_msun': m_lens_msun, 'z_lens': z_lens,
            'y1': y1, 'y2': y2, 'gamma': gamma, 'beta': beta, 'kappa': kappa}


@dataclass(frozen=True)
class _FiducialEnvelope:
    """
    Memoized fiducial SACR-C envelope for the ratio layer.

    Built once per fiducial cell (`_fiducial_key`) and reused by every
    candidate that snaps into the cell.  Carries the fiducial's
    leave-one-out envelope nodes, the w-independent partition geometry
    (delays, ``real_mask``, ``critical_delay``), and the Re/Im
    cubic-in-``ln w`` spline of the envelope used to divide out the
    fiducial from the candidate (`envelope`).

    Attributes
    ----------
    partition : ChangRefsdalPartition
        The fiducial seed engine evaluation, carrying ``real_mask`` (for
        the image-count guard) and ``critical_delay`` ``tau_c_fid``.
    coarse_w : np.ndarray
        Fiducial envelope node grid (strictly increasing, positive).
    envelope_nodes : np.ndarray
        Fiducial envelope ``E_fid(w)`` at ``coarse_w`` (complex).
    spline_real, spline_imag : CubicSpline
        Real/imaginary cubic-in-``ln w`` not-a-knot splines of
        ``envelope_nodes``.
    """

    partition: object
    coarse_w: np.ndarray
    envelope_nodes: np.ndarray
    spline_real: CubicSpline
    spline_imag: CubicSpline

    def envelope(self, w):
        """
        Fiducial envelope ``E_fid`` at ``w`` (scalar or array).

        Evaluates the Re/Im cubic-in-``ln w`` splines.  The candidate and
        fiducial share ``m_lens``/``z_lens`` exactly, so the candidate
        ``w`` grid lies inside the fiducial spline's ``ln w`` support --
        no extrapolation.

        Parameters
        ----------
        w : float or np.ndarray
            Dimensionless frequency (positive).

        Returns
        -------
        complex or np.ndarray
            ``E_fid(w)`` with the shape of ``w``.
        """
        ln_w = np.log(w)
        return self.spline_real(ln_w) + 1j * self.spline_imag(ln_w)


class LensedBinningError(ValueError):
    """
    Relative-binning bins are too coarse for the requested image delays.

    Raised when ``pi * Delta_f_bin * delta_t_max >= bin_delay_tol`` (the
    lens-aware bin criterion is violated), either at construction for the
    declared ``delta_t_max`` or at evaluation for a candidate whose image
    delays exceed ``delta_t_max``.
    """


def _edge_linear_coefficients(values_at_edges, fbin):
    """
    Per-bin linear (value-at-center, slope) coefficients from edge values.

    A smooth function sampled at the bin edges ``fbin`` is approximated
    within bin ``b`` by ``c0_b + c1_b * (f - f_b)`` with ``f_b`` the bin
    center.  ``c0_b`` is the midpoint value of the linear interpolant and
    ``c1_b`` its slope.

    Parameters
    ----------
    values_at_edges : np.ndarray
        Shape ``(..., n_edges)``; values at ``fbin`` (last axis).
    fbin : np.ndarray
        Shape ``(n_edges,)`` bin-edge frequencies [Hz], increasing.

    Returns
    -------
    c0, c1 : np.ndarray
        Shape ``(..., n_bins)`` center values and slopes [x], [x/Hz].
    """
    lower = values_at_edges[..., :-1]
    upper = values_at_edges[..., 1:]
    widths = np.diff(fbin)
    c0 = 0.5 * (lower + upper)
    c1 = (upper - lower) / widths
    return c0, c1


def _leave_one_out_errors(abscissa, values):
    """
    Held-out (leave-one-out) interpolation error at each interior node.

    For every interior node ``i`` the value ``values[i]`` is predicted
    from the four nearest OTHER nodes by cubic Lagrange interpolation in
    the given abscissa (``ln w``), and the error is the magnitude of the
    difference from the true value.  This uses only node data -- never a
    dense reference -- so it is a self-certifying, config-independent
    estimate of the interpolation error, computed in O(n) with a
    fixed four-point stencil.  Being local (it ignores the smoothing of
    the more distant nodes the global spline also sees) it OVERestimates
    the true global-spline reconstruction error, which is the safe
    direction for a refinement stop criterion.

    Endpoints have no leave-one-out prediction (removing them would force
    extrapolation) and are assigned zero error, so they are never chosen
    as a refinement target.

    Parameters
    ----------
    abscissa : np.ndarray
        Shape ``(n,)`` strictly increasing node abscissa (``ln w``).
    values : np.ndarray
        Shape ``(n,)`` complex node values ``E(w)``.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` non-negative held-out errors; zero at the two
        endpoints.
    """
    n_nodes = abscissa.size
    errors = np.zeros(n_nodes)
    indices = np.arange(n_nodes)
    for node in range(1, n_nodes - 1):
        others = indices[indices != node]
        stencil = others[np.argsort(np.abs(others - node))[:4]]
        stencil.sort()
        xs = abscissa[stencil]
        prediction = 0.0 + 0.0j
        for k in range(xs.size):
            basis = 1.0
            for m in range(xs.size):
                if m != k:
                    basis *= (abscissa[node] - xs[m]) / (xs[k] - xs[m])
            prediction += values[stencil[k]] * basis
        errors[node] = abs(values[node] - prediction)
    return errors


def _loo_stop_for_lens(lens):
    """
    Leave-one-out stop tolerance for a candidate's lens geometry.

    A pure function of the candidate's lens parameters and NOTHING else
    (no data, no frequency grid, no fiducial state), so keying the
    refinement stop on it preserves the memoized-fiducial contract
    exactly.  Returns the tighter `_LOO_STOP_STRONG` in the strong-shear/
    saddle region -- where the deep-cancellation troughs of ``F(w)`` live
    and the ``max|F|`` error-currency normalization under-resolves them --
    and the certified-fast `_LOO_STOP_FAST` elsewhere.

    The key is the mass-sheet-reduced shear ``gamma' = gamma / (1 - kappa)``
    (NOT ``abs(gamma)``): the rescued cancellation family ``gamma = 0.405,
    kappa = 0.57`` has ``gamma' = 0.94``, which an ``abs(gamma)`` key would
    wrongly leave on the fast stop.  In the ``kappa = 0`` sampled space
    ``gamma' == gamma``, so the crown fixture (``gamma' = 0.20``) stays on
    the fast stop and its node count / warm cost are byte-identical.

    Parameters
    ----------
    lens : dict
        Lens parameters; must carry keys ``gamma`` and ``kappa``.

    Returns
    -------
    float
        `_LOO_STOP_STRONG` if ``gamma' >= _STRONG_SHEAR_STOP_THRESHOLD``,
        else `_LOO_STOP_FAST`.
    """
    gamma_prime = lens['gamma'] / (1.0 - lens['kappa'])
    if gamma_prime >= _STRONG_SHEAR_STOP_THRESHOLD:
        return _LOO_STOP_STRONG
    return _LOO_STOP_FAST


def _data_term(a_moments, rho0, rho1, kbar0, kbar1, tau, f_center):
    """
    Contract the data term ``(d|h_L)`` per detector, mode-then-image.

    Implements the ``p <= 2`` moment expansion of
    ``sum_{a,m,b} A^(p)_{m,d,b} * [conj(r_m K_a) exp(-2*pi*1j*f*tau_a)]^(p)``
    (paper Eq. data-term-rb generalized to harmonic modes and split into a
    mode reduction followed by an image reduction).

    Parameters
    ----------
    a_moments : sequence of np.ndarray
        ``[A^(0), A^(1), A^(2)]``, each shape ``(n_m, n_det, n_bins)``.
    rho0, rho1 : np.ndarray
        Shape ``(n_m, n_det, n_bins)`` center value and slope of the
        conjugated unlensed component ratio ``conj(r_m)``.
    kbar0, kbar1 : np.ndarray
        Shape ``(n_img, n_bins)`` center value and slope of the
        conjugated candidate kernel ``conj(K_a)``.
    tau : np.ndarray
        Shape ``(n_img,)`` data-frame relative image delays [s]
        (image delay minus the common linear-free time shift).
    f_center : np.ndarray
        Shape ``(n_bins,)`` bin-center frequencies [Hz].

    Returns
    -------
    np.ndarray
        Shape ``(n_det,)`` complex ``(d|h_L)`` per detector (ASD-drift
        not yet applied).
    """
    a0, a1, a2 = a_moments

    # Mode reduction: G^(p,q)_{d,b} = sum_m A^(p)_{m,d,b} rho_q_{m,d,b}
    g00 = np.einsum('mdb,mdb->db', a0, rho0)
    g10 = np.einsum('mdb,mdb->db', a1, rho0)
    g11 = np.einsum('mdb,mdb->db', a1, rho1)
    g20 = np.einsum('mdb,mdb->db', a2, rho0)
    g21 = np.einsum('mdb,mdb->db', a2, rho1)

    tau_a = tau[:, np.newaxis, np.newaxis]  # (n_img, 1, 1)
    # In-bin polynomial of conj(K_a) exp(-2 pi i (f-f_b) tau_a) times the
    # mode-reduced tensors, grouped by the kernel coefficients kbar0/kbar1.
    coeff_k0 = (g00 + g11)[np.newaxis] - _TWO_PI_I * tau_a * (g10 + g21
                                                             )[np.newaxis]
    coeff_k1 = (g10 + g21)[np.newaxis] - _TWO_PI_I * tau_a * g20[np.newaxis]

    phase = np.exp(-_TWO_PI_I * f_center[np.newaxis, :] * tau[:, np.newaxis])
    term = phase[:, np.newaxis, :] * (
        coeff_k0 * kbar0[:, np.newaxis, :]
        + coeff_k1 * kbar1[:, np.newaxis, :])
    return term.sum(axis=(0, 2))


def _norm_term(b_moments, r0, r1, rho0, rho1, k0, k1, kbar0, kbar1,
               delays, f_center):
    """
    Contract the norm term ``(h_L|h_L)`` per detector, mode-then-image.

    Implements the ``p <= 3`` moment expansion of the double component
    sum (paper Eq. norm-term-rb generalized to harmonic modes), factored
    into a mode-pair reduction ``N^(p,q)`` followed by an image-pair
    reduction.  All ``M^2`` ordered mode pairs and ``n_img^2`` ordered
    image pairs are summed explicitly (the result is real).

    Parameters
    ----------
    b_moments : sequence of np.ndarray
        ``[B^(0..3)]``, each shape ``(n_m, n_m, n_det, n_bins)`` over
        ordered mode pairs ``(m, m')``.
    r0, r1, rho0, rho1 : np.ndarray
        Shape ``(n_m, n_det, n_bins)`` center/slope of the component ratio
        ``r_m`` and its conjugate ``rho = conj(r_m)``.
    k0, k1, kbar0, kbar1 : np.ndarray
        Shape ``(n_img, n_bins)`` center/slope of the candidate kernel
        ``K_a`` and its conjugate ``kbar = conj(K_a)``.
    delays : np.ndarray
        Shape ``(n_img,)`` relative image delays [s] (the common time
        shift cancels in the norm term).
    f_center : np.ndarray
        Shape ``(n_bins,)`` bin-center frequencies [Hz].

    Returns
    -------
    np.ndarray
        Shape ``(n_det,)`` real ``(h_L|h_L)`` per detector (ASD-drift not
        yet applied).
    """
    b0, b1, b2, b3 = b_moments

    def reduce_pairs(bp, x_m, y_mprime):
        """Contract ``sum_{m,m'} B^(p) x_m y_m'`` -> ``(n_det, n_bins)``."""
        return np.einsum('mMdb,mdb,Mdb->db', bp, x_m, y_mprime)

    # Mode reduction: N^(p,q) with q the order of the mode-pair ratio
    # mu_q, q in {0,1,2}: mu0 = r0 rho0', mu1 = r1 rho0' + r0 rho1',
    # mu2 = r1 rho1'.
    n00 = reduce_pairs(b0, r0, rho0)
    n11 = reduce_pairs(b1, r1, rho0) + reduce_pairs(b1, r0, rho1)
    n22 = reduce_pairs(b2, r1, rho1)
    n10 = reduce_pairs(b1, r0, rho0)
    n21 = reduce_pairs(b2, r1, rho0) + reduce_pairs(b2, r0, rho1)
    n32 = reduce_pairs(b3, r1, rho1)
    n20 = reduce_pairs(b2, r0, rho0)
    n31 = reduce_pairs(b3, r1, rho0) + reduce_pairs(b3, r0, rho1)
    n30 = reduce_pairs(b3, r0, rho0)

    # Collect by image-side in-bin order s (coefficient of nu_s):
    m0 = n00 + n11 + n22  # (n_det, n_bins)
    m1 = n10 + n21 + n32
    m2 = n20 + n31
    m3 = n30

    # Image reduction over ordered channel pairs (a, c).
    kpair0 = k0[:, np.newaxis, :] * kbar0[np.newaxis, :, :]
    kpair1 = (k1[:, np.newaxis, :] * kbar0[np.newaxis, :, :]
              + k0[:, np.newaxis, :] * kbar1[np.newaxis, :, :])
    kpair2 = k1[:, np.newaxis, :] * kbar1[np.newaxis, :, :]

    delta = delays[:, np.newaxis] - delays[np.newaxis, :]  # (n_img, n_img)
    shift = _TWO_PI_I * delta[:, :, np.newaxis]            # (n_img, n_img, 1)
    nu0 = kpair0
    nu1 = kpair1 + shift * kpair0
    nu2 = kpair2 + shift * kpair1
    nu3 = shift * kpair2

    phase = np.exp(_TWO_PI_I * delta[:, :, np.newaxis]
                   * f_center[np.newaxis, np.newaxis, :])

    # (n_img, n_img, n_det, n_bins): mode-reduced tensors times image nu_s.
    contrib = phase[:, :, np.newaxis, :] * (
        m0[np.newaxis, np.newaxis] * nu0[:, :, np.newaxis, :]
        + m1[np.newaxis, np.newaxis] * nu1[:, :, np.newaxis, :]
        + m2[np.newaxis, np.newaxis] * nu2[:, :, np.newaxis, :]
        + m3[np.newaxis, np.newaxis] * nu3[:, :, np.newaxis, :])
    return contrib.sum(axis=(0, 1, 3)).real


class LensedRelativeBinningLikelihood(BaseLinearFree):
    """
    Fast relative-binning likelihood for Chang--Refsdal microlensed CBCs.

    Subclasses `BaseLinearFree` to reuse the frequency-bin machinery, the
    ASD-drift correction, and the linear-free common-time-shift idiom.  The
    reference waveform ``par_dic_0`` is *unlensed*; the lensing enters only
    at evaluation through the per-candidate channel decomposition: its
    single smooth SACR-C envelope is interpolated, the analytic switched
    saddle kernels are rebuilt in closed form, and the image-delay phases
    are kept analytic (see the module docstring).

    Parameters
    ----------
    event_data : data.EventData
        Conditioned strain and whitening filter.
    waveform_generator : waveform.WaveformGenerator or \
            lensing.waveform.LensedWaveformGenerator
        Unlensed waveform generator (a `LensedWaveformGenerator` is
        accepted and unwrapped to its embedded unlensed generator; the
        lens is supplied per candidate through ``par_dic``).
    par_dic_0 : dict
        Reference (unlensed) waveform parameters, close to the maximum
        likelihood; keys per ``waveform_generator.params``.
    delta_t_max : float
        Largest relative image delay [s] the bins must support.  Sets the
        lens-aware bin criterion and is checked against each candidate.
    fbin : array_like or None
        Bin-edge frequencies [Hz].  Pass this or ``pn_phase_tol``.
    pn_phase_tol : float or None
        Post-Newtonian phase tolerance [rad] for automatic bin selection.
        Defaults to a small value if neither ``fbin`` nor ``pn_phase_tol``
        is given.
    spline_degree : int
        Spline degree used by the base class for the linear-free fit
        weights (the lensed summaries use per-bin moments, not splines).
    bin_delay_tol : float
        Tolerance [rad] in the lens-aware bin criterion.
    kernel_subsamples : int
        Number of frequency sub-samples per coarse bin at which the
        (closed-form reconstructed) candidate channel kernels are reduced
        to their per-bin (value, slope) coefficients by least squares.  Must
        be ``>= 2``; the default 2 is the plain bin-edge secant, which is
        accurate once the channel kernels are bounded.  Larger values fit
        a line over interior sub-samples and are a robustness margin
        against pure geometric phase oscillation near a caustic, not a
        correctness requirement.  The reference summaries and the
        mode-then-image contraction are unaffected.  Note the engine is
        NOT evaluated at these sub-samples -- the candidate kernels are
        reconstructed there in closed form from the single interpolated
        SACR-C envelope (see ``_amplification_coefficients``).
    amplification_surrogate : lensing.surrogate.LensAmplificationSurrogate \
            or None
        Optional trained envelope emulator (Build 8a).  When supplied and
        the candidate lies inside the surrogate's certified box,
        `_amplification_coefficients` serves the amplification from a cheap
        geometry-only partition plus the emulated envelope, short-circuiting
        the entire per-candidate engine cost (seed evaluation, fiducial
        cache, ratio/LOO paths).  A candidate that is out of the surrogate's
        domain, in a different image-count region, near the caustic, or that
        the surrogate declines to serve falls through to the exact path with
        no behavioural change.  The default ``None`` disables the fast path,
        leaving every evaluation byte-identical to the pure-engine build.

    Notes
    -----
    The engine evaluates only the single smooth SACR-C transition
    envelope ``E(w)`` on a small leave-one-out-adaptive coarse ``w`` node
    grid (seed ``_LOO_SEED_NODES``, gamma'-keyed stop -- `_LOO_STOP_FAST`
    for ``gamma' < _STRONG_SHEAR_STOP_THRESHOLD``, else `_LOO_STOP_STRONG`
    -- ceiling ``_LOO_MAX_NODES``); the candidate channel kernels ``K_a``
    are then
    reconstructed at the dense bin sub-samples in closed form (the
    analytic switched saddle kernels ``S_a * H_a`` plus the interpolated
    envelope, `reconstruct_from_envelope`).  The node budget is
    self-certifying and config-independent -- there is no coarse-grid size
    to tune -- which is why no ``n_kernel_nodes`` argument exists.

    Raises
    ------
    LensedBinningError
        If ``pi * max(Delta_f_bin) * delta_t_max >= bin_delay_tol``.
    ValueError
        If ``kernel_subsamples < 2``.
    """

    def __init__(self, event_data, waveform_generator, par_dic_0,
                 delta_t_max, *, fbin=None, pn_phase_tol=None,
                 spline_degree=3, bin_delay_tol=_DEFAULT_BIN_DELAY_TOL,
                 kernel_subsamples=_DEFAULT_KERNEL_SUBSAMPLES,
                 amplification_surrogate=None):
        if isinstance(waveform_generator, LensedWaveformGenerator):
            base_generator = waveform_generator.waveform_generator
        else:
            base_generator = waveform_generator

        if delta_t_max <= 0:
            raise ValueError('`delta_t_max` must be positive.')
        if kernel_subsamples < 2:
            raise ValueError(
                '`kernel_subsamples` must be >= 2 (2 reproduces the plain '
                f'bin-edge secant); got {kernel_subsamples}.')

        self.delta_t_max = float(delta_t_max)
        self.bin_delay_tol = float(bin_delay_tol)
        self.kernel_subsamples = int(kernel_subsamples)
        # Optional trained envelope emulator; stored under the constructor
        # name so `JSONMixin.get_init_dict` reads it back (see the
        # `get_init_dict` override for the None-vs-fitted serialization).
        self.amplification_surrogate = amplification_surrogate

        # Populated by ``_set_summary`` (triggered by the ``fbin`` setter
        # inside ``super().__init__``).
        self._moment_ops = None
        self._f_center = None
        self.n_bins = None
        self._h0_edges = None
        self._a_moments = None
        self._b_moments = None
        # Per-bin kernel sub-sample grid and least-squares reduction
        # weights (populated by ``_build_kernel_subsampling``).
        self._kernel_dense_f = None
        self._kernel_fit_value = None
        self._kernel_fit_slope = None
        # Ratio-layer fiducial-envelope cache, keyed by `_fiducial_key`.
        # Transient in-memory runtime state: not a constructor argument,
        # so `get_init_dict` (JSONMixin serialization) never captures it.
        self._fid_cache = {}
        # Testing-only seam: when True, `_amplification_coefficients`
        # bypasses the ratio layer and always takes the direct SACR-C
        # path, so a test can compare the two paths on one candidate.  Not
        # a constructor argument and never set by the hot path.
        self._force_direct = False

        if fbin is None and pn_phase_tol is None:
            pn_phase_tol = _DEFAULT_PN_PHASE_TOL

        super().__init__(event_data, base_generator, par_dic_0,
                         fbin=fbin, pn_phase_tol=pn_phase_tol,
                         spline_degree=spline_degree)

    def __getstate__(self):
        """
        Pickle state, dropping the transient derived caches.

        ``_fid_cache`` is a pure memoization of a deterministic function
        of the candidate on a fixed lattice (see ``_fiducial_key``), so a
        forked/unpickled worker rebuilds bit-identical values on first
        evaluation -- roughly one direct SACR-C eval per lattice cell per
        worker, which is acceptable -- and determinism is preserved.
        The behavioural testing seam ``_force_direct`` is NOT derived
        state and is kept, so a pickled instance evaluates identically to
        its parent.  The trained ``amplification_surrogate`` (small flat
        ndarrays; sampler workers need it) rides along in the ``__dict__``
        copy and is preserved.
        """
        state = self.__dict__.copy()
        state.pop('_fid_cache', None)
        return state

    def __setstate__(self, state):
        """Restore pickle state with the derived caches reset."""
        self.__dict__.update(state)
        self._fid_cache = {}

    def get_init_dict(self, **kwargs):
        """
        JSON init dict, deferring surrogate serialization.

        With the default ``amplification_surrogate=None`` the key is
        dropped so the serialized JSON is byte-identical to the pure-engine
        build (a None-surrogate instance round-trips unchanged).  JSON
        serialization of a *fitted* surrogate is deferred to a later build
        (sampling is out of scope here); a non-None surrogate raises rather
        than emitting an unserializable object.
        """
        init_dict = super().get_init_dict(**kwargs)
        if init_dict.get('amplification_surrogate') is None:
            init_dict.pop('amplification_surrogate', None)
        else:
            raise NotImplementedError(
                'JSON serialization of a fitted `amplification_surrogate` '
                'is deferred to a later build; pickle preserves it for '
                'sampler workers.  Serialize with `amplification_surrogate='
                'None` or omit the surrogate for JSON round-trips.')
        return init_dict

    # -- Parameters ------------------------------------------------------

    @property
    def params(self):
        """Sorted waveform parameters plus the lens parameters."""
        return sorted(set(self.waveform_generator.params) | set(_LENS_PARAMS))

    # -- Summary setup (FFTs / heavy precompute allowed here) ------------

    def _set_summary(self):
        """
        Compute the delay-free frequency-moment summaries.

        Builds ``A^(p)`` (data) and ``B^(p)`` (norm) for the *unlensed*
        fiducial waveform at all detectors, plus the fiducial strain at
        bin edges used to form candidate component ratios.  ASD-drift and
        the linear-free fit weights come from the base class.  Also
        validates the lens-aware bin criterion.
        """
        super()._set_summary()  # asd_drift + linear-free helper weights
        self._build_moment_operators()
        self._build_kernel_subsampling()
        self._validate_bin_delay_criterion()

        # Fiducial unlensed detector strain per |m|, at full resolution and
        # at bin edges.  Use all spin components regardless of
        # ``disable_precession`` so the reference is stable.
        with utils.temporarily_change_attributes(self.waveform_generator,
                                                 disable_precession=False):
            h0_f = self._get_h_f(self.par_dic_0, by_m=True)
            h0_edges = self.waveform_generator.get_strain_at_detectors(
                self.fbin, self.par_dic_0, by_m=True)

        # Undo the big coarse-time shift so the reference is smooth and
        # nonzero at high frequency, stall the ringdown, then reapply it
        # (mirrors ``RelativeBinningLikelihood._set_summary``).
        shift_f = np.exp(-2j * np.pi * self.event_data.frequencies
                         * self.event_data.tcoarse)
        shift_edges = np.exp(-2j * np.pi * self.fbin
                             * self.event_data.tcoarse)
        h0_f = h0_f * shift_f.conj()
        h0_edges = h0_edges * shift_edges.conj()
        self._stall_ringdown(h0_f, h0_edges)
        h0_f = h0_f * shift_f
        h0_edges = h0_edges * shift_edges
        self._h0_edges = h0_edges

        # Data moments A^(p)_{m,d,b} = 4 df sum_{f in b}
        #   blued(f) conj(h0_{m,d}(f)) (f - f_b)^p.
        integrand_dh = self.event_data.blued_strain[np.newaxis] * h0_f.conj()
        self._a_moments = self._bin_moments(integrand_dh, _DATA_MAX_MOMENT)

        # Norm moments B^(p)_{mm',d,b} over all ordered mode pairs.
        wht2 = self.event_data.wht_filter ** 2
        integrand_hh = (h0_f[:, np.newaxis] * h0_f[np.newaxis].conj() * wht2)
        self._b_moments = self._bin_moments(integrand_hh, _NORM_MAX_MOMENT)

    def _build_moment_operators(self):
        """
        Build sparse ``(n_bins, n_rfft)`` frequency-moment operators.

        ``moment_ops[p] @ g`` yields ``4 df sum_{f in b} g(f) (f-f_b)^p``
        with hard bin membership ``f in [fbin[b], fbin[b+1])`` and ``f_b``
        the bin center.
        """
        frequencies = self.event_data.frequencies
        fbin = self.fbin
        n_bins = len(fbin) - 1
        f_center = 0.5 * (fbin[:-1] + fbin[1:])

        in_band = (frequencies >= fbin[0]) & (frequencies <= fbin[-1])
        cols = np.nonzero(in_band)[0]
        bin_of = np.clip(
            np.searchsorted(fbin, frequencies[cols], side='right') - 1,
            0, n_bins - 1)
        offsets = frequencies[cols] - f_center[bin_of]
        prefactor = 4.0 * self.event_data.df

        self._moment_ops = []
        for power in range(_NORM_MAX_MOMENT + 1):
            data = prefactor * offsets ** power
            self._moment_ops.append(scipy.sparse.csr_matrix(
                (data, (bin_of, cols)), shape=(n_bins, len(frequencies))))
        self._f_center = f_center
        self.n_bins = n_bins

    def _build_kernel_subsampling(self):
        """
        Precompute the per-bin sub-sample grid and least-squares weights.

        The candidate channel kernels ``K_a(w)`` are reduced to their
        per-bin ``(value, slope)`` coefficients by fitting a line over
        ``kernel_subsamples`` frequencies inside each bin.  With the
        default ``kernel_subsamples == 2`` this is the plain bin-edge
        secant; larger values add interior sub-samples.  Once the channel
        kernels are bounded (the near-caustic ``K_a`` "blow-up" was the
        engine's channel-switch neighbourhood bug, since fixed in the
        Chang--Refsdal channels, not an edge-secant aliasing failure) the
        secant is accurate for the gated configurations, so the extra
        interior sub-samples are a robustness margin against pure
        geometric phase oscillation near a caustic, not a correctness
        requirement.

        Only the candidate kernel coefficients are affected: the
        reference frequency-moment summaries and the mode-then-image
        contraction are unchanged, so the ``F -> 1`` normalization and
        the additive ``M^2 + n_img^2`` cost are preserved.

        The sub-sample offsets are symmetric about each bin center, so
        ``sum_j offset_j = 0`` and the fit decouples into a value weight
        ``1 / n_sub`` (the mean) and a slope weight
        ``offset_j / sum_k offset_k**2``.  Both are candidate independent
        and depend only on ``fbin``.
        """
        n_sub = self.kernel_subsamples
        lower = self.fbin[:-1][:, np.newaxis]      # (n_bins, 1)
        widths = np.diff(self.fbin)[:, np.newaxis]  # (n_bins, 1)
        # Midpoints of ``n_sub`` equal sub-intervals: strictly interior,
        # strictly increasing across bins, symmetric about the center.
        fractions = (np.arange(n_sub) + 0.5) / n_sub
        dense_f = lower + widths * fractions[np.newaxis, :]  # (n_bins, n_sub)
        offsets = dense_f - self._f_center[:, np.newaxis]    # (n_bins, n_sub)

        self._kernel_dense_f = dense_f.reshape(-1)
        self._kernel_fit_value = np.full((self.n_bins, n_sub), 1.0 / n_sub)
        self._kernel_fit_slope = (
            offsets / np.sum(offsets ** 2, axis=1, keepdims=True))

    def _bin_moments(self, integrand, max_moment):
        """
        Project an integrand onto the frequency-moment summaries.

        Parameters
        ----------
        integrand : np.ndarray
            Shape ``(..., n_rfft)`` complex; oscillatory part of the
            integral (last axis is the FFT grid).
        max_moment : int
            Highest moment order to compute.

        Returns
        -------
        list of np.ndarray
            ``[moment_0, ..., moment_max]``, each shape ``(..., n_bins)``.
        """
        *pre_shape, n_rfft = integrand.shape
        flat = integrand.reshape(-1, n_rfft)
        moments = []
        for power in range(max_moment + 1):
            projected = (self._moment_ops[power] @ flat.T).T
            moments.append(np.asarray(projected).reshape(*pre_shape,
                                                         self.n_bins))
        return moments

    def _validate_bin_delay_criterion(self):
        """
        Raise `LensedBinningError` if the bins cannot resolve the delays.

        The linear in-bin expansion of the image-delay phase requires
        ``pi * Delta_f_bin * delta_t_max < bin_delay_tol`` for the widest
        bin.
        """
        widest_bin = float(np.max(np.diff(self.fbin)))
        criterion = np.pi * widest_bin * self.delta_t_max
        if criterion >= self.bin_delay_tol:
            raise LensedBinningError(
                'Relative-binning bins are too coarse for the requested '
                f'delta_t_max={self.delta_t_max:.6g} s: '
                f'pi*Delta_f_bin*delta_t_max = {criterion:.4g} '
                f'>= bin_delay_tol = {self.bin_delay_tol:.4g}. '
                'Use finer bins (smaller pn_phase_tol) or a smaller '
                'delta_t_max.')

    # -- Amplification decomposition (coarse-node engine call) -----------

    def _lens_params(self, par_dic):
        """Return the lens sub-dictionary, validating required keys."""
        missing = [key for key in _LENS_PARAMS if key not in par_dic]
        if missing:
            raise KeyError(
                f'`par_dic` is missing lens parameters {missing}; expected '
                f'{list(_LENS_PARAMS)}.')
        return {key: par_dic[key] for key in _LENS_PARAMS}

    def _evaluate_envelope(self, lens, new_w, pad_w):
        """
        Evaluate the SACR-C partition and envelope at ``new_w``.

        A fresh `ChangRefsdalChannels` uses the deterministic initial
        label assignment, so the geometry fields (delays, critical delay,
        assignment, real mask) and the demodulated envelope ``E(w)`` are
        identical whether a node is evaluated alone or in a larger grid;
        this is what lets the leave-one-out loop grow the grid one small
        batch at a time and stitch the envelope values together.

        `ChangRefsdalChannels` requires a grid of at least two strictly
        increasing points, so a lone new node is padded with ``pad_w``
        (an already-evaluated node, re-evaluated harmlessly) and the
        padding is dropped from the returned values.

        Parameters
        ----------
        lens : dict
            Lens parameters, keys ``gamma, beta, kappa, y1, y2``.
        new_w : np.ndarray
            Dimensionless frequencies to evaluate (positive), 1-D.
        pad_w : float
            An already-evaluated node used only to satisfy the engine's
            two-point minimum when ``new_w`` has a single element.

        Returns
        -------
        partition : ChangRefsdalPartition
            The engine output on the evaluated grid (its w-independent
            geometry is the same at every call).
        envelope : np.ndarray
            Envelope ``E(w)`` at ``new_w`` (padding removed).
        exact_total : np.ndarray
            Exact amplification total at ``new_w`` (padding removed), used
            to normalize the leave-one-out error by ``max|F|``.
        """
        new_w = np.atleast_1d(np.asarray(new_w, dtype=float))
        grid = new_w
        if grid.size < 2:
            grid = np.unique(np.concatenate([grid, np.atleast_1d(pad_w)]))
        partition = ChangRefsdalChannels(grid).evaluate(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])
        keep = np.searchsorted(grid, new_w)
        return partition, partition.envelope[keep], partition.exact_total[keep]

    def _envelope_loo_nodes(self, lens, dense_w, *, seed=None):
        """
        Leave-one-out-adaptive coarse ``w`` nodes for the SACR-C envelope.

        Only the single smooth transition envelope ``E(w)`` is
        interpolated (the analytic switched saddle kernels are rebuilt in
        closed form), so the node grid is chosen to resolve ``E`` alone.
        Seeded with `_LOO_SEED_NODES` log-spaced nodes spanning the
        in-band range ``[dense_w.min(), dense_w.max()]`` (so the endpoints
        coincide with the dense grid's -- no extrapolation downstream, and
        the worst-cancellation node ``w_max`` is always evaluated), the
        grid is refined by repeatedly splitting the two intervals flanking
        the node of largest held-out (leave-one-out) error until that
        error, measured relative to the peak amplification magnitude
        ``max|F|`` (the reconstruction gate's currency), drops below the
        gamma'-keyed stop (`_loo_stop_for_lens`: `_LOO_STOP_FAST` in the
        certified fast region ``gamma' < _STRONG_SHEAR_STOP_THRESHOLD``,
        the tighter `_LOO_STOP_STRONG` in the strong-shear/saddle region),
        or the count reaches `_LOO_MAX_NODES`.

        The held-out error is a local cubic estimate (`_leave_one_out_errors`)
        that uses only node data -- no dense truth -- so the stop is
        self-certifying per evaluation and the resulting node count is
        config-independent by construction: there is no coarse-grid size
        to tune.  Because the local estimate overestimates the true
        global-spline reconstruction error, the stop is conservative and
        drives the true error well inside the ``1e-3`` reconstruction gate.

        The macro-geometry domain refusals (over-critical / Type III and
        the ``det A = 0`` parity boundary) make the first engine evaluation
        raise `geometry.LensDomainError`, and an uncertifiable or
        above-ceiling wave-branch contraction raises
        `operator.CancellationError` / `SchwingerCertificationError`; all
        propagate unswallowed, matching the brute-force path so the two
        refuse symmetrically.

        Parameters
        ----------
        lens : dict
            Lens parameters, keys ``gamma, beta, kappa, y1, y2``.
        dense_w : np.ndarray
            Dense bin sub-sample dimensionless frequencies (positive,
            strictly increasing); only its min and max seed the grid.
        seed : tuple or None
            Optional precomputed seed evaluation
            ``(partition, coarse_w, envelope_nodes, exact_total_nodes)``
            reused by the dispatch so the candidate seed is engine-
            evaluated once for the guard check and the direct path (no
            double engine work).  When ``None`` the seed is evaluated
            here, reproducing the standalone behaviour exactly.

        Returns
        -------
        partition : ChangRefsdalPartition
            The seed engine evaluation, carrying the w-independent
            geometry the closed-form reconstruction needs.
        coarse_w : np.ndarray
            Strictly increasing positive envelope node grid.
        envelope_nodes : np.ndarray
            Envelope ``E(w)`` at ``coarse_w`` (complex).
        """
        if seed is None:
            w_max = float(dense_w.max())
            coarse_w = np.geomspace(
                float(dense_w.min()), w_max, _LOO_SEED_NODES)
            partition, env_nodes, ftot_nodes = self._evaluate_envelope(
                lens, coarse_w, pad_w=w_max)
        else:
            partition, coarse_w, env_nodes, ftot_nodes = seed

        def node_error(node_w, node_env, node_ftot):
            loo = _leave_one_out_errors(np.log(node_w), node_env)
            scale = max(float(np.max(np.abs(node_ftot))),
                        _ENVELOPE_SCALE_FLOOR)
            return loo, scale

        coarse_w, env_nodes, _ = self._refine_envelope_grid(
            lens, coarse_w, env_nodes, ftot_nodes, node_error)
        return partition, coarse_w, env_nodes

    def _refine_envelope_grid(self, lens, coarse_w, env_nodes, ftot_nodes,
                              node_error):
        """
        Adaptive leave-one-out refinement of an engine node grid.

        The refinement loop shared by the direct envelope grid
        (`_envelope_loo_nodes`) and the ratio grid (`_ratio_loo_nodes`):
        repeatedly split the two intervals flanking the node of largest
        held-out error (geometric midpoints in ``w``), re-evaluate the
        SACR-C envelope there, and stop once the worst held-out error
        (normalized to the ``max|F|`` reconstruction currency by
        ``node_error``) drops below the gamma'-keyed stop
        (`_loo_stop_for_lens`, a pure function of ``lens``) or the count
        reaches `_LOO_MAX_NODES`.  Only the interpolated object and its error
        currency differ between the two callers, so they are supplied via
        ``node_error``; the placement, engine re-evaluation, and node
        bookkeeping are identical and live here once.

        Parameters
        ----------
        lens : dict
            Lens parameters, keys ``gamma, beta, kappa, y1, y2``.
        coarse_w : np.ndarray
            Seed node grid (strictly increasing positive), at least the
            `_LOO_SEED_NODES` seed points.
        env_nodes : np.ndarray
            Engine envelope ``E(w)`` at ``coarse_w`` (complex).
        ftot_nodes : np.ndarray
            Exact amplification total ``F(w)`` at ``coarse_w`` (complex),
            used to normalize the held-out error.
        node_error : callable
            ``node_error(coarse_w, env_nodes, ftot_nodes) -> (errors,
            scale)`` returning the per-node held-out error array (already
            in ``max|F|`` currency) and the scalar normalizing magnitude.

        Returns
        -------
        coarse_w, env_nodes, ftot_nodes : np.ndarray
            The refined node grid and the engine envelope / exact-total
            values on it (strictly increasing in ``coarse_w``).
        """
        # The stop is keyed on the candidate's mass-sheet-reduced shear
        # ``gamma' = gamma/(1-kappa)`` (a pure function of ``lens``), so
        # the strong-shear/saddle region gets deep-trough resolution while
        # the certified fast region keeps its byte-identical node count.
        loo_stop = _loo_stop_for_lens(lens)
        while True:
            n_nodes = coarse_w.size
            errors, scale = node_error(coarse_w, env_nodes, ftot_nodes)
            worst = int(np.argmax(errors))
            if errors[worst] / scale < loo_stop or n_nodes >= _LOO_MAX_NODES:
                break

            # Split the two intervals flanking the worst node (geometric
            # midpoints in ``w`` == arithmetic midpoints in ``ln w``).
            flanks = []
            if worst - 1 >= 0:
                flanks.append(np.sqrt(coarse_w[worst - 1] * coarse_w[worst]))
            if worst + 1 < n_nodes:
                flanks.append(np.sqrt(coarse_w[worst] * coarse_w[worst + 1]))
            new_w = np.array(
                [w for w in flanks
                 if not np.any(np.isclose(w, coarse_w, rtol=1e-9))])
            if new_w.size == 0:
                break  # placement already saturated around the worst node
            new_w = new_w[:_LOO_MAX_NODES - n_nodes]

            _, new_env, new_ftot = self._evaluate_envelope(
                lens, new_w, pad_w=coarse_w[-1])
            coarse_w = np.concatenate([coarse_w, new_w])
            env_nodes = np.concatenate([env_nodes, new_env])
            ftot_nodes = np.concatenate([ftot_nodes, new_ftot])
            order = np.argsort(coarse_w)
            coarse_w = coarse_w[order]
            env_nodes = env_nodes[order]
            ftot_nodes = ftot_nodes[order]

        return coarse_w, env_nodes, ftot_nodes

    def _reconstruct_kernels(self, dense_w, coarse_w, envelope_nodes,
                             partition):
        """
        Closed-form SACR-C reconstruction of the channel kernels.

        Interpolates the single smooth envelope ``E(w)`` from the coarse
        nodes onto the dense sub-samples with a not-a-knot cubic spline in
        ``ln w`` (real and imaginary parts separately -- the certified
        interpolant), evaluates the analytic switched saddle kernels
        ``S_a(w) * H_a(w)`` in closed form at every dense frequency, and
        rebuilds

            K_a(w) = S_a*H_a + u_a(w) * exp(-1j*w*(tau_a - tau_c)) * E,

        via `reconstruct_from_envelope`.  The saddle kernels ``H_a`` come
        from `geometry.image_kernel` (`_physical_kernels`) and the switch
        ``S_a`` from `_channel_switch`, both closed-form functions of ``w``
        and the (w-independent) geometry; only ``E`` is interpolated, and
        the carrier phases are reduced mod ``2*pi`` inside the gauge
        algebra (F001).  The reconstruction is pure vectorized numpy -- no
        njit is introduced on this path.

        Parameters
        ----------
        dense_w : np.ndarray
            Dense bin sub-sample dimensionless frequencies (positive,
            strictly increasing, within ``[coarse_w.min, coarse_w.max]``).
        coarse_w : np.ndarray
            Envelope node grid (strictly increasing positive).
        envelope_nodes : np.ndarray
            Envelope ``E(w)`` at ``coarse_w`` (complex).
        partition : ChangRefsdalPartition
            Carries the w-independent geometry (``delays``, ``assignment``,
            ``images``, ``matrix``, ``real_mask``, ``critical_delay``).

        Returns
        -------
        np.ndarray
            Shape ``(n_dense, n_channels)`` complex channel kernels ``K_a``.
        """
        ln_coarse = np.log(coarse_w)
        ln_dense = np.log(dense_w)
        spline_real = CubicSpline(ln_coarse, envelope_nodes.real,
                                  bc_type='not-a-knot')
        spline_imag = CubicSpline(ln_coarse, envelope_nodes.imag,
                                  bc_type='not-a-knot')
        envelope_dense = spline_real(ln_dense) + 1j * spline_imag(ln_dense)
        return self._kernels_from_dense_envelope(
            dense_w, envelope_dense, partition)

    def _kernels_from_dense_envelope(self, dense_w, envelope_dense, partition):
        """
        Rebuild the channel kernels from a dense envelope, in closed form.

        The saddle/switch/`reconstruct_from_envelope` core of the SACR-C
        reconstruction, extracted so it can be reused both from
        `_reconstruct_kernels` (which supplies ``envelope_dense`` by cubic
        spline of the coarse envelope nodes) and, later, from the ratio
        layer (which supplies ``envelope_dense`` from the candidate ratio
        times the fiducial envelope).  Evaluates the analytic switched
        saddle kernels ``S_a(w) * H_a(w)`` at every dense frequency
        (`geometry.image_kernel` via `_physical_kernels`; the switch
        ``S_a`` via `_channel_switch`), then rebuilds

            K_a(w) = S_a*H_a + u_a(w) * exp(-1j*w*(tau_a - tau_c)) * E,

        via `reconstruct_from_envelope`.  Pure vectorized numpy -- no njit
        is introduced on this path.

        Parameters
        ----------
        dense_w : np.ndarray
            Dense bin sub-sample dimensionless frequencies (positive,
            strictly increasing).
        envelope_dense : np.ndarray
            Envelope ``E(w)`` evaluated at ``dense_w`` (complex).
        partition : ChangRefsdalPartition
            Carries the w-independent geometry (``delays``, ``assignment``,
            ``images``, ``matrix``, ``real_mask``, ``critical_delay``).

        Returns
        -------
        np.ndarray
            Shape ``(n_dense, n_channels)`` complex channel kernels ``K_a``.
        """
        saddle_dense = _physical_kernels(
            dense_w, partition.assignment, partition.images, partition.matrix)
        switch_dense = _channel_switch(
            dense_w, partition.delays, partition.real_mask,
            partition.critical_delay)
        kernels, _total = reconstruct_from_envelope(
            dense_w, envelope_dense, partition.delays, saddle_dense,
            switch_dense, partition.critical_delay)
        return kernels

    def _surrogate_coefficients(self, par_dic):
        """
        Surrogate fast-path amplification coefficients, or ``None``.

        Serves the candidate amplification from the attached
        `LensAmplificationSurrogate` WITHOUT any exact-engine total: a
        cheap geometry-only partition
        (`ChangRefsdalChannels.geometry_partition`) supplies the channel
        delays, analytic saddle kernels, switch and critical delay; the
        surrogate supplies the smooth envelope ``E(w)``; and
        `reconstruct_from_envelope` rebuilds the channel kernels, reduced
        to the same per-bin ``(value, slope)`` coefficients as the exact
        path.

        The partition's certified physical geometry is fed to the
        surrogate query: the caustic distance (`caustic_distance`), the
        caustic arc angle (`caustic_theta`, gauge -- used only for the
        surrogate's cusp-window exclusion), and the real-image count
        (`real_mask.sum()`).  The surrogate's multi-chart guard stack
        (`LensAmplificationSurrogate.serve`) keys chart selection on those
        physical quantities and returns ``served=False`` -- signalling the
        caller to fall through to the exact path -- when no chart covers
        the candidate (out of the trained ``w`` / ``gamma`` box, wrong
        image-count region, inside a chart's caustic floor, or in a cusp
        window).

        Returns ``None`` -- signalling the caller to fall through to the
        exact path -- when the candidate carries ``kappa != 0`` (the
        surrogate is a ``kappa = 0`` surface), the dense ``w`` grid is
        non-positive, the cheap `may_serve` pre-check fails, or the
        surrogate declines to serve.  A `geometry.LensDomainError` raised
        by the geometry-only partition is NOT caught: it propagates
        exactly as the exact path's seed evaluation would raise it,
        preserving the refusal set.  (The surrogate is never queried where
        the engine would raise an `operator.CancellationError` /
        `SchwingerCertificationError`: each chart's domain gate excludes
        those points via its training-refusal exclusion balls.)

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, partition)`` in the same shape as
            `_amplification_coefficients`, or ``None`` to fall through.
        """
        lens = self._lens_params(par_dic)

        # The surrogate is a kappa = 0 surface BY CONSTRUCTION (the
        # sampled space eliminates kappa; the emulator's axes carry no
        # kappa dimension).  A general API candidate may carry kappa != 0,
        # and serving it the kappa = 0 envelope would be finite-but-wrong —
        # the exact never-serve-where-wrong violation the domain gate
        # exists to prevent (Inspector latent finding INS-8a-001).  Fall
        # through to the exact engine, which handles kappa fully.
        if lens['kappa'] != 0.0:
            return None

        dense_w = dimensionless_frequency(
            self._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        if not np.all(dense_w > 0):
            # Let the exact path raise the LensedBinningError, unchanged.
            return None

        surrogate = self.amplification_surrogate

        # Cheap pre-check on (gamma, w-band) BEFORE building the geometry
        # partition: a candidate no chart's (gamma, log w) box can contain
        # is unservable, so skip the partition work entirely.
        log_w = np.log(dense_w)
        if not surrogate.may_serve(
                lens['gamma'], float(log_w.min()), float(log_w.max())):
            return None

        # Cheap geometry-only partition (no exact total).  A candidate-side
        # `geometry.LensDomainError` propagates UNSWALLOWED.
        geom = ChangRefsdalChannels(dense_w).geometry_partition(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])

        # Full multi-chart guard stack: chart selection keys on the
        # certified physical caustic distance + real-image count (theta is
        # used only for cusp exclusion).  Recomputes NO geometry.
        envelope_dense, served = surrogate.serve(
            dense_w, gamma=lens['gamma'], y1=lens['y1'], y2=lens['y2'],
            beta=lens['beta'], eta=geom.caustic_distance,
            theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()))
        if not served:
            return None

        kernels, _total = reconstruct_from_envelope(
            dense_w, envelope_dense, geom.delays, geom.saddle_kernels,
            geom.switch, geom.critical_delay)
        k0, k1 = self._reduce_dense_kernels(kernels)
        delays = self._image_delays(lens, geom)
        return delays, k0, k1, geom

    def _amplification_coefficients(self, par_dic):
        """
        Candidate amplification coefficients (ratio-layer dispatch).

        0. Surrogate fast path (Build 8a).  If an
           ``amplification_surrogate`` is attached and the candidate is
           inside its certified box, the amplification is served from a
           cheap geometry-only partition plus the emulated envelope
           (`_surrogate_coefficients`), short-circuiting the entire
           per-candidate engine cost below.  A miss (no surrogate, out of
           domain, image-count mismatch, near the caustic, or a surrogate
           that declines) falls through with no behavioural change, so
           ``amplification_surrogate=None`` is byte-identical to the
           pure-engine path.

        Evaluates the candidate SACR-C seed partition once, then routes to
        the fast ratio path or the direct path:

        1. Engine-evaluate the candidate on the `_LOO_SEED_NODES` seed
           grid (a single engine call; a candidate-side
           `geometry.LensDomainError` or `operator.CancellationError`
           propagates UNSWALLOWED, matching ``lnlike_bruteforce``).
        2. Look up or build the fiducial envelope for the candidate's
           fiducial cell (`_fiducial_key`).  ONLY the fiducial build is
           wrapped in ``try/except (LensDomainError, CancellationError)``:
           a candidate inside the certified domain must not be refused
           because its snapped fiducial happens to fall outside, so a
           refusing fiducial falls back to the direct path.
        3. Two guards, either of which falls back to the direct path:
           image-count mismatch (``real_mask.sum()`` differs between
           candidate and fiducial), or an unhealthy fiducial envelope
           (``min|E_fid| / max|E_fid| < _ENVELOPE_HEALTH_FLOOR`` on the
           in-band dense grid, guarding the ratio's division).
        4. Otherwise take the ratio path (`_ratio_coefficients`), which
           interpolates only the ultra-smooth candidate/fiducial ratio.

        The candidate seed evaluation is reused across the guard check and
        both the ratio and direct paths, so the engine is hit once per
        candidate for the seed regardless of route.

        Setting the testing-only attribute ``self._force_direct`` bypasses
        the ratio layer entirely (used to compare the two paths on one
        candidate); the hot path never sets it.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        delays : np.ndarray
            Shape ``(n_channels,)`` relative image delays [s].
        k0, k1 : np.ndarray
            Shape ``(n_channels, n_bins)`` per-bin center value and slope
            [1/Hz] of the candidate kernel ``K_a``.
        partition : ChangRefsdalPartition
            The candidate seed engine evaluation (w-independent geometry).
        """
        # SINGLE surrogate intercept.  A trained envelope emulator
        # supersedes any in-place kernel swap: it short-circuits the WHOLE
        # per-candidate cost (seed eval, fiducial cache, ratio/LOO), which
        # stay intact below as the fallback.  Named refusals from the
        # geometry-only partition are NEVER caught here -- they propagate
        # exactly as the exact seed evaluation's would.
        if self.amplification_surrogate is not None:
            served = self._surrogate_coefficients(par_dic)
            if served is not None:
                return served

        lens = self._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            self._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        if not np.all(dense_w > 0):
            raise LensedBinningError(
                'All kernel sub-sample frequencies must map to positive '
                'dimensionless frequency w = xi*f; got a non-positive value.')

        # Candidate seed engine evaluation (single call).  A candidate-side
        # `geometry.LensDomainError` / `operator.CancellationError` from
        # its own seed nodes propagates unswallowed here.
        w_max = float(dense_w.max())
        seed_w = np.geomspace(float(dense_w.min()), w_max, _LOO_SEED_NODES)
        partition_cand, seed_env, seed_ftot = self._evaluate_envelope(
            lens, seed_w, pad_w=w_max)
        seed = (partition_cand, seed_w, seed_env, seed_ftot)

        if self._force_direct:  # testing-only bypass
            return self._amplification_coefficients_direct(par_dic, seed=seed)

        key = _fiducial_key(lens)
        try:
            fiducial = self._get_or_build_fiducial(key, _lens_from_key(key))
        except (LensDomainError, CancellationError):
            # Refusal symmetry: a refusing SNAPPED fiducial must not veto a
            # candidate that is itself inside the certified domain.
            return self._amplification_coefficients_direct(par_dic, seed=seed)

        # Guard 1: the candidate and fiducial must have the same number of
        # real images, else the ratio's carriers do not correspond.
        if (int(partition_cand.real_mask.sum())
                != int(fiducial.partition.real_mask.sum())):
            return self._amplification_coefficients_direct(par_dic, seed=seed)

        # Guard 2: the fiducial envelope must stay away from zero across
        # the in-band dense grid, else dividing by it is ill-conditioned.
        e_fid_dense = fiducial.envelope(dense_w)
        magnitude = np.abs(e_fid_dense)
        max_magnitude = float(np.max(magnitude))
        if (max_magnitude <= 0.0
                or float(np.min(magnitude)) / max_magnitude
                < _ENVELOPE_HEALTH_FLOOR):
            return self._amplification_coefficients_direct(par_dic, seed=seed)

        return self._ratio_coefficients(
            lens, dense_w, partition_cand, fiducial, seed_w, seed_env,
            seed_ftot)

    def _amplification_coefficients_direct(self, par_dic, *, seed=None):
        """
        Candidate kernels: coarse-node envelope, closed-form reconstruct.

        Evaluates the Chang--Refsdal engine only for the single smooth
        SACR-C transition envelope ``E(w)`` on a small
        leave-one-out-adaptive coarse ``w`` node grid
        (`_envelope_loo_nodes`, ``<= _LOO_MAX_NODES`` points), then
        reconstructs each candidate channel kernel ``K_a(w)`` at the dense
        ``n_bins * kernel_subsamples`` bin sub-samples in CLOSED FORM: the
        analytic switched saddle kernels ``S_a * H_a`` evaluated directly
        at every sub-sample plus the interpolated envelope
        (`_reconstruct_kernels`).  Those dense kernels are reduced to
        per-bin center value and slope by the same least-squares fit as
        before.  The frequency-independent relative image delays are read
        from the same engine evaluation and kept analytic.

        This decouples the engine cost from the waveform bin grid and from
        any fixed coarse-grid size: because the SACR-C envelope is
        beat-free by construction, far fewer nodes certify a
        reconstruction than the old channel-kernel grid, and the node
        count is chosen adaptively per evaluation (self-certifying,
        config-independent) rather than fixed.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.
        seed : tuple or None
            Optional precomputed candidate seed evaluation
            ``(partition, coarse_w, envelope_nodes, exact_total_nodes)``
            forwarded to `_envelope_loo_nodes`, so the fallback path
            reuses the seed engine evaluation the dispatch already made
            for the guards (no double engine work).  When ``None`` the
            seed is evaluated here (standalone direct call).

        Returns
        -------
        delays : np.ndarray
            Shape ``(n_channels,)`` relative image delays [s].
        k0, k1 : np.ndarray
            Shape ``(n_channels, n_bins)`` per-bin center value and slope
            [1/Hz] of the candidate kernel ``K_a``.
        partition : ChangRefsdalPartition
            The seed engine evaluation (carrying the w-independent
            geometry); retained for API compatibility and diagnostics --
            the hot path reconstructs ``K_a`` from the closed form, not
            from ``partition.kernels``.

        Notes
        -----
        `geometry.LensDomainError` (Type III / parity boundary, raised by
        the first engine evaluation) and `operator.CancellationError` /
        `SchwingerCertificationError` (uncertifiable or above-ceiling
        contraction, raised at the worst-cancellation node ``w_max`` that
        the seed always evaluates) propagate unswallowed, exactly as in
        ``lnlike_bruteforce``, so the two paths refuse symmetrically.
        """
        lens = self._lens_params(par_dic)
        dense_w = dimensionless_frequency(
            self._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        if not np.all(dense_w > 0):
            raise LensedBinningError(
                'All kernel sub-sample frequencies must map to positive '
                'dimensionless frequency w = xi*f; got a non-positive value.')

        # Coarse-node envelope (LOO-adaptive); the first engine call
        # raises the engine's named refusals (`geometry.LensDomainError`
        # for Type III / the parity boundary, `operator.CancellationError`
        # / `SchwingerCertificationError` for an uncertifiable or
        # above-ceiling contraction), matching the brute-force path.
        partition, coarse_w, envelope_nodes = self._envelope_loo_nodes(
            lens, dense_w, seed=seed)

        delays = self._image_delays(lens, partition)

        # Closed-form dense reconstruction of the channel kernels, then the
        # same per-bin least-squares (value, slope) reduction as before --
        # only the kernel source changed (from a spline of the beat-laden
        # kernels to the analytic saddles plus the smooth envelope).
        dense_kernels = self._reconstruct_kernels(
            dense_w, coarse_w, envelope_nodes, partition)
        k0, k1 = self._reduce_dense_kernels(dense_kernels)
        return delays, k0, k1, partition

    def _image_delays(self, lens, partition):
        """
        Frequency-independent relative image delays [s] from a partition.

        Converts the engine's dimensionless channel delays ``tau_a``
        (minimum-relative convention) to detector-frame relative image
        delays ``dt_a = xi * tau_a / (2*pi)``, with ``xi = w / f`` the
        dimensionless-frequency slope set by the lens mass and redshift.

        Parameters
        ----------
        lens : dict
            Lens parameters (uses ``m_lens_msun`` and ``z_lens``).
        partition : ChangRefsdalPartition
            Carries the dimensionless channel delays ``partition.delays``.

        Returns
        -------
        np.ndarray
            Shape ``(n_channels,)`` relative image delays [s].
        """
        xi = float(dimensionless_frequency(
            1.0, lens['m_lens_msun'], lens['z_lens']))
        return xi * partition.delays / (2.0 * np.pi)

    def _reduce_dense_kernels(self, dense_kernels):
        """
        Reduce dense channel kernels to per-bin (value, slope) coefficients.

        Reshapes the dense ``n_bins * kernel_subsamples`` channel kernels
        to ``(n_bins, kernel_subsamples, n_channels)`` and applies the
        precomputed per-bin least-squares (value, slope) weights
        (`_build_kernel_subsampling`) -- the reduction shared by the
        direct and ratio paths.

        Parameters
        ----------
        dense_kernels : np.ndarray
            Shape ``(n_bins * kernel_subsamples, n_channels)`` complex
            channel kernels at the dense bin sub-samples.

        Returns
        -------
        k0, k1 : np.ndarray
            Shape ``(n_channels, n_bins)`` per-bin center value and slope.
        """
        n_channels = dense_kernels.shape[1]
        kernels = dense_kernels.reshape(
            self.n_bins, self.kernel_subsamples, n_channels)
        k0 = np.einsum('bj,bja->ab', self._kernel_fit_value, kernels)
        k1 = np.einsum('bj,bja->ab', self._kernel_fit_slope, kernels)
        return k0, k1

    # -- Ratio layer (candidate/fiducial heterodyne) ---------------------

    def _get_or_build_fiducial(self, key, lens_at_key):
        """
        Return the memoized fiducial envelope for a cell, building on miss.

        The fiducial is keyed on `_fiducial_key` alone (never on the raw
        candidate parameters), so the cache is deterministic in the cell
        and the result is independent of which candidate first populated
        it.  On a miss the fiducial envelope is engine-evaluated at the
        SNAPPED lens parameters (`_envelope_loo_nodes`) and its Re/Im
        cubic-in-``ln w`` splines are built; a `geometry.LensDomainError`
        / `operator.CancellationError` from the snapped configuration
        propagates to the caller, which falls back to the direct path.

        Parameters
        ----------
        key : tuple
            The `_fiducial_key` cell key.
        lens_at_key : dict
            The fiducial lens sub-dictionary reconstructed from ``key``
            (`_lens_from_key`).

        Returns
        -------
        _FiducialEnvelope
            The (cached) fiducial envelope record.
        """
        cached = self._fid_cache.get(key)
        if cached is not None:
            return cached

        dense_w = dimensionless_frequency(
            self._kernel_dense_f, lens_at_key['m_lens_msun'],
            lens_at_key['z_lens'])
        partition, coarse_w, envelope_nodes = self._envelope_loo_nodes(
            lens_at_key, dense_w)
        ln_coarse = np.log(coarse_w)
        spline_real = CubicSpline(ln_coarse, envelope_nodes.real,
                                  bc_type='not-a-knot')
        spline_imag = CubicSpline(ln_coarse, envelope_nodes.imag,
                                  bc_type='not-a-knot')
        fiducial = _FiducialEnvelope(
            partition, coarse_w, envelope_nodes, spline_real, spline_imag)
        self._fid_cache[key] = fiducial
        return fiducial

    def _ratio_loo_nodes(self, lens, fiducial, dtau_c, seed_w, seed_env,
                         seed_ftot):
        """
        Leave-one-out-adaptive nodes for the candidate/fiducial ratio.

        Forms the ultra-smooth bare ratio

            rho_bare(w) = exp(1j*w*dtau_c) * E_cand(w) / E_fid(w),

        with ``dtau_c = tau_c_cand - tau_c_fid`` the critical-carrier
        delay difference (the residual carrier the candidate/fiducial
        demodulation mismatch leaves, removed here so ``rho`` is
        beat-free), and refines a node grid on ``rho`` with the shared
        leave-one-out loop (`_refine_envelope_grid`).  Seeded with the
        reused candidate seed evaluation, so the engine is not re-hit for
        the seed.  The held-out error on ``rho`` is weighted by
        ``|E_fid|`` to express it in the candidate-envelope (``max|F|``)
        currency of the reconstruction gate.

        Because the reconstruction multiplies ``rho`` back by the SAME
        fiducial spline and undoes the ``dtau_c`` carrier, ``dtau_c`` and
        ``E_fid`` cancel in the exact-``rho`` limit: they precondition the
        interpolated object toward flatness (fewer nodes) without changing
        the reconstructed candidate envelope.

        Parameters
        ----------
        lens : dict
            Candidate lens parameters (for engine re-evaluation of
            ``E_cand`` at refinement nodes).
        fiducial : _FiducialEnvelope
            The fiducial envelope record (supplies ``E_fid``).
        dtau_c : float
            Critical-carrier delay difference ``tau_c_cand - tau_c_fid``.
        seed_w, seed_env, seed_ftot : np.ndarray
            The reused candidate seed grid, envelope ``E_cand``, and exact
            total ``F_cand`` at the seed nodes.

        Returns
        -------
        coarse_w : np.ndarray
            Strictly increasing positive ratio node grid.
        rho_nodes : np.ndarray
            Bare ratio ``rho_bare(w)`` at ``coarse_w`` (complex).
        """
        def node_error(node_w, node_env, node_ftot):
            e_fid = fiducial.envelope(node_w)
            rho = np.exp(1j * node_w * dtau_c) * node_env / e_fid
            loo = _leave_one_out_errors(np.log(node_w), rho)
            scale = max(float(np.max(np.abs(node_ftot))),
                        _ENVELOPE_SCALE_FLOOR)
            return loo * np.abs(e_fid), scale

        coarse_w, env_nodes, _ = self._refine_envelope_grid(
            lens, np.asarray(seed_w, dtype=float),
            np.asarray(seed_env, dtype=complex),
            np.asarray(seed_ftot, dtype=complex), node_error)
        e_fid = fiducial.envelope(coarse_w)
        rho_nodes = np.exp(1j * coarse_w * dtau_c) * env_nodes / e_fid
        return coarse_w, rho_nodes

    def _ratio_coefficients(self, lens, dense_w, partition_cand, fiducial,
                            seed_w, seed_env, seed_ftot):
        """
        Candidate kernels via the candidate/fiducial ratio path.

        Interpolates only the ultra-smooth bare ratio ``rho_bare``
        (`_ratio_loo_nodes`) with a Re/Im cubic-in-``ln w`` spline,
        rebuilds the candidate envelope

            E_cand(w) = exp(-1j*w*dtau_c) * rho(w) * E_fid(w),

        and reconstructs the channel kernels from it in closed form
        (`_kernels_from_dense_envelope`).

        The reconstruction uses the CANDIDATE partition's geometry
        (delays, saddle kernels, switch, and critical delay
        ``tau_c_cand``), NOT the fiducial's -- the fiducial only supplies
        the smooth divisor ``E_fid`` and the ``dtau_c`` carrier, which
        cancel against themselves so the reconstructed candidate envelope
        is exact in the exact-``rho`` limit.  Using the fiducial's
        ``tau_c`` here would be a correctness error.

        Parameters
        ----------
        lens : dict
            Candidate lens parameters.
        dense_w : np.ndarray
            Dense bin sub-sample dimensionless frequencies (positive,
            strictly increasing).
        partition_cand : ChangRefsdalPartition
            The candidate seed partition (supplies the reconstruction
            geometry and ``tau_c_cand``).
        fiducial : _FiducialEnvelope
            The fiducial envelope record (supplies ``E_fid`` and
            ``tau_c_fid``).
        seed_w, seed_env, seed_ftot : np.ndarray
            The reused candidate seed grid, envelope, and exact total.

        Returns
        -------
        delays : np.ndarray
            Shape ``(n_channels,)`` relative image delays [s].
        k0, k1 : np.ndarray
            Shape ``(n_channels, n_bins)`` per-bin center value and slope.
        partition : ChangRefsdalPartition
            The candidate seed partition (``partition_cand``).
        """
        dtau_c = float(partition_cand.critical_delay
                       - fiducial.partition.critical_delay)
        coarse_w, rho_nodes = self._ratio_loo_nodes(
            lens, fiducial, dtau_c, seed_w, seed_env, seed_ftot)

        ln_coarse = np.log(coarse_w)
        ln_dense = np.log(dense_w)
        rho_real = CubicSpline(ln_coarse, rho_nodes.real,
                               bc_type='not-a-knot')
        rho_imag = CubicSpline(ln_coarse, rho_nodes.imag,
                               bc_type='not-a-knot')
        rho_dense = rho_real(ln_dense) + 1j * rho_imag(ln_dense)

        envelope_dense = (np.exp(-1j * dense_w * dtau_c) * rho_dense
                          * fiducial.envelope(dense_w))
        dense_kernels = self._kernels_from_dense_envelope(
            dense_w, envelope_dense, partition_cand)

        k0, k1 = self._reduce_dense_kernels(dense_kernels)
        delays = self._image_delays(lens, partition_cand)
        return delays, k0, k1, partition_cand

    def _check_candidate_delays(self, delays):
        """Raise if a candidate's relative delays exceed ``delta_t_max``."""
        largest = float(np.max(np.abs(delays)))
        if largest > self.delta_t_max:
            raise LensedBinningError(
                f'Candidate relative image delay {largest:.6g} s exceeds the '
                f'certified delta_t_max = {self.delta_t_max:.6g} s; the bins '
                'do not resolve it. Rebuild with a larger delta_t_max and '
                'finer bins.')

    # -- Hot path (no FFTs, vectorized contractions) ---------------------

    def _candidate_bin_ratios(self, par_dic):
        """
        Per-bin linear coefficients of the unlensed component ratio.

        Returns ``(r0, r1, dt_linearfree)`` where ``r_m = r0 + r1*(f-f_b)``
        is the candidate/fiducial detector-strain ratio in the linear-free
        (time-aligned) frame, and ``dt_linearfree`` is the common time
        shift back to the approximant convention.
        """
        h_edges = self.waveform_generator.get_strain_at_detectors(
            self.fbin, par_dic, by_m=True)
        # Pass a copy: get_hplus_hcross may update ZERO_INPLANE_SPINS in
        # place when precession is disabled.
        _, dt_linearfree = self._get_linearfree_hplus_hcross_dt(
            dict(par_dic), by_m=True)

        # Align the raw candidate strain to the fiducial to keep the ratio
        # smooth across bins; the shift is undone as a common delay later.
        align = np.exp(2j * np.pi * self.fbin * dt_linearfree)
        ratio_edges = h_edges * align / self._h0_edges
        r0, r1 = _edge_linear_coefficients(ratio_edges, self.fbin)
        return r0, r1, dt_linearfree

    def _get_dh_hh_no_asd_drift(self, par_dic):
        """
        Return ``(d|h_L)`` (complex) and ``(h_L|h_L)`` (real) per detector.

        No ASD-drift correction is applied (it multiplies both at the
        evaluation stage, as in `CBCLikelihood`).
        """
        r0, r1, dt_linearfree = self._candidate_bin_ratios(par_dic)
        rho0, rho1 = r0.conj(), r1.conj()

        # Reduce the candidate kernels inside each bin to (value, slope).
        # At the default kernel_subsamples == 2 this is the plain bin-edge
        # secant, which is accurate now that the channel kernels are
        # bounded; interior sub-samples are only a robustness margin.
        delays, k0, k1, _ = self._amplification_coefficients(par_dic)
        self._check_candidate_delays(delays)
        kbar0, kbar1 = k0.conj(), k1.conj()

        # Data frame carries the common linear-free shift; the norm term
        # is invariant to it (it cancels in delta = dt_a - dt_c).
        tau = delays - dt_linearfree

        d_h = _data_term(self._a_moments, rho0, rho1, kbar0, kbar1, tau,
                         self._f_center)
        h_h = _norm_term(self._b_moments, r0, r1, rho0, rho1, k0, k1,
                         kbar0, kbar1, delays, self._f_center)
        return d_h, h_h

    def lnlike_and_metadata(self, par_dic):
        """
        Return ``(lnl, metadata)`` using lensed relative binning.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        lnl : float
            Log likelihood ``sum_d [(d|h)_d.real - (h|h)_d/2] / drift_d^2``.
        metadata : dict
            ``{'lnl': lnl}`` for postprocessing.
        """
        d_h, h_h = self._get_dh_hh_no_asd_drift(par_dic)
        lnl = float((d_h.real - h_h / 2) @ self.asd_drift ** -2)
        return lnl, {'lnl': lnl}

    def get_blob(self, metadata):
        """Return the metadata dict as posterior-sample columns."""
        return metadata

    # -- Brute-force reference (no relative binning) ---------------------

    def lnlike_bruteforce(self, par_dic):
        """
        Exact log likelihood on the full FFT grid, no relative binning.

        Builds the lensed strain ``F(w(f)) * h(f)`` through a
        `LensedWaveformGenerator` (consuming the exact amplification
        ``exact_total``) and evaluates the inner products directly.  This
        is the same-generator reference the relative-binning ``lnlike``
        must reproduce; a macro-saddle `geometry.LensDomainError` or an
        `operator.CancellationError` propagates unswallowed.

        Parameters
        ----------
        par_dic : dict
            Waveform and lens parameters, keys per ``self.params``.

        Returns
        -------
        float
            Log likelihood.
        """
        lens = self._lens_params(par_dic)
        lensed_generator = LensedWaveformGenerator(
            self.waveform_generator,
            m_lens_msun=lens['m_lens_msun'], z_lens=lens['z_lens'],
            y=(lens['y1'], lens['y2']), gamma=lens['gamma'],
            beta=lens['beta'], kappa=lens['kappa'])

        event_data = self.event_data
        fslice = event_data.fslice
        frequencies = event_data.frequencies

        strain = self.waveform_generator.get_strain_at_detectors(
            frequencies[fslice], par_dic, by_m=False)  # (n_det, n_band)
        amplification = lensed_generator.amplification(frequencies[fslice])

        lensed_strain = np.zeros(
            (len(event_data.detector_names), len(frequencies)),
            dtype=np.complex128)
        lensed_strain[:, fslice] = amplification * strain

        d_h = self._compute_d_h(lensed_strain)
        h_h = self._compute_h_h(lensed_strain)
        return float(np.sum(d_h.real) - np.sum(h_h) / 2)
