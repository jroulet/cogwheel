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

   ::

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
the exact ``det A = 0`` parity boundary) and
`SchwingerCertificationError` (the wave branch cannot certify its
accuracy, on either parity, and above its certified ceiling).

Conventions
-----------
Frequencies in Hz, times in GPS seconds, delays in seconds; lens mass
``m_lens_msun`` in solar masses.  Inner products follow `CBCLikelihood`
(ASD-drift applied at evaluation, not baked into the summaries).
"""
from __future__ import annotations

import cmath
import math
import types
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse
from scipy.interpolate import CubicSpline

from cogwheel import utils
from cogwheel.likelihood.relative_binning import BaseLinearFree
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.channels import (
    _channel_switch, _physical_kernels, reconstruct_from_envelope,
    reconstruct_farfield, farfield_ghost_term, farfield_w_floor,
    _frame_phase, _FARFIELD_KERNEL_FAMILY, FARFIELD_DIFFRACTIVE,
    FARFIELD_KERNEL_SUM, FARFIELD_KERNEL_SUM_MINUS_GHOST,
    KNOWN_FARFIELD_DEFINITIONS, KNOWN_INTERIOR_DEFINITIONS)
from cogwheel.lensing.chang_refsdal.geometry import (
    LensDomainError, GhostDomainError, macro_matrix)
from cogwheel.lensing.chang_refsdal._schwinger import (
    W_CEILING_SCHWINGER, W_CEILING_SCHWINGER_QD, SchwingerCertificationError)
from cogwheel.lensing.chang_refsdal._diffractive import (
    DiffractiveDomainError, diffractive_amplification, w_low_fit,
    _reduced_shear, _caustic_rho)
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    HypergeometricDomainError, prefactor_c)
from cogwheel.lensing.chang_refsdal._born import _born_factors
from cogwheel.lensing.chang_refsdal.operator import RHO_END
from cogwheel.lensing.waveform import (LensedWaveformGenerator,
                                       dimensionless_frequency)
from cogwheel.lensing.ppgo_map import (ASTROID_WALL, SADDLE_WALL, UNKNOWN,
                                       CERTIFICATION_BAR, caustic_rho,
                                       get_certified_ppgo_map)
from cogwheel.lensing.born_residual_chart import BornResidualChart
from cogwheel.lensing.low_w_diffractive_chart import LowWDiffractiveChart

__all__ = ['LensedRelativeBinningLikelihood', 'LensedBinningError']

#: Sentinel for the ``born_residual_chart`` constructor argument: when the
#: caller omits it, the shipped chart artifact is auto-loaded (refusing to
#: ``None`` on any load anomaly).  An *explicit* ``None`` is the pure-engine
#: opt-out (no chart attached, byte-identical to the no-chart path) and must
#: stay distinguishable from "argument not supplied", which is why a plain
#: ``None`` default cannot serve here.
_AUTO_BORN_CHART = object()

#: Sentinel for the ``low_w_diffractive_chart`` constructor argument,
#: mirroring `_AUTO_BORN_CHART`: when the caller omits it, the shipped
#: low-w diffractive residual chart is auto-loaded (refusing to ``None`` on
#: any load anomaly).  An *explicit* ``None`` is the pure-engine opt-out;
#: a caller-supplied in-memory instance is stored verbatim.
_AUTO_LOW_W_CHART = object()

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
#: from the strong-shear/saddle region (``gamma' >= 0.5`` ->
#: `_LOO_STOP_STRONG`).  The key is ``gamma'`` (NOT ``abs(gamma)``): the
#: rescued cancellation family ``gamma = 0.405, kappa = 0.57`` has ``gamma' =
#: 0.94`` -- an ``abs(gamma) >= 0.5`` key would wrongly leave it on the fast
#: stop and fail the accuracy gate.  In the ``kappa = 0`` sampled space
#: ``gamma' == gamma``, so the crown fixture stays on the fast stop unchanged.
_STRONG_SHEAR_STOP_THRESHOLD = 0.5

#: c3-led certificate admission for the tier-1 macro-saddle far-field serve
#: rung (`_saddle_farfield_analytic`).  The rung serves a 2-image saddle
#: exterior with a ZERO residual envelope; the omitted physics is the
#: leading stationary-phase remainder, which decays as ``w**-3`` with the
#: SAME SHAPE as the c3 term ``geometry.ppgo_error_estimate`` computes
#: per-image.  A draw is admitted iff the safety-factored certificate,
#: evaluated at the band FLOOR ``w_lo`` (worst case -- ``w**-3`` is largest
#: there, so a pass there certifies the whole band), clears the production
#: bar: ``_SADDLE_FARFIELD_SAFETY * ppgo_error_estimate(...) <=
#: _SADDLE_FARFIELD_CERT_BAR``.  The certificate's ``None`` return (divergent
#: ``mu``/``c3`` at a merging pair near the critical curve) is the PRIMARY
#: coalescence discriminator; ``S = 20``, ``bar = 1e-3`` at ``w_lo`` are
#: Professor-authorized from the 672-point calibration set (they clear the
#: measured max-error leg by 21x and the p90 leg by 2.1x with zero false
#: admits) -- do NOT re-derive them.  This replaces the retired scalar
#: ``rho`` floor (a proxy the measured failure geometry did not respect) and
#: the delta_tau resolution leg (whose ``delta_taus > 0`` tie filter refused
#: symmetry-tied mirror pairs that are spatially far apart and serve fine).
_SADDLE_FARFIELD_SAFETY = 20.0
_SADDLE_FARFIELD_CERT_BAR = 1e-3

#: Image-separation backstop (defense-in-depth) for the saddle far-field
#: gate.  SECONDARY to the certificate's ``None`` return: it guards the
#: residual case of a finite-but-optimistic certificate near a merge.  We
#: require the minimum pairwise Euclidean separation among the real image
#: positions (source-plane, Einstein-radius units) to be ``>= 0.05`` -- a
#: symmetry-tied mirror pair at ``+/-x`` has separation ``2|x| >= 0.1`` and
#: passes, while a genuinely coalescing pair falls below the floor and is
#: refused.  It discriminates coalescence by SPATIAL separation, never by
#: delay coincidence.
_SADDLE_FARFIELD_MIN_IMAGE_SEP = 0.05

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

#: Minimum Airy parameter xi = (3*w*Δτ/4)^(2/3) below which a fold-corrected
#: ppGO pair is not well resolved.  Retained as the fold arm's resolution
#: threshold; the raw-ppGO interior handoff no longer gates on it (Build
#: ppgo_interior_certificate replaced that leg with the c3 certificate below,
#: because every 4-image interior config fails this leg yet is served
#: certifiably under the certificate).
_XI_FOLD_THRESHOLD = 4.0

#: Safety factor on the raw-ppGO ``w**-3`` (c3) leading-omitted-term
#: certificate before it is compared against ``CERTIFICATION_BAR`` on the true
#: caustic interior.  Fact 3 (handoff ppgo_interior_certificate) measured the
#: ratio true_error / certificate over 1248 interior samples: median 0.587,
#: p99 0.953, MAX 0.980, 0.0% optimistic -- so S = 1.0 already suffices on the
#: measured interior; S = 2.0 is a modest margin (never 10x) that still admits
#: certifiable service (max served interior error 4.8e-5 < the 1e-4 bar).
_PPGO_INTERIOR_SAFETY = 2.0


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


def _saddle_farfield_analytic_serves(real_images, source, matrix, w_lo):
    """
    Whether the far-from-caustic macro saddle may be served analytically
    with a zero residual envelope.

    SINGLE SOURCE OF TRUTH for the tier-1 saddle-analytic serve gate.  Both
    the live serve rung (`_saddle_farfield_analytic`) and the census
    band-splitting (WP-2) call this exact predicate, so the served set and
    the counted set can never skew.

    The gate is a c3-led certificate with a separation backstop:

    1. Certificate (accuracy, PRIMARY).  The zero-envelope serve's true
       remainder decays as ``w**-3`` with the same shape as the per-image
       c3 term ``ppgo_error_estimate`` computes.  We admit iff the
       safety-factored estimate at the band FLOOR clears the production bar,
       ``_SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR``.  The
       band floor ``w_lo`` is the worst case (``w**-3`` largest), so a pass
       there certifies the whole band.  A ``None`` estimate -- divergent
       ``mu``/``c3`` at a genuinely merging pair near the critical curve --
       is the PRIMARY coalescence discriminator and refuses.

    2. Separation backstop (defense-in-depth, SECONDARY).  Require the
       minimum pairwise Euclidean separation among the real image positions
       to be ``>= _SADDLE_FARFIELD_MIN_IMAGE_SEP``.  This guards the residual
       case of a finite-but-optimistic certificate near a merge; a
       symmetry-tied mirror pair at ``+/-x`` (separation ``2|x| >= 0.1``)
       passes, a coalescing pair falls below the floor and refuses.

    Parameters
    ----------
    real_images : np.ndarray
        Shape ``(k, 2)``, the REAL image positions (source-plane,
        Einstein-radius units).  ``geom.images`` is already the real-only
        array (``geometry.find_images``); pass it directly -- do NOT index
        it with the length-4 channel mask ``geom.real_mask``.
    source : np.ndarray
        Shape ``(2,)``, the source position.  Passed to
        ``ppgo_error_estimate`` for interface symmetry.
    matrix : np.ndarray
        Shape ``(2, 2)``, the macro matrix.
    w_lo : float
        Band-floor dimensionless frequency ``min(dense_w)``, at which the
        ``w**-3`` remainder is largest.

    Returns
    -------
    bool
        Whether the analytic far-field rung may serve the whole band.
    """
    from cogwheel.lensing.chang_refsdal.geometry import ppgo_error_estimate
    images = np.asarray(real_images, dtype=float)
    if len(images) < 2:
        # Cannot be a resolved 2-image exterior.
        return False
    est = ppgo_error_estimate(images, source, matrix, w_lo)
    if est is None:
        # Divergent mu/c3 near the critical curve -- a genuinely merging
        # pair.  This is the primary coalescence discriminator.
        return False
    # Minimum pairwise Euclidean image separation (separation backstop),
    # shared with the band-split serve rung via `_saddle_min_image_sep`.
    min_sep = _saddle_min_image_sep(images)
    if min_sep is None:
        # Unreachable: >= 2 images guaranteed above.  Kept as an explicit
        # guard so the None-returning helper contract is honoured locally
        # (and the type is narrowed to float for the comparison below).
        return False
    return (min_sep >= _SADDLE_FARFIELD_MIN_IMAGE_SEP
            and _SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR)


def _saddle_min_image_sep(real_images):
    """Minimum pairwise Euclidean separation among real image positions.

    Source-plane, Einstein-radius units.  Shared by the saddle far-field
    serve gate (`_saddle_farfield_analytic_serves`) and the band-split
    serve rung (`_saddle_farfield_analytic`) so both apply the IDENTICAL
    separation backstop.  Returns ``None`` when fewer than two real images
    (no pair to separate).

    Parameters
    ----------
    real_images : np.ndarray
        Shape ``(k, 2)`` REAL image positions (``geom.images``; already
        real-only -- do NOT index with the length-4 channel mask
        ``geom.real_mask``).

    Returns
    -------
    float or None
        The minimum pairwise separation, or ``None`` for fewer than two
        real images.
    """
    images = np.asarray(real_images, dtype=float)
    if len(images) < 2:
        return None
    diffs = images[:, None, :] - images[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    iu = np.triu_indices(len(images), k=1)
    return float(np.min(dists[iu]))


def _born_carrier_certificate_serves(lens, w_lo, w_hi, real_images):
    """Whether the Born far exterior may be served carrier-only, or not.

    SINGLE SOURCE OF TRUTH for the beyond-the-chart-box Born serve gate.
    When the trained ``born_residual_chart`` does not cover a far-exterior
    query -- past the trained ``(gamma, rho)`` grid, OR the entire macro
    saddle region the astroid-only artifact never covers -- the residual is
    served as identically ZERO and ONLY the lead-only carrier
    (`_born.born_lead_carrier`, reconstructed via `born_carrier_from_partition`)
    is kept.  This predicate decides when that carrier-only truncation is
    accurate enough to admit, mirroring `_saddle_farfield_analytic_serves`.

    The gate is a carrier-relative truncation certificate with a saddle
    resolution fence and an image-separation backstop:

    1. Domain (matches the census / chart axes).  Refuse if ``kappa != 0``
       or ``beta != 0`` (the chart and the certificate are ``kappa = 0``,
       ``beta = 0`` surfaces) or ``gamma == 0`` (no shear, no caustic frame).

    2. Certificate (accuracy, PRIMARY).  The lead-only carrier omits a term
       that is LINEAR in ``w`` (``|delta| = hypot(a0, 0.5*w*b1) / q2r``,
       `_born.born_carrier_omitted_term`), so its worst case over the band is
       at the CEILING ``w_hi`` -- the OPPOSITE convention to the saddle-c3
       gate's band-FLOOR ``w_lo`` (evaluating at ``w_lo`` would silently
       under-certify).  Admit iff the safety-factored estimate clears the
       SAME production bar the saddle gate uses,
       ``_SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR`` (no
       Born-specific bar/safety).  A degenerate geometry returns ``inf`` and
       refuses.

    3. Separation backstop (defense-in-depth).  Require the minimum pairwise
       real-image separation ``>= _SADDLE_FARFIELD_MIN_IMAGE_SEP`` (shared
       with the saddle far-field gate via `_saddle_min_image_sep`); fewer
       than two real images refuses.

    4. Macro-saddle resolution fence.  For the macro saddle (``gamma > 1``,
       ``det A < 0``) additionally require the band FLOOR to resolve the
       closest real-image pair, ``w_lo * delta_min >= RHO_END`` (=4.0), with
       ``delta_min`` the smallest pairwise real-image Fermat-delay separation
       (`operator._real_delay_min_separation`).  Positive parity gets NO such
       fence here: its low-``w`` floor is served by the diffractive ``F_P``
       rung, out of scope for this gate.

    Parameters
    ----------
    lens : dict
        Lens parameters from `_lens_params`; ``kappa``, ``beta``, ``gamma``,
        ``y1``, ``y2`` are read.
    w_lo, w_hi : float
        Band floor and ceiling dimensionless frequencies (``min``/``max`` of
        the dense grid).  The certificate is evaluated at ``w_hi`` (worst
        case), the saddle resolution fence at ``w_lo`` (worst case).
    real_images : np.ndarray
        Shape ``(k, 2)`` REAL image positions (``geom.images``; already
        real-only -- do NOT index with the length-4 channel mask
        ``geom.real_mask``).

    Returns
    -------
    bool
        Whether the carrier-only far-field serve may serve the whole band.
    """
    from cogwheel.lensing.chang_refsdal._born import born_carrier_omitted_term
    from cogwheel.lensing.chang_refsdal.operator import (
        _real_delay_min_separation)

    # Domain assumptions matching the chart axes and the buried-path guard.
    if lens['kappa'] != 0.0 or lens['beta'] != 0.0 or lens['gamma'] == 0.0:
        return False

    # Carrier-relative truncation certificate at the band CEILING ``w_hi``
    # (the omitted term grows linearly in ``w``).  A degenerate geometry
    # returns ``inf`` and fails the bar.
    est = born_carrier_omitted_term(w_hi, lens['y1'], lens['y2'],
                                    lens['gamma'])
    if not (_SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR):
        return False

    # Separation backstop (defense-in-depth); < 2 real images refuses.
    min_sep = _saddle_min_image_sep(real_images)
    if min_sep is None or min_sep < _SADDLE_FARFIELD_MIN_IMAGE_SEP:
        return False

    # Macro-saddle resolution fence at the band FLOOR ``w_lo``.  Positive
    # parity (gamma <= 1) has no such fence here.
    if lens['gamma'] > 1.0:
        source = np.array([lens['y1'], lens['y2']])
        matrix = macro_matrix(lens['gamma'], lens['beta'], lens['kappa'])
        delta_min = _real_delay_min_separation(source, matrix)
        if not (w_lo * delta_min >= RHO_END):
            return False

    return True


def _saddle_c3_split_point(real_images, source, matrix):
    """Certificate split frequency for the saddle far-field band split.

    The zero-envelope analytic serve's true remainder decays as the c3
    leading stationary-phase term ``est(w) = C / w**3`` with ``C`` a
    frequency-INDEPENDENT geometry factor (`geometry.ppgo_error_estimate`;
    Professor ruling).  The smallest ``w`` at which the safety-factored
    certificate clears the production bar,

        _SADDLE_FARFIELD_SAFETY * C / w_split**3 == _SADDLE_FARFIELD_CERT_BAR,

    is the EXACT cube-root inversion

        w_split = w_ref * (S * est(w_ref) / bar) ** (1 / 3),

    which is INDEPENDENT of the reference ``w_ref`` because ``est`` is a
    pure ``C / w**3`` law (so ``w_ref = 1.0`` yields ``C`` directly).  No
    bisection, no hardcoded split.  Above ``w_split`` the analytic zero
    envelope already clears the bar; at or below it the exact engine must
    serve.  Evaluating at a fixed ``w_ref`` makes the split point a
    property of the geometry alone, independent of the dense grid; the
    caller compares it against the grid's ``(w_lo, w_hi)`` and the exact
    engine ceiling.

    A ``None`` return from ``ppgo_error_estimate`` -- divergent ``mu`` /
    ``c3`` at a genuinely merging pair near the critical curve -- is the
    PRIMARY coalescence discriminator and propagates as ``None`` here: a
    merging pair must REFUSE the whole draw and never enter a band split.
    The ``None``-ness depends only on the ``w``-independent ``mu`` / ``c3``
    finiteness, so it agrees with the gate's own ``est(w_lo)`` check.

    Parameters
    ----------
    real_images : np.ndarray
        Shape ``(k, 2)`` REAL image positions (``geom.images``; already
        real-only -- do NOT index with the length-4 channel mask).
    source : np.ndarray
        Shape ``(2,)`` source position (interface symmetry with the gate).
    matrix : np.ndarray
        Shape ``(2, 2)`` macro matrix.

    Returns
    -------
    float or None
        The split frequency ``w_split``, or ``None`` for a merging pair.
    """
    from cogwheel.lensing.chang_refsdal.geometry import ppgo_error_estimate
    w_ref = 1.0
    est = ppgo_error_estimate(
        np.asarray(real_images, dtype=float), source, matrix, w_ref)
    if est is None:
        return None
    return w_ref * (_SADDLE_FARFIELD_SAFETY * est
                    / _SADDLE_FARFIELD_CERT_BAR) ** (1.0 / 3.0)


def _band_split_mask(dense_w, split):
    """Band-split boolean and below-split node mask for a dense ``w`` grid.

    Shared arithmetic for every band-split serve rung.  A band split is
    active only when ``split`` is provided AND lies STRICTLY inside the
    dense band ``(w_lo, w_hi)``; a split at or outside an endpoint is a
    no-op.  ``below_mask`` marks the nodes the below-split rung serves;
    without an active split it is all-``True`` (nothing is masked out), so
    the caller's serve is byte-identical to the un-split result.

    Convention (load-bearing): ALL band-split rungs zero the reconstructed
    envelope ABOVE the split via ``envelope[~below_mask] = 0.0``; only what
    POPULATES the below-split envelope differs per rung, so this helper
    shares the mask arithmetic only -- never a serve-below callable.

    Parameters
    ----------
    dense_w : np.ndarray
        Dimensionless frequency grid for the kernel subsamples.
    split : float or None
        Certified trusted-floor frequency at which to split the band, or
        ``None`` for no split.

    Returns
    -------
    tuple
        ``(band_split, below_mask)`` where ``band_split`` is a ``bool`` and
        ``below_mask`` is a boolean ``np.ndarray`` of ``dense_w``'s shape.
    """
    w_lo = float(dense_w.min())
    w_hi = float(dense_w.max())
    band_split = split is not None and w_lo < split < w_hi
    below_mask = ((dense_w <= split) if band_split
                  else np.ones(dense_w.shape, dtype=bool))
    return band_split, below_mask


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

    # Mode-pair reduction -- the dominant cost of the lensed likelihood
    # (quadratic in the mode count via the ``(n_m, n_m, n_det, n_bins)``
    # moment tensors).  Each B^(p) is contracted against the ordered outer
    # product ``r_p[m] * rho_q[m']`` for ``p, q in {0, 1}``.  The original
    # form summed the joint ``(m, m')`` index in a single three-operand
    # einsum, repeated once per required ``(p, q)`` combination (twelve
    # quadratic passes).  Here the left-mode contraction is hoisted:
    # contract the ``m`` index first (batched over the two left vectors
    # ``r0, r1``), then contract the ``m'`` index (batched over the two
    # right vectors ``rho0, rho1``).  Each B^(p) then costs one quadratic
    # einsum plus cheap linear dot products, and the intermediate is reused
    # across all ``(p, q)`` entries.  This is a pure re-association of the
    # same double sum: the mathematics and every downstream branch/mask
    # decision are unchanged; only the floating-point accumulation order
    # differs (~1e-15 relative, well inside the 1e-10 preservation bound).
    r_stack = np.stack((r0, r1))        # (2, n_m, n_det, n_bins)
    rho_stack = np.stack((rho0, rho1))  # (2, n_m, n_det, n_bins)

    def bilinear(bp):
        """Ordered mode-pair form ``Q[p, q, d, b]`` for one B^(p) tensor."""
        # left[p, m', d, b] = sum_m bp[m, m', d, b] r_p[m, d, b]
        left = np.einsum('mMdb,pmdb->pMdb', bp, r_stack)
        # Q[p, q, d, b] = sum_m' left[p, m', d, b] rho_q[m', d, b]
        return np.einsum('pMdb,qMdb->pqdb', left, rho_stack)

    q0, q1, q2, q3 = bilinear(b0), bilinear(b1), bilinear(b2), bilinear(b3)

    # Assemble N^(p,q) with q the order of the mode-pair ratio mu_q,
    # q in {0,1,2}: mu0 = r0 rho0', mu1 = r1 rho0' + r0 rho1', mu2 = r1 rho1'
    # (each entry q*[p, q] has shape ``(n_det, n_bins)``, identical to the
    # previous ``reduce_pairs`` output).
    n00 = q0[0, 0]
    n11 = q1[1, 0] + q1[0, 1]
    n22 = q2[1, 1]
    n10 = q1[0, 0]
    n21 = q2[1, 0] + q2[0, 1]
    n32 = q3[1, 1]
    n20 = q2[0, 0]
    n31 = q3[1, 0] + q3[0, 1]
    n30 = q3[0, 0]

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
                 amplification_surrogate=None,
                 born_residual_chart=_AUTO_BORN_CHART,
                 low_w_diffractive_chart=_AUTO_LOW_W_CHART):
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
        # Optional trained Born residual chart; same serialization
        # pattern as `amplification_surrogate` (see `get_init_dict`).
        # Default (`_AUTO_BORN_CHART` sentinel): auto-load the shipped
        # artifact, refusing to `None` on any load anomaly (mirrors
        # `use_certified_ppgo_map`'s refuse-to-None pattern) so a corrupt /
        # absent artifact degrades to the pure-engine path instead of
        # raising.  An explicit `None` is the pure-engine opt-out; an
        # explicit instance is stored verbatim.
        #
        # Record whether the chart came from the auto-load default so
        # `get_init_dict` can round-trip it faithfully: once the sentinel is
        # consumed here the resolved `self.born_residual_chart` (a chart, or
        # `None` after a refused auto-load) no longer reveals the original
        # intent.
        self._born_residual_chart_is_default = (
            born_residual_chart is _AUTO_BORN_CHART)
        if born_residual_chart is _AUTO_BORN_CHART:
            try:
                self.born_residual_chart = BornResidualChart.load()
            except (OSError, ValueError, KeyError) as error:
                warnings.warn(
                    f'Born-residual chart unavailable ({error}); the Born '
                    f'weak-deflection rung will fall through to the exact '
                    f'engine. Regenerate with '
                    f'scripts/train_born_residual.py.', RuntimeWarning)
                self.born_residual_chart = None
        else:
            self.born_residual_chart = born_residual_chart

        # Optional trained low-w diffractive residual chart; same
        # auto-load / refuse-to-None / round-trip pattern as
        # `born_residual_chart` (see the comments and `get_init_dict`
        # override there).  Default (`_AUTO_LOW_W_CHART` sentinel):
        # auto-load the shipped artifact, refusing to `None` on any load
        # anomaly so an absent / corrupt artifact degrades to the
        # `_low_w_diffractive_serve` fall-through instead of raising.
        self._low_w_diffractive_chart_is_default = (
            low_w_diffractive_chart is _AUTO_LOW_W_CHART)
        if low_w_diffractive_chart is _AUTO_LOW_W_CHART:
            try:
                self.low_w_diffractive_chart = LowWDiffractiveChart.load()
            except (OSError, ValueError, KeyError) as error:
                warnings.warn(
                    f'Low-w diffractive chart unavailable ({error}); the '
                    f'near-fold-shell and wall-band draws will fall through '
                    f'to the exact engine. Regenerate with '
                    f'scripts/train_low_w_diffractive_chart.py.',
                    RuntimeWarning)
                self.low_w_diffractive_chart = None
        else:
            self.low_w_diffractive_chart = low_w_diffractive_chart

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

        ``born_residual_chart`` round-trips three ways, keyed on the
        recorded construction intent (`_born_residual_chart_is_default`),
        NOT on the resolved chart (which cannot tell the auto-loaded default
        apart from a caller-supplied copy of the same artifact):

        * auto-loaded default (or a refused-to-``None`` auto-load) -> the key
          is dropped so reconstruction re-defaults to the auto-load sentinel
          and re-serves via the Born path;
        * an explicit ``None`` opt-out -> ``None`` is emitted verbatim so the
          reconstructed likelihood stays pure-engine (it must NOT silently
          re-auto-load a chart);
        * a caller-supplied in-memory chart -> raises, because the chart has
          no source path to reference and its interpolation tables are not
          embedded in the init dict (pickle preserves it for sampler
          workers).
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
        if self._born_residual_chart_is_default:
            init_dict.pop('born_residual_chart', None)
        elif self.born_residual_chart is None:
            init_dict['born_residual_chart'] = None
        else:
            raise NotImplementedError(
                'JSON serialization of a caller-supplied in-memory '
                '`born_residual_chart` is unsupported: the chart carries no '
                'source path to reference and its interpolation tables are '
                'not embedded in the init dict.  Reconstruct with the shipped '
                'auto-loaded default by omitting `born_residual_chart`, or '
                'opt out of the Born rung with `born_residual_chart=None`.  '
                'Pickle preserves an in-memory chart for sampler workers.')
        if self._low_w_diffractive_chart_is_default:
            init_dict.pop('low_w_diffractive_chart', None)
        elif self.low_w_diffractive_chart is None:
            init_dict['low_w_diffractive_chart'] = None
        else:
            raise NotImplementedError(
                'JSON serialization of a caller-supplied in-memory '
                '`low_w_diffractive_chart` is unsupported: the chart carries '
                'no source path to reference and its interpolation tables '
                'are not embedded in the init dict.  Reconstruct with the '
                'shipped auto-loaded default by omitting '
                '`low_w_diffractive_chart`, or opt out of the low-w '
                'diffractive rung with `low_w_diffractive_chart=None`.  '
                'Pickle preserves an in-memory chart for sampler workers.')
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
        above-ceiling wave-branch quadrature raises
        `SchwingerCertificationError`; both propagate unswallowed,
        matching the brute-force path so the two refuse symmetrically.

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

    def _ppgo_cell_coords(self, lens):
        """
        ``(parity, gamma, rho)`` locating this draw in the ppGO map grid,
        or ``None``.

        The parity / shear / caustic-frame rho coordinate shared by the
        two per-cell map queries -- `_ppgo_band_split` reads ``w_trust`` and
        `_ppgo_cell_ceiling` reads ``w_ceiling`` -- so both land in the SAME
        cell from ONE derivation (DRY: one caustic-reach convention).

        The caustic-frame rho coordinate is obtained from the single
        authoritative converter `ppgo_map.caustic_rho`, which returns
        ``rho = |y| / caustic_reach`` with ``caustic_reach`` from
        `ppgo_map.caustic_geometry` -- the SAME authoritative reach the map
        was built with.  ``kappa`` is assumed ``0`` (the caller has already
        refused ``kappa != 0``); the map's caustic-reach convention is a
        ``kappa = 0`` surface.

        Returns ``None`` when the caustic reach is undefined
        (`LensDomainError`), so the caller keeps whole-band behaviour.

        Parameters
        ----------
        lens : dict
            Lens parameters (`_lens_params`); ``gamma``, ``y1``, ``y2``
            are used.

        Returns
        -------
        tuple or None
            ``(parity, gamma, rho)`` -- ``parity`` one of
            ``'positive'`` / ``'saddle'`` -- or ``None``.
        """
        gamma = float(lens['gamma'])
        # kappa == 0 here (the caller refused kappa != 0): the macro image
        # is a minimum (positive parity) for gamma < 1 and a saddle for
        # gamma > 1.  The gamma == 1 parity boundary lies inside the map's
        # guard band and returns UNKNOWN, so no split is attempted there.
        parity = 'positive' if gamma < 1.0 else 'saddle'
        try:
            rho = caustic_rho(
                gamma, float(np.hypot(lens['y1'], lens['y2'])), kappa=0.0)
        except LensDomainError:
            return None
        # The saddle rho<1 decision now lives solely in
        # CertifiedPpgoMap.w_trust / w_ceiling (single authoritative
        # source; F080 per-cell allowlist).  A still-refused cell returns
        # UNKNOWN downstream, so _ppgo_band_split / _ppgo_cell_ceiling are
        # unchanged there; only the allowlisted cell newly flows through.
        return parity, gamma, rho

    def _ppgo_band_split(self, lens):
        """
        Trusted dispatch floor ``w_trust`` for a per-node band split, or
        ``None``.

        Queries the process-global certified-ppGO map
        (`get_certified_ppgo_map`) for the margin-inflated trusted floor
        ``w_trust`` at this draw's ``(parity, gamma, caustic-frame rho)``
        cell (`_ppgo_cell_coords`).  Above ``w_trust`` the bare point-mass
        geometric-optics (ppGO) reconstruction is certified accurate, so
        the dense ``w`` band may be split: chart-served below, bare ppGO
        above -- but only up to the cell's measured ceiling, enforced by
        the caller via `_ppgo_cell_ceiling` (`_surrogate_coefficients`).
        ``w_trust`` is read from the map -- never a hardcoded constant.

        Returns ``None`` -- meaning "do NOT band-split; keep today's
        whole-band behaviour" -- when no map is installed, the caustic
        reach is undefined, or the cell is `UNKNOWN` (out of grid, beyond
        the Schwinger wall, or a parity-invalid band).  ``kappa`` is
        assumed ``0`` (the caller has already refused ``kappa != 0``); the
        map and its caustic-reach convention are ``kappa = 0`` surfaces.

        Parameters
        ----------
        lens : dict
            Lens parameters (`_lens_params`); ``gamma``, ``y1``, ``y2``
            are used.

        Returns
        -------
        float or None
            The trusted dispatch floor ``w_trust`` (raw ``w`` units), or
            ``None`` to keep whole-band behaviour.
        """
        ppgo_map = get_certified_ppgo_map()
        if ppgo_map is None:
            return None
        coords = self._ppgo_cell_coords(lens)
        if coords is None:
            return None
        w_trust = ppgo_map.w_trust(*coords)
        if w_trust is UNKNOWN:
            return None
        return float(w_trust)

    def _ppgo_cell_ceiling(self, lens):
        """
        Measured ``w`` ceiling of this draw's ppGO map cell, or ``None``.

        Reads ``w_ceiling`` from the SAME ``(parity, gamma, rho)`` cell as
        `_ppgo_band_split` reads ``w_trust`` (`_ppgo_cell_coords`).  The
        certified range is ``[w_cert, w_ceiling]``; above the ceiling the
        exact reference is UNKNOWN, so the band-split guard caps the split
        at ``min(parity_wall, cell_ceiling)`` and refuses / charts above it
        (Build 8h-b).

        Returns ``None`` -- meaning "no measured ceiling constraint; keep
        HEAD behaviour" -- when no map is installed, the caustic reach is
        undefined, or the cell is `UNKNOWN` (out of grid,
        beyond-``rho_measured_max``, beyond the Schwinger wall, or a
        parity-invalid band).  A certified cell always carries a finite
        ceiling, so a non-``None`` ``w_trust`` from `_ppgo_band_split`
        implies a non-``None`` ceiling here (both gate on the same
        certified cell).

        Parameters
        ----------
        lens : dict
            Lens parameters (`_lens_params`); ``gamma``, ``y1``, ``y2``
            are used.

        Returns
        -------
        float or None
            The measured ceiling ``w_ceiling`` (raw ``w`` units), or
            ``None`` to keep HEAD behaviour.
        """
        ppgo_map = get_certified_ppgo_map()
        if ppgo_map is None:
            return None
        coords = self._ppgo_cell_coords(lens)
        if coords is None:
            return None
        ceiling = ppgo_map.w_ceiling(*coords)
        if ceiling is UNKNOWN:
            return None
        return float(ceiling)

    def _engine_farfield_total(self, lens, sub_w):
        """Exact amplification total ``F`` at ``sub_w`` via the engine host.

        Thin wrapper over `_evaluate_envelope` that returns only the exact
        total: in the `FARFIELD_DIFFRACTIVE` gauge the envelope IS ``F``, so
        no channel subtraction is needed.  The engine self-certifies with
        its paired N/2N quadrature (``_CERTIFICATION_TOL = 3e-10``); a
        `SchwingerCertificationError` is that certificate declining, which
        this maps to ``None`` for a clean fall-through to refusal.

        Parameters
        ----------
        lens : dict
            Lens parameters (``gamma, beta, kappa, y1, y2``).
        sub_w : np.ndarray
            Positive dimensionless frequencies to host, 1-D.

        Returns
        -------
        np.ndarray or None
            Exact total ``F`` at ``sub_w``, or ``None`` if the engine's
            quadrature certificate declined.
        """
        try:
            _partition, _envelope, exact = self._evaluate_envelope(
                lens, sub_w, pad_w=float(sub_w.max()))
        except SchwingerCertificationError:
            return None
        return exact

    def _low_w_diffractive_chart_serve(self, lens, dense_w, geom):
        """Serve the low-w band from the trained diffractive residual chart.

        Charts the low-w diffractive residual for the positive-parity
        far-field exterior directly, bypassing the `w_low_fit` band split
        and the exact engine entirely for the band the chart owns: the
        near-fold shell (`w_low_fit` declines there -> ``None``) and the
        wall band (`w_low_fit` would split into a tiny analytic sub-band
        plus an engine host).  Mirrors `_low_w_diffractive_serve`'s
        re-modulation and reconstruction tail exactly -- the same
        ``FARFIELD_DIFFRACTIVE`` gauge, ``t_min`` demod/re-mod pair, and
        ``(delays, k0, k1, geom)`` result -- so a chart serve is a drop-in
        replacement for the two-rung serve over the band it covers.

        The residual ``r_pure`` interpolated by the chart is the reduced
        point-mass kernel in units of the macro amplitude times the exact
        point-mass prefactor ``C(w)``: the full amplitude reconstructs as
        ``F = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full * r_pure``,
        with ``mass_sheet_phase = exp(0.5j*w*(log(lam) - kappa*s))`` and
        ``sqrt_mu_full`` from `_born_factors` -- the SAME mass-sheet
        decomposition as the test oracle ``_engine_reference_kappa``
        (``mass_sheet_phase * f_pure / lam``) with ``1/lam`` folded into
        ``sqrt_mu_full``.  `_schwinger.f_schwinger` is NEVER called on this
        path: the chart is the sole serve-time source for the band it owns.

        Parameters
        ----------
        lens : dict
            Lens parameters (``gamma, beta, kappa, y1, y2``).
        dense_w : np.ndarray
            Full dimensionless-frequency grid, 1-D, strictly positive.
        geom : ChangRefsdalGeometryPartition
            Geometry-only partition over ``dense_w`` (delays, saddle
            kernels, real mask, ``t_min``) already in hand.

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, geom)`` on a full-band chart serve, or
            ``None`` to fall through (chart absent, out of coverage, a
            reduced-shear domain refusal, or a draw in a per-cell declined
            cell the training oracle flagged as unable to meet the
            certification bar).
        """
        chart = self.low_w_diffractive_chart
        if chart is None:
            return None

        gamma = float(lens['gamma'])
        beta = float(lens['beta'])
        kappa = float(lens['kappa'])
        try:
            lam, gamma_prime = _reduced_shear(gamma, kappa)
        except DiffractiveDomainError:
            return None
        if gamma_prime == 0.0:
            # No shear -> no caustic frame; `_caustic_rho` would divide by
            # a degenerate zero-radius caustic.  Fall through to the
            # `w_low_fit` path, which returns 0.0 (series exact) here.
            return None

        root = math.sqrt(lam)
        y1 = float(lens['y1'])
        y2 = float(lens['y2'])
        yp0, yp1 = y1 / root, y2 / root
        s = yp0 * yp0 + yp1 * yp1
        if not s > 0.0:
            return None
        z_eig = cmath.exp(-1j * beta) * complex(yp0, yp1)
        theta = math.atan2(z_eig.imag, z_eig.real)

        rho = _caustic_rho(abs(gamma_prime), s, theta)

        if not chart.covers(gamma_prime, rho, dense_w):
            return None

        if chart.declined(gamma_prime, rho, theta):
            # A measured per-cell decline: the served two-sided error cannot
            # meet the certification bar in this cell (near-fold resonance),
            # so fall through to the exact engine -- never an amplitude scale.
            return None

        r_fit = chart.evaluate(dense_w, gamma_prime, rho, theta) * chart.derate
        sqrt_mu_full = _born_factors(y1, y2, gamma, beta, kappa)[0]

        mass_sheet_phase = np.exp(
            0.5j * dense_w * (math.log(lam) - kappa * s))
        prefactor = np.array(
            [prefactor_c(float(w)) for w in dense_w], dtype=complex)
        farfield = mass_sheet_phase * prefactor * sqrt_mu_full * r_fit

        # Demodulate F by t_min into the far-field envelope and reconstruct
        # via the SAME tail as `_low_w_diffractive_serve` (FARFIELD_DIFFRACTIVE).
        envelope = farfield * np.exp(1j * _frame_phase(dense_w, geom.t_min))
        kernels, _total = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_DIFFRACTIVE, geom.t_min)
        k0, k1 = self._reduce_dense_kernels(kernels)
        delays = self._image_delays(lens, geom)
        return delays, k0, k1, geom

    def _low_w_diffractive_serve(self, lens, dense_w, geom, w_lo, w_hi):
        """Serve the far-field diffractive bottom below ``farfield_w_floor``.

        Replaces the F070 refusal -- below the far-field kernel-sum floor
        the stored envelope DIVERGES, so the kernel-sum family cannot be
        served there -- with the two-rung diffractive serve, routed through
        the one gauge that is finite at ALL ``w``: `FARFIELD_DIFFRACTIVE`
        (switch ``0`` everywhere, "subtract nothing", the envelope IS the
        bounded smooth ``F``).  The WHOLE dense band is reconstructed in that
        single gauge; the chart's upper-band kernel-sum serve is
        deliberately NOT preserved here (gauge-mixing / chart preservation is
        deferred to WP2b/WP2c), so this rung either serves the whole band or
        returns ``None`` -- a byte-identical fall-through to the prior F070
        refusal path.

        Rung P (positive parity, ``det A > 0`` i.e. ``gamma < 1``): the
        low-``w`` truncation certificate `w_low_fit` admits the analytic
        series ``F_P`` (`diffractive_amplification`) on ``w < w_low``; the
        exact engine hosts ``[w_low, w_hi]``.  A single `_band_split_mask`
        at ``w_low`` (analytic below / engine above).  The c3/Born nested
        split at ``w_split`` is deferred to WP2b/WP2c.  ``w_low`` is an
        O(1) parametric surface fitted to the engine-honest truncation
        ceiling (the largest ``w`` whose order-``M`` series stays within
        `CERTIFICATION_BAR` of the exact Schwinger engine), de-rated to be
        conservative on its calibration grid -- so admission here is
        certificate-verified, never merely closed-form.

        Rung S (macro saddle, ``det A < 0`` i.e. ``gamma > 1``): the Fermat
        moments diverge, so there is NO analytic series.  The exact engine
        HOSTS the whole low-``w`` band, self-certified by its paired N/2N
        quadrature.  Reachability is capped PER DRAW at ``W_reach =
        min(w_split, W_CEILING_SCHWINGER)`` (the cheap direct-double engine
        ceiling, read from `_schwinger`, never a hardcoded ``60``); a band
        topping out above ``W_reach`` refuses.  ``w_split`` is the c3
        certificate split (`_saddle_c3_split_point`), ``None`` for a merging
        pair (no cap beyond the engine ceiling; the engine's own certificate
        gates the near-caustic corner).  This is engine hosting WITH a
        certificate -- a strict improvement over refusal, NOT an analytic
        closure -- so the census must count it as engine demand (WP3).

        Both rungs demodulate ``F`` by ``geom.t_min`` (READ from the
        geometry partition, never recomputed) into the frame-invariant
        far-field envelope ``E = F * exp(+1j w t_min)`` and hand it to
        `reconstruct_farfield` under `FARFIELD_DIFFRACTIVE`, which
        re-modulates by ``exp(-1j w t_min)`` and rebuilds the channel
        kernels exactly as the trained far-field serve mirror does.

        Parameters
        ----------
        lens : dict
            Lens parameters (``gamma, beta, kappa, y1, y2, m_lens_msun,
            z_lens``).  ``kappa`` and ``beta`` are used AS GIVEN and may be
            nonzero: Rung P handles ``kappa != 0`` / ``beta != 0`` through
            the reduced-shear map (``lam = 1 - kappa``; reduced shear
            ``gamma' = gamma / lam``; the source is rescaled by ``sqrt(lam)``
            and rotated by ``beta`` into the eigenframe, with a ``1 / lam``
            amplitude prefactor, so ``F_P -> sqrt(mu_macro)`` as ``w -> 0``)
            inside `_diffractive.diffractive_amplification` /
            `w_low_fit`, which receive ``lens['beta']`` and
            ``lens['kappa']`` verbatim.
            There is NO upstream ``kappa == 0`` / ``beta == 0`` guard: the
            sole gate on the calling path (the ``_amplification_coefficients``
            diffractive intercept, ~L3157-3161) is
            ``int(geom.real_mask.sum()) == 2`` -- positive-parity image-count
            selection of the far-field exterior -- and it forwards ``lens``
            unmodified.  Admission onto the Rung-P analytic sub-band is
            therefore governed SOLELY by the truncation certificate
            `w_low_fit` (which returns ``None`` on degenerate geometry, on
            draws inside the near-fold shell (``rho`` in
            ``[_DIFFRACTIVE_FIT_FENCE_RHO_LO, 1 +
            _DIFFRACTIVE_FIT_FENCE_DELTA]``) -- falling through
            byte-identically to the wall refusal -- and raises at the
            reduced-shear wall), not by any ``kappa`` /
            ``beta`` precondition.
        dense_w : np.ndarray
            Full dimensionless-frequency grid, 1-D, strictly positive.
        geom : ChangRefsdalGeometryPartition
            Geometry-only partition over ``dense_w`` (delays, saddle
            kernels, real mask, ``t_min``, images) already in hand.
        w_lo, w_hi : float
            ``dense_w`` band extremes (``dense_w.min()`` / ``.max()``).

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, geom)`` on a full-band serve, or ``None`` to
            fall through to the prior F070 refusal path byte-identically.
        """
        gamma = float(lens['gamma'])
        farfield = np.zeros(dense_w.shape, dtype=complex)

        if gamma < 1.0:
            # Rung P: consult the trained low-w diffractive residual chart
            # FIRST.  The chart owns the near-fold shell (where `w_low_fit`
            # declines -> None) and the wall band (which `w_low_fit` would
            # split into a tiny analytic sub-band plus an engine host), so
            # a full-band chart serve short-circuits the split below.  On
            # any miss (chart absent or out of coverage) it returns None
            # and the `w_low_fit` split runs unchanged.
            served = self._low_w_diffractive_chart_serve(lens, dense_w, geom)
            if served is not None:
                return served

            # Rung P: analytic F_P below the truncation certificate w_low,
            # exact engine host above it.
            y = (lens['y1'], lens['y2'])
            try:
                w_low = w_low_fit(
                    y, gamma, lens['beta'], lens['kappa'], w_hi=w_hi)
            except DiffractiveDomainError:
                return None
            if w_low is None or w_low <= w_lo:
                # Null case: no admissible analytic sub-band inside the
                # band.  Refuse exactly as the F070 fall-through did.
                return None
            _split, below_low = _band_split_mask(dense_w, w_low)
            try:
                for idx in np.flatnonzero(below_low):
                    farfield[idx] = diffractive_amplification(
                        float(dense_w[idx]), y, gamma,
                        lens['beta'], lens['kappa'])
            except HypergeometricDomainError:
                return None
            above_low = ~below_low
            if above_low.any():
                host = self._engine_farfield_total(lens, dense_w[above_low])
                if host is None:
                    return None
                farfield[above_low] = host
        else:
            # Rung S: no analytic series (divergent Fermat moments); host
            # the exact engine over the whole band under the per-draw
            # reachability cap.  ``geom.images`` is ALREADY real-only -- do
            # NOT index it with the length-4 channel real_mask.
            source = np.array([lens['y1'], lens['y2']], dtype=float)
            matrix = macro_matrix(gamma, lens['beta'], lens['kappa'])
            w_split = _saddle_c3_split_point(
                np.asarray(geom.images, dtype=float), source, matrix)
            w_reach = (W_CEILING_SCHWINGER if w_split is None
                       else min(w_split, W_CEILING_SCHWINGER))
            if w_hi > w_reach:
                return None
            host = self._engine_farfield_total(lens, dense_w)
            if host is None:
                return None
            farfield[:] = host

        # Demodulate F by t_min (READ from geom) into the frame-invariant
        # far-field envelope and reconstruct via the diffractive serve
        # mirror.  `_frame_phase` (w t_min reduced mod 2*pi) is the SAME
        # authoritative phase `reconstruct_farfield` re-modulates with, so
        # the demod/re-mod pair cancels to machine precision.
        envelope = farfield * np.exp(1j * _frame_phase(dense_w, geom.t_min))
        kernels, _total = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_DIFFRACTIVE, geom.t_min)
        k0, k1 = self._reduce_dense_kernels(kernels)
        delays = self._image_delays(lens, geom)
        return delays, k0, k1, geom

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

        Per-node band split (Build 8h-a WP2).  When a certified-ppGO map
        is installed (`_ppgo_band_split`) and this draw's cell is
        certified, the dense ``w`` band is split at the map's trusted
        floor ``w_trust``: nodes at or below ``w_trust`` are served by the
        chart (over the sub-band slice, so whole-band containment is
        checked against the sub-band), and nodes above ``w_trust`` are
        served by the bare point-mass ppGO -- the far-field kernel sum
        with the wave correction ``E_ff = 0``, which telescopes to the
        existing image-kernel sum (no new formula).  Both segments share
        the ONE full-band partition, so their per-channel kernels are
        w-ordered by construction and reduce through the unchanged
        `_reduce_dense_kernels`.  Beyond-wall / out-of-grid cells return
        `UNKNOWN` from the map, so no split is attempted and any node no
        rung certifies still raises its named refusal.  The split applies
        only to the far-field label; a tube candidate is never band-split
        (the ppGO telescoping identity is a far-field-gauge identity).
        When the map is None or the cell is `UNKNOWN` the flow is
        byte-identical to the whole-band path.

        Returns ``None`` -- signalling the caller to fall through to the
        exact path -- when the candidate carries ``kappa != 0`` (the
        surrogate is a ``kappa = 0`` surface), the dense ``w`` grid is
        non-positive, the cheap `may_serve` pre-check fails, or the
        surrogate declines to serve.  A `geometry.LensDomainError` raised
        by the geometry-only partition is NOT caught: it propagates
        exactly as the exact path's seed evaluation would raise it,
        preserving the refusal set.  (The surrogate is never queried where
        the engine would raise a `SchwingerCertificationError`: each
        chart's domain gate excludes those points via its
        training-refusal exclusion balls.)

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

        # The surrogate is likewise a beta = 0 surface BY CONSTRUCTION.
        # ``serve`` de-rotates the source position into the shear eigenframe
        # (via y1, y2) but passes ``theta`` (the caustic arc angle)
        # UN-rotated, so the served geometry corresponds to
        # ``theta = theta_eig + beta`` exactly; the emulator was trained only
        # on the beta = 0 surface, so serving a beta != 0 candidate would use
        # a mis-rotated caustic angle — finite-but-wrong, the same
        # never-serve-where-wrong violation the kappa guard prevents.  Fall
        # through to the exact engine, which handles beta fully.  Latent only
        # because production pins beta = 0.0 (Inspector latent finding).
        if lens['beta'] != 0.0:
            return None

        dense_w = dimensionless_frequency(
            self._kernel_dense_f, lens['m_lens_msun'], lens['z_lens'])
        if not np.all(dense_w > 0):
            # Let the exact path raise the LensedBinningError, unchanged.
            return None

        surrogate = self.amplification_surrogate

        # Per-node band split (Build 8h-a WP2).  If a certified-ppGO map
        # is installed AND this draw's cell is certified, split the dense
        # w band at the trusted floor ``w_trust`` (read from the map, never
        # a hardcoded constant): nodes at or below w_trust are chart-served,
        # nodes above are served by the bare point-mass ppGO reconstruction
        # (the far-field kernel sum with the wave correction ``E_ff = 0``,
        # which telescopes to the existing image-kernel sum).  When the map
        # is None or the cell is UNKNOWN, w_trust is None and the flow is
        # byte-identical to the whole-band path (surrogate None short-
        # circuits even earlier, via the caller's ``is not None`` guard).
        w_trust = self._ppgo_band_split(lens)
        w_lo = float(dense_w.min())
        w_hi = float(dense_w.max())
        # F005 / beyond-ceiling guard (INS-8haf-002, Build 8h-b): the map
        # certifies a CELL by geometry, but certification only exists BELOW
        # the parity's Schwinger wall AND only over the cell's MEASURED
        # range ``[w_cert, w_ceiling]`` -- the exact reference does not
        # exist above either bound.  The effective ceiling is therefore the
        # tighter of the two, ``min(parity_wall, cell_ceiling)`` (the cell
        # ceiling read from the SAME cell as ``w_trust``).  A draw whose
        # band tops out beyond that must NOT be band-split (bare ppGO would
        # silently serve uncertified beyond-wall / beyond-measured-ceiling
        # nodes); fall through to the whole-band path, which refuses loudly
        # exactly as HEAD does.  When the cell ceiling is UNKNOWN the
        # effective ceiling is the wall alone -- byte-identical to HEAD.
        if w_trust is not None:
            wall = (ASTROID_WALL if float(lens['gamma']) < 1.0
                    else SADDLE_WALL)
            cell_ceiling = self._ppgo_cell_ceiling(lens)
            eff_ceiling = (wall if cell_ceiling is None
                           else min(wall, cell_ceiling))
            if w_hi > eff_ceiling:
                w_trust = None
        band_split = w_trust is not None and w_lo < w_trust < w_hi

        # The chart segment is the sub-band the surrogate must actually
        # serve, so whole-band containment (`may_serve` / `select_chart`)
        # is checked against THIS slice, not the full band.  Without a
        # split it is the whole band (byte-identical to HEAD).
        below_mask = ((dense_w <= w_trust) if band_split
                      else np.ones(dense_w.shape, dtype=bool))
        chart_w = dense_w[below_mask]
        if chart_w.size == 0:
            return None

        # Cheap pre-check on (gamma, chart sub-band) BEFORE building the
        # geometry partition.  Weakened from the full band to the chart
        # sub-band (Build 8h-a WP2): a band-splittable candidate whose FULL
        # band overflows every chart's log-w box can still have a servable
        # lower sub-band.  This stays a pure performance gate -- select_chart
        # below still returns None (fall through) if no chart covers the
        # sub-band, so it never serves where a chart would refuse.
        log_chart_w = np.log(chart_w)
        if not surrogate.may_serve(
                lens['gamma'], float(log_chart_w.min()),
                float(log_chart_w.max())):
            return None

        # Cheap geometry-only partition over the FULL dense band (no exact
        # total).  A candidate-side `geometry.LensDomainError` propagates
        # UNSWALLOWED.  The chart segment and the ppGO segment both reduce
        # through this ONE partition, so their per-channel kernels share
        # the same minimum-relative delays and channel structure and the
        # w-ordered "concatenation" is structural: the intact full-band
        # array carries the chart envelope below w_trust and E_ff = 0 above,
        # so no reordering of the bin/sub-sample grid is ever needed.
        geom = ChangRefsdalChannels(dense_w).geometry_partition(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])

        # Full multi-chart guard stack on the CHART sub-band slice only:
        # chart selection keys on the certified physical caustic distance +
        # real-image count (theta is used only for cusp exclusion) and its
        # whole-band containment is thus enforced against the sub-band.
        # Recomputes NO geometry.
        envelope_chart, served, definition = surrogate.serve(
            chart_w, gamma=lens['gamma'], y1=lens['y1'], y2=lens['y2'],
            beta=lens['beta'], eta=geom.caustic_distance,
            theta=geom.caustic_theta,
            image_count=int(geom.real_mask.sum()))

        # LOW-END WINDOW GUARD (F070).  `_log_w_band_serveable` leaves the
        # low end open and clamps `w < chart.w_min` to the first grid point,
        # justified by the envelope being flat below the first Airy fringe.
        # That holds for the SACR-C envelope and is FALSE for the kernel-sum
        # family, which DIVERGES into the diffractive bottom below the
        # region's `farfield_w_floor`.  The trainer clips every exterior tile
        # to [w_floor, w_trust]; nothing re-checked it at serve time, so a
        # correctly tiled kernel-sum chart queried below its floor passed
        # every admission gate and served 468x max|F| wrong.  Refuse instead
        # and fall through -- the geometry needed is already in hand, so this
        # costs one min-delay scan.
        if served and definition in _FARFIELD_KERNEL_FAMILY:
            w_floor = farfield_w_floor(geom.delays, geom.real_mask)
            if float(chart_w.min()) < w_floor:
                # F070 low-w window.  The kernel-sum family DIVERGES below
                # the floor, so the chart cannot serve here (Build 8h; once
                # served 468x max|F| wrong).  Attempt the two-rung
                # diffractive serve instead: analytic ``F_P`` below the
                # ``w_low`` truncation certificate for positive parity, and
                # an exact-engine host with a per-draw reachability cap for
                # macro saddle -- reconstructed over the WHOLE band in the
                # finite `FARFIELD_DIFFRACTIVE` gauge.  On any refusal it
                # returns None and we fall through byte-identically to the
                # prior F070 refusal path (served = False).
                low_w_result = self._low_w_diffractive_serve(
                    lens, dense_w, geom, w_lo, w_hi)
                if low_w_result is not None:
                    return low_w_result
                served = False
                envelope_chart = None

        if not served:
            # Fact-4 slot (Born weak-deflection rung): serve the analytic
            # carrier + trained residual chart for configurations in the
            # Born exterior (rho > 1.0).  Falls through to the exact engine
            # when the chart is not attached or doesn't cover the config.
            born_chart = self.born_residual_chart
            if born_chart is None:
                return None
            abs_y = math.hypot(lens['y1'], lens['y2'])
            try:
                rho = caustic_rho(lens['gamma'], abs_y, lens['kappa'])
            except (ValueError, LensDomainError):
                return None
            if rho <= 1.0 or not born_chart.covers(lens['gamma'], rho):
                # --- Raw-ppGO interior handoff (Build ppgo_interior_certificate) ---
                # Serve raw geometric optics on the TRUE caustic interior,
                # keyed on the EXACT interior predicate: exactly four real
                # images.  ``ppgo_map.caustic_rho`` normalises ``|y|`` by the
                # caustic's MAXIMUM angular reach, so ``rho <= 1`` is NECESSARY
                # but not sufficient for the interior (F073 -- it admits
                # exterior sources at 58.7% of points); the four-real-image
                # count is the exact, free predicate (0/2400 disagreements vs
                # the closed-form caustic).  Four real roots also prove the
                # ghost is exactly zero (geometry.GhostAbsentError), so NO
                # ghost term enters and no per-serve ghost_kernel is called.
                # Admission uses the raw-ppGO leading-omitted-term certificate
                # -- the ``w**-3`` (c3) estimate DERIVED from what this rung
                # serves, not the fold arm's uniform estimate -- with a modest
                # safety factor ``_PPGO_INTERIOR_SAFETY``.  On the measured
                # interior it is conservative with zero optimistic points (max
                # true/certificate ratio 0.980 over 1248 samples, Fact 3).
                # The former fold-resolution leg (xi_min >= _XI_FOLD_THRESHOLD)
                # is dropped from this rung: over the interior evidence sweep
                # EVERY 4-image config fails that leg (deep-interior fold pairs
                # give xi_min < 4), yet the c3 certificate at S = 2.0 admits
                # 230 of those band points with MAX true error 4.8e-5 <
                # CERTIFICATION_BAR (1e-4) and NONE over the bar -- so the
                # fold-resolution leg only suppressed certifiable service.
                # Near-caustic merging configs (sqrt|mu| divergent) self-refuse
                # through ``ppgo_error_estimate`` (None or an over-bar estimate)
                # and never reach the serve.  Reconstruction mirrors the Born
                # rung (reconstruct_farfield with FARFIELD_KERNEL_SUM).
                if int(geom.real_mask.sum()) == 4:
                    try:
                        from cogwheel.lensing.chang_refsdal.geometry import (
                            ppgo_error_estimate)
                        from cogwheel.lensing.chang_refsdal.operator import (
                            geometric_amplification)
                        source = np.array([lens['y1'], lens['y2']],
                                          dtype=float)
                        matrix = macro_matrix(
                            lens['gamma'], lens['beta'], lens['kappa'])
                        real = np.asarray(geom.real_mask, dtype=bool)
                        real_images = np.asarray(geom.images)[real]
                        w_min = float(dense_w.min())
                        est = ppgo_error_estimate(
                            real_images, source, matrix, w_min)
                        if (est is not None
                                and est * _PPGO_INTERIOR_SAFETY
                                <= CERTIFICATION_BAR):
                            # All gates pass -- serve RAW ppGO (F069): the
                            # far-field envelope is the exact minus-relative
                            # amplification with the resolved ppGO channels
                            # subtracted, demodulated to the caustic frame.
                            f_total = np.atleast_1d(
                                geometric_amplification(
                                    dense_w, source, lens['gamma']))
                            f_minrel = f_total * np.exp(
                                -1j * dense_w * geom.t_min)
                            ppgo_sum = np.sum(
                                geom.saddle_kernels[:, real]
                                * np.exp(
                                    1j * dense_w[:, None]
                                    * geom.delays[real][None, :]),
                                axis=1)
                            envelope = (
                                (f_minrel - ppgo_sum)
                                * np.exp(1j * dense_w * geom.t_min))
                            kernels, _total = reconstruct_farfield(
                                dense_w, envelope, geom.delays,
                                geom.saddle_kernels, geom.real_mask,
                                FARFIELD_KERNEL_SUM, geom.t_min)
                            k0, k1 = self._reduce_dense_kernels(kernels)
                            delays = self._image_delays(lens, geom)
                            return delays, k0, k1, geom
                    except (LensDomainError, ValueError,
                            ZeroDivisionError):
                        pass  # Structural refusal: fall through.
                return None
            # Trained-band refusal (mirrors _born_residual_analytic): the
            # residual interpolator cubic-extrapolates off the trained
            # ``log_w_grid`` (fill_value=None).  This buried rung serves the
            # WHOLE ``dense_w`` band, so decline and fall through to the exact
            # engine when that band escapes the trained frequency range rather
            # than serve a finite-but-wrong extrapolated residual.
            if not born_chart.covers(lens['gamma'], rho, dense_w):
                return None
            # Build a duck-typed namespace adapter for
            # born_carrier_from_partition (which reads attributes by name).
            partition_ns = types.SimpleNamespace(
                w=dense_w,
                source=np.array([lens['y1'], lens['y2']]),
                gamma=lens['gamma'],
                beta=lens['beta'],
                kappa=lens['kappa'],
                matrix=macro_matrix(
                    lens['gamma'], lens['beta'], lens['kappa']),
                t_min=geom.t_min,
                delays=geom.delays,
                saddle_kernels=geom.saddle_kernels,
                real_mask=geom.real_mask,
                images=geom.images)
            # Deferred import avoids cycle risk (born_carrier_from_partition's
            # module imports channels which may circle back at module load).
            from cogwheel.lensing.chang_refsdal.channels import (
                born_carrier_from_partition)
            carrier = born_carrier_from_partition(partition_ns)
            residual = born_chart.evaluate(dense_w, lens['gamma'], rho)
            f_total = carrier + residual
            # Reconstruct channel kernels via the far-field reconstruction
            # path: extract the far-field envelope by subtracting the
            # resolved ppGO channels and demodulating.
            real = np.asarray(geom.real_mask, dtype=bool)
            ppgo = np.sum(
                geom.saddle_kernels[:, real]
                * np.exp(1j * dense_w[:, None] * geom.delays[real][None, :]),
                axis=1)
            envelope = (f_total - ppgo) * np.exp(1j * dense_w * geom.t_min)
            kernels, _total = reconstruct_farfield(
                dense_w, envelope, geom.delays, geom.saddle_kernels,
                geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)
            k0, k1 = self._reduce_dense_kernels(kernels)
            delays = self._image_delays(lens, geom)
            return delays, k0, k1, geom

        if definition in KNOWN_FARFIELD_DEFINITIONS:
            # Far-field serve mirror (Build 8h-b3-fin S1-2): reconstruct
            # ``F`` by inverting the SAME w-windowed window-class label the
            # chart was trained on, dispatched on its envelope-definition
            # tag through `reconstruct_farfield` (which mirrors
            # `farfield_envelope_from_partition`: the switch policy is
            # `_farfield_switch(definition)` and the carrier is parked at
            # ``tau_c = 0``).  On a band split the wave correction above
            # ``w_trust`` is certified below the ppGO bar, so ``E_ff = 0``
            # there and the upper-band kernels collapse to the bare
            # image-kernel sum -- the kernel-sum-gauge telescoping identity.
            # That identity holds ONLY for the kernel-sum family (real
            # switch = 1), NOT for the diffractive (switch = 0) gauge, so a
            # diffractive band split is refused (fall through to the exact
            # path).  Without a split ``envelope_dense`` equals the served
            # envelope exactly (below_mask is all True), and for the legacy
            # `FARFIELD_KERNEL_SUM` tag this path is identical to HEAD up to
            # the frame demod/re-mod round-trip (label demodulated by
            # ``exp(+1j w t_min)``, reconstruct_farfield re-modulates by
            # ``exp(-1j w t_min)``) -- differences are ~machine eps.
            if definition == FARFIELD_DIFFRACTIVE and band_split:
                return None
            envelope_dense = np.zeros(dense_w.shape, dtype=complex)
            envelope_dense[below_mask] = envelope_chart
            if definition == FARFIELD_KERNEL_SUM_MINUS_GHOST:
                # The mid-band ghost label subtracted the decaying
                # complex-saddle ghost ``G`` analytically; re-add it over
                # the chart region with the SAME primitive and gate
                # (`farfield_ghost_term`, source/matrix rebuilt exactly as
                # the training partition did) so ``F`` telescopes to machine
                # precision.  The ghost gate is the GEOMETRIC separation
                # ``min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN``, which is
                # frequency-independent: it reads only ``(source, matrix)``,
                # so this serve mirror and the training label reach a
                # PROVABLY IDENTICAL admit/refuse decision for a fixed config
                # -- the old ``w_min * Im tau_c`` decay gate, keyed on the
                # caller's ``w`` array, could re-add a ghost the label never
                # subtracted; the separation gate removes that skew by
                # construction.  ``G`` lives in the mid-band window only,
                # never in the bare ppGO band above ``w_trust`` (where
                # ``envelope_dense`` is already zero).  A GhostDomainError
                # (ghost inseparable from a real image near a cusp / inside
                # the caustic) refuses symmetrically with the training label:
                # fall through to the exact path.
                # ``geom.t_min`` is the frame origin the geometry partition
                # already solved for; passing it spares a second image-quartic
                # solve on this per-likelihood-evaluation path.
                source = np.array([lens['y1'], lens['y2']], dtype=float)
                matrix = macro_matrix(
                    lens['gamma'], lens['beta'], lens['kappa'])
                try:
                    ghost = farfield_ghost_term(
                        chart_w, source, matrix, t_min=geom.t_min,
                        real_images=geom.images)
                except GhostDomainError:
                    return None
                # Re-add the ghost in the SAME frame-invariant (demodulated)
                # convention the stored far-field label uses (Build 8h-d2):
                # the training label subtracted the min-relative ghost BEFORE
                # the ``exp(+1j w t_min)`` demodulation, so the serve mirror
                # adds it back with the same ``+t_min`` tilt.
                # `reconstruct_farfield` then de-tilts label-plus-ghost
                # together back to the min-relative frame in one
                # ``exp(-1j w t_min)`` multiply, so the telescoping stays exact
                # to machine precision.
                envelope_dense[below_mask] += ghost * np.exp(
                    1j * chart_w * geom.t_min)
            kernels, _total = reconstruct_farfield(
                dense_w, envelope_dense, geom.delays, geom.saddle_kernels,
                geom.real_mask, definition, geom.t_min)
        elif band_split:
            # A tube OR a whole-interior SACR-C chart carries the
            # caustic-region ``tau_c``-demodulated envelope reconstructed with
            # the geometry's own switch / critical delay; the far-field-gauge
            # ppGO telescoping identity above does NOT hold for that gauge, so
            # a band split of such a chart is not served -- fall through to the
            # exact path.  Tubes and interior charts live where the map returns
            # UNKNOWN (near-caustic / inside the caustic), so band_split is
            # normally already False for them; this guard is belt-and-braces.
            return None
        else:
            # SACR-C caustic-region envelope, dispatched here for BOTH a tube
            # chart (``definition is None``) and a whole-interior chart
            # (``definition in KNOWN_INTERIOR_DEFINITIONS``, the S2-3
            # `INTERIOR_SACR_C` tag): reconstruct with the geometry's OWN
            # switch and critical delay, i.e. add the switched near-merged
            # channels back at ``tau_c``.  The reconstruction algebra is
            # identical for tube and interior (`reconstruct_from_envelope`),
            # so no separate branch is warranted; the far-field family is the
            # only case handled above.  ``definition`` here is guaranteed to
            # be ``None`` or an interior tag by the loader's tag validation.
            assert (definition is None
                    or definition in KNOWN_INTERIOR_DEFINITIONS), (
                f'unexpected non-interior envelope tag {definition!r} reached '
                'the caustic-region reconstruction branch')
            kernels, _total = reconstruct_from_envelope(
                dense_w, envelope_chart, geom.delays, geom.saddle_kernels,
                geom.switch, geom.critical_delay)
        k0, k1 = self._reduce_dense_kernels(kernels)
        delays = self._image_delays(lens, geom)
        return delays, k0, k1, geom

    def _ppgo_above_ceiling(self, lens, dense_w):
        """Split-band ppGO serve when w_max exceeds the Schwinger QD ceiling.

        Fires only when ``w_max > W_CEILING_SCHWINGER_QD`` (=150), where the
        exact engine hard-refuses the above-ceiling nodes.  Admits only when
        the LOWEST above-ceiling node is resolved
        (``W_CEILING_SCHWINGER_QD * min_delta_tau >= RHO_END``), which
        guarantees EVERY above-ceiling node -- the ones the engine refuses
        and fold_ppgo must carry -- is resolved and the fold-corrected ppGO
        carrier is accurate there.

        The band is split at the ceiling (`_band_split_mask` with
        ``split = W_CEILING_SCHWINGER_QD``): the exact engine serves every
        node at or below 150 via `_engine_envelope_below_split` (always
        engine-reachable), and the fold-corrected ppGO carrier serves every
        node above 150.  The two envelopes are zeroed on each other's nodes
        and summed, so they stitch over the full band with no double count,
        then reconstructed under ``FARFIELD_KERNEL_SUM``.  When the whole
        band is above the ceiling (``w_lo >= 150``) the engine serves
        nothing and fold_ppgo carries the entire band -- byte-identical to
        HEAD's whole-band ppGO serve.

        On any gate miss returns ``None`` -- the caller falls through to the
        exact engine, which raises ``SchwingerCertificationError`` unchanged
        from HEAD (the deferred 2b residual for an unresolved above-ceiling
        corner, not a bug).

        Parameters
        ----------
        lens : dict
            Lens parameters from `_lens_params`.
        dense_w : np.ndarray
            Dimensionless frequency grid for the kernel subsamples.

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, partition)`` on success, ``None`` to fall
            through.
        """
        w_max = float(dense_w.max())
        if w_max <= W_CEILING_SCHWINGER_QD:
            return None

        geom = ChangRefsdalChannels(dense_w).geometry_partition(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])

        real = np.asarray(geom.real_mask, dtype=bool)
        real_delays = np.asarray(geom.delays)[real]
        if len(real_delays) < 2:
            return None
        sorted_delays = np.sort(real_delays)
        delta_taus = np.diff(sorted_delays)
        positive_deltas = delta_taus[delta_taus > 0]
        if len(positive_deltas) == 0:
            return None
        min_delta_tau = float(np.min(positive_deltas))
        # Ceiling-keyed resolution gate: min_delta_tau is a per-DRAW
        # geometric constant and w is linear in f, so the analytic-eligible
        # set collapses to a single threshold at the ceiling.  Admit only
        # when the LOWEST above-ceiling node (w just above the ceiling) is
        # resolved, which guarantees EVERY above-ceiling node is resolved
        # and fold_ppgo is accurate there.  Keying on the ceiling (not
        # w_lo) is conservative when the whole band is above the ceiling
        # (w_lo > 150): a draw whose above-ceiling corner is unresolved
        # returns None and falls through to the exact engine ->
        # SchwingerCertificationError (the deferred 2b residual, not a bug).
        if W_CEILING_SCHWINGER_QD * min_delta_tau < RHO_END:
            return None

        # Split the band at the Schwinger ceiling: the exact engine serves
        # every node at or below the ceiling (always engine-reachable) and
        # the fold-corrected ppGO carrier serves every node above it.
        split = float(W_CEILING_SCHWINGER_QD)
        band_split, below_mask = _band_split_mask(dense_w, split)
        if not band_split:
            # The entry guard fixes w_hi > split, so an inactive split can
            # only mean split <= w_lo: the WHOLE band is above the ceiling
            # and the exact engine serves nothing.  `_band_split_mask`
            # returns an all-True below_mask for an inactive split -- its
            # convention serves below-populator rungs, whose trusted floor
            # sits ABOVE the band; this ceiling rung's engine populates
            # BELOW the split (the opposite polarity), so collapse the mask
            # to all-False and let fold_ppgo carry every node (the deep-
            # massive-lens asymptote, byte-identical to HEAD's whole-band
            # ppGO serve).
            below_mask = np.zeros(dense_w.shape, dtype=bool)

        # Exact-engine far-field envelope below the ceiling (0 above); when
        # the whole band is above the ceiling there is nothing to serve, so
        # skip the engine and leave the below contribution identically 0.
        if band_split:
            engine_below = self._engine_envelope_below_split(
                lens, dense_w, below_mask)
        else:
            engine_below = np.zeros(dense_w.shape, dtype=complex)

        from cogwheel.lensing.chang_refsdal._airy_fold import (
            fold_ppgo_correction)

        source = np.array([lens['y1'], lens['y2']], dtype=float)
        f_total = np.atleast_1d(fold_ppgo_correction(
            dense_w, source, lens['gamma'],
            beta=lens['beta'], kappa=lens['kappa']))

        finite_mask = np.isfinite(f_total)
        f_total = np.where(finite_mask, f_total, 0.0)

        f_minrel = f_total * np.exp(-1j * dense_w * geom.t_min)

        ppgo_sum = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * dense_w[:, None] * real_delays[None, :]),
            axis=1)

        # Fold-corrected ppGO envelope above the ceiling; zero it on the
        # below-split nodes the exact engine already populates so the two
        # carriers stitch over the full band with no double count.
        fold_envelope = (f_minrel - ppgo_sum) * np.exp(
            1j * dense_w * geom.t_min)
        fold_envelope[below_mask] = 0.0

        envelope = engine_below + fold_envelope

        kernels, _total = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        k0, k1 = self._reduce_dense_kernels(kernels)
        delays = self._image_delays(lens, geom)
        return delays, k0, k1, geom

    def _engine_envelope_below_split(self, lens, dense_w, below_mask):
        """Full-length far-field envelope: exact engine below split, 0 above.

        Builds the ``FARFIELD_KERNEL_SUM``-gauge far-field envelope from a
        fresh EXACT-Schwinger evaluation on the below-split sub-band
        ``dense_w[below_mask]`` and scatters it into a full-length array,
        leaving the above-split nodes ``0.0`` -- the analytic zero the
        tier-1 serve carries.  The zeroed above-split region telescopes to
        the bare ppGO image-kernel sum under `reconstruct_farfield`, so the
        split stitches the exact engine below onto the analytic carriers
        above with no field discontinuity.

        The engine geometry is ``w``-INDEPENDENT (deterministic initial
        label assignment; see `_evaluate_envelope`), so the sub-band
        partition's delays / kernels / ``t_min`` / real mask match the
        full-band ``geom`` the caller reconstructs against, and the
        envelope VALUES are frame-consistent node by node.  The label comes
        from the single authoritative far-field producer
        `farfield_envelope_from_partition` -- NOT the SACR-C
        ``partition.envelope``, whose ``critical_delay`` carrier is a
        DIFFERENT gauge that `reconstruct_farfield` does not invert; the
        producer's ``exp(+1j w t_min)`` demodulation is the matched inverse
        of `reconstruct_farfield`'s re-modulation (both via the reduced
        ``_frame_phase``), so the round trip is machine-precision.

        Every below-split node satisfies ``w <= w_split <=
        W_CEILING_SCHWINGER_QD``, so the exact engine is always reachable
        (no ``SchwingerCertificationError``); a merging pair never reaches
        here (refused at the gate).  A geometry ``LensDomainError``
        propagates unswallowed, mirroring the sibling rungs.

        Parameters
        ----------
        lens : dict
            Lens parameters from `_lens_params`.
        dense_w : np.ndarray
            Full dimensionless frequency grid for the kernel subsamples.
        below_mask : np.ndarray
            Boolean mask (``dense_w`` shape) marking the below-split nodes
            the exact engine serves; the complement is left ``0.0``.

        Returns
        -------
        np.ndarray
            Full-length (``dense_w`` shape) complex far-field envelope: the
            exact-engine ``FARFIELD_KERNEL_SUM`` label on the below-split
            nodes, ``0.0`` above.
        """
        sub_w = dense_w[below_mask]
        # Pad with the FULL-band ceiling (a node strictly above the split,
        # since w_split < w_hi in the split case) so a single below-split
        # node still satisfies the engine's two-point minimum.  The padding
        # node is dropped by ``keep`` and the geometry is w-independent, so
        # it cannot perturb the served sub-band values.  (This uses the
        # full-band max rather than the sub-band max so the size-1 sub-band
        # gets a DISTINCT second node -- padding with the sub-band's own
        # single value would collapse the grid back to one point.)
        partition, _sacrc_env, _exact_total = self._evaluate_envelope(
            lens, sub_w, pad_w=float(dense_w.max()))
        # Deferred import avoids module-load cycle risk (mirrors the Born
        # rung's `born_carrier_from_partition` import).
        from cogwheel.lensing.chang_refsdal.channels import (
            farfield_envelope_from_partition)
        ff_envelope = farfield_envelope_from_partition(
            partition, FARFIELD_KERNEL_SUM)
        keep = np.searchsorted(partition.w, sub_w)
        envelope = np.zeros(dense_w.shape, dtype=complex)
        envelope[below_mask] = ff_envelope[keep]
        return envelope

    def _diffractive_bottom_ceiling(self, lens, *, w_hi=None):
        """Low-``w`` diffractive truncation certificate ``w_low``, or ``None``.

        Thin wrapper over `w_low_fit`: the fitted, conservative frequency
        below which the positive-parity diffractive series ``F_P``
        (`diffractive_amplification`) is certified to the truncation bar.
        It supplies the NESTED low split shared by the c3 and Born
        band-split rungs -- the boundary between the analytic diffractive
        bottom (Rung P) and the engine/chart host.

        The optional ``w_hi`` band cap is forwarded verbatim to `w_low_fit`
        (it caps the fitted ceiling), so a whole-band certificate returns
        ``w_hi`` and collapses the host region to empty.  The null-split
        handling -- the whole band below or above the certificate boundary
        -- is owned by the CALL SITES via ``_band_split_mask(dense_w,
        w_low)`` plus the ``w_low >= dense_w.max()`` whole-band branch;
        this wrapper returns the raw fitted ceiling only.

        Returns ``None`` at the macro-saddle parity wall (``gamma >= 1``,
        `DiffractiveDomainError`), where there is NO positive-parity series
        -- so a saddle-c3 split's nested bottom is empty and its entire
        below-split region is hosted by the exact Schwinger engine (Rung
        S).  Also returns ``None`` for degenerate geometry (propagated from
        `w_low_fit`) and ``0.0`` when there is no shear (series exact);
        both collapse the nested bottom to empty via the whole-band /
        empty-bottom branches at the call sites.  The near-fold-shell
        decline of `w_low_fit` (draws with ``rho`` in
        ``[_DIFFRACTIVE_FIT_FENCE_RHO_LO, 1 +
        _DIFFRACTIVE_FIT_FENCE_DELTA]``) likewise passes through as
        ``None`` and falls through byte-identically to the wall refusal.

        Parameters
        ----------
        lens : dict
            Lens parameters from `_lens_params`.
        w_hi : float or None
            Optional band cap forwarded to `w_low_fit` (``None`` ->
            unbounded above).

        Returns
        -------
        float or None
            ``w_low``; ``w_hi`` when the fitted ceiling reaches ``w_hi``;
            ``None`` at the parity wall, on a degenerate solve, or inside
            the near-fold shell (``rho`` in
            ``[_DIFFRACTIVE_FIT_FENCE_RHO_LO, 1 +
            _DIFFRACTIVE_FIT_FENCE_DELTA]``).
        """
        try:
            return w_low_fit(
                (lens['y1'], lens['y2']), lens['gamma'],
                lens['beta'], lens['kappa'], w_hi=w_hi)
        except DiffractiveDomainError:
            return None

    def _saddle_farfield_analytic(self, lens, dense_w):
        """Tier-1 far-from-caustic macro-saddle serve with a c3 band split.

        Serves the resolvable far-from-caustic macro saddle (``gamma > 1``)
        from the switched analytic channels, splitting the dense ``w`` band
        at the per-draw c3 certificate frequency ``w_split``
        (`_saddle_c3_split_point`): ABOVE ``w_split`` the residual envelope
        is ZERO -- the analytic carriers alone reconstruct the amplification
        to within the certified bar (the omitted ``w**-3`` stationary-phase
        remainder, whose shape the c3 term carries, has decayed below the
        bar); AT OR BELOW ``w_split`` the exact Schwinger engine supplies
        the ``FARFIELD_KERNEL_SUM`` far-field envelope
        (`_engine_envelope_below_split`).  Both regions reconstruct under
        the ``FARFIELD_KERNEL_SUM`` tag over the FULL ``dense_w`` band --
        which parks ``tau_c = 0`` and hardcodes ``S_a = 1`` on the
        saturated set -- so the exact engine below stitches onto the
        analytic carriers above with no field discontinuity.

        The whole-draw admit / refuse envelope is unchanged from HEAD; only
        the previously-refused-but-splittable middle is newly served:

        * Whole-band admit (``w_split <= w_lo``, i.e. the certificate
          already clears the bar at the band floor): the gate
          `_saddle_farfield_analytic_serves` returns ``True`` and the whole
          band is served with a ZERO residual envelope -- no engine call,
          BYTE-IDENTICAL to HEAD.
        * Whole-draw refuse: a genuinely merging pair near the critical
          curve (``ppgo_error_estimate`` -> ``None``, so ``w_split`` is
          ``None``) or an under-separated pair (separation backstop) is
          REFUSED, and a certificate that fails across the whole reachable
          band (``w_split >= w_hi``) or would split beyond the exact
          engine's ceiling (``w_split > W_CEILING_SCHWINGER_QD``) falls
          through.  Each returns ``None`` and the caller falls through to
          the exact seed engine, BYTE-IDENTICAL to HEAD.
        * Band split (``w_lo < w_split < w_hi`` and ``w_split <= 150``):
          exact engine below, analytic zero above.  This is the new serving
          that revives the saddle-c3 route on draws HEAD refused whole.

        The split point is the EXACT cube-root inversion of the certificate
        (`_saddle_c3_split_point`); it is never hardcoded.  The rung never
        uses ``geom.switch`` / ``geom.critical_delay``: the
        ``FARFIELD_KERNEL_SUM`` tag is the switched-analytic sum on the
        saturated set and needs no channel handover.  A geometry
        ``LensDomainError`` propagates unswallowed, mirroring
        `_ppgo_above_ceiling`.

        Parameters
        ----------
        lens : dict
            Lens parameters from `_lens_params`.
        dense_w : np.ndarray
            Dimensionless frequency grid for the kernel subsamples.

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, partition)`` on success, ``None`` to fall
            through.
        """
        geom = ChangRefsdalChannels(dense_w).geometry_partition(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])

        real_images = np.asarray(geom.images)  # already real-only (find_images)
        source = np.array([lens['y1'], lens['y2']], dtype=float)
        matrix = macro_matrix(lens['gamma'], lens['beta'], lens['kappa'])
        w_lo = float(dense_w.min())
        w_hi = float(dense_w.max())

        # Whole-band admit fast path (single source of truth, shared with
        # the census).  A ``True`` here is the HEAD admit: the c3
        # certificate clears the bar at the band floor -- equivalently the
        # split point ``w_split <= w_lo`` -- so the whole band is served
        # with a ZERO residual envelope and no engine call.  The reconstruct
        # below is then BYTE-IDENTICAL to HEAD.
        if _saddle_farfield_analytic_serves(real_images, source, matrix, w_lo):
            envelope = np.zeros(dense_w.shape, dtype=complex)
        else:
            # Gate miss: either a genuine whole-draw refusal (BYTE-IDENTICAL
            # HEAD fall-through) or a certificate that fails at the band
            # floor but admits a band split.  The split point discriminates,
            # replacing the whole-band accuracy bar the gate applies at
            # ``w_lo``.  The est-None (merging pair) and separation-backstop
            # refusals mirror the gate's own two whole-draw refusals.
            min_sep = _saddle_min_image_sep(real_images)
            if min_sep is None:
                # Fewer than two real images: not a resolved 2-image
                # exterior (mirrors the gate's ``len < 2`` refusal).
                return None
            if min_sep < _SADDLE_FARFIELD_MIN_IMAGE_SEP:
                return None
            w_split = _saddle_c3_split_point(real_images, source, matrix)
            if (w_split is None or w_split >= w_hi
                    or w_split > W_CEILING_SCHWINGER_QD):
                # Merging pair (None), certificate fails across the whole
                # band, or the split would fall beyond the exact engine's
                # ceiling: fall through to the exact engine (HEAD refuse).
                return None
            # Reached only with a gate miss AND separation OK, so the
            # accuracy bar failed at ``w_lo`` and ``w_lo < w_split``
            # strictly; with ``w_split < w_hi`` and ``w_split <= 150`` the
            # split lies STRICTLY inside the band.  ``_band_split_mask``
            # therefore returns ``band_split = True`` and marks the
            # below-split nodes; the exact engine serves them and the
            # above-split residual stays zero.
            _band_split, below_mask = _band_split_mask(dense_w, w_split)
            # NESTED low split.  Conceptually the below-split region splits
            # again at the diffractive certificate ``w_low`` (the band-aware
            # honest ceiling): the analytic bottom ``[w_lo, honest_ceiling)``
            # (Rung P) and the exact engine host ``[honest_ceiling, w_split)``.
            # For the macro saddle (``gamma > 1``)
            # there is NO positive-parity diffractive series --
            # `_diffractive_bottom_ceiling` returns ``None`` at the parity
            # wall -- so ``band_split_low`` is ``False``, ``bottom_mask`` is
            # empty, and ``host_mask`` is the WHOLE below-split region.  Rung
            # S (f_schwinger) therefore hosts it in a single engine call,
            # BYTE-IDENTICAL to the un-nested serve.  The two sequential
            # `_band_split_mask` calls reuse the shared split arithmetic (no
            # third copy); the bottom/host boolean composition is the only
            # inline logic.
            w_low = self._diffractive_bottom_ceiling(
                lens, w_hi=float(dense_w.max()))
            band_split_low, below_low = _band_split_mask(dense_w, w_low)
            if w_low is not None and w_low >= float(dense_w.max()):
                # Whole band certified: the analytic diffractive bottom
                # serves the ENTIRE below-split region and the engine/chart
                # host collapses to empty.
                bottom_mask = below_mask
            else:
                bottom_mask = ((below_low & below_mask) if band_split_low
                               else np.zeros(dense_w.shape, dtype=bool))
            host_mask = below_mask & ~bottom_mask
            envelope = self._engine_envelope_below_split(
                lens, dense_w, host_mask)

        kernels, _total = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)

        k0, k1 = self._reduce_dense_kernels(kernels)
        delays = self._image_delays(lens, geom)
        return delays, k0, k1, geom

    def _born_residual_analytic(self, lens, dense_w):
        """First-class Born weak-deflection analytic serve, or ``None``.

        Serves the Born (weak-deflection) exterior directly from the
        attached ``born_residual_chart`` -- WITHOUT any surrogate -- as the
        analytic carrier ``born_carrier_from_partition`` plus (in box) the
        trained residual, reconstructed under the ``FARFIELD_KERNEL_SUM``
        tag.  It is the reachability lift of the buried Born rung in
        `_surrogate_coefficients`: the same carrier + interpolated-residual
        decomposition, but reachable on the production (surrogate-free)
        path.

        GATE.  Serves ONLY when a chart is attached AND ``kappa == 0`` AND
        ``beta == 0`` AND the caustic-frame ``rho = caustic_rho(...) > 2.0``
        (far exterior, two real images).  Interior / near-caustic / tube
        (``rho <= 2``) always falls through to the exact engine -- the
        carrier-only lift below never captures it.

        TRAINED-FLOOR BAND SPLIT.  When the far-exterior query IS box-covered
        but the served host sub-band drops BELOW the chart's trained
        ``log_w`` floor (a low-edge escape), the band is split a second time
        at ``trained_floor = exp(log_w_grid[0])`` -- the low edge of the
        trained coverage read from the shipped artifact, never a literal:
        the chart serves the trained sub-band ``[trained_floor, w_trust]`` it
        was actually trained on, the exact Schwinger engine hosts the
        untrained remainder ``[w_low, trained_floor)`` below it (via
        `_engine_envelope_below_split`), the analytic diffractive bottom
        ``[w_lo, w_low)`` (Rung P) is unchanged, and the bare ppGO carrier
        serves above ``w_trust``.  The two tiers share the
        ``FARFIELD_KERNEL_SUM`` gauge so they stitch with no field
        discontinuity.  This route runs ONLY when the chart sub-band is
        genuinely covered by the trained range (a strict sub-band); a
        high-edge escape or a disjoint trained range skips it and falls
        through to the carrier-only lift below.

        BEYOND-THE-BOX CARRIER-ONLY LIFT.  When the far-exterior query is
        NOT covered by the trained ``(gamma, rho, log_w)`` box -- past the
        astroid-only ``gamma_grid`` (a macro ``gamma > 1`` saddle query the
        artifact never trained), past the trained ``rho`` reach, or when the
        served sub-band escapes the trained ``log_w`` range at the HIGH edge
        / disjointly (the trained-floor split above does not apply) -- the
        residual is served as identically ZERO and ONLY the lead carrier is
        kept, gated by the module-level ``_born_carrier_certificate_serves``
        (carrier-relative truncation certificate at the band ceiling, a
        saddle-only ``w_lo * delta_min >= RHO_END`` resolution fence and the
        shared min-image-separation backstop).  On a certificate refusal the
        rung declines and falls through to the exact engine, exactly as HEAD
        did on the bare ``covers()`` miss.  The FULLY-in-box serve
        (box-covered AND the served sub-band inside the trained log-w range)
        keeps the interpolated residual and is BYTE-IDENTICAL to HEAD.

        The ``kappa == 0`` / ``beta == 0`` guards mirror the buried-path
        guard precedence (``KappaBetaGuardPrecedenceTestCase``): the chart
        axes AND the certificate are the ``kappa = 0``, ``beta = 0``
        reference surface, so a ``kappa != 0`` or ``beta != 0`` candidate
        CANNOT be represented and MUST fall through to the exact engine --
        serving a ``kappa = 0`` residual/carrier for a ``kappa != 0`` config
        would be a silent finite-but-wrong accuracy bug.  These stay
        explicit flat guards (no shared optional-geometry helper).

        MAP CONSULT.  When a certified-ppGO map is installed and this
        draw's cell is certified with a trusted floor ``w_trust`` strictly
        inside the band and at or below the effective ceiling
        ``min(parity_wall, cell_ceiling)``, the dense ``w`` band is split:
        the Born carrier + residual serves the nodes at or below
        ``w_trust`` and the bare point-mass ppGO serves above (``E_ff = 0``,
        which telescopes to the image-kernel sum in the ``FARFIELD_KERNEL_SUM``
        gauge), mirroring the surrogate-path band-split arithmetic.  When no
        map is installed, the cell is ``UNKNOWN``, or the split does not lie
        strictly inside the band, the WHOLE band is served by Born and the
        result is byte-identical to the un-split Born serve (the null-split
        identity the test battery pins): ``below_mask`` is all-``True`` so
        ``chart_w`` carries identical float values to ``dense_w``.

        A geometry ``LensDomainError`` propagates unswallowed, mirroring
        `_saddle_farfield_analytic` / `_ppgo_above_ceiling`; the cheap gates
        are checked BEFORE the geometry solve so a gate miss costs nothing.
        The reconstruction TAIL (carrier, Rung P, demodulation, kernel
        reduction) is factored into `_born_reconstruct`, fed the
        interpolated residual (in box) or a zero residual (carrier-only).

        Parameters
        ----------
        lens : dict
            Lens parameters from `_lens_params`.
        dense_w : np.ndarray
            Dimensionless frequency grid for the kernel subsamples.

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, partition)`` on success, ``None`` to fall
            through to the exact seed engine.
        """
        born_chart = self.born_residual_chart
        if born_chart is None:
            return None

        # The chart is a kappa = 0, beta = 0 surface BY CONSTRUCTION (its
        # axes carry neither dimension) and the carrier-only certificate is
        # the same reference surface.  A candidate with kappa != 0 or
        # beta != 0 CANNOT be represented, and serving it the kappa = 0 /
        # beta = 0 residual/carrier would be finite-but-wrong -- the exact
        # never-serve-where-wrong violation the guard exists to prevent.
        # Fall through to the exact engine, which handles both fully.
        if lens['kappa'] != 0.0 or lens['beta'] != 0.0:
            return None

        # Unlensed/macro-trivial limit: at gamma == 0 there is no caustic
        # (reach 0), and `caustic_rho` raises a raw ZeroDivisionError there
        # rather than a domain error -- measured 2026-08-14 when the F -> 1
        # zero-noise anchors hit this rung through the auto-attached chart.
        # No caustic frame means no Born exterior; fall through. Tiny
        # nonzero gamma refuses via LensDomainError / covers() below.
        if lens['gamma'] == 0.0:
            return None

        abs_y = math.hypot(lens['y1'], lens['y2'])
        try:
            rho = caustic_rho(lens['gamma'], abs_y, lens['kappa'])
        except (ValueError, LensDomainError):
            return None

        # Interior / near-caustic / tube (rho <= 2): no Born far exterior.
        # Fall through to the exact engine BEFORE the geometry solve,
        # byte-identical to HEAD.  The carrier-only lift below NEVER captures
        # this branch -- the tube case (covered but rho <= 2) keeps its
        # engine fallthrough (pinned invariant).
        if rho <= 2.0:
            return None

        # Cheap geometry-only partition (same construction the sibling
        # `_saddle_farfield_analytic` rung uses).  A geometry
        # `LensDomainError` propagates unswallowed -- and the seed engine
        # path below would raise the identical error, so propagating it here
        # preserves the refusal set exactly.
        geom = ChangRefsdalChannels(dense_w).geometry_partition(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])

        # Map consult: split the band at the certified trusted floor
        # ``w_trust`` only when it lies strictly inside the band AND the
        # band tops out at or below the effective ceiling
        # ``min(parity_wall, cell_ceiling)`` (bare ppGO must never serve
        # beyond-wall / beyond-measured-ceiling nodes).  Mirrors the
        # surrogate-path band-split arithmetic.
        w_trust = self._ppgo_band_split(lens)
        w_hi = float(dense_w.max())
        if w_trust is not None:
            wall = (ASTROID_WALL if float(lens['gamma']) < 1.0
                    else SADDLE_WALL)
            cell_ceiling = self._ppgo_cell_ceiling(lens)
            eff_ceiling = (wall if cell_ceiling is None
                           else min(wall, cell_ceiling))
            if w_hi > eff_ceiling:
                w_trust = None

        # ``below_mask`` marks the nodes the Born envelope actually serves;
        # above ``w_trust`` the reconstructed envelope is zeroed (E_ff = 0,
        # telescoping to the bare ppGO image-kernel sum).  Without a split it
        # is all-True, so nothing is zeroed and the serve is the whole-band
        # Born result.  ``chart_w`` is that served sub-band, used ONLY for
        # the trained-band refusal below -- the carrier / residual / ppGO
        # serve runs over the FULL ``dense_w``.
        _band_split, below_mask = _band_split_mask(dense_w, w_trust)

        # NESTED low split.  The below-split region ``[w_lo, w_trust)`` is
        # split again at the diffractive certificate ``w_low`` (the band-aware
        # honest ceiling): the analytic diffractive bottom
        # ``[w_lo, honest_ceiling)`` (Rung P, `F_P`) replaces the chart there,
        # and the trained carrier + residual host the middle
        # ``[honest_ceiling, w_trust)``.  ``chart_w`` -- the trained-band
        # refusal probe -- is the HOST sub-band only: the analytic bottom
        # does not consult the chart, so a draw whose bottom escapes the
        # trained ``log_w`` range is no longer refused whole.
        w_low = self._diffractive_bottom_ceiling(
            lens, w_hi=float(dense_w.max()))
        band_split_low, below_low = _band_split_mask(dense_w, w_low)
        if w_low is not None and w_low >= float(dense_w.max()):
            # Whole band certified: the analytic diffractive bottom serves
            # the ENTIRE below-split region and the engine/chart host
            # collapses to empty.
            bottom_mask = below_mask
        else:
            bottom_mask = ((below_low & below_mask) if band_split_low
                           else np.zeros(dense_w.shape, dtype=bool))
        host_mask = below_mask & ~bottom_mask
        chart_w = dense_w[host_mask]

        # Serve decision.  Three routes, in order of specificity:
        #
        #  1. FULLY in box (box-covered AND the served host sub-band inside
        #     the trained log-w range): the interpolated residual over the
        #     whole band -- BYTE-IDENTICAL to HEAD.
        #  2. TRAINED-FLOOR band split (box-covered but the host sub-band
        #     drops BELOW the trained log-w floor -- a low-edge escape): the
        #     chart serves the trained sub-band ``[trained_floor, w_trust]``
        #     it was actually trained on and the exact engine hosts the
        #     untrained remainder ``[w_low, trained_floor)`` below it,
        #     instead of refusing the whole band.
        #  3. Beyond-box / high-edge escape / disjoint trained range: a
        #     certificate-gated CARRIER-ONLY serve (residual identically
        #     ZERO, only the lead carrier reconstructed), else fall through
        #     to the exact engine -- BYTE-IDENTICAL to HEAD.
        covered = born_chart.covers(lens['gamma'], rho)
        trained_band_escape = (
            covered and host_mask.any()
            and not born_chart.covers(lens['gamma'], rho, chart_w))

        # Route 1: fully in box.
        if covered and not trained_band_escape:
            residual = born_chart.evaluate(dense_w, lens['gamma'], rho)
            return self._born_reconstruct(
                lens, dense_w, geom, residual, below_mask, bottom_mask)

        # Route 2: trained-floor band split (direction (a), low-edge escape).
        # ``trained_floor`` is the low edge of the trained log-w coverage,
        # read from the shipped artifact (``log_w_grid[0]``) -- NEVER a
        # literal.  Split the host region again at ``trained_floor`` with a
        # second ``_band_split_mask`` call (shared arithmetic; no bespoke
        # 3-region helper): the engine hosts BELOW ``trained_floor`` and the
        # chart serves AT or ABOVE it (the same inverted polarity the
        # saddle-c3 and ppGO band splits use).  ``below_floor`` includes the
        # exact ``trained_floor`` node on the engine side; the chart
        # sub-band is strictly above it and therefore strictly inside the
        # trained log-w range.
        if trained_band_escape:
            trained_floor = math.exp(float(born_chart.log_w_grid[0]))
            band_split_floor, below_floor = _band_split_mask(
                dense_w, trained_floor)
            engine_mask = host_mask & below_floor
            chart_mask = host_mask & ~below_floor
            # Serve the split ONLY when it is a genuine strict sub-band: the
            # inner split is active (``trained_floor`` strictly inside the
            # band -- rejects the null-fallback all-True ``below_floor``),
            # BOTH tiers are non-empty, AND the chart sub-band is fully
            # covered by the trained log-w range.  A high-edge escape or a
            # disjoint range leaves ``chart_mask`` uncovered; skip the wrong
            # populator and fall through to Route 3 (BYTE-IDENTICAL to HEAD).
            if (band_split_floor and engine_mask.any() and chart_mask.any()
                    and born_chart.covers(lens['gamma'], rho,
                                          dense_w[chart_mask])):
                residual = np.zeros(dense_w.shape, dtype=complex)
                residual[chart_mask] = born_chart.evaluate(
                    dense_w[chart_mask], lens['gamma'], rho)
                engine_envelope = self._engine_envelope_below_split(
                    lens, dense_w, engine_mask)
                return self._born_reconstruct(
                    lens, dense_w, geom, residual, below_mask, bottom_mask,
                    engine_envelope=engine_envelope, engine_mask=engine_mask)

        # Route 3: certificate-gated carrier-only serve, else fall through.
        w_lo = float(dense_w.min())
        if not _born_carrier_certificate_serves(
                lens, w_lo, w_hi, geom.images):
            return None
        residual = np.zeros(dense_w.shape, dtype=complex)
        return self._born_reconstruct(
            lens, dense_w, geom, residual, below_mask, bottom_mask)

    def _born_reconstruct(self, lens, dense_w, geom, residual,
                          below_mask, bottom_mask,
                          engine_envelope=None, engine_mask=None):
        """Reconstruct the Born far-field kernels from a supplied residual.

        Pure reconstruction TAIL shared by the in-box serve (fed the
        interpolated ``born_residual_chart`` residual) and the beyond-box
        carrier-only serve (fed an identically ZERO residual).  Builds the
        analytic carrier via `born_carrier_from_partition`, adds the caller's
        residual, overwrites the diffractive bottom ``[w_lo, w_low)`` with
        the diffractive series ``F_P`` (Rung P), demodulates into the
        ``FARFIELD_KERNEL_SUM`` gauge zeroing the envelope above the trusted
        floor, and reduces to the two dense kernels.

        Owns NO serve decision and NO residual zeroing: the caller decides
        in-box vs carrier-only and supplies ``residual`` (the interpolated
        chart residual, or an identically zero array).  Passing the zero
        array reduces ``f_total = carrier + residual`` to the bare carrier
        with no perturbation of the in-box null-identity byte-path.

        Parameters
        ----------
        lens : dict
            Lens parameters from `_lens_params`.
        dense_w : np.ndarray
            FULL dimensionless frequency grid.  The carrier / residual / ppGO
            are ALL served over this full band so their length matches
            ``geom.saddle_kernels`` (N rows) and ``geom.delays`` --
            ``born_carrier_from_partition`` -> ``reconstruct_farfield``
            validate ``saddle_kernels.shape[0] == w.size`` and raise a shape
            ``ValueError`` on a sub-slice against a full-length geometry
            (INS-2-001).  The band split is applied by ZEROING the
            reconstructed envelope, never by sub-slicing ``w``.
        geom : object
            Geometry-only partition from
            ``ChangRefsdalChannels.geometry_partition``.
        residual : np.ndarray
            Complex residual over ``dense_w`` -- the interpolated chart
            residual (in box) or an identically zero array (carrier-only).
        below_mask : np.ndarray
            Boolean nodes the Born envelope serves; the reconstructed
            envelope is zeroed above (bare ppGO telescopes in).  All-True
            without a band split.
        bottom_mask : np.ndarray
            Boolean nodes on the analytic diffractive bottom overwritten by
            ``F_P``.  Empty at the saddle wall / null split, leaving the
            whole-below-split serve byte-identical to HEAD.
        engine_envelope : np.ndarray, optional
            Full-length (``dense_w`` shape) exact-engine far-field envelope
            in the ``FARFIELD_KERNEL_SUM`` gauge (from
            `_engine_envelope_below_split`), used only on the trained-floor
            band split.  When supplied, it OVERWRITES the reconstructed
            envelope on ``engine_mask`` -- the untrained sub-band
            ``[w_low, trained_floor)`` the chart cannot serve.  ``None`` on
            every non-split serve (carrier-only and fully-in-box), leaving
            the byte-path untouched.
        engine_mask : np.ndarray, optional
            Boolean nodes (``dense_w`` shape) the ``engine_envelope``
            overwrites; required when ``engine_envelope`` is given and
            ignored otherwise.

        Returns
        -------
        tuple or None
            ``(delays, k0, k1, partition)`` on success, or ``None`` when the
            diffractive bottom hits a `HypergeometricDomainError`.
        """
        # Duck-typed namespace adapter for born_carrier_from_partition
        # (reads attributes by name), built over the FULL ``dense_w`` band.
        partition_ns = types.SimpleNamespace(
            w=dense_w,
            source=np.array([lens['y1'], lens['y2']]),
            gamma=lens['gamma'],
            beta=lens['beta'],
            kappa=lens['kappa'],
            matrix=macro_matrix(
                lens['gamma'], lens['beta'], lens['kappa']),
            t_min=geom.t_min,
            delays=geom.delays,
            saddle_kernels=geom.saddle_kernels,
            real_mask=geom.real_mask,
            images=geom.images)
        # Deferred import avoids cycle risk (born_carrier_from_partition's
        # module imports channels which may circle back at module load).
        from cogwheel.lensing.chang_refsdal.channels import (
            born_carrier_from_partition)
        carrier = born_carrier_from_partition(partition_ns)
        f_total = carrier + residual

        # Rung P: overwrite the total amplification on the analytic bottom
        # ``[w_lo, w_low)`` with the diffractive series ``F_P`` (which IS the
        # full amplification, tending to ``sqrt(mu_macro)`` as ``w -> 0``),
        # discarding the extrapolated / zero chart carrier + residual there.
        # ``F_P`` is expressed in the SAME absolute-frame amplification as
        # ``f_total``, so the shared ``(f_total - ppgo) * exp(1j w t_min)``
        # demodulation below carries the analytic bottom into the
        # ``FARFIELD_KERNEL_SUM`` gauge with no field discontinuity at
        # ``w_low``.  Empty when the nested bottom collapsed (saddle parity
        # wall / null split), leaving the whole-below-split serve
        # BYTE-IDENTICAL to HEAD.
        if bottom_mask.any():
            born_y = (lens['y1'], lens['y2'])
            try:
                for idx in np.flatnonzero(bottom_mask):
                    f_total[idx] = diffractive_amplification(
                        float(dense_w[idx]), born_y, lens['gamma'],
                        lens['beta'], lens['kappa'])
            except HypergeometricDomainError:
                return None

        # Extract the far-field envelope over the full band by subtracting
        # the resolved ppGO channels and demodulating, THEN zero it above
        # ``w_trust`` (E_ff = 0, which telescopes to the bare ppGO
        # image-kernel sum in the FARFIELD_KERNEL_SUM gauge).  Without a
        # split ``below_mask`` is all-True so nothing is zeroed and the
        # serve is the whole-band Born result (the null-split identity).
        real = np.asarray(geom.real_mask, dtype=bool)
        ppgo = np.sum(
            geom.saddle_kernels[:, real]
            * np.exp(1j * dense_w[:, None] * geom.delays[real][None, :]),
            axis=1)
        envelope = (f_total - ppgo) * np.exp(1j * dense_w * geom.t_min)
        envelope[~below_mask] = 0.0

        # Trained-floor tier overlay.  On the trained-floor band split the
        # exact-engine far-field envelope replaces the (extrapolated,
        # zeroed-residual) chart carrier on the untrained sub-band
        # ``[w_low, trained_floor)`` -- the nodes the chart was never
        # trained on.  Both sides are the SAME ``FARFIELD_KERNEL_SUM`` gauge
        # (see `_engine_envelope_below_split`), so the exact engine below
        # stitches onto the chart carrier + residual above with no field
        # discontinuity.  ``None`` on every non-split serve, leaving the
        # carrier-only and fully-in-box byte-paths untouched.
        if engine_envelope is not None:
            envelope[engine_mask] = engine_envelope[engine_mask]

        kernels, _total = reconstruct_farfield(
            dense_w, envelope, geom.delays, geom.saddle_kernels,
            geom.real_mask, FARFIELD_KERNEL_SUM, geom.t_min)
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
           `geometry.LensDomainError` or `SchwingerCertificationError`
           propagates UNSWALLOWED, matching ``lnlike_bruteforce``).
        2. Look up or build the fiducial envelope for the candidate's
           fiducial cell (`_fiducial_key`).  ONLY the fiducial build is
           wrapped in ``try/except LensDomainError``:
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

        # ppGO above-ceiling intercept (Build exterior_followup WP4).
        # Some draws exceed the Schwinger QD ceiling (w > 150) where the
        # exact engine hard-refuses.  When the image pair is resolved
        # (w_lo * min_delta_tau >= RHO_END), the fold-corrected ppGO
        # carrier is accurate across the whole band and serves without
        # any per-candidate engine cost.  On gate miss, fall through to
        # the exact engine -> SchwingerCertificationError unchanged.
        if float(dense_w.max()) > W_CEILING_SCHWINGER_QD:
            ppgo = self._ppgo_above_ceiling(lens, dense_w)
            if ppgo is not None:
                return ppgo

        # Tier-1 far-from-caustic macro-saddle analytic intercept.  For
        # the resolvable FAR-FROM-CAUSTIC macro saddle (gamma > 1) the
        # switched analytic channel sum carries the whole band to within
        # the certified bar (p90 ~5e-5, max ~7e-4 at the rho floor), so it
        # serves with a zero envelope and no per-candidate engine cost.
        # The gamma > 1 guard keeps the astroid (gamma <= 1) path
        # byte-identical; the two saddle rungs are mutually exclusive in
        # effect (ppGO already served the w_max > 150 resolvable case
        # above).  On gate miss, fall through to the exact seed engine.
        if lens['gamma'] > 1.0:
            served = self._saddle_farfield_analytic(lens, dense_w)
            if served is not None:
                return served

        # First-class Born weak-deflection analytic intercept (WP-B).  When
        # a `born_residual_chart` is attached and the candidate sits in the
        # Born exterior (kappa == 0, beta == 0, caustic-frame rho > 2,
        # covered by the chart box), serve the analytic carrier + trained
        # residual directly -- no surrogate, no per-candidate engine cost --
        # consulting the certified ppGO map to band-split the upper band to
        # bare ppGO where certified.  Positioned LAST among the analytic
        # intercepts (after the saddle far-field block, before the
        # seed/fiducial/ratio engine path).  On any gate miss it returns
        # None and falls through to the exact seed engine, byte-identical to
        # the no-chart path.
        served = self._born_residual_analytic(lens, dense_w)
        if served is not None:
            return served

        # Low-w diffractive intercept (WP2c).  The far-field kernel-sum
        # gauge DIVERGES below its per-draw floor `channels.farfield_w_floor`
        # (the F070 window), so every analytic rung above declines the band
        # bottom; absent a surrogate chart to trip the buried F070 serve in
        # `_surrogate_coefficients`, such a draw falls through to the exact
        # engine.  Intercept it here with the SAME two-rung diffractive serve
        # (`_low_w_diffractive_serve`): analytic ``F_P`` below the ``w_low``
        # truncation certificate for the positive-parity exterior, and a
        # reachability-capped engine host for the macro saddle, reconstructed
        # whole-band in the finite ``FARFIELD_DIFFRACTIVE`` gauge.  Gated to
        # the FAR-FIELD EXTERIOR (exactly two real images): the diffractive
        # series is a weak-deflection expansion and would be finite-but-wrong
        # on the four-image caustic interior -- a regime the far-field-chart
        # admission excludes on the surrogate path but which is ungated here,
        # so the image-count guard supplies that exclusion directly.  Only
        # attempted when the band actually dips below the floor
        # (``w_lo < w_floor``); otherwise the far-field rungs / engine own the
        # band and this rung has no job.  The geometry-only partition mirrors
        # the surrogate path's build and lets a `geometry.LensDomainError`
        # propagate UNSWALLOWED, exactly as the seed engine path below would
        # raise it, so the refusal set is preserved.  On any refusal the serve
        # returns None and we fall through byte-identically to the exact seed
        # engine.  Ordered LAST among the analytic intercepts by w-band
        # disjointness (it owns the sub-floor bottom), not by priority.
        w_lo = float(dense_w.min())
        w_hi = float(dense_w.max())
        geom = ChangRefsdalChannels(dense_w).geometry_partition(
            gamma=lens['gamma'], y=(lens['y1'], lens['y2']),
            beta=lens['beta'], kappa=lens['kappa'])
        if int(geom.real_mask.sum()) == 2:
            w_floor = farfield_w_floor(geom.delays, geom.real_mask)
            if w_lo < w_floor:
                served = self._low_w_diffractive_serve(
                    lens, dense_w, geom, w_lo, w_hi)
                if served is not None:
                    return served

        # Candidate seed engine evaluation (single call).  A candidate-side
        # `geometry.LensDomainError` / `SchwingerCertificationError` from
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
        except LensDomainError:
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
        the first engine evaluation) and `SchwingerCertificationError`
        (uncertifiable or above-ceiling quadrature, raised at the
        worst-cancellation node ``w_max`` that
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
        # for Type III / the parity boundary,
        # `SchwingerCertificationError` for an uncertifiable or
        # above-ceiling quadrature), matching the brute-force path.
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
        from the snapped configuration propagates to the caller, which
        falls back to the direct path.

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
        must reproduce; a macro-saddle `geometry.LensDomainError` or a
        `SchwingerCertificationError` propagates unswallowed.

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
