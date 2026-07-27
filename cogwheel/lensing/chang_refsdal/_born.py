"""Born (weak-deflection) analytic amplification for the low-w far zone.

WHAT
    The last rung of the Chang-Refsdal serving ladder: an *analytic*
    microlensing amplification valid in the low-frequency far zone --
    large reduced impact ``|y'|`` and small ``w`` -- where the exact
    engine is still certifiable but a *trained* cover cannot be
    prior-universal (the low-w far zone varies on the Einstein scale, so
    trained tiles there are prior-sized).  This module supplies

    * `born_amplification` -- the total wave-optics amplification
      ``F_born`` expanded about ``sqrt(mu_macro)``;
    * `born_gate` -- the measured validity boundary as a named refusal;
    * `born_envelope` -- the demodulated far-field envelope, built with
      the SAME `switched_analytic_channels` projection the trained
      far-field charts use, so the serve mirror is byte-for-byte the
      existing one; and
    * `BornDomainError` -- the rung's named refusal.

WHY
    Owner ruling 2026-07-23 (NON-NEGOTIABLE): zero-quadrature serving
    plus prior-universality leave no other cover for the low-w far zone.
    Born is needed for prior-universality, NOT because the oracle fails
    -- every Born claim in the target annulus ``3.0 < |y| <= 4.2426`` is
    certifiable against the exact engine (``w * |y| <= 60``).

STATUS -- NOT WIRED INTO THE SERVE PATH (Inspector INS-c1-001).  The
    two-term series carries an unpinned O(1) numerator
    (`_born_factors` ``b1 = 1.0``, a placeholder the Professor has not yet
    derived a closed form for), so ``born_amplification`` disagrees with
    the exact `operator.F_op` by up to ~13% across the target annulus even
    where `born_gate` PASSES -- guard A is calibrated to the same
    placeholder and so fails to refuse where inaccurate.  This module is
    therefore kept for its primitives and for the oracle-accuracy tests
    that measure the gap, but `likelihood._surrogate_coefficients` does
    NOT call it: the low-w far zone falls through to the exact engine.
    Re-enable the serve slot ONLY once the Professor pins ``b1`` AND an
    oracle-accuracy gate against `operator.F_op` passes at a stated
    tolerance.

CONVENTIONS (each expressed ONCE here, asserted at its consumer -- the
delay-frame bug fixed in Build 8h-b7 was a convention held implicitly at
four sites; this module adds no fifth):

    * The expansion origin is ``sqrt(mu_macro)`` with
      ``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)``, NOT ``1``.  A
      series about ``1`` is wrong wherever the shear is nonzero, i.e.
      everywhere this rung serves.
    * The small parameter is ``1 / |y'|**2`` (Einstein-radius squared
      over reduced impact squared), realised as ``1 / Q2r``; the leading
      correction carries ``w`` in the NUMERATOR, so it vanishes as
      ``w -> 0`` and the leading term is ``w``-independent.
    * `born_amplification` returns the total in the ABSOLUTE Fermat-delay
      frame (matching `operator.F_op`); `born_envelope` demodulates to
      the min-relative-delay frame using ``geom.t_min`` -- the SAME frame
      origin the geometry partition already carries.

NAMING HAZARD: ``far-field`` / ``FarFieldChart`` / ``farfield_*`` in this
package mean "trained chart OUTSIDE the caustic in the far-field GAUGE",
NOT the weak-deflection far field this module implements.  No far-field
gauge name is overloaded for the Born term.
"""

from __future__ import annotations

import math
import cmath

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry, channels

#: Held-out accuracy budget for the Born series (guard A): the estimated
#: magnitude of the first neglected ``O(w**2 / Q2r**2)`` term must stay
#: below this for the two-term series to be trusted.
EPS_BORN = 1e-4

#: Positive-parity convergence margin (guard B): the reduced shear
#: ``gamma_p = |gamma| / (1 - kappa)`` must stay this far below the parity
#: wall ``gamma_p = 1`` (``det A = 0``).  The series' convergence radius
#: IS the parity wall, so the macro image degenerates as ``gamma_p -> 1``.
DELTA_GAMMA_P = 5e-3


class BornDomainError(geometry.LensDomainError):
    """Lens parameters fall outside the measured Born validity region.

    Subclasses `geometry.LensDomainError` so the Born rung refuses
    symmetrically with the exact path and the other analytic rungs
    (`SchwingerCertificationError`, `GhostDomainError`): a caller that
    already handles `LensDomainError` need not special-case Born.
    """


def _born_factors(y1: float, y2: float, gamma: float, beta: float,
                  kappa: float) -> tuple[float, float, float, float]:
    """Frequency-independent geometry factors of the Born series.

    Every returned quantity is ``w``-independent, so it is computed once
    and reused across a whole frequency grid (`born_envelope`) or a
    single node (`born_amplification`).  Kept a pure ``float64`` scalar
    computation with no Python objects, so it stays ``numba``-compatible
    if the amplification is later hoisted onto an accelerated path (do
    NOT add ``fastmath`` -- the phase must stay reproducible).

    The macro matrix ``A = (1 - kappa) I - gamma Q(beta)`` is built inline
    (mirroring `geometry.macro_matrix`) rather than called, to avoid a
    NumPy object on the scalar path; `born_gate` calls the real
    `geometry.macro_matrix` for the degenerate-axis refusal.

    Parameters
    ----------
    y1, y2 : float
        Source position in the lens plane.
    gamma, beta, kappa : float
        External shear magnitude, shear orientation (radians), and
        convergence.

    Returns
    -------
    sqrt_mu : float
        ``sqrt(mu_macro) = 1 / sqrt((1 - kappa)**2 - gamma**2)``, the
        ``w -> 0`` amplification (positive parity, so the radicand is
        positive; the caller must have gated the parity wall).
    phi_geo : float
        Geometric (Fermat) phase at the macro image ``x0 = A^{-1} y``,
        identical to ``geometry.delay(x0, y, A)`` (same Fermat
        convention, evaluated inline).  Absolute-delay frame.
    q2r : float
        Reduced macro-image squared impact ``|x0'|**2`` in mass-sheet
        coordinates (``x0' = sqrt(1 - kappa) * x0``); the ``O(|y'|**2)``
        denominator whose reciprocal is the series' small parameter.
    b1 : float
        The ``O(1)`` reduced-coordinate numerator of the leading
        correction ``1j * (w / 2) * b1 / q2r``.
    """
    lam = 1.0 - kappa
    cos2b = math.cos(2.0 * beta)
    sin2b = math.sin(2.0 * beta)

    # A = lam * I - gamma * Q(beta), Q(beta) = [[cos2b, sin2b],
    #                                           [sin2b, -cos2b]].
    a11 = lam - gamma * cos2b
    a12 = -gamma * sin2b
    a22 = lam + gamma * cos2b
    det_a = a11 * a22 - a12 * a12  # == lam**2 - gamma**2 == 1 / mu_macro

    sqrt_mu = 1.0 / math.sqrt(det_a)

    # Macro image x0 = A^{-1} y (leading weak-deflection image position).
    x0_1 = (a22 * y1 - a12 * y2) / det_a
    x0_2 = (-a12 * y1 + a11 * y2) / det_a

    # phi_geo = geometry.delay(x0, y, A).  With A x0 = y the quadratic
    # term collapses: x0 @ A @ x0 == x0 @ y, hence
    # 0.5 x0.A.x0 - y.x0 = -0.5 (x0 . y).
    x0_dot_y = x0_1 * y1 + x0_2 * y2
    y_sq = y1 * y1 + y2 * y2
    r0_sq = x0_1 * x0_1 + x0_2 * x0_2
    phi_geo = -0.5 * x0_dot_y + 0.5 * y_sq - 0.5 * math.log(r0_sq)

    # Reduced (mass-sheet) macro-image squared impact.  The reduced
    # problem has A' = A / lam and y' = y / sqrt(lam), so its macro image
    # is x0' = A'^{-1} y' = sqrt(lam) * x0 and |x0'|**2 = lam * |x0|**2.
    q2r = lam * r0_sq

    # BLOCKED-derivation: the exact O(1) numerator b1 of the leading
    # correction is owned by the Professor and is NOT recorded anywhere
    # in the repository (searched .claude/spec, FINDINGS, every handoff).
    # The MANDATED STRUCTURE is honoured exactly -- imaginary correction
    # 1j*(w/2)*b1/q2r, w in the numerator, smallness 1/|y'|**2 = 1/q2r --
    # but the numeric value below is a documented placeholder pending the
    # pinned derivation; the Test Developer's oracle-accuracy gate against
    # the exact engine is what fixes it.  Single authoritative site: edit
    # here only.  UNVERIFIED (see change report).
    b1 = 1.0

    return sqrt_mu, phi_geo, q2r, b1


def born_amplification(w: float, y1: float, y2: float, gamma: float,
                       beta: float = 0.0, kappa: float = 0.0) -> complex:
    """Total Born amplification ``F_born`` at one frequency.

    The weak-deflection wave-optics amplification, expanded about
    ``sqrt(mu_macro)``::

        F_born = sqrt(mu_macro) * exp(1j*w*phi_geo)
                 * (1 + 1j*(w/2)*b1/Q2r + O(w**2 / Q2r**2)),

    with ``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)``, ``phi_geo`` the
    geometric Fermat phase at the macro image, and ``(b1, Q2r)`` the
    ``O(1)`` geometry factors of `_born_factors`.  The correction carries
    ``w`` in the numerator, so ``F_born -> sqrt(mu_macro)`` as ``w -> 0``
    (the macro-magnification limit) and the leading term is
    ``w``-independent.  Positive parity only; a ``c1 = 1 / (2 w)`` form
    (diverging as ``w -> 0``) would be the inverted-power bug the
    Professor explicitly ruled out.

    Returned in the ABSOLUTE Fermat-delay frame, matching
    `operator.F_op` (which `channels._exact_total` later demodulates by
    ``exp(-1j*w*t_min)``); `born_envelope` applies that same
    demodulation.

    Pure ``float64`` / ``complex128`` scalar arithmetic (``numba``-ready,
    no ``fastmath``).  Assumes the configuration has passed `born_gate`:
    the positive-parity radicand ``(1 - kappa)**2 - gamma**2 > 0`` is a
    precondition (a saddle host makes ``math.sqrt`` raise -- fail loud).

    Parameters
    ----------
    w : float
        Dimensionless frequency (``> 0``).
    y1, y2 : float
        Source position in the lens plane.
    gamma, beta, kappa : float
        External shear magnitude, orientation (radians), convergence.

    Returns
    -------
    complex
        The total amplification ``F_born(w)``.
    """
    sqrt_mu, phi_geo, q2r, b1 = _born_factors(y1, y2, gamma, beta, kappa)
    correction = 1.0 + 1j * (0.5 * w) * b1 / q2r
    return sqrt_mu * cmath.exp(1j * w * phi_geo) * correction


def born_gate(w: float, y1: float, y2: float, gamma: float,
              beta: float, kappa: float) -> None:
    """Refuse configurations outside the measured Born validity region.

    Two physically distinct, load-bearing guards, both raising
    `BornDomainError` (a named refusal, never a silent ``nan``):

    * **Guard A -- series convergence.**  The estimated magnitude of the
      first neglected ``O(w**2 / Q2r**2)`` term must stay below
      `EPS_BORN`.  ``Q2r`` shrinks as the source approaches the shear
      anisotropy, so this is the boundary where the two-term series stops
      being accurate.
    * **Guard B -- parity-wall margin.**  The reduced shear
      ``gamma_p = |gamma| / (1 - kappa)`` must stay at least
      `DELTA_GAMMA_P` below ``1``.  The series' convergence radius is the
      parity wall ``gamma_p = 1`` (``det A = 0``), where the macro image
      degenerates; positive parity only, so the macro saddle
      (``gamma_p > 1``) is refused here by construction.

    The exact degenerate axes (``1 - kappa <= 0`` and
    ``|gamma| == 1 - kappa``) are refused first by delegating to
    `geometry.macro_matrix`, whose `geometry.LensDomainError` propagates
    unchanged.

    Parameters
    ----------
    w : float
        Dimensionless frequency (``> 0``).
    y1, y2 : float
        Source position in the lens plane.
    gamma, beta, kappa : float
        External shear magnitude, orientation (radians), convergence.

    Raises
    ------
    geometry.LensDomainError
        Degenerate macro axis (``1 - kappa <= 0`` or
        ``|gamma| == 1 - kappa``), from `geometry.macro_matrix`.
    BornDomainError
        Guard A (series non-convergence) or guard B (parity-wall margin).
    """
    # Degenerate-axis refusal, shared with every other wave-branch path.
    geometry.macro_matrix(gamma, beta=beta, kappa=kappa)

    lam = 1.0 - float(kappa)
    gamma_p = abs(float(gamma)) / lam

    # Guard B: parity-wall convergence margin.
    if gamma_p >= 1.0 - DELTA_GAMMA_P:
        raise BornDomainError(
            f'Born rung refuses (kappa, gamma) = ({kappa}, {gamma}): '
            f'reduced shear gamma_p = |gamma| / (1 - kappa) = {gamma_p} '
            f'>= 1 - DELTA_GAMMA_P = {1.0 - DELTA_GAMMA_P}. The series '
            f'converges only inside the positive-parity margin; the macro '
            f'image degenerates at the parity wall gamma_p = 1.')

    # Guard A: first-neglected-term estimate against the accuracy budget.
    _, _, q2r, b1 = _born_factors(float(y1), float(y2), float(gamma),
                                  float(beta), float(kappa))
    next_term = 0.5 * (0.5 * float(w) * abs(b1) / q2r) ** 2
    if next_term >= EPS_BORN:
        raise BornDomainError(
            f'Born rung refuses (w, y1, y2, gamma_p) = '
            f'({w}, {y1}, {y2}, {gamma_p}): estimated next-order term '
            f'|O(w^2 / Q2r^2)| = {next_term:.3e} >= EPS_BORN = {EPS_BORN}. '
            f'The reduced impact Q2r = {q2r:.3e} is too small (source too '
            f'close to the shear anisotropy) for the two-term series.')


def born_envelope(dense_w: np.ndarray, y1: float, y2: float, gamma: float,
                  beta: float, kappa: float,
                  geom: 'channels.ChangRefsdalGeometryPartition'
                  ) -> np.ndarray:
    """Demodulated far-field envelope of the Born amplification.

    Reconstructs the analytic ``F_born`` into ONE demodulated transition
    envelope through the SAME `channels.switched_analytic_channels`
    projection the trained far-field charts use
    (`channels.farfield_envelope_from_partition`), with the analytic
    ``F_born`` grid replacing the expensive exact operator total.  The
    switch, weights, kernels, delays and parked ``tau_c = 0`` carrier are
    taken from the geometry partition and its far-field kernel-sum tag, so
    the serve mirror (`channels.reconstruct_farfield`) needs no new gauge
    math and cannot drift from the exact-total far-field label.

    The Born total is evaluated in the absolute frame and demodulated to
    the min-relative-delay frame by ``exp(-1j * w * geom.t_min)`` -- the
    SAME frame origin `channels._exact_total` uses and that
    ``geom.delays`` / ``geom.saddle_kernels`` are already expressed in.
    ``t_min`` is read ONLY from ``geom.t_min`` and never recomputed (the
    Build 8h-b7 delay-frame lesson).

    Parameters
    ----------
    dense_w : np.ndarray
        Dimensionless frequency grid; must equal ``geom.w`` (the grid the
        partition's kernels and switch are sampled on).
    y1, y2 : float
        Source position in the lens plane.
    gamma, beta, kappa : float
        External shear magnitude, orientation (radians), convergence.
    geom : channels.ChangRefsdalGeometryPartition
        Geometry-only partition carrying ``w``, ``delays``,
        ``saddle_kernels``, ``real_mask`` and ``t_min`` in the
        min-relative-delay frame.

    Returns
    -------
    np.ndarray
        Shape ``(n_w,)`` complex Born far-field envelope.

    Raises
    ------
    ValueError
        If ``dense_w`` does not match ``geom.w`` (the kernels and switch
        would be sampled on a different grid than the Born total).
    """
    dense_w = np.asarray(dense_w, dtype=float)
    if not np.array_equal(dense_w, np.asarray(geom.w, dtype=float)):
        raise ValueError(
            'born_envelope requires dense_w to equal geom.w: the '
            'partition kernels and switch are sampled on geom.w, so the '
            'Born total must be evaluated on the same grid.')

    # SINGLE authoritative delay-frame origin (Build 8h-b7 lesson): the
    # envelope must sit in the SAME min-relative-delay frame the geometry
    # partition already carries.  Read ONLY from geom.t_min; never
    # recompute an independent t_min here.
    t_min = float(geom.t_min)
    assert np.isfinite(t_min), \
        'born_envelope must demodulate in a finite geom.t_min frame'

    # Frequency-independent factors computed once, then broadcast over w.
    sqrt_mu, phi_geo, q2r, b1 = _born_factors(
        float(y1), float(y2), float(gamma), float(beta), float(kappa))
    correction = 1.0 + 1j * (0.5 * dense_w) * b1 / q2r
    f_born = sqrt_mu * np.exp(1j * dense_w * phi_geo) * correction

    # Demodulate the absolute-frame total into the relative-delay frame,
    # exactly as channels._exact_total does for the exact operator total.
    total = f_born * np.exp(-1j * dense_w * t_min)

    # Reuse the far-field kernel-sum label verbatim: constant switch
    # (S_a = 1 on real channels), tau_c = 0 parked carrier, SACR-C
    # apportionment weights.  No far-field gauge math is reimplemented.
    switch = channels._farfield_switch(
        geom.real_mask, dense_w.shape[0], channels.FARFIELD_KERNEL_SUM)
    _, envelope = channels.switched_analytic_channels(
        dense_w, total, geom.delays, geom.saddle_kernels, switch, 0.0,
        channels._envelope_weights(switch))
    return envelope
