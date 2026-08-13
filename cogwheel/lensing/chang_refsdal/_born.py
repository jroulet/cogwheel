"""Born (weak-deflection) analytic carrier for the exterior region.

WHAT
    The last rung of the Chang-Refsdal serving ladder for the far
    exterior region (caustic-relative coordinates).
    The served object is an *analytic carrier*; a driver-trained chart
    interpolates the cheap RESIDUAL ``F_exact - F_carrier`` (the same
    carrier + interpolated-remainder decomposition as SACR-C and the
    far-field label).  This module supplies

    * `born_lead_carrier` -- the SERVE object: the LEAD-ONLY carrier
      ``sqrt(mu_macro) * exp(1j*w*phi_geo)`` with NO ``a0``/``b1``
      correction (F025: the resolved-image correction splines far worse
      and violates F009 below the band split, so it serves nowhere);
    * `born_amplification` -- the resolved-image (``w*Delta_tau >> 1``)
      two-term diagnostic ``F_born`` expanded about ``sqrt(mu_macro)``,
      carrying the real ``a0`` and imaginary ``b1`` corrections;
    * `born_gate` -- the validity boundary as a named refusal
      (parity-wall margin and band-split guard);
    * `born_envelope` -- the demodulated far-field envelope of the
      diagnostic total, built with the SAME `switched_analytic_channels`
      projection the trained far-field charts use; and
    * `BornDomainError` -- the rung's named refusal.

WHY
    F023's original premise is BACKWARDS.  It is NOT that the far zone
    "varies on the Einstein scale" so a trained cover cannot help: once
    the analytic carrier ``exp(1j*w*phi_geo)`` is demodulated, the
    Einstein-scale variation lives in that CLOSED-FORM phase, and the
    residual ``F_exact - F_carrier`` splines cheaply -- ~4 y-nodes at low
    ``w`` and MORE at mid/high ``w`` (F023).  The old label
    "low-frequency far zone" is therefore a misnomer: this is a MID-``w``
    RESOLVED-IMAGE expansion, keyed on ``w * Delta_tau`` (the Fermat-delay
    difference of the two real images), NOT on ``w`` or ``|y|`` alone.
    Born is needed for prior-universality, NOT because the oracle fails
    -- every claim in the target exterior region is certifiable against the exact
    engine (``w * |y| <= 60``).

STATUS -- CARRIER SHIPS; LIVE SERVE AWAITS THE TRAINED RESIDUAL CHART.
    `born_lead_carrier` and the band-split gate (`born_gate`) are correct
    and shippable primitives, but the residual chart that the carrier is
    subtracted from is a TRAIN_TIER driver artifact that has not yet been
    trained.  Until it exists, `likelihood._surrogate_coefficients` does
    NOT wire this slot: the Born exterior falls through to the exact engine,
    which is certifiable throughout (``w * |y| <= 60``).  ``a0``/``b1``
    are the correct resolved-image physics and the macro-limit diagnostic
    (`born_amplification`); they are deliberately kept OUT of the serve
    carrier (F025).
CONVENTIONS (each single-sourced elsewhere; this module adds no new
site for any of them):

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

from cogwheel.lensing.chang_refsdal import geometry, channels, operator

#: Upper edge of the smooth-switch window and geometric-optics resolution
#: onset, imported from `operator` as the ONE authoritative home of the
#: SACR-C resolution scale ``RHO_END = 4``.  Do not introduce a second
#: literal in any module that MAY import `operator`.  `_gauge.py` is the
#: one exception: it is a pinned dependency-free leaf
#: (`test_lensing_gauge.GaugeIndependenceTestCase` forbids it importing
#: `operator`), so it carries a guarded duplicate pinned equal by a test
#: instead -- see FINDINGS F068.
#: Guard A refuses once the two real images are resolved,
#: ``w * Delta_tau >= RHO_END``, the same band split SACR-C switches on.
RHO_END = operator.RHO_END

#: EPS_BORN was the RETIRED T1 accuracy bar of the standalone two-term
#: series (guard A's old ``O(w**2 / Q2r**2)`` estimate vs 1e-4).  The
#: production criterion is no longer that the standalone series hits an
#: accuracy target -- the carrier is subtracted and a chart interpolates
#: the residual (F025) -- so the constant is retired and guard A is
#: re-keyed to the band split ``w * Delta_tau >= RHO_END``.

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
                  kappa: float) -> tuple[float, float, float, float, float]:
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
        The MAGNITUDE ``sqrt(|mu_macro|) = 1 / sqrt(|(1 - kappa)**2 -
        gamma**2|)``, the ``w -> 0`` amplification amplitude.  This is the
        unsigned magnitude for BOTH parities: on the macro saddle
        (``det_a < 0``) the radicand's absolute value is taken here and the
        caller (`born_lead_carrier`) applies the Morse phase ``-1j``
        separately (F024/F009-S).
    phi_geo : float
        Geometric (Fermat) phase at the macro image ``x0 = A^{-1} y``,
        identical to ``geometry.delay(x0, y, A)`` (same Fermat
        convention, evaluated inline).  Absolute-delay frame.
    q2r : float
        Reduced macro-image squared impact ``|x0'|**2`` in mass-sheet
        coordinates (``x0' = sqrt(1 - kappa) * x0``); the ``O(|y'|**2)``
        denominator whose reciprocal is the series' small parameter.
    b1 : float
        The ``O(1)`` reduced-coordinate numerator of the IMAGINARY,
        ``w``-linear leading correction ``1j * (w / 2) * b1 / q2r``.  The
        matrix form is ``b1 = -lam * (2 lam r0_sq - x0.y) / (det_a r0_sq)``
        (F023); a pure point mass (``gamma = kappa = 0``) gives exactly
        ``b1 = -1``.
    a0 : float
        The ``O(1)`` reduced-coordinate numerator of the REAL,
        ``w``-independent resolved-image correction ``a0 / q2r`` (valid
        only for ``w * Delta_tau >> 1``; it must NOT enter the serve
        carrier, F009/F025).  ``a0 = -lam * (lam r0_sq - x0.y) /
        (det_a r0_sq)``; a pure point mass gives exactly ``a0 = 0``.  The
        pair satisfies the macro-limit invariant
        ``b1 - a0 == -lam**2 * mu_macro``.
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

    # Magnitude of the amplification for BOTH parities: abs() makes the
    # radicand safe on the macro saddle (det_a < 0), where the Morse phase
    # is applied by the caller (F024).  Positive parity (det_a > 0) is
    # bit-identical to the old 1/sqrt(det_a) (abs is the identity there).
    sqrt_mu = 1.0 / math.sqrt(abs(det_a))

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

    # Closed-form O(1) numerators of the resolved-image expansion (F023,
    # Professor-confirmed).  b1 is the imaginary w-linear correction, a0
    # the real w-independent one; they equal the matrix forms
    # -lam * x0^T A^{-1} x0 / |x0|**2 evaluated with A x0 = y.  Point mass
    # (gamma = kappa = 0): b1 = -1, a0 = 0; invariant b1 - a0 =
    # -lam**2 * mu_macro.  Serve uses NEITHER (F025) -- these feed the
    # resolved-image diagnostic `born_amplification`/`born_envelope` only.
    b1 = -lam * (2.0 * lam * r0_sq - x0_dot_y) / (det_a * r0_sq)
    a0 = -lam * (lam * r0_sq - x0_dot_y) / (det_a * r0_sq)

    return sqrt_mu, phi_geo, q2r, b1, a0


def born_amplification(w: float, y1: float, y2: float, gamma: float,
                       beta: float = 0.0, kappa: float = 0.0) -> complex:
    """Resolved-image Born amplification diagnostic ``F_born``.

    The weak-deflection wave-optics amplification in the RESOLVED-IMAGE
    regime (``w * Delta_tau >> 1``), expanded about ``sqrt(mu_macro)``::

        F_born = sqrt(mu_macro) * exp(1j*w*phi_geo)
                 * (1 + a0/Q2r + 1j*(w/2)*b1/Q2r + O(w**2 / Q2r**2)),

    with ``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)``, ``phi_geo`` the
    geometric Fermat phase at the macro image, and ``(b1, a0, Q2r)`` the
    ``O(1)`` geometry factors of `_born_factors`.  This is a DIAGNOSTIC /
    macro-limit form, NOT the serve object: the ``a0`` constant offset
    violates F009 below the band split (``F(w->0) = sqrt(mu_macro)``
    exactly, whereas ``1 + a0/Q2r`` does not), so the SERVE path uses the
    lead-only carrier `born_lead_carrier` instead (F025).  A
    ``c1 = 1 / (2 w)`` form (diverging as ``w -> 0``) would be the
    inverted-power bug the Professor explicitly ruled out.

    Returned in the ABSOLUTE Fermat-delay frame, matching
    `operator.F_op` (which `channels._exact_total` later demodulates by
    ``exp(-1j*w*t_min)``); `born_envelope` applies that same
    demodulation.

    Pure ``float64`` / ``complex128`` scalar arithmetic (``numba``-ready,
    no ``fastmath``).  This resolved-image DIAGNOSTIC is POSITIVE-PARITY
    ONLY: the radicand ``(1 - kappa)**2 - gamma**2 > 0`` is a precondition,
    enforced by an explicit `BornDomainError` guard (the ``abs()`` in
    `_born_factors` no longer lets a saddle host fail via ``math.sqrt``;
    extending the ``a0``/``b1`` correction to the saddle is out of scope).

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
        The resolved-image amplification diagnostic ``F_born(w)``.
    """
    # Fail-loud positive-parity precondition: the a0/b1 resolved-image
    # correction is derived only for det_a > 0.  Guard on the SAME radicand
    # _born_factors uses; det_a == 0 is the measure-zero parity wall.
    if (1.0 - kappa) ** 2 - gamma ** 2 <= 0.0:
        raise BornDomainError(
            f'born_amplification is positive-parity only: radicand '
            f'(1 - kappa)**2 - gamma**2 = {(1.0 - kappa) ** 2 - gamma ** 2} '
            f'<= 0 for (kappa, gamma) = ({kappa}, {gamma}). The a0/b1 '
            f'resolved-image correction is not derived on the macro saddle.')
    sqrt_mu, phi_geo, q2r, b1, a0 = _born_factors(y1, y2, gamma, beta, kappa)
    correction = 1.0 + a0 / q2r + 1j * (0.5 * w) * b1 / q2r
    return sqrt_mu * cmath.exp(1j * w * phi_geo) * correction


def born_lead_carrier(w: float, y1: float, y2: float, gamma: float,
                      beta: float = 0.0, kappa: float = 0.0) -> complex:
    """Lead-only Born carrier ``morse * sqrt(|mu_macro|) * exp(1j*w*phi_geo)``.

    THE SERVE OBJECT for the Born exterior.  It carries ONLY the analytic
    lead term -- NO ``a0``/``b1`` resolved-image correction -- because a
    driver-trained chart interpolates the residual ``F_exact - F_carrier``
    and the lead-only carrier is what makes that residual splinable
    (F025): the ``(a0, b1)`` carrier inflates the azimuthal node count
    2.5x-11x over the swept gamma range, and ``a0`` breaks the exact limit
    ``F(w -> 0) = sqrt(mu_macro)`` (F009).  ``phi_geo`` supplies the
    Einstein-scale phase variation in closed form, so the demodulated
    residual varies slowly.

    Serves BOTH parities.  On the macro saddle (``det_a = (1 - kappa)**2 -
    gamma**2 < 0``, macro Morse index 1) the carrier origin carries the
    exact Morse phase ``-1j`` (F024/F009-S): the Fresnel prefactor is
    ``(2*pi*i/w) |det A|^(-1/2) exp(-i*pi*n/2)`` and the Morse phase does
    NOT cancel in the carrier itself.  ``|F_carrier| = sqrt(|mu_macro|)``
    is therefore ``w``-INDEPENDENT for both parities, but the TOTAL phase
    is not (F009-S drift ``w * [tau_G + 0.5*ln(w/2) + c0]`` lives in
    ``exp(1j*w*phi_geo)``).

    Returned in the ABSOLUTE Fermat-delay frame, matching
    `operator.F_op`; downstream demodulation is the caller's (mirroring
    `born_amplification`).  Pure ``float64`` / ``complex128`` scalar
    arithmetic (``numba``-ready, no ``fastmath``).  Assumes the
    configuration has passed `born_gate`.

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
        The lead-only carrier ``morse * sqrt(|mu_macro|) *
        exp(1j*w*phi_geo)``, with ``morse = -1j`` on the macro saddle and
        ``1.0`` at positive parity.
    """
    sqrt_mu, phi_geo, _, _, _ = _born_factors(y1, y2, gamma, beta, kappa)

    # Morse phase of the macro image.  det_a is beta-independent (A's
    # eigenvalues are lam -/+ gamma), so the parity is fixed by (gamma,
    # kappa) alone.  For the macro SADDLE (det_a < 0, Morse index 1) the
    # carrier origin is the EXACT literal -1j (F009-S); cmath.exp(-1j*pi/2)
    # is NOT used -- it injects a ~6e-17 real part that rotates into |F|
    # and breaks the w-independent-magnitude acceptance.  For positive
    # parity (det_a > 0) morse is the float 1.0, so the product is
    # bit-identical to sqrt_mu * exp(1j*w*phi_geo).
    det_a = (1.0 - kappa) ** 2 - gamma ** 2
    morse = (-1j) ** 1 if det_a < 0.0 else 1.0
    return morse * sqrt_mu * cmath.exp(1j * w * phi_geo)


def born_gate(w: float, y1: float, y2: float, gamma: float,
              beta: float, kappa: float) -> None:
    """Refuse configurations outside the Born validity region.

    Two physically distinct, load-bearing guards, all raising
    `BornDomainError` (a named refusal, never a silent ``nan``):

    * **Guard B -- two-sided parity-wall margin.**  The reduced shear
      ``gamma_p = |gamma| / (1 - kappa)`` must stay at least
      `DELTA_GAMMA_P` away from ``1`` on EITHER side
      (``|gamma_p - 1| > DELTA_GAMMA_P``).  The convergence radius is the
      parity wall ``gamma_p = 1`` (``det A = 0``), where the macro image
      degenerates on both parities; the wall strip is refused so the
      positive branch (``gamma_p < 1``) and the saddle branch
      (``gamma_p > 1``) each see ``det A`` safely away from zero.
    * **Guard A -- band split (re-keyed).**  Refuse once the two real
      images are RESOLVED, ``w * Delta_tau >= RHO_END``, with
      ``Delta_tau`` the difference of their FULL Fermat delays
      (`geometry.find_images` + `geometry.delay`, including the ``-ln|x|``
      term; NOT ``phi_geo`` nor ``w * r0_sq`` -- F024).  Above the split
      the served lead-only carrier is superseded by the two-real-image
      ppGO + ghost branch, so the Born carrier rung declines there.

    The exact degenerate axes (``1 - kappa <= 0`` and
    ``|gamma| == 1 - kappa``) are refused first by delegating to
    `geometry.macro_matrix`, whose `geometry.LensDomainError` propagates
    unchanged; its returned matrix feeds `geometry.find_images`.

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
        Guard B (two-sided parity-wall margin) or guard A (band split).
    """
    # Degenerate-axis refusal, shared with every other wave-branch path;
    # its matrix is reused by the band-split image finder below.
    matrix = geometry.macro_matrix(gamma, beta=beta, kappa=kappa)

    lam = 1.0 - float(kappa)
    gamma_p = abs(float(gamma)) / lam

    # Guard B: two-sided parity-wall margin.  Refuse the strip
    # |gamma_p - 1| <= DELTA_GAMMA_P straddling the parity wall gamma_p = 1
    # (det A = 0), where the macro image degenerates on BOTH parities.  For
    # positive parity (gamma_p < 1) this is byte-identical to the old
    # one-sided gamma_p >= 1 - DELTA_GAMMA_P refusal; it additionally
    # refuses the wall strip just above 1 so the saddle branch below only
    # sees gamma_p safely past the wall.
    if abs(gamma_p - 1.0) <= DELTA_GAMMA_P:
        raise BornDomainError(
            f'Born rung refuses (kappa, gamma) = ({kappa}, {gamma}): '
            f'reduced shear gamma_p = |gamma| / (1 - kappa) = {gamma_p} '
            f'>= 1 - DELTA_GAMMA_P = {1.0 - DELTA_GAMMA_P}. The series '
            f'converges only inside the positive-parity margin; the macro '
            f'image degenerates at the parity wall gamma_p = 1.')

    # Guard A: band split.  Delta_tau is the FULL Fermat-delay difference
    # of the two real images (includes -ln|x|); resolved => decline.
    source = np.array([float(y1), float(y2)], dtype=float)
    images = geometry.find_images(source, matrix)
    if len(images) < 2:
        raise BornDomainError(
            f'Born rung refuses (w, y1, y2, gamma) = '
            f'({w}, {y1}, {y2}, {gamma}): the band split needs the two '
            f'real images, but geometry.find_images returned '
            f'{len(images)}. The configuration should yield two real '
            f'images for the Born carrier to serve.')
    delays = [geometry.delay(image, source, matrix) for image in images]
    delta_tau = max(delays) - min(delays)
    if float(w) * delta_tau >= RHO_END:
        raise BornDomainError(
            f'Born rung refuses (w, y1, y2, gamma) = '
            f'({w}, {y1}, {y2}, {gamma}): the two real images are resolved, '
            f'w * Delta_tau = {float(w) * delta_tau} >= RHO_END = '
            f'{RHO_END}. Above the band split the lead-only carrier is '
            f'superseded by the two-real-image ppGO + ghost branch.')


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
    BornDomainError
        If the radicand ``(1 - kappa)**2 - gamma**2 <= 0`` (macro saddle):
        this positive-parity diagnostic's ``a0``/``b1`` correction is not
        derived there.
    ValueError
        If ``dense_w`` does not match ``geom.w`` (the kernels and switch
        would be sampled on a different grid than the Born total).
    """
    # Fail-loud positive-parity precondition: this resolved-image
    # DIAGNOSTIC envelope carries the a0/b1 correction, derived only for
    # det_a > 0.  Guard on the SAME radicand _born_factors uses (the
    # ``abs()`` there no longer surfaces a saddle host via math.sqrt);
    # det_a == 0 is the measure-zero parity wall.
    if (1.0 - kappa) ** 2 - gamma ** 2 <= 0.0:
        raise BornDomainError(
            f'born_envelope is positive-parity only: radicand '
            f'(1 - kappa)**2 - gamma**2 = {(1.0 - kappa) ** 2 - gamma ** 2} '
            f'<= 0 for (kappa, gamma) = ({kappa}, {gamma}). The a0/b1 '
            f'resolved-image correction is not derived on the macro saddle.')

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
    # This is the resolved-image DIAGNOSTIC envelope (a0 + b1 correction),
    # not the serve object; the served carrier is `born_lead_carrier`.
    sqrt_mu, phi_geo, q2r, b1, a0 = _born_factors(
        float(y1), float(y2), float(gamma), float(beta), float(kappa))
    correction = 1.0 + a0 / q2r + 1j * (0.5 * dense_w) * b1 / q2r
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
