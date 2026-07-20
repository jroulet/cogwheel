"""Topology-stable four-channel Chang-Refsdal decomposition.

WHAT
----
`ChangRefsdalChannels` continues a universal FOUR-label partition

    F(w) = sum_a exp(1j * w * tau_a) * K_a(w)

along a continuous path in lens parameters, exposing each channel's
Fermat delay ``tau_a`` and kernel ``K_a(w)`` together with the exact
amplification total.  The label count is fixed at four regardless of
how many real images exist, so the decomposition does not jump when a
caustic is crossed and images are created or destroyed.  This is the
entry point the microlensed relative-binning likelihood (Build 2)
consumes.

WHY FOUR STABLE LABELS
----------------------
A Chang-Refsdal lens has either two or four real images (see the Morse
census in the geometry module); the count changes discontinuously
across a caustic.  A relative-binning summary that indexed physical
images directly would gain or lose a channel at the crossing.  Instead
every parameter point uses the same four computational channels:

* Real images are continued between neighbouring points by a
  brute-force minimum-cost ASSIGNMENT on lens-plane markers (the
  problem is bounded at four labels, so the exhaustive solver is a
  deliberate, adequate choice -- no linear-assignment machinery).
* Empty labels are parked at the NEAREST CRITICAL POINT
  (`geometry.nearest_caustic_point`); they become the newly born images
  when the source crosses into the caustic, so a channel is always
  present to receive them.
* A smooth switch ``S_a(w)`` hands each resolved channel over to its
  physical stationary-phase kernel ``H_a`` (`geometry.image_kernel`),
  each label switching on ITS OWN separation from the CRITICAL carrier,
  ``S_a(w) = smootherstep(w * |tau_a - tau_c|, RHO_START, RHO_END)``.
  The switch buys smoothness, not accuracy.

THE SACR-C DECOMPOSITION
------------------------
This module builds the switched-analytic + single-envelope
decomposition of the design report (Build 3f).  Rather than splitting
the whole amplification total among the four channels through an
artificial cluster gauge, it carries

    F(w) = sum_a exp(1j*w*tau_a) * S_a(w) * H_a(w)
           + exp(1j*w*tau_c) * E(w),

where ``tau_c`` is the delay of the parked critical carrier (the
`geometry.nearest_caustic_point` delay, relative to the minimum image),
``H_a`` is the analytic saddle kernel of the resolved image, and

    E(w) := exp(-1j*w*tau_c) * (F - sum_a exp(1j*w*tau_a) * S_a * H_a)

is the SINGLE smooth transition envelope, demodulated at ``tau_c``.
Because the demodulation distance and the switch scale are the same
quantity ``w * |tau_a - tau_c|``, only channels with ``S_a < 1``
contribute O(1) content to ``E``, and their phase against the ``tau_c``
carrier is bounded by ``RHO_END`` -- so ``E`` is beat-free by
construction and is the one object the likelihood interpolates
(`switched_analytic_channels` in `_gauge`).

To preserve the four physical labels the likelihood consumes, the
kernels are returned in the equivalent per-frequency-weight (four
channel) form

    K_a(w) = S_a*H_a + u_a(w) * exp(-1j*w*(tau_a - tau_c)) * E,
    u_a(w) = (1 - S_a + eta) / sum_b (1 - S_b + eta),

so that ``F = sum_a exp(1j*w*tau_a) * K_a`` still holds identically for
any weights summing to one (``eta = _ENVELOPE_WEIGHT_FLOOR``).  A fifth
envelope channel was deliberately NOT introduced: it would change
``_N_CHANNELS`` and the switch neighbour set and label-continuity
behaviour that the crossing-scenario tests depend on.

WHY IT DELEGATES
----------------
The exact residual projection (now carrying the ``tau_c`` critical
carrier) lives once in `_gauge`; this module calls it rather than
carrying a second copy, and the public `reconstruct_from_envelope`
wraps the forward reconstruction for the likelihood.  The
wave/geometric evaluation gate and the smooth-switch window live once
in `operator`; this module imports ``RHO_START``, ``RHO_END`` and
`select_branch` rather than re-deriving the thresholds.  The exact
total is evaluated with the contour-free operator `operator.F_op` where
the wave branch is certified and with `operator.geometric_amplification`
once `operator.select_branch` reports the stationary-phase branch is
legitimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._gauge import (
    channels_from_envelope, envelope_total, reconstructed_total,
    smootherstep, switched_analytic_channels)
from cogwheel.lensing.chang_refsdal.operator import (
    RHO_START, RHO_END, MAX_ORDER, F_op_grid, cancellation_exponent,
    geometric_amplification, select_branch)

__all__ = ['ChangRefsdalChannels', 'ChangRefsdalGeometryPartition',
           'ChangRefsdalPartition', 'real_image_delays',
           'reconstruct_from_envelope']

#: The fixed number of topology-stable labels.
_N_CHANNELS = 4

#: Floor added to every channel's envelope weight ``1 - S_a`` so that a
#: fully resolved channel (``S_a = 1``) still carries a small, non-zero
#: share of the transition envelope and the per-frequency weights are
#: always normalizable.  This is the SACR-C ``eta`` of the design report
#: (Sec. 3): resolved channels keep an ``O(eta)`` envelope weight, while
#: unresolved and virtual channels (``S_a -> 0``) carry the bulk.
_ENVELOPE_WEIGHT_FLOOR = 1e-2

#: Sentinel written into ``operator_orders`` for a frequency evaluated
#: on the geometric (stationary-phase) branch, which has no operator
#: order to report.
_GEOMETRIC_ORDER = -1

#: Floor on the marker length scale used to normalize assignment costs,
#: so a configuration collapsing toward the origin cannot make every
#: continuation cost blow up.
_MARKER_SCALE_FLOOR = 0.3


def _validate_frequencies(w: Sequence[float]) -> np.ndarray:
    """Return ``w`` as a validated strictly increasing positive grid.

    Parameters
    ----------
    w : Sequence[float]
        Candidate dimensionless frequency grid.

    Returns
    -------
    np.ndarray
        The validated 1-D array.

    Raises
    ------
    ValueError
        If ``w`` is not one-dimensional with at least two points, is not
        everywhere positive, or is not strictly increasing.
    """
    array = np.asarray(w, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(
            'The frequency grid must be a 1-D array with at least two '
            f'points, got shape {array.shape}.')
    if np.any(array <= 0.0):
        raise ValueError(
            'The frequency grid must be strictly positive, got a '
            f'minimum of {float(array.min())}.')
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(
            'The frequency grid must be strictly increasing.')
    return array


def _initial_assignment(positions: np.ndarray,
                        n_images: int) -> np.ndarray:
    """Deterministic first labeling: real images by polar angle.

    Real labels are ordered by polar angle (ties broken by radius);
    empty labels follow.  This is the deterministic RESET convention a
    path uses at a fresh start or after a far proposal.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(n_images, 2)`` image positions.
    n_images : int
        Number of real images.

    Returns
    -------
    np.ndarray
        Length-``_N_CHANNELS`` image index per channel; ``-1`` marks a
        virtual (empty) channel.
    """
    order = np.lexsort((np.linalg.norm(positions, axis=1),
                        np.arctan2(positions[:, 1], positions[:, 0])))
    assignment = np.full(_N_CHANNELS, -1, dtype=int)
    assignment[:n_images] = order
    return assignment


def _continued_assignment(prev_markers: np.ndarray,
                          positions: np.ndarray,
                          n_images: int) -> np.ndarray:
    """Continue labels from a previous point by min-cost assignment.

    Exhaustively searches the ``n_images``-out-of-``_N_CHANNELS`` label
    choices and the permutation of images onto them, minimizing the
    squared lens-plane displacement from the previous markers.  The
    search is deliberately brute force: with at most four labels there
    are at most 24 candidates, and an exact solver is both adequate and
    transparent here.

    Parameters
    ----------
    prev_markers : np.ndarray
        Shape ``(_N_CHANNELS, 2)`` markers from the previous point.
    positions : np.ndarray
        Shape ``(n_images, 2)`` current image positions.
    n_images : int
        Number of real images.

    Returns
    -------
    np.ndarray
        Length-``_N_CHANNELS`` image index per channel; ``-1`` marks a
        virtual channel.
    """
    scale = max(float(np.median(np.linalg.norm(prev_markers, axis=1))),
                _MARKER_SCALE_FLOOR)
    best_cost = np.inf
    best = np.full(_N_CHANNELS, -1, dtype=int)
    for labels in permutations(range(_N_CHANNELS), n_images):
        cost = 0.0
        for image_index, channel in enumerate(labels):
            offset = positions[image_index] - prev_markers[channel]
            cost += float(offset @ offset) / scale**2
        if cost < best_cost:
            best_cost = cost
            best = np.full(_N_CHANNELS, -1, dtype=int)
            for image_index, channel in enumerate(labels):
                best[channel] = image_index
    return best


def _assign_labels(prev_markers: np.ndarray | None,
                   images: list[np.ndarray],
                   virtual_position: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Assign real images to channels and park empty channels.

    Parameters
    ----------
    prev_markers : np.ndarray or None
        Markers from the previous path point, or ``None`` to reset to
        the deterministic initial labeling.
    images : list of np.ndarray
        Real image positions.
    virtual_position : np.ndarray
        Shape ``(2,)`` lens-plane position (the nearest critical point)
        where empty channels are parked.

    Returns
    -------
    assignment : np.ndarray
        Length-``_N_CHANNELS`` image index per channel; ``-1`` marks a
        virtual channel.
    markers : np.ndarray
        Shape ``(_N_CHANNELS, 2)`` lens-plane marker per channel: the
        image position for a real channel, ``virtual_position`` for a
        virtual one.

    Raises
    ------
    ValueError
        If more than ``_N_CHANNELS`` real images are supplied, which the
        four-label topology cannot represent.
    """
    n_images = len(images)
    if n_images > _N_CHANNELS:
        raise ValueError(
            f'A Chang-Refsdal lens yields at most {_N_CHANNELS} real '
            f'images, but {n_images} were supplied; the topology-stable '
            'partition cannot label them.')
    positions = np.asarray(images, dtype=float).reshape(n_images, 2)
    if prev_markers is None:
        assignment = _initial_assignment(positions, n_images)
    else:
        assignment = _continued_assignment(prev_markers, positions,
                                            n_images)

    markers = np.tile(np.asarray(virtual_position, dtype=float),
                      (_N_CHANNELS, 1))
    for channel, image_index in enumerate(assignment):
        if image_index >= 0:
            markers[channel] = positions[image_index]
    return assignment, markers


def _labeled_delays(assignment: np.ndarray,
                    image_delays: np.ndarray,
                    virtual_delay: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Fill per-channel delays, parking virtual channels at the caustic.

    Parameters
    ----------
    assignment : np.ndarray
        Image index per channel (``-1`` for virtual).
    image_delays : np.ndarray
        Real-image Fermat delays, relative to the minimum.
    virtual_delay : float
        Delay of the parked (critical-point) carrier, relative to the
        minimum.

    Returns
    -------
    delays : np.ndarray
        Length-``_N_CHANNELS`` delay per channel.
    real_mask : np.ndarray
        Length-``_N_CHANNELS`` boolean: ``True`` where the channel holds
        a real image.
    """
    delays = np.full(_N_CHANNELS, float(virtual_delay), dtype=float)
    real_mask = np.zeros(_N_CHANNELS, dtype=bool)
    for channel, image_index in enumerate(assignment):
        if image_index >= 0:
            delays[channel] = image_delays[image_index]
            real_mask[channel] = True
    return delays, real_mask


def _physical_kernels(w: np.ndarray,
                      assignment: np.ndarray,
                      images: list[np.ndarray],
                      matrix: np.ndarray) -> np.ndarray:
    """Stationary-phase target kernel per channel.

    Real channels carry `geometry.image_kernel`; virtual channels carry
    zeros, which are inert because their switch value is zero.

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency grid.
    assignment : np.ndarray
        Image index per channel (``-1`` for virtual).
    images : list of np.ndarray
        Real image positions.
    matrix : np.ndarray
        Shape ``(2, 2)`` macro matrix.

    Returns
    -------
    np.ndarray
        Shape ``(n_w, _N_CHANNELS)`` complex kernels.
    """
    kernels = np.zeros((w.shape[0], _N_CHANNELS), dtype=complex)
    for channel, image_index in enumerate(assignment):
        if image_index >= 0:
            kernels[:, channel] = geometry.image_kernel(
                w, images[image_index], matrix)
    return kernels


def _channel_switch(w: np.ndarray,
                    delays: np.ndarray,
                    real_mask: np.ndarray,
                    critical_delay: float) -> np.ndarray:
    """Per-channel smooth hand-over switch on the criticality separation.

    Each real channel switches on its OWN separation from the CRITICAL
    (parked) carrier ``tau_c`` -- the SACR-C criticality-separation rule
    of the design report (Sec. 3), which supersedes the F008
    full-cluster nearest-neighbour rule where the report certifies it:

        delta_a = |tau_a - tau_c|,   S_a(w) = smootherstep(
            w * delta_a, RHO_START, RHO_END).

    The switch scale is then exactly the demodulation distance of the
    transition envelope ``E`` against its ``tau_c`` carrier, so any
    channel whose switch has not completed contributes only bounded-phase
    (``<= RHO_END`` rad) content to ``E`` -- the beat-free guarantee the
    old full-cluster rule lacked.  Images merging AT the critical point
    have ``tau_a -> tau_c``, so ``delta_a`` shrinks and the switch stays
    in the artificial gauge exactly as F008 intends (at least as
    conservatively, since ``delta_a = |tau_a - tau_c| ~ delta_pair / 2``
    for a genuine merger); ACCIDENTAL delay degeneracies between
    non-merging images no longer stall the switch, because they only
    matter when they also sit near ``tau_c``, where the demodulated phase
    in ``E`` is equally tiny and harmless.  Virtual channels never switch
    (``S_a = 0`` for a virtual label).

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency grid.
    delays : np.ndarray
        Per-channel delays, indexed by cluster label ``0 .. _N_CHANNELS - 1``.
    real_mask : np.ndarray
        Boolean mask of real channels.
    critical_delay : float
        Delay ``tau_c`` of the parked critical carrier (relative to the
        minimum image delay), the separation reference every real
        channel switches against.

    Returns
    -------
    np.ndarray
        Shape ``(n_w, _N_CHANNELS)`` switch in ``[0, 1]``.
    """
    switch = np.zeros((w.shape[0], _N_CHANNELS), dtype=float)
    real_ids = np.flatnonzero(real_mask)
    for channel in real_ids:
        separation = abs(float(delays[channel]) - float(critical_delay))
        switch[:, channel] = smootherstep(w * separation,
                                          RHO_START, RHO_END)
    return switch


def _envelope_weights(switch: np.ndarray) -> np.ndarray:
    """Raw per-frequency envelope-apportionment weights ``1 - S_a + eta``.

    The SACR-C weight policy (design report Sec. 3): every channel's
    unnormalized share of the transition envelope is ``1 - S_a + eta``,
    so unresolved and virtual channels (``S_a -> 0``) carry the bulk
    while a fully resolved channel (``S_a = 1``) keeps only the
    ``eta = _ENVELOPE_WEIGHT_FLOOR`` floor.  The floor guarantees a
    strictly positive, normalizable weight at every frequency; `_gauge`
    normalizes these across channels so the reconstruction identity holds
    exactly.  The single authoritative home of this policy -- the
    likelihood reconstructs densely through `reconstruct_from_envelope`,
    which reuses it rather than re-deriving the weights.

    Parameters
    ----------
    switch : np.ndarray
        Per-channel switch ``S_a`` in ``[0, 1]``, shape
        ``(n_w, _N_CHANNELS)``.

    Returns
    -------
    np.ndarray
        Non-negative raw weights of the same shape.
    """
    return 1.0 - switch + _ENVELOPE_WEIGHT_FLOOR


def _min_delay_separation(delays: np.ndarray,
                          real_mask: np.ndarray) -> float:
    """Smallest pairwise Fermat-delay separation among real channels.

    This is ``delta_min`` for `operator.select_branch`, the wave vs.
    geometric branch gate -- a DIFFERENT quantity from the per-channel
    switch separation ``delta_j`` of `_channel_switch` (Eq.
    delay-separation), and it is deliberately minimised over REAL images
    ONLY.  The two must not be conflated: the switch chooses a channel
    gauge, this gate chooses an evaluation method, and the operator
    module docstring warns their thresholds must not leak into each
    other.

    Real-only is correct HERE because the geometric branch replaces the
    wave operator ``F_op`` with the stationary-phase sum, which runs
    over the real images alone (a virtual label parked at the critical
    point carries ``H_j = 0`` and contributes no stationary point).  The
    asymptote is therefore legitimate exactly when the actual stationary
    points -- the real images -- are mutually resolved
    (``w * delta_min >= RHO_END``); the proximity of a parked virtual
    label neither adds nor removes a saddle and so must not enter this
    gate.  The paper defines no branch-gate separation of its own (it
    uses the exact projection throughout, Eq. exact-reconstruction);
    the resolution criterion is owned by `operator.select_branch`, and
    real-only matches it.  Fewer than two real images means nothing is
    resolved, so zero is returned and the resolution condition fails,
    keeping the wave branch.

    Parameters
    ----------
    delays : np.ndarray
        Per-channel delays.
    real_mask : np.ndarray
        Boolean mask of real channels.

    Returns
    -------
    float
        The minimum pairwise separation among real channels, or ``0.0``
        if fewer than two real channels exist.
    """
    real_delays = delays[real_mask]
    if real_delays.size < 2:
        return 0.0
    differences = np.abs(real_delays[:, None] - real_delays[None, :])
    upper = differences[np.triu_indices(real_delays.size, k=1)]
    return float(np.min(upper))


def _exact_total(w: np.ndarray, source: np.ndarray, gamma: float,
                 beta: float, kappa: float, t_min: float,
                 delta_min: float, max_order: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the exact amplification total over the grid.

    Per frequency, `operator.select_branch` chooses between the
    contour-free wave operator and the stationary-phase
    `operator.geometric_amplification`.  The wave-branch nodes are
    evaluated together in a SINGLE `operator.F_op_grid` call (the
    operator table and per-order weight vectors are ``w``-independent, so
    they are built once and reused across the node set); the geometric
    nodes are evaluated per node.  Every result is shifted by
    ``exp(-1j * w * t_min)`` so its carrier matches the channels'
    minimum-relative delays.

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency grid.
    source : np.ndarray
        Shape ``(2,)`` source position.
    gamma, beta, kappa : float
        Shear magnitude, shear orientation, convergence.
    t_min : float
        Minimum absolute Fermat delay, subtracted to form relative
        delays.
    delta_min : float
        Smallest pairwise real-channel delay separation, for the branch
        gate.
    max_order : int
        Operator-series order cap forwarded to `operator.F_op_grid`.

    Returns
    -------
    total : np.ndarray
        Exact amplification total in the relative-delay convention.
    orders : np.ndarray
        Operator order used per frequency (``_GEOMETRIC_ORDER`` on the
        geometric branch).
    converged : np.ndarray
        Whether each wave-branch evaluation met the stopping rule
        (``True`` on the geometric branch).
    """
    n_w = w.shape[0]
    total = np.empty(n_w, dtype=complex)
    orders = np.empty(n_w, dtype=int)
    converged = np.empty(n_w, dtype=bool)

    # The branch decision stays PER NODE (`cancellation_exponent` +
    # `select_branch`).  Geometric nodes are evaluated inline, exactly as
    # before; wave nodes are only collected here and evaluated together
    # in one batched `F_op_grid` call below, since within one lens
    # configuration only ``w`` varies and the operator table / weight
    # vectors are ``w``-independent (see `operator.F_op_grid`).
    # On a macro-saddle host the L = w*|y'| cancellation bookkeeping does
    # not exist (`cancellation_exponent` is positive-parity-only by
    # design; the Schwinger channel is L_S = pi*w/4, y-independent), and
    # the operator's saddle arm owns the per-node geometric-vs-wave
    # routing internally (resolved AND above the w <= 60 ceiling ->
    # stationary phase; otherwise Schwinger).  Delegate every saddle
    # node to the batched operator call; the positive-parity branch
    # decision below is byte-identical to before.
    saddle_host = not 1.0 - kappa > abs(gamma)

    wave_indices = []
    for index in range(n_w):
        frequency = float(w[index])
        if saddle_host:
            wave_indices.append(index)
            continue
        exponent = cancellation_exponent(frequency, source, gamma, kappa)
        if select_branch(frequency, delta_min, exponent) == 'geometric':
            value = complex(geometric_amplification(
                frequency, source, gamma, beta=beta, kappa=kappa))
            orders[index] = _GEOMETRIC_ORDER
            converged[index] = True
            total[index] = value * np.exp(-1j * frequency * t_min)
        else:
            wave_indices.append(index)

    if wave_indices:
        wave_idx = np.asarray(wave_indices, dtype=int)
        w_wave = w[wave_idx]
        # A single batched wave-branch evaluation.  Any uncertifiable
        # node raises `operator.CancellationError` from inside this call
        # and propagates unswallowed -- identical to the former per-node
        # `F_op` raise -- so the RB and brute paths refuse symmetrically.
        values_wave, orders_wave, converged_wave = F_op_grid(
            w_wave, source, gamma, beta=beta, kappa=kappa,
            max_order=max_order)
        # Same per-node relative-delay carrier as the geometric branch,
        # applied elementwise over the wave subset.
        total[wave_idx] = values_wave * np.exp(-1j * w_wave * t_min)
        orders[wave_idx] = orders_wave
        converged[wave_idx] = converged_wave

    return total, orders, converged


def real_image_delays(gamma: float, y: Sequence[float], *,
                      beta: float = 0.0,
                      kappa: float = 0.0) -> np.ndarray:
    """Sorted real-image relative Fermat delays at one parameter point.

    Frequency-INDEPENDENT geometry only: the real macro images and their
    Fermat delays via `geometry.macro_matrix`, `geometry.find_images` and
    `geometry.delay`.  No operator sweep (`F_op`) is performed, so this is
    cheap enough to place spline nodes without paying an engine
    evaluation.  The returned delays are dimensionless (in the ``w``
    convention, ``tau = w-frame Fermat delay``) and relative to the
    minimum-delay image, matching `ChangRefsdalPartition.delays` on the
    real channels.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    y : Sequence[float]
        Shape ``(2,)`` source position.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.

    Returns
    -------
    np.ndarray
        Real-image relative Fermat delays ``tau_a`` (dimensionless),
        sorted increasing; the smallest is ``0``.

    Raises
    ------
    geometry.LensDomainError
        For the two `geometry.macro_matrix` refusals -- Type III
        ``1 - kappa <= 0`` and the ``det A = 0`` parity boundary
        ``abs(gamma) == 1 - kappa`` -- or the image census guard, exactly
        as in `ChangRefsdalChannels.evaluate` and on the brute-force
        strain path, so the paths refuse symmetrically.  Both parities
        (positive parity and the macro saddle) return normally.
    """
    source = np.asarray(y, dtype=float)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(source, matrix)
    absolute_delays = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    return np.sort(absolute_delays - absolute_delays.min())


def reconstruct_from_envelope(w: np.ndarray | float,
                              envelope: np.ndarray | complex,
                              delays: np.ndarray,
                              saddle_kernels: np.ndarray,
                              switch: np.ndarray,
                              critical_delay: float
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild SACR-C channels and total from an interpolated envelope.

    The forward SACR-C reconstruction the microlensed likelihood uses on
    its hot path: having interpolated the single smooth envelope ``E(w)``
    from coarse engine nodes onto a dense frequency grid and evaluated
    the analytic saddle kernels ``H_a`` (`geometry.image_kernel`) and the
    switch ``S_a`` (`_channel_switch`) at the same frequencies, rebuild

        K_a(w) = S_a*H_a + u_a(w) * exp(-1j*w*(tau_a - tau_c)) * E,
        F(w)   = sum_a exp(1j*w*tau_a) * S_a*H_a + exp(1j*w*tau_c) * E,

    with the SACR-C per-frequency weights ``u_a`` derived HERE from
    `_envelope_weights` -- the single authoritative home of the
    ``1 - S_a + eta`` weight policy, so the likelihood never re-derives
    the apportionment.  The reconstruction identity
    ``F = sum_a exp(1j*w*tau_a) * K_a`` holds exactly for any envelope;
    only ``E`` is approximated by interpolation, never the algebra
    (`_gauge.channels_from_envelope`).

    Parameters
    ----------
    w : float or np.ndarray
        Dimensionless frequency, scalar or 1-D grid.
    envelope : complex or np.ndarray
        The transition envelope ``E(w)`` (typically interpolated), with
        the shape of ``w``.
    delays : np.ndarray
        Shape ``(_N_CHANNELS,)`` channel delays ``tau_a``, in the
        minimum-relative convention of `ChangRefsdalPartition.delays`.
    saddle_kernels : np.ndarray
        Analytic saddle kernels ``H_a`` (`geometry.image_kernel`),
        shape ``(_N_CHANNELS,)`` for scalar ``w`` or
        ``(n_w, _N_CHANNELS)`` for a grid; zero for virtual channels.
    switch : np.ndarray
        Per-channel switch ``S_a`` in ``[0, 1]``, same shape as
        ``saddle_kernels``.
    critical_delay : float
        Delay ``tau_c`` of the parked critical carrier, in the same
        minimum-relative convention as ``delays``.

    Returns
    -------
    kernels : np.ndarray
        Channel kernels ``K_a``, the per-image ``(tau_a, K_a)``
        decomposition, shape matching ``saddle_kernels``.
    total : np.ndarray
        The reconstructed amplification total ``F``, with the shape of
        ``w``.
    """
    weights = _envelope_weights(np.asarray(switch, dtype=float))
    return channels_from_envelope(
        w, envelope, delays, saddle_kernels, switch, critical_delay,
        weights)


@dataclass(frozen=True)
class ChangRefsdalPartition:
    """One evaluated four-channel partition at a parameter point.

    Attributes
    ----------
    w : np.ndarray
        Dimensionless frequency grid.
    source : np.ndarray
        Shape ``(2,)`` source position.
    gamma, beta, kappa : float
        Shear magnitude, shear orientation, convergence.
    delays : np.ndarray
        Shape ``(4,)`` channel delays ``tau_a``, relative to the minimum
        image delay ``t_min``.
    kernels : np.ndarray
        Shape ``(n_w, 4)`` channel kernels ``K_a(w)``; the per-image
        ``(tau_a, K_a)`` decomposition the likelihood consumes, with
        ``F = sum_a exp(1j*w*tau_a) * K_a`` holding exactly.
    envelope : np.ndarray
        Shape ``(n_w,)`` transition envelope ``E(w)``, demodulated at
        the critical carrier ``tau_c`` -- the SINGLE smooth object the
        likelihood interpolates (beat-free by construction).
    saddle_kernels : np.ndarray
        Shape ``(n_w, 4)`` analytic saddle kernels ``H_a(w)``
        (`geometry.image_kernel`), zero for virtual channels.
    switch : np.ndarray
        Shape ``(n_w, 4)`` per-channel switch ``S_a(w)`` in ``[0, 1]``,
        keyed on the criticality separation ``|tau_a - tau_c|``.
    critical_delay : float
        Delay ``tau_c`` of the parked critical carrier, relative to the
        minimum image delay ``t_min``.
    matrix : np.ndarray
        Shape ``(2, 2)`` macro matrix, exposed so the likelihood can
        re-evaluate ``H_a`` at dense frequencies.
    images : np.ndarray
        Shape ``(n_images, 2)`` real image positions in the lens plane.
    assignment : np.ndarray
        Shape ``(4,)`` real-image index per channel (``-1`` for a
        virtual channel), the map from channels to `images`.
    exact_total : np.ndarray
        Shape ``(n_w,)`` exact amplification total in the same
        relative-delay convention; the channels reconstruct it exactly.
    real_mask : np.ndarray
        Shape ``(4,)`` boolean: ``True`` where the channel holds a real
        image rather than a parked virtual label.
    markers : np.ndarray
        Shape ``(4, 2)`` lens-plane marker per channel, used to continue
        labels to the next point.
    t_min : float
        Minimum absolute Fermat delay subtracted from every delay.
    critical_theta : float
        Polar angle of the nearest critical point.
    critical_image : np.ndarray
        Shape ``(2,)`` nearest critical point in the lens plane.
    critical_source : np.ndarray
        Shape ``(2,)`` caustic point it maps to.
    caustic_distance : float
        Source-plane distance from ``source`` to the caustic.
    operator_orders : np.ndarray
        Shape ``(n_w,)`` operator order per frequency
        (``-1`` on the geometric branch).
    operator_converged : np.ndarray
        Shape ``(n_w,)`` wave-branch convergence flag per frequency.
    """

    w: np.ndarray
    source: np.ndarray
    gamma: float
    beta: float
    kappa: float
    delays: np.ndarray
    kernels: np.ndarray
    envelope: np.ndarray
    saddle_kernels: np.ndarray
    switch: np.ndarray
    critical_delay: float
    matrix: np.ndarray
    images: np.ndarray
    assignment: np.ndarray
    exact_total: np.ndarray
    real_mask: np.ndarray
    markers: np.ndarray
    t_min: float
    critical_theta: float
    critical_image: np.ndarray
    critical_source: np.ndarray
    caustic_distance: float
    operator_orders: np.ndarray
    operator_converged: np.ndarray

    @property
    def reconstructed(self) -> np.ndarray:
        """Coherent channel sum ``sum_a exp(1j*w*tau_a) * K_a(w)``.

        Returns
        -------
        np.ndarray
            The reconstructed total, which equals `exact_total` to the
            scale-aware floating-point tolerance of the projection.
        """
        return reconstructed_total(self.w, self.delays, self.kernels)

    @property
    def reconstruction_error(self) -> float:
        """Max absolute deviation of `reconstructed` from `exact_total`.

        Returns
        -------
        float
            ``max_w |reconstructed - exact_total|``.  Diagnostic only;
            near a fold the achievable value scales with the kernel
            magnitude and is not a flat constant.
        """
        return float(np.max(np.abs(self.reconstructed - self.exact_total)))

    @property
    def envelope_reconstruction(self) -> np.ndarray:
        """Total rebuilt from the smooth envelope and analytic saddles.

        Reconstructs ``F`` through the SACR-C identity
        ``F = sum_a S_a H_a exp(1j*w*tau_a) + exp(1j*w*tau_c) E(w)``
        using the stored envelope, saddle kernels, and switch weights.
        This is the object the likelihood rebuilds after interpolating
        only the single smooth envelope ``E(w)`` onto a dense grid.

        Returns
        -------
        np.ndarray
            The reconstructed total, equal to `exact_total` to the
            scale-aware floating-point tolerance of the projection.
        """
        return envelope_total(
            self.w, self.delays, self.saddle_kernels, self.switch,
            self.critical_delay, self.envelope)


@dataclass(frozen=True)
class ChangRefsdalGeometryPartition:
    """Geometry-only Chang-Refsdal partition -- no exact amplification.

    The CHEAP half of `ChangRefsdalChannels.evaluate`: the label-continued
    channel geometry a surrogate envelope needs to be reconstructed to
    channels via `reconstruct_from_envelope`, computed WITHOUT the
    expensive exact operator/Schwinger total (`_exact_total`) or the
    SACR-C envelope build.  It carries exactly the arguments
    `reconstruct_from_envelope` consumes alongside an interpolated
    envelope, plus the caustic distance the microlensed likelihood reads
    for its in-domain check.

    Attributes
    ----------
    w : np.ndarray
        Dimensionless frequency grid the kernels and switch are sampled
        on.
    delays : np.ndarray
        Shape ``(4,)`` channel delays ``tau_a``, relative to the minimum
        image delay ``t_min``.
    saddle_kernels : np.ndarray
        Shape ``(n_w, 4)`` analytic saddle kernels ``H_a(w)``
        (`geometry.image_kernel`), zero for virtual channels.
    switch : np.ndarray
        Shape ``(n_w, 4)`` per-channel switch ``S_a(w)`` in ``[0, 1]``,
        keyed on the criticality separation ``|tau_a - tau_c|``.
    critical_delay : float
        Delay ``tau_c`` of the parked critical carrier, relative to the
        minimum image delay ``t_min``.
    real_mask : np.ndarray
        Shape ``(4,)`` boolean: ``True`` where the channel holds a real
        image rather than a parked virtual label.
    caustic_distance : float
        Source-plane distance from the source to the caustic; the
        in-domain proximity the likelihood reads.
    """

    w: np.ndarray
    delays: np.ndarray
    saddle_kernels: np.ndarray
    switch: np.ndarray
    critical_delay: float
    real_mask: np.ndarray
    caustic_distance: float


class ChangRefsdalChannels:
    """Topology-stable four-channel Chang-Refsdal amplification.

    Continues a universal four-label partition
    ``F(w) = sum_a exp(1j*w*tau_a) * K_a(w)`` along a path in lens
    parameters.  Call `evaluate` repeatedly to continue labels from the
    previous point, `reset` to return to the deterministic initial
    labeling, or `evaluate_path` for a fresh, fully-continued sweep.

    Parameters
    ----------
    w : Sequence[float]
        Dimensionless frequency grid: 1-D, strictly positive, strictly
        increasing.
    max_order : int, optional
        Operator-series order cap forwarded to `operator.F_op`.
        Defaults to `operator.MAX_ORDER`.

    Raises
    ------
    ValueError
        If ``w`` is not a valid frequency grid.
    """

    def __init__(self, w: Sequence[float], *,
                 max_order: int = MAX_ORDER) -> None:
        self._w = _validate_frequencies(w)
        self._max_order = int(max_order)
        self._markers: np.ndarray | None = None

    @property
    def w(self) -> np.ndarray:
        """The validated dimensionless frequency grid."""
        return self._w

    def reset(self) -> None:
        """Forget the previous point's labels.

        The next `evaluate` uses the deterministic initial labeling
        (real images by polar angle) rather than continuing from a
        stored marker set.  This is the reset convention for a far
        proposal, whose total agrees with a continued evaluation because
        the total is label-invariant.
        """
        self._markers = None

    def evaluate(self, *, gamma: float, y: Sequence[float],
                 beta: float = 0.0,
                 kappa: float = 0.0) -> ChangRefsdalPartition:
        """Evaluate the four-channel partition at one parameter point.

        Continues labels from the previous `evaluate` call unless `reset`
        was called (or this is the first call), in which case the
        deterministic initial labeling is used.

        Parameters
        ----------
        gamma : float
            External shear magnitude.
        y : Sequence[float]
            Shape ``(2,)`` source position.
        beta : float, optional
            External shear orientation, radians.
        kappa : float, optional
            External convergence.

        Returns
        -------
        ChangRefsdalPartition
            The channel delays, kernels, exact total, and bookkeeping
            for this point.

        Raises
        ------
        geometry.LensDomainError
            Raised by name for the two macro-matrix refusals -- Type III
            ``1 - kappa <= 0`` and the ``det A = 0`` parity boundary
            ``abs(gamma) == 1 - kappa`` -- and by the downstream image
            census / fold-degenerate metric guards.  BOTH parities are
            served: positive parity ``1 - kappa > abs(gamma)`` and the
            macro saddle ``0 < 1 - kappa < abs(gamma)`` flow through the
            same parity-blind SACR-C construction, the saddle wave branch
            being routed to `f_schwinger` by the operator parity dispatch.
        operator.CancellationError
            If the operator-series contraction is uncertifiable (strong
            shear / high ``w``; FINDINGS F005).
        SchwingerCertificationError
            If the saddle Schwinger evaluator fails its paired-rule
            certification (above the ``w <= 60`` ceiling; FINDINGS F013).
        """
        source = np.asarray(y, dtype=float)
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        caustic = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)

        images = geometry.find_images(source, matrix)
        absolute_delays = np.array(
            [geometry.delay(image, source, matrix) for image in images],
            dtype=float)
        t_min = float(absolute_delays.min())
        relative_delays = absolute_delays - t_min

        assignment, markers = _assign_labels(
            self._markers, images, caustic.image)
        self._markers = markers.copy()

        virtual_delay = geometry.delay(caustic.image, source, matrix)
        critical_delay = virtual_delay - t_min
        delays, real_mask = _labeled_delays(
            assignment, relative_delays, critical_delay)

        physical = _physical_kernels(self._w, assignment, images, matrix)
        switch = _channel_switch(self._w, delays, real_mask,
                                 critical_delay)
        delta_min = _min_delay_separation(delays, real_mask)

        exact_total, orders, converged = _exact_total(
            self._w, source, gamma, beta, kappa, t_min, delta_min,
            self._max_order)

        # SACR-C decomposition: switched analytic saddle trials
        # ``S_a * H_a`` plus one envelope ``E`` demodulated at the
        # critical carrier ``tau_c = critical_delay``, projected exactly
        # onto the four physical labels with per-frequency weights
        # ``1 - S_a + eta`` (`_gauge.switched_analytic_channels`).  The
        # four-channel per-frequency-weight form keeps ``_N_CHANNELS = 4``
        # so the crossing-scenario / label-continuity tests are
        # unaffected; a fifth envelope channel is deliberately avoided.
        kernels, envelope = switched_analytic_channels(
            self._w, exact_total, delays, physical, switch,
            critical_delay, _envelope_weights(switch))

        return ChangRefsdalPartition(
            w=self._w,
            source=source,
            gamma=float(gamma),
            beta=float(beta),
            kappa=float(kappa),
            delays=delays,
            kernels=kernels,
            envelope=envelope,
            saddle_kernels=physical,
            switch=switch,
            critical_delay=float(critical_delay),
            matrix=matrix,
            images=np.asarray(images, dtype=float).reshape(-1, 2),
            assignment=assignment,
            exact_total=exact_total,
            real_mask=real_mask,
            markers=markers,
            t_min=t_min,
            critical_theta=float(caustic.theta),
            critical_image=np.asarray(caustic.image, dtype=float),
            critical_source=np.asarray(caustic.source, dtype=float),
            caustic_distance=float(caustic.distance),
            operator_orders=orders,
            operator_converged=converged)

    def geometry_partition(self, *, gamma: float, y: Sequence[float],
                           beta: float = 0.0,
                           kappa: float = 0.0
                           ) -> ChangRefsdalGeometryPartition:
        """Cheap channel geometry without the exact amplification total.

        Computes exactly the geometry `evaluate` builds BEFORE the
        expensive exact total -- the macro-matrix domain check, the
        nearest caustic, the per-image delays, the analytic saddle
        kernels ``H_a``, the criticality switch ``S_a``, the critical
        carrier delay ``tau_c``, and the real-image mask -- and returns
        them WITHOUT ever evaluating the operator/Schwinger total
        (`_exact_total`) or building the SACR-C envelope.  It exists so a
        surrogate envelope ``E(w)`` (Build 8a) can be reconstructed to
        channels through `reconstruct_from_envelope`, which consumes
        precisely ``delays``, ``saddle_kernels``, ``switch`` and
        ``critical_delay``.

        Labels are continued from the previous `evaluate` /
        `geometry_partition` call exactly as `evaluate` does (call `reset`
        for the deterministic initial labeling), so at a given point with
        the same continuation state the geometry is identical to
        `evaluate`'s -- this method reproduces those lines verbatim and
        merely stops short of the exact total.

        Parameters
        ----------
        gamma : float
            External shear magnitude.
        y : Sequence[float]
            Shape ``(2,)`` source position.
        beta : float, optional
            External shear orientation, radians.
        kappa : float, optional
            External convergence.

        Returns
        -------
        ChangRefsdalGeometryPartition
            The channel delays, saddle kernels, switch, critical delay,
            real-image mask, and caustic distance for this point.

        Raises
        ------
        geometry.LensDomainError
            Raised by name for the SAME macro-matrix refusals as
            `evaluate` -- Type III ``1 - kappa <= 0`` and the
            ``det A = 0`` parity boundary ``abs(gamma) == 1 - kappa`` --
            and the downstream image-census / fold-degenerate-metric
            guards.  The cheap decidable domain refusals stay live at this
            API boundary because `geometry.macro_matrix` is evaluated
            first, exactly as in `evaluate`; only the expensive exact
            total, whose ``operator.CancellationError`` /
            ``SchwingerCertificationError`` refusals `evaluate` can raise,
            is skipped here.
        """
        source = np.asarray(y, dtype=float)
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        caustic = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)

        images = geometry.find_images(source, matrix)
        absolute_delays = np.array(
            [geometry.delay(image, source, matrix) for image in images],
            dtype=float)
        t_min = float(absolute_delays.min())
        relative_delays = absolute_delays - t_min

        assignment, markers = _assign_labels(
            self._markers, images, caustic.image)
        self._markers = markers.copy()

        virtual_delay = geometry.delay(caustic.image, source, matrix)
        critical_delay = virtual_delay - t_min
        delays, real_mask = _labeled_delays(
            assignment, relative_delays, critical_delay)

        physical = _physical_kernels(self._w, assignment, images, matrix)
        switch = _channel_switch(self._w, delays, real_mask,
                                 critical_delay)

        return ChangRefsdalGeometryPartition(
            w=self._w,
            delays=delays,
            saddle_kernels=physical,
            switch=switch,
            critical_delay=float(critical_delay),
            real_mask=real_mask,
            caustic_distance=float(caustic.distance))

    def evaluate_path(self, path: Iterable[dict]
                      ) -> list[ChangRefsdalPartition]:
        """Evaluate a continuous path of parameter points, continued.

        Resets first, then evaluates each point continuing labels from
        the last, so the labeling is consistent along the whole path.

        Parameters
        ----------
        path : Iterable[dict]
            Keyword dicts for `evaluate` (``gamma``, ``y`` and optional
            ``beta``, ``kappa``).

        Returns
        -------
        list of ChangRefsdalPartition
            One partition per point, in path order.
        """
        self.reset()
        return [self.evaluate(**point) for point in path]
