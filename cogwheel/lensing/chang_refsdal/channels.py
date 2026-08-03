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
from typing import Callable, Iterable, Sequence

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._gauge import (
    channels_from_envelope, envelope_total, reconstructed_total,
    smootherstep, switched_analytic_channels)
from cogwheel.lensing.chang_refsdal.operator import (
    RHO_START, RHO_END, F_op_grid, cancellation_exponent,
    geometric_amplification, select_branch)

__all__ = ['ChangRefsdalChannels', 'ChangRefsdalGeometryPartition',
           'ChangRefsdalPartition', 'farfield_envelope_from_partition',
           'born_carrier_from_partition',
           'real_image_delays', 'reconstruct_from_envelope',
           'reconstruct_farfield', 'farfield_ghost_term', 'farfield_w_floor',
           'FARFIELD_KERNEL_SUM', 'FARFIELD_DIFFRACTIVE',
           'FARFIELD_KERNEL_SUM_MINUS_GHOST', 'KNOWN_FARFIELD_DEFINITIONS',
           'INTERIOR_SACR_C', 'KNOWN_INTERIOR_DEFINITIONS']

#: The fixed number of topology-stable labels.
_N_CHANNELS = 4

#: Far-field envelope-definition tags -- the ``w``-windowed exterior label
#: classes, each subtracting only the analytic terms valid on its window
#: (Build 8h-b3-fin S1-2).  ``FARFIELD_KERNEL_SUM`` (window iii, the legacy
#: Build 8g-b label) subtracts the real-image kernels and hands the band
#: above ``w_trust`` to the point-mass ppGO band split;
#: ``FARFIELD_DIFFRACTIVE`` (window i, the diffractive bottom below
#: ``w_floor``) subtracts NOTHING and fits the bounded smooth ``F`` object (``F
#: -> 1`` as ``w -> 0``); ``FARFIELD_KERNEL_SUM_MINUS_GHOST`` (window ii, the
#: mid band ``[w_floor, w_trust)``) additionally subtracts the decaying
#: complex-saddle ghost ``G`` so the stored remainder is smooth across the
#: fold.  The tags NAME the reconstruction algebra, so they live here (the
#: physics label home) and the surrogate imports them for its known-tag set and
#: per-tile stamping.
FARFIELD_KERNEL_SUM = 'farfield_full_kernel_sum'
FARFIELD_DIFFRACTIVE = 'farfield_diffractive_bare_total'
FARFIELD_KERNEL_SUM_MINUS_GHOST = 'farfield_kernel_sum_minus_ghost'

#: Every far-field envelope-definition tag with a reconstruction path.  A
#: chart whose tag is absent from this set is hard-refused at load BEFORE any
#: numerics (the atomicity invariant: no tag can be written without its
#: reconstruction branch existing here).
KNOWN_FARFIELD_DEFINITIONS = frozenset({
    FARFIELD_KERNEL_SUM, FARFIELD_DIFFRACTIVE,
    FARFIELD_KERNEL_SUM_MINUS_GHOST})

#: Interior (inside-the-caustic) envelope-definition tag.  Unlike the far-field
#: tags -- which SUBTRACT individually-divergent near-merged image kernels and
#: therefore fail generically inside the astroid/deltoid (the near-merged pair
#: has a small ``w * |tau_a - tau_c|`` so its subtracted kernel blows up) --
#: the interior tag stamps tiles that store the FULL SACR-C
#: ``tau_c``-demodulated envelope ``E`` (the same object
#: `reconstruct_from_envelope` consumes): the switch is ON, near-merged images
#: are SWITCHED INTO the bounded envelope instead of subtracted, and
#: demodulation is the unimodular multiply ``e^{-i w tau_c}`` with ``tau_c``
#: the finite/smooth critical (virtual) delay.  The label has NO ``1/(tau_a -
#: tau_c)`` and NO ``Im tau_c`` denominator, so it is bounded at every interior
#: point the charts admit -- including near-cusp directions, the regime SACR-C
#: was built for -- and needs no interior carve-out.  Reconstruction is the
#: EXISTING `reconstruct_from_envelope` (SACR-C caustic-region path with the
#: geometry's own switch and critical delay); no new reconstruction algebra is
#: introduced (Build S2-3, frozen WP8 amended to whole-interior, Professor R4
#: ruling).
INTERIOR_SACR_C = 'interior_sacr_c_envelope'

#: Every interior envelope-definition tag with a reconstruction path.  A chart
#: whose interior tag is absent here is hard-refused at load BEFORE any
#: numerics (same atomicity invariant as `mem:KNOWN_FARFIELD_DEFINITIONS`: no
#: tag can be stamped without its reconstruction branch -- here the SACR-C
#: `reconstruct_from_envelope` -- existing).
KNOWN_INTERIOR_DEFINITIONS = frozenset({INTERIOR_SACR_C})

#: The two far-field tags whose gauge forces the switch to ``1`` on every real
#: channel (``F`` minus the real kernels).  ``FARFIELD_DIFFRACTIVE`` is the odd
#: one out: it forces the switch to ``0`` everywhere (subtract nothing), so the
#: ``E_ff = 0`` ppGO telescoping identity does NOT hold for it and it is never
#: band-split.
_FARFIELD_KERNEL_FAMILY = frozenset({
    FARFIELD_KERNEL_SUM, FARFIELD_KERNEL_SUM_MINUS_GHOST})

#: Half the SACR-C resolution scale ``RHO_END`` (in radians of accumulated
#: phase), used by the ``w_floor`` window boundary and as the numerator in
#: the ghost decay threshold ``_GHOST_DECAY_IM_THRESHOLD``.  The window
#: boundary: ``w_floor`` is the smallest ``w`` resolving the closest
#: real-image pair (``w * min|tau_a - tau_b| >= RHO_END/2``).  A PHYSICS
#: constant derived from geometry, not a fit knob.
_FARFIELD_WINDOW_RADIANS = RHO_END / 2.0

#: Minimum lens-plane separation between the decaying complex-saddle ('ghost')
#: image and its nearest REAL image, in Einstein-radius units, below which the
#: mid-band label must NOT subtract the ghost.  The currency is the bare
#: (un-normalized) complex Euclidean distance ``min_a |x_a - x_c|``, where the
#: ghost position ``x_c`` is complex (its imaginary part carries the
#: coalescence information and MUST be kept).  This is a GEOMETRIC,
#: frequency-independent criterion: unlike the old decay gate
#: ``w_min * Im tau_c`` it reads only the configuration, so the training label
#: and the serve mirror reach a provably identical admit/refuse decision for a
#: fixed config (no ``w``-array skew).  The single-saddle stationary-phase
#: expansion the ghost kernel relies on is valid only where the ghost is
#: RESOLVED from every real image; near a cusp the ghost coalesces with a real
#: image and the expansion breaks even though the ghost has not decayed (decay
#: and separation disagree exactly there).  The driver measured a clean gap
#: (2026-07-27): configurations that must be refused reach separation down to
#: 0.24, while configurations that must be admitted stay at 1.33 and above.
#: The threshold sits inside that gap and is deliberately biased toward REFUSE:
#: a false admit injects a silent lnL bias into the served ``F`` (never caught,
#: since the ghost is baked into the label), whereas a false refuse only drops
#: to the exact engine, which is never wrong.  CAVEAT: if any refuse-required
#: (fact-2) config ever sweeps to separation > 0.7, or any admit-required
#: (fact-1) config to separation < 0.7, this threshold must move to the
#: geometric mean of the measured refuse-max and admit-min.
#:
#: Part 0 resolution (Build 7): this constant is a LENS-PLANE quantity in
#: Einstein-radius units.  The Einstein radius is the physical scale in the
#: image plane (unlike the source plane where the caustic is the only scale and
#: Part 0 applies).  The cusp-coalescence geometry that sets the single-saddle
#: expansion validity boundary is O(1) in Einstein units regardless of gamma —
#: measured gap (refuse max 0.29, admit min 0.94) stable across gamma 0.30–0.90
#: on both parities, verified by the live tripwire assertions SEP_REFUSE_MAX=0.5
#: and SEP_ADMIT_MIN=1.0 in test_lensing_ghost_gate.py.  NOT a prior-box-derived
#: constant; NOT subsumed by the decay gate (_GHOST_DECAY_IM_THRESHOLD, which
#: guards the orthogonal near-axis non-decaying failure mode).
_GHOST_SEPARATION_MIN = 0.7

#: Minimum imaginary-delay decay required for the ghost to be subtracted:
#: ``Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD``.  This is the frequency-
#: INDEPENDENT form of the retired condition ``w_min * Im tau_c >=
#: _FARFIELD_WINDOW_RADIANS``, pinning ``w_min`` to the chart band floor
#: (approximately ``5``, the typical lowest dimensionless frequency at which
#: exterior charts are defined).  The threshold
#: ``_FARFIELD_WINDOW_RADIANS / 5 = 2.0 / 5.0 = 0.4`` catches the near-axis
#: pathology (F027: ``Im tau_c -> 0`` means pure oscillation, no decay) while
#: admitting off-axis configs where the ghost has genuinely decayed (well
#: off-axis configs typically have ``Im tau_c > 0.4``, e.g. 0.69–0.87 at
#: typical admitted test configs; on-axis near-cusp configs have
#: ``Im tau_c ~ 0.001``).  Being a FIXED constant it is identical at train
#: and serve — no ``w``-array skew is possible.
_GHOST_DECAY_IM_THRESHOLD = _FARFIELD_WINDOW_RADIANS / 5.0

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
                 delta_min: float
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

    # Third leg of the authoritative gate: distance to the caustic (F031),
    # w-independent so measured once.  MUST be passed -- the default `inf`
    # silently satisfies the leg and desynchronises this path from the
    # operator grids.  Pinned by
    # `test_lensing_operator.py::BranchGateTestCase::
    # test_thresholds_have_one_home`.  A refusing caustic search gives
    # `eta = 0.0` -> 'wave', the conservative direction.
    eta = 0.0
    if not saddle_host:
        try:
            eta = float(geometry.nearest_caustic_point(
                gamma, beta, source, kappa=kappa).distance)
        except geometry.LensDomainError:
            eta = 0.0

    wave_indices = []
    for index in range(n_w):
        frequency = float(w[index])
        if saddle_host:
            wave_indices.append(index)
            continue
        exponent = cancellation_exponent(frequency, source, gamma, kappa)
        if select_branch(frequency, delta_min, exponent, eta) == 'geometric':
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
        # node raises `SchwingerCertificationError` from inside this call
        # and propagates unswallowed -- identical to the former per-node
        # `F_op` raise -- so the RB and brute paths refuse symmetrically.
        values_wave, orders_wave, converged_wave = F_op_grid(
            w_wave, source, gamma, beta=beta, kappa=kappa)
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
    # Frame origin ``t_min`` from THE authoritative helper (the fourth site
    # of the 8h-b7 min-subtraction convention), not an inline ``.min()``
    # rebuild.  ``_frame_delays`` returns ``t_min = min(absolute_delays)``
    # as a real float, so ``np.sort(absolute_delays - t_min)`` is
    # byte-identical to the previous expression while sharing the single
    # source of the frame convention.
    _images, absolute_delays, t_min = _frame_delays(source, matrix)
    return np.sort(absolute_delays - t_min)


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


def farfield_w_floor(delays: np.ndarray, real_mask: np.ndarray) -> float:
    """Diffractive/mid-band window boundary ``w_floor`` from geometry.

    The smallest dimensionless frequency at which the CLOSEST real image
    pair is resolved to half the SACR-C resolution scale,

    ::

        w_floor = (RHO_END / 2) / min_{a != b real} |tau_a - tau_b|,

    so that ``w_floor * min|tau_a - tau_b| == RHO_END / 2`` -- the same
    radian currency as the mid-band ghost gate.  Below ``w_floor`` no real
    pair separates and the exterior label is the bounded diffractive-bottom
    object (`FARFIELD_DIFFRACTIVE`, subtract nothing); at and above it the
    real kernels may be subtracted (`FARFIELD_KERNEL_SUM_MINUS_GHOST`).  A
    PHYSICS constant per region derived from geometry, not a fit knob.

    Parameters
    ----------
    delays : np.ndarray
        Per-channel delays ``tau_a`` (`ChangRefsdalPartition.delays`).
    real_mask : np.ndarray
        Boolean mask of real channels.

    Returns
    -------
    float
        The window boundary ``w_floor``; ``inf`` when fewer than two real
        images exist (no pair to resolve, so the whole exterior is the
        diffractive bottom).
    """
    separation = _min_delay_separation(delays, real_mask)
    if separation <= 0.0:
        return float('inf')
    return _FARFIELD_WINDOW_RADIANS / separation


def _farfield_switch(real_mask: np.ndarray, n_w: int,
                     definition: str) -> np.ndarray:
    """Per-tag far-field switch: ``1`` on real channels, or all ``0``.

    The far-field gauge parks the critical carrier at ``tau_c = 0`` and uses
    a constant switch (no criticality demodulation), so the switch reduces to
    a tag choice: the kernel-sum family (`_FARFIELD_KERNEL_FAMILY`) forces
    ``S_a = 1`` on every real channel (subtract the real kernels), while
    `FARFIELD_DIFFRACTIVE` forces ``S_a = 0`` everywhere (subtract nothing).
    Virtual channels always carry ``S_a = 0`` (their saddle kernel is a
    structural zero, so forcing it would be a no-op anyway).

    Parameters
    ----------
    real_mask : np.ndarray
        Shape ``(_N_CHANNELS,)`` boolean mask of real channels.
    n_w : int
        Number of frequencies.
    definition : str
        A far-field envelope-definition tag in `KNOWN_FARFIELD_DEFINITIONS`.

    Returns
    -------
    np.ndarray
        Shape ``(n_w, _N_CHANNELS)`` switch in ``{0, 1}``.

    Raises
    ------
    ValueError
        If ``definition`` is not a known far-field tag.
    """
    if definition not in KNOWN_FARFIELD_DEFINITIONS:
        raise ValueError(
            f'Unknown far-field envelope-definition tag {definition!r} '
            f'(known: {sorted(KNOWN_FARFIELD_DEFINITIONS)}).')
    switch = np.zeros((n_w, _N_CHANNELS), dtype=float)
    if definition in _FARFIELD_KERNEL_FAMILY:
        switch[:, np.asarray(real_mask, dtype=bool)] = 1.0
    return switch


def _frame_delays(source: np.ndarray, matrix: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """``(images, absolute_delays, t_min)`` -- the relative-delay frame.

    THE authoritative construction of the min-subtraction convention every
    channel kernel is carried in: `ChangRefsdalChannels.evaluate`,
    `ChangRefsdalChannels.geometry_partition` and `farfield_ghost_term` all
    obtain ``t_min`` from here, so no two sites can drift apart.  Returning
    the images and absolute delays alongside it keeps the partition builders
    from re-solving the image quartic just to recover them.

    Parameters
    ----------
    source : np.ndarray
        Shape ``(2,)`` source position.
    matrix : np.ndarray
        Shape ``(2, 2)`` symmetric macro matrix (`geometry.macro_matrix`).

    Returns
    -------
    images : np.ndarray
        The real images (`geometry.find_images`).
    absolute_delays : np.ndarray
        Their absolute Fermat delays, in image order.
    t_min : float
        ``min(absolute_delays)`` -- the frame origin subtracted from every
        delay.  Real, so subtracting it leaves ``Im tau_c`` invariant.
    """
    images = geometry.find_images(source, matrix)
    absolute_delays = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    return images, absolute_delays, float(absolute_delays.min())


def _frame_t_min(source: np.ndarray, matrix: np.ndarray) -> float:
    """Minimum real-image Fermat delay: the partition frame's origin.

    Thin accessor over `_frame_delays` for callers that hold only
    ``(source, matrix)`` and cannot be handed a precomputed ``t_min`` --
    it costs an image-quartic solve, so prefer passing `partition.t_min`
    or `geometry_partition.t_min` where one is already in hand.
    """
    return _frame_delays(source, matrix)[2]


def farfield_ghost_term(w: np.ndarray, source: np.ndarray,
                        matrix: np.ndarray, *,
                        t_min: float | None = None,
                        real_images: np.ndarray | None = None) -> np.ndarray:
    """Decaying complex-saddle ghost contribution ``G(w)`` to the total.

    The carrier-restored ghost term the mid-band label subtracts and the
    likelihood serve mirror re-adds, carried in the partition's
    min-subtracted (relative-delay) frame,

        G(w) = C_c(w) * exp(1j * w * (tau_c - t_min)),   Im tau_c >= 0,

    where the carrier-free kernel ``C_c`` and complex Fermat delay
    ``tau_c`` come from `geometry.ghost_kernel` (decaying-member selection
    and the sqrt branch pinned there), and ``t_min`` is the minimum
    real-image Fermat delay (`_frame_t_min`).  The real channel kernels the
    ghost is subtracted alongside -- via `switched_analytic_channels` in
    `farfield_envelope_from_partition` -- are carried at ``tau_a - t_min``,
    so the ghost MUST be shifted by the same ``-t_min`` to sit in the same
    frame; otherwise the subtracted/re-added ``G`` is off by
    ``exp(-1j * w * t_min)`` and the mid-band residual ``|R - G| / |F|``
    does not collapse.  Because ``t_min`` is real, the shift is a pure
    phase.  This is the SINGLE authoritative ghost primitive: the training
    label (`farfield_envelope_from_partition`) and the serve mirror both
    call it, so the subtracted and re-added ``G`` cannot diverge in member
    selection, carrier convention, or frame.

    The min-subtraction correction lives HERE, in the composition layer,
    NOT in `geometry.ghost_kernel`: ``tau_c`` is a holomorphic,
    gauge-independent property of the ghost saddle, whereas ``t_min`` is
    the partition's gauge choice -- a property of the SET of real images,
    unknown to (and none of the business of) the single-saddle primitive.
    `geometry.ghost_kernel` / `geometry._ghost_delay` therefore stay RAW
    and byte-identical, honouring their documented contract.

    The mid-band ghost gate is a GEOMETRIC SEPARATION criterion, enforced
    here on the lens-plane image positions: the single-saddle
    stationary-phase expansion the ghost kernel relies on is valid ONLY
    where the ghost image ``x_c`` is resolved from every real image, so the
    ghost is subtracted only where

    ::

        min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN,

    a bare complex Euclidean distance over the real images ``x_a``
    (`geometry.find_images`) and the complex ghost position ``x_c``.  The
    imaginary part of ``x_c`` is KEPT (it carries the near-cusp coalescence
    information; dropping it would corrupt the separation).  Unlike the
    former decay gate ``w_min * Im tau_c``, this criterion contains no
    ``w``: it is a pure property of the configuration, so the training
    label and the serve mirror -- which historically saw different ``w``
    sub-bands -- now reach a PROVABLY IDENTICAL admit/refuse decision for a
    fixed ``(source, matrix)``, removing the train/serve skew by
    construction.  Near a cusp the ghost coalesces with a real image
    (``min_a |x_a - x_c| -> 0``) while it may not have decayed at all;
    the separation gate refuses there where the old decay gate wrongly
    admitted.  The distinct self-merge pathology (the ghost at its OWN
    fold) is caught earlier and independently by ``_GHOST_DET_FLOOR``
    inside `geometry._ghost_kernel`.

    A complementary DECAY gate ensures the ghost has actually decayed:

    ::

        Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD,

    a FIXED constant (``_FARFIELD_WINDOW_RADIANS / 5.0 = 0.4``), derived
    by pinning the retired condition ``w_min * Im tau_c >=
    _FARFIELD_WINDOW_RADIANS`` to the chart band floor (``w ~ 5``, the
    typical lowest dimensionless frequency of exterior charts).  Being a
    pure constant, this gate is a property of the CONFIGURATION alone —
    it reads nothing from the ``w`` array, so the training label and the
    serve mirror reach a provably identical admit/refuse decision for a
    fixed ``(source, matrix)``, with no ``w``-array skew by construction.
    Near a principal axis ``Im(tau_c) -> 0`` and the ghost is pure
    oscillation, no longer a small exponentially-decaying correction
    (F027).

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency grid (1-D); builds the ghost kernel and
        carrier.  Does NOT enter the admit/refuse gate (the decay gate
        uses the fixed constant ``_GHOST_DECAY_IM_THRESHOLD``).
    source : np.ndarray
        Shape ``(2,)`` source position.
    matrix : np.ndarray
        Shape ``(2, 2)`` symmetric macro matrix (`geometry.macro_matrix`).

    Returns
    -------
    np.ndarray
        Shape ``(n_w,)`` complex ghost contribution ``G(w)`` in the
        min-subtracted frame.

    Raises
    ------
    geometry.GhostDomainError
        If no complex-saddle pair exists, the continuation is degenerate
        (both from `geometry.ghost_kernel`), the decay gate
        ``Im(tau_c) >= _GHOST_DECAY_IM_THRESHOLD`` fails (the ghost has
        not decayed enough to be a small correction — F027), or the
        separation gate
        ``min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN`` fails (the ghost is
        inseparable from a real image near a cusp, so the single-saddle
        expansion is invalid and it must not be subtracted).  Subclasses
        `geometry.LensDomainError`, so it refuses symmetrically with the
        exact path.
    """
    w = np.asarray(w, dtype=float)
    source = np.asarray(source, dtype=float)
    contribution = geometry.ghost_kernel(w, source, matrix)
    x_c = contribution.position
    # Geometric separation gate: refuse where the ghost is inseparable from
    # a real image near a cusp.  The norm is complex (``|.|`` over the
    # complex ``x_a - x_c``): dropping Im(x_c) would corrupt the near-cusp
    # separation.
    #
    # The images are PASSED where the caller already has them -- both the
    # training label and the likelihood serve mirror hold a partition that
    # carries them -- because re-solving the image quartic here costs ~234 us,
    # 51% of this function, on a per-likelihood-evaluation path.  Same
    # treatment as ``t_min``: carry the value, do not re-derive it.
    if real_images is None:
        real_images = geometry.find_images(source, matrix)
    if len(real_images) == 0:
        raise geometry.GhostDomainError(
            f'No real image for source ({source[0]!r}, {source[1]!r}) to '
            f'separate the ghost from: the single-saddle expansion cannot '
            f'be validated, so the ghost must not be subtracted.')
    # Decay gate (F027): the ghost must have decayed at least
    # _GHOST_DECAY_IM_THRESHOLD of imaginary delay.  This is the frequency-
    # INDEPENDENT form: a pure property of the configuration, identical at
    # train and serve, with no w-array skew.  Near a principal axis
    # Im(tau_c)->0 and the ghost is pure oscillation, no longer a small
    # correction.
    im_tau_c = float(contribution.delay.imag)
    if not im_tau_c >= _GHOST_DECAY_IM_THRESHOLD:
        raise geometry.GhostDomainError(
            f'Ghost decay Im(tau_c) = {im_tau_c!r} is below the '
            f'threshold {_GHOST_DECAY_IM_THRESHOLD!r}: the ghost has '
            f'not decayed enough to be a small correction, so it must '
            f'not be subtracted (F027).')
    separation = min(
        float(np.sqrt(np.sum(np.abs(x_a - x_c) ** 2))) for x_a in real_images)
    if not separation >= _GHOST_SEPARATION_MIN:
        raise geometry.GhostDomainError(
            f'Ghost separation min_a |x_a - x_c| = {separation!r} is below '
            f'the threshold {_GHOST_SEPARATION_MIN!r}: single-saddle '
            f'expansion invalid: ghost inseparable from a real image near a '
            f'cusp, so it must not be subtracted.')
    # Carry the ghost in the partition's min-subtracted frame: the real
    # kernels it is subtracted alongside use tau_a - t_min, so the ghost
    # carries tau_c - t_min.  t_min is real, so it is a pure phase and does
    # not affect the (frequency-independent) separation gate above.
    if t_min is None:
        t_min = _frame_t_min(source, matrix)
    return contribution.kernel * np.exp(1j * w * (contribution.delay - t_min))


def _frame_phase(w: np.ndarray, t_min: float) -> np.ndarray:
    """Frame-demodulation phase ``w * t_min`` reduced modulo ``2*pi``.

    The far-field label is demodulated by ``exp(+1j w t_min)`` and
    `reconstruct_farfield` re-modulates by ``exp(-1j w t_min)`` (Build 8h-d2,
    frame invariance).  This is the SINGLE authoritative source of that phase,
    called by BOTH sites so the convention has exactly one expression (Build
    8h-d2, INS-4-003).  The reduction modulo ``2*pi`` is load-bearing: ``exp``
    is ``2*pi``-periodic, so it leaves the demodulation mathematically
    unchanged, but it bounds the argument to ``[0, 2*pi)`` so the two
    exponentials round against a small argument instead of one growing like
    ``|w t_min|``.  That keeps the demod/re-mod pair exact to machine precision
    wherever the reconstruction is WELL CONDITIONED.

    Precision caveat near a fold (Build 8h-d2, INS-4-003).  The reduction does
    NOT restore machine precision when the far-field envelope is huge relative
    to the reconstructed total -- i.e. next to a fold, where the analytic
    kernel sum nearly cancels the envelope.  There the re-modulation multiply
    injects a rounding of order ``eps * |E_tilde|`` into the envelope that the
    independently-formed kernel sum cannot cancel, so the residual in ``F``
    scales as ``eps * |E_tilde| / max|F|`` (the catastrophic-cancellation
    amplification), independent of ``|w t_min|``.  For the worst measured
    cusp-adjacent fixture (``|E_tilde| ~ 2.6e5``, ``max|F| ~ 2.8``) that floor
    is ``~1.7e-11`` -- above a ``1e-11`` telescoping bound.  Removing it
    requires reconstructing in the min-relative frame from the small envelope
    (avoiding the round-trip multiply on the large one), a data-flow change out
    of scope for this phase-reduction helper.

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency grid.
    t_min : float
        Minimum real-image Fermat delay of the config (the frame origin).

    Returns
    -------
    np.ndarray
        ``(w * t_min) mod 2*pi``, broadcast to the shape of ``w``.
    """
    return np.mod(np.asarray(w, dtype=float) * float(t_min), 2.0 * np.pi)


def reconstruct_farfield(w: np.ndarray,
                         envelope: np.ndarray,
                         delays: np.ndarray,
                         saddle_kernels: np.ndarray,
                         real_mask: np.ndarray,
                         definition: str,
                         t_min: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Serve-mirror reconstruction, dispatching the switch on the window tag.

    The far-field twin of `reconstruct_from_envelope`: rebuild the channel
    kernels and total from an interpolated far-field envelope, choosing the
    switch policy from the window-class tag (`_farfield_switch`) with the
    critical carrier parked at ``tau_c = 0`` -- exactly mirroring
    `farfield_envelope_from_partition`.  The caller MUST supply the envelope
    already carrying the window's analytic content in the SAME frame-invariant
    (demodulated) convention the stored label uses: for
    `FARFIELD_KERNEL_SUM_MINUS_GHOST` the analytic ghost term
    (`farfield_ghost_term`) must have been re-added over the chart region
    BEFORE this call and carried with the same ``+t_min`` tilt as the stored
    label (the ghost lives in the mid-band window only, never in the bare
    ppGO band above ``w_trust``), so that the internal ``exp(-1j w t_min)``
    de-tilt returns label and ghost together to the min-relative frame.

    Frame convention (Build 8h-d2).  `farfield_envelope_from_partition`
    demodulates the label by ``exp(+1j w t_min)`` to make it frame-invariant;
    this function re-modulates by ``exp(-1j w t_min)`` FIRST, returning to the
    min-relative delay frame the SACR-C reconstruction below works in.  Both
    directions take the phase from `_frame_phase` (``w t_min`` reduced modulo
    ``2*pi``), so the round trip reproduces the pre-demodulation field to
    machine precision wherever the reconstruction is well conditioned.  It does
    NOT reach machine precision next to a fold: there ``|E_tilde|`` dwarfs
    ``max|F|`` and the re-modulation multiply injects an ``eps * |E_tilde|``
    rounding the kernel sum cannot cancel, leaving a residual ``~ eps *
    |E_tilde| / max|F|`` that the phase reduction cannot remove (see
    `_frame_phase`; Build 8h-d2, INS-4-003).  ``t_min`` is a REQUIRED
    positional (no default) so a stale caller that has not migrated hard-fails
    rather than silently reconstructing in the wrong frame.

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency, 1-D grid.
    envelope : np.ndarray
        The interpolated (and, for the mid band, ghost-restored)
        frame-invariant far-field envelope, with the shape of ``w``.
    delays : np.ndarray
        Shape ``(_N_CHANNELS,)`` channel delays ``tau_a``.
    saddle_kernels : np.ndarray
        Analytic saddle kernels ``H_a``, shape ``(n_w, _N_CHANNELS)``.
    real_mask : np.ndarray
        Shape ``(_N_CHANNELS,)`` boolean mask of real channels.
    definition : str
        A far-field envelope-definition tag in `KNOWN_FARFIELD_DEFINITIONS`.
    t_min : float
        Minimum real-image Fermat delay of the served config
        (`ChangRefsdalPartition.t_min` / geometry ``t_min``), the frame
        origin the demodulation is inverted against.

    Returns
    -------
    kernels : np.ndarray
        Channel kernels ``K_a``.
    total : np.ndarray
        The reconstructed amplification total ``F``.
    """
    w = np.asarray(w, dtype=float)
    # Invert the frame-invariance demodulation applied by
    # `farfield_envelope_from_partition`: the stored label is
    # ``E_tilde = E_ff_minrel * exp(+1j w t_min)``, so re-modulate by
    # ``exp(-1j w t_min)`` to return to the min-relative delay frame.  The
    # phase is `_frame_phase` (``w t_min`` reduced modulo ``2*pi``) -- the SAME
    # authoritative expression the producer used -- so the demod/re-mod pair
    # cancels to machine precision regardless of ``|w t_min|`` (Build 8h-d2,
    # INS-4-003).  For `FARFIELD_KERNEL_SUM_MINUS_GHOST` the caller has already
    # added the ghost in the demodulated frame, so this single multiply
    # de-tilts label and ghost together.
    envelope = np.asarray(envelope) * np.exp(-1j * _frame_phase(w, t_min))
    switch = _farfield_switch(real_mask, w.shape[0], definition)
    return reconstruct_from_envelope(
        w, envelope, delays, saddle_kernels, switch, 0.0)


def farfield_envelope_from_partition(
        partition: 'ChangRefsdalPartition',
        definition: str = FARFIELD_KERNEL_SUM) -> np.ndarray:
    """Far-field training label for one w-windowed window class.

    The exterior (far-field) surrogate label, DISTINCT from the
    caustic-region transition envelope `ChangRefsdalPartition.envelope`.
    In the exterior every image is well resolved, so the criticality
    switch ``S_a`` and the ``tau_c`` demodulation carrier that the
    caustic-region envelope needs do no useful work; worse, on the
    astroid diagonals (the lobe-equidistance lines) the
    ``nearest_caustic_point`` carrier flips lobes, a well-resolved image
    spuriously reads as near-critical, its switch turns off, and its full
    oscillation is left UN-subtracted -- the stored envelope jumps by
    ~1500x mid-tile and no spline can fit it (Build 8g-b measurement).

    This removes that dependence entirely for far-field charts by reusing
    the existing SACR-C projection (`switched_analytic_channels`, no new
    gauge math) with the switch chosen per window class by
    `_farfield_switch` and the carrier parked at ``tau_c = 0`` (a constant
    carrier -- no caustic point consulted).  The three window classes, each
    subtracting only the analytic terms valid on its w window, are:

    * `FARFIELD_DIFFRACTIVE` -- the diffractive bottom ``[w_small, w_floor)``
      where no real pair separates.  Switch ``0`` everywhere: subtract
      nothing, so the label is the bounded smooth ``F`` object (``F -> 1``
      limit; no kernel divergence).
    * `FARFIELD_KERNEL_SUM` -- the mid band ``[w_floor, w_trust)`` with the
      real kernels subtracted,

          E_ff(w) = F(w) - sum_{a real} H_a(w) * exp(1j*w*tau_a),

      the full post-geometric-optics remainder, measured smooth and
      ``~1e-4`` in magnitude on the good side of every flip line.
    * `FARFIELD_KERNEL_SUM_MINUS_GHOST` -- the mid band with the decaying
      complex-saddle ghost ``G`` additionally subtracted
      (`farfield_ghost_term`, gated on the GEOMETRIC separation ``min_a |x_a -
      x_c| >= _GHOST_SEPARATION_MIN``: the ghost is subtracted only where it is
      resolved from every real image, so the single-saddle expansion is valid.
      The gate is frequency-independent, so the training label here and the
      serve mirror decide admit/refuse identically for a fixed config; near a
      cusp the ghost coalesces with a real image and the gate refuses).

    The window boundary ``w_floor`` is `farfield_w_floor`; ``w_trust`` (the
    upper mid-band edge, above which the bare ppGO band-split serves) is a
    tiling concern, not this label's.  Mixed-tag charts are legal: each
    tile carries its own envelope-definition tag and the serve mirror
    (`reconstruct_farfield`, `farfield_ghost_term`) reconstructs each via
    its own path.

    The range-reduced carriers keep the ``F - sum_a H_a`` subtraction at
    machine precision.  This is the SINGLE authoritative source of the
    far-field label, so the trained object, the held-out eps gate
    reference, and the census reference can never diverge for a given tag.

    Parameters
    ----------
    partition : ChangRefsdalPartition
        A fully evaluated partition (its ``exact_total`` is required);
        the geometry-only `ChangRefsdalGeometryPartition` has no exact
        total and cannot be used here.  For `FARFIELD_KERNEL_SUM_MINUS_GHOST`
        the partition's ``source`` and ``matrix`` feed the ghost extractor.
    definition : str
        The window-class envelope-definition tag; one of
        `KNOWN_FARFIELD_DEFINITIONS`.  Defaults to `FARFIELD_KERNEL_SUM`,
        which reproduces the historical mid-band kernel-sum label
        byte-identically.

    Returns
    -------
    np.ndarray
        Shape ``(n_w,)`` complex far-field envelope for the window class,
        DEMODULATED by ``exp(+1j w t_min)`` (Build 8h-d2) so the stored label
        is frame-INVARIANT.  The demodulation phase is `_frame_phase` (``w
        t_min`` reduced modulo ``2*pi``); `reconstruct_farfield` inverts it
        with the SAME range-reduced phase and its required ``t_min`` argument
        before rebuilding, so the round trip reproduces the pre-demodulation
        field to machine precision (the reduction is what keeps the two
        large-argument exponentials cancelling; see `_frame_phase`).

    Raises
    ------
    ValueError
        If ``definition`` is not a known far-field tag.
    geometry.GhostDomainError
        For `FARFIELD_KERNEL_SUM_MINUS_GHOST` when the ghost gate refuses
        (source too close to a principal axis, inside the caustic, or the
        continuation is degenerate).  Subclasses `geometry.LensDomainError`,
        so it refuses symmetrically with the exact path.
    """
    switch = _farfield_switch(
        partition.real_mask, partition.w.shape[0], definition)
    _, envelope = switched_analytic_channels(
        partition.w, partition.exact_total, partition.delays,
        partition.saddle_kernels, switch, 0.0, _envelope_weights(switch))
    if definition == FARFIELD_KERNEL_SUM_MINUS_GHOST:
        envelope = envelope - farfield_ghost_term(
            partition.w, partition.source, partition.matrix,
            t_min=partition.t_min, real_images=partition.images)
    # Demodulate the residual ``t_min`` carrier so the STORED far-field label
    # is frame-INVARIANT (Build 8h-d2).  ``switched_analytic_channels`` and
    # ``farfield_ghost_term`` above both carry the envelope in the partition's
    # min-relative delay frame (``tau_a - t_min``); that frame origin
    # ``t_min`` varies from node to node, so the un-demodulated label winds by
    # ``w * t_min`` across a tile -- the dominant spatial phase of the fitted
    # object -- and a spline mixes incompatible frames (Build 8h-d2 defect 3).
    # Multiplying by ``exp(+1j w t_min)`` removes that frame dependence,
    # mirroring the interior SACR-C ``tau_c`` demodulation and the Born
    # ``exp(-1j w t_min)`` convention.  `reconstruct_farfield` re-modulates by
    # ``exp(-1j w t_min)`` before rebuilding.  Both sites take the phase from
    # `_frame_phase` (reduced modulo ``2*pi``) so the demod/re-mod pair cancels
    # to machine precision even for large ``w t_min`` near a fold (Build 8h-d2,
    # INS-4-003) -- node-exact telescoping.
    return envelope * np.exp(1j * _frame_phase(partition.w, partition.t_min))


def born_carrier_from_partition(
        partition: 'ChangRefsdalPartition', *,
        split_constant: float = RHO_END,
        lead_carrier: Callable[..., complex] | None = None) -> np.ndarray:
    """Band-split analytic carrier ``F_carrier`` for the far annulus.

    The training/serve carrier of the Born (weak-deflection) annulus rung
    (``3.0 < |y| <= 4.2426`` at positive parity, ``gamma < 3/4``): the
    analytic object a driver-trained chart interpolates the RESIDUAL
    ``F_exact - F_carrier`` against, the same carrier + interpolated-remainder
    decomposition as SACR-C and the far-field label.  The carrier does NOT
    have to hit an accuracy target on its own; the criterion is how cheaply
    the residual splines (F025).  It is returned in the SAME min-relative
    delay frame as `ChangRefsdalPartition.exact_total`, so the driver forms
    the residual by a direct subtraction.

    The carrier is split per-frequency at ``w * Delta_tau = split_constant``
    (``RHO_END = 4`` by default, the SAME resolution scale SACR-C switches on
    and `_born.born_gate` guard A keys on), where ``Delta_tau`` is the FULL
    Fermat-delay difference of the two real images read straight from
    ``partition.delays`` on its real channels (already min-subtracted, so the
    difference is frame-invariant).  ``Delta_tau`` is NOT reconstructed from
    ``phi_geo``, NOT ``w * r0_sq`` (which coincides with ``Delta_tau`` only on
    the positive-parity branch, F024), and the images are NOT re-solved -- the
    partition already carries the delays.  Keying the split on the invariant
    ``w * Delta_tau`` means the out-of-scope macro-saddle branch (``gamma > 1``,
    where ``r0_sq / (2 Delta_tau)`` spans 0.16 to 35.6, F024) adds a branch
    (a different ``lead_carrier``), not a re-key.

    * **Below the split** (``w * Delta_tau < split_constant``): the LEAD-ONLY
      carrier ``sqrt(mu_macro) * exp(1j*w*phi_geo)`` from ``lead_carrier``
      (`_born.born_lead_carrier` by default), demodulated to the min-relative
      frame by ``exp(-1j*w*t_min)`` with ``t_min`` read ONLY from
      ``partition.t_min`` (never recomputed; Build 8h-b7).  NO ``a0``/``b1``
      resolved-image correction, NO second image, NO ppGO, NO ghost: the ppGO
      ``1/w**2`` kernel would inflate the residual by five orders of magnitude
      below ``w ~ 0.05`` (F023), and ``a0`` breaks the exact limit
      ``F(w->0) = sqrt(mu_macro)`` (F009).
    * **Above the split** (``w * Delta_tau >= split_constant``): the resolved
      two-real-image geometric-optics sum plus the decaying complex-saddle
      ghost where admitted, demodulated the same way.  This is
      `geometric_amplification` over both real images at the full C1/C2 image
      kernels (``partition.saddle_kernels``) demodulated to the min-relative
      frame, PLUS `farfield_ghost_term`, reconstructed through the SACR-C
      switched-analytic projection (`reconstruct_farfield` ->
      `switched_analytic_channels`) in the resolved (``S_a = 1``) limit -- the
      SAME `FARFIELD_KERNEL_SUM_MINUS_GHOST` serve mirror the far-field label
      uses.  Reusing that one authoritative reconstruction (rather than a
      second, independent geometric-optics sum that could drift) keeps every
      real image's own carrier ``exp(1j*w*(tau_a - t_min))`` -- crucially the
      second image's ``exp(1j*w*Delta_tau)`` -- so the above-split residual is
      NOT demodulated by a single macro carrier, which would inflate the
      azimuthal node count (F024).

    Ghost absence is tolerated: `farfield_ghost_term`'s admission gate is the
    frequency-independent geometric separation
    ``min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN``, and its kernel can raise
    `geometry.GhostDomainError` / `geometry.LensDomainError` (the F023 witness
    ``|y|=3.6, theta=0.5, gamma=0.25, kappa=0.3, beta=0.5``, where
    `geometry.find_images` returns two real images but the ghost continuation
    refuses).  The ghost is an ADDITIVE correction, never a precondition: when
    it raises or is not admitted the carrier continues with the bare ppGO sum
    alone, so the returned array is always finite over the whole grid.

    Parameters
    ----------
    partition : ChangRefsdalPartition
        A partition of the served config; its ``w``, ``source``, ``gamma``,
        ``beta``, ``kappa``, ``matrix``, ``delays``, ``saddle_kernels``,
        ``real_mask``, ``images`` and ``t_min`` are read.  ``exact_total`` is
        NOT read -- the carrier is analytic -- so the cheap geometry-only build
        is sufficient in principle, but a `ChangRefsdalPartition` is taken to
        mirror `farfield_envelope_from_partition` and carry the source/matrix
        the ghost and lead carrier need.
    split_constant : float, optional
        The ``w * Delta_tau`` band-split threshold; defaults to `RHO_END`.
        Parameterised so the macro-saddle branch can re-scale the split
        without a rewrite.
    lead_carrier : callable, optional
        The below-split lead carrier, a scalar callable
        ``(w, y1, y2, gamma, beta, kappa) -> complex`` returning the carrier
        in the ABSOLUTE Fermat-delay frame (this function demodulates it).
        Defaults to `_born.born_lead_carrier`.  Parameterised so the
        macro-saddle branch (whose carrier is also lead-only) can supply its
        own lead term without a rewrite.

    Returns
    -------
    np.ndarray
        Shape ``(n_w,)`` complex analytic carrier ``F_carrier(w)`` on
        ``partition.w``, in the min-relative delay frame of
        ``partition.exact_total``.

    Raises
    ------
    ValueError
        If the partition carries fewer than two real images, so the
        two-real-image band split ``Delta_tau`` is undefined (the far annulus
        at ``gamma < 3/4`` is exterior to the caustic and must yield two).
    geometry.LensDomainError
        Propagated from `geometric_amplification` / `reconstruct_farfield` if
        the served census is inconsistent -- refusing symmetrically with the
        exact path.  A ghost-only `geometry.GhostDomainError` does NOT
        propagate: it is caught and the ppGO sum serves alone.
    """
    # Deferred import: ``_born`` imports ``channels`` at module load, so a
    # top-level ``import _born`` here would close the cycle.  Importing inside
    # the call keeps ``_born`` a pure scalar leaf module (the composition lives
    # here) and resolves the default carrier lazily.
    if lead_carrier is None:
        from cogwheel.lensing.chang_refsdal import _born
        lead_carrier = _born.born_lead_carrier

    w = np.asarray(partition.w, dtype=float)
    source = np.asarray(partition.source, dtype=float)
    y1, y2 = float(source[0]), float(source[1])
    gamma, beta, kappa = (
        float(partition.gamma), float(partition.beta), float(partition.kappa))
    t_min = float(partition.t_min)

    # Macro parity: det A = (1 - kappa)**2 - gamma**2.  On the macro saddle
    # (det A < 0) both real images are already Morse-index 1, so the above-split
    # branch serves the pure two-real-image ppGO sum and REFUSES the complex
    # ghost (see the above-split block).  The below-split lead-only carrier is
    # parity-agnostic: the default ``_born.born_lead_carrier`` applies the Morse
    # -1j for the saddle internally, so one code path serves both parities.
    is_saddle = (1.0 - kappa) ** 2 - gamma ** 2 < 0.0

    # Delta_tau: the FULL Fermat-delay difference of the two real images, read
    # straight from the partition's real-channel delays (F024/Professor: NOT
    # ``phi_geo``, NOT ``w * r0_sq``, NOT a re-solve).  ``partition.delays`` is
    # min-subtracted, so the max-minus-min difference is the frame-invariant
    # full-delay span.
    real_delays = partition.delays[np.asarray(partition.real_mask, dtype=bool)]
    if real_delays.size < 2:
        raise ValueError(
            f'born_carrier_from_partition needs two real images to form the '
            f'band-split Delta_tau, but the partition has {real_delays.size} '
            f'real channel(s).  The far annulus at gamma < 3/4 is exterior to '
            f'the caustic and should yield two real images.')
    delta_tau = float(real_delays.max() - real_delays.min())

    below = w * delta_tau < float(split_constant)
    carrier = np.empty(w.shape, dtype=complex)

    # Below the split: lead-only carrier, demodulated to the min-relative
    # frame.  ``lead_carrier`` is a scalar entry point, evaluated only on the
    # sub-band it serves (the above-split values would be discarded).
    if below.any():
        w_below = w[below]
        lead_absolute = np.array(
            [lead_carrier(float(w_i), y1, y2, gamma, beta, kappa)
             for w_i in w_below], dtype=complex)
        carrier[below] = lead_absolute * np.exp(-1j * w_below * t_min)

    # Above the split: the resolved two-image ppGO sum + ghost, reconstructed
    # through the switched-analytic projection in the ``S_a = 1`` far-field
    # gauge.  The ghost enters in the frame-invariant convention the stored
    # label uses (``exp(+1j w t_min)`` tilt); ``reconstruct_farfield``
    # re-modulates by ``exp(-1j w t_min)`` with the SAME reduced ``_frame_phase``
    # so label and ghost telescope back to the min-relative frame together.
    if not below.all():
        above = ~below
        if is_saddle:
            # Macro-saddle branch (det A < 0): pure two-real-image ppGO, the
            # complex ghost REFUSED.  NOT because the ghost is wrong here --
            # F027 measured the opposite: ``geometry._branch_pinned_amplitude``
            # pins its sqrt branch against ``exp(-0.5j*pi)`` with a
            # positive-parity PHRASING, but the reference is parity-independent
            # (``tr Hess = 2*lam > 0`` forbids index-2 images on both branches,
            # so every A2 fold annihilates an (index 0, index 1) pair), and
            # ``+G`` beats bare ppGO at every ``w <= 1.41`` on the saddle.
            #
            # The refusal is owed to a MISSING FENCE, not a wrong sign: the
            # ghost's decay gate was retired in Build 8h-d1 and replaced by a
            # geometric separation gate only.  Near a principal axis
            # ``Im tau_c -> 0``, so the ghost stops decaying with ``w`` while
            # the ppGO residual keeps shrinking; measured at ``w = 5``,
            # ``|y| = 4.2426``, ``gamma = 1.60``, ``|G|/|F|`` climbs 5.2e-6 ->
            # 1.0e-1 as theta falls 0.90 -> 0.02, degrading the residual up to
            # 64x versus ppGO alone (F027).  Nothing currently refuses that
            # regime.  Until a band-floor-pinned decay gate exists the
            # conservative arm is to omit an ADDITIVE correction, which costs
            # accuracy where the ghost would have helped but can never inject a
            # silent bias.  A ZERO envelope with `FARFIELD_KERNEL_SUM`
            # (``S_a = 1`` on the real channels) reconstructs exactly the pure
            # ppGO coherent sum over the two real images, demodulated to the
            # min-relative frame (Professor).
            _kernels, ppgo = reconstruct_farfield(
                w, np.zeros(w.shape, dtype=complex), partition.delays,
                partition.saddle_kernels, partition.real_mask,
                FARFIELD_KERNEL_SUM, t_min)
            carrier[above] = ppgo[above]
        else:
            try:
                ghost = farfield_ghost_term(
                    w, source, partition.matrix,
                    t_min=t_min, real_images=partition.images)
                envelope = ghost * np.exp(1j * _frame_phase(w, t_min))
            except geometry.LensDomainError:
                # Ghost unadmitted (separation gate) or its continuation
                # refused: the ghost is an ADDITIVE correction, so serve the
                # bare ppGO sum.
                envelope = np.zeros(w.shape, dtype=complex)
            _kernels, ppgo_plus_ghost = reconstruct_farfield(
                w, envelope, partition.delays, partition.saddle_kernels,
                partition.real_mask, FARFIELD_KERNEL_SUM_MINUS_GHOST, t_min)
            carrier[above] = ppgo_plus_ghost[above]

            # --- Fold correction (additive) ---
            # Replace the merging fold pair's ppGO contribution with the
            # uniform Airy form.  The correction is
            #   (F_airy - F_ppgo_pair) * exp(-1j * w * t_min)
            # where both F_airy and F_ppgo_pair are in the ABSOLUTE delay
            # frame, and the exp(-1j*w*t_min) demodulates to the min-relative
            # frame that carrier[] lives in.  On any structural refusal the
            # correction is zero (carrier stays as-is, byte-identical to the
            # uncorrected path).
            #
            # NOTE (maintenance): this block duplicates the fold-correction
            # logic of `_airy_fold.fold_ppgo_correction`.  The duplication is
            # intentional: `fold_ppgo_correction` re-solves the geometry from
            # scratch (needed for its standalone public interface), whereas this
            # block reuses pre-computed ``partition.images`` / ``partition.matrix``
            # to avoid a redundant `geometric_amplification` call.  If the
            # correction formula or its structural gates change, BOTH locations
            # must be updated together.  See INS-c8-003.
            from cogwheel.lensing.chang_refsdal._airy_fold import (
                _merging_fold_pair, _image_at_delay,
                _fold_amplitudes, _soft_axis_cubic, airy_fold_value)

            try:
                matrix = partition.matrix
                images = partition.images
                pair = _merging_fold_pair(images, source, matrix)
                if pair is not None:
                    tau_plus, tau_minus = pair
                    fold_delta_tau = tau_minus - tau_plus
                    if fold_delta_tau > 0.0:
                        tau_bar = 0.5 * (tau_plus + tau_minus)

                        nearest = geometry.nearest_caustic_point(
                            gamma, beta, source, kappa=kappa)
                        b3 = _soft_axis_cubic(nearest.image, nearest.soft_axis)
                        if b3 is not None:
                            amplitudes = _fold_amplitudes(
                                nearest.hard_eigenvalue, b3)
                            if amplitudes is not None:
                                p_amplitude, q_amplitude, sigma = amplitudes

                                # Airy values (absolute frame) for above-split
                                # frequencies only.
                                w_above = w[above]
                                airy_vals = np.empty(
                                    w_above.shape, dtype=complex)
                                for idx, w_i in enumerate(w_above):
                                    xi_i = (3.0 * w_i * fold_delta_tau
                                            / 4.0) ** (2.0 / 3.0)
                                    airy_vals[idx] = airy_fold_value(
                                        w_i, tau_bar, xi_i,
                                        p_amplitude, q_amplitude, sigma)

                                # Pair ppGO in absolute frame.
                                image_plus = _image_at_delay(
                                    images, source, matrix, tau_plus)
                                image_minus = _image_at_delay(
                                    images, source, matrix, tau_minus)
                                if (image_plus is not None
                                        and image_minus is not None):
                                    pair_ppgo = np.zeros(
                                        w_above.shape, dtype=complex)
                                    for img, tau_a in (
                                            (image_plus, tau_plus),
                                            (image_minus, tau_minus)):
                                        pair_ppgo += (
                                            np.exp(1j * w_above * tau_a)
                                            * geometry.image_kernel(
                                                w_above, img, matrix))

                                    # Correction demodulated to min-relative.
                                    correction = (
                                        (airy_vals - pair_ppgo)
                                        * np.exp(-1j * w_above * t_min))

                                    # Guard non-finite Airy: keep carrier
                                    # untouched where airy produced non-finite.
                                    finite_mask = np.isfinite(airy_vals)
                                    if finite_mask.any():
                                        carrier[above] = np.where(
                                            finite_mask,
                                            carrier[above] + correction,
                                            carrier[above])
            except (geometry.LensDomainError, ValueError,
                    ZeroDivisionError):
                # Structural refusal: no correction, carrier unchanged.
                pass

    return carrier


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
    caustic_theta : float
        LENS-plane critical-curve parameter: the angle in ``[0, 2*pi)``
        that parametrizes the critical curve at the closest caustic point
        (`geometry.NearestCausticPoint.theta`).  This is NOT the
        source-plane caustic angle ``theta_c``.  A GAUGE coordinate
        carried for the surrogate's cusp-window exclusion test (Build 8c);
        no engine logic consumes it.
    t_min : float
        Minimum absolute Fermat delay subtracted to form ``delays`` (the
        relative-delay frame origin, `_frame_delays`).  Carried -- as
        `ChangRefsdalPartition` already carries it -- so a consumer that
        must express another term in this frame, such as the ghost
        carrier in `farfield_ghost_term`, can be handed the value instead
        of re-solving the image quartic to recover it.
    images : np.ndarray
        The real image positions (`geometry.find_images`) this partition was
        built from, carried for the same reason as ``t_min``: the ghost
        separation gate needs them, and re-solving the quartic on the serve
        path costs ~234 us per call (51% of `farfield_ghost_term`).
    """

    w: np.ndarray
    delays: np.ndarray
    saddle_kernels: np.ndarray
    switch: np.ndarray
    critical_delay: float
    real_mask: np.ndarray
    caustic_distance: float
    caustic_theta: float
    t_min: float
    images: np.ndarray


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

    Raises
    ------
    ValueError
        If ``w`` is not a valid frequency grid.
    """

    def __init__(self, w: Sequence[float]) -> None:
        self._w = _validate_frequencies(w)
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
        SchwingerCertificationError
            The wave-branch refusal on BOTH parities: if the Schwinger
            evaluator fails its paired-rule certification (strong shear /
            high ``w``, FINDINGS F005; above the ``w <= 60`` ceiling,
            FINDINGS F013).
        """
        source = np.asarray(y, dtype=float)
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        caustic = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)

        images, absolute_delays, t_min = _frame_delays(source, matrix)
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
            self._w, source, gamma, beta, kappa, t_min, delta_min)

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
            total, whose ``SchwingerCertificationError`` refusal
            `evaluate` can raise, is skipped here.
        """
        source = np.asarray(y, dtype=float)
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        caustic = geometry.nearest_caustic_point(
            gamma, beta, source, kappa=kappa)

        images, absolute_delays, t_min = _frame_delays(source, matrix)
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
            caustic_distance=float(caustic.distance),
            caustic_theta=float(caustic.theta),
            t_min=t_min,
            images=images)

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
