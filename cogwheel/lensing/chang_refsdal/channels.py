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
* A smooth switch hands each resolved channel from its artificial
  cluster gauge to its physical stationary-phase target, each label
  switching on ITS OWN delay separation.  The switch buys smoothness,
  not accuracy.
* A residual projection, reused verbatim from `_gauge`, makes the four
  channel kernels sum to the exact operator total at every frequency,
  including through the transition.  Because the total is symmetric in
  the labels, relabelling (whether from a reset or from continuation)
  can never change it -- only the smoothness of individual channels
  depends on the labels being continued consistently.

WHY IT DELEGATES
----------------
The exact residual projection lives once in `_gauge`; this module
calls it rather than carrying a second copy.  The wave/geometric
evaluation gate and the smooth-switch window live once in `operator`;
this module imports ``RHO_START``, ``RHO_END`` and `select_branch`
rather than re-deriving the thresholds.  The exact total is evaluated
with the contour-free operator `operator.F_op` where the wave branch is
certified and with `operator.geometric_amplification` once
`operator.select_branch` reports the stationary-phase branch is
legitimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._gauge import (
    exact_transition_channels, reconstructed_total, smootherstep)
from cogwheel.lensing.chang_refsdal.operator import (
    RHO_START, RHO_END, MAX_ORDER, F_op, cancellation_exponent,
    geometric_amplification, select_branch)

__all__ = ['ChangRefsdalChannels', 'ChangRefsdalPartition',
           'real_image_delays']

#: The fixed number of topology-stable labels.
_N_CHANNELS = 4

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
                    real_mask: np.ndarray) -> np.ndarray:
    """Per-channel smooth hand-over switch.

    Each real channel switches on its OWN delay separation ``delta_j``
    from the NEAREST cluster member of any kind -- a real image OR a
    virtual label parked at the critical point -- per the paper's
    delay-separation rule (Eq. delay-separation, ``eq:delay-separation``)

        delta_j = min_{k in C, k != j} |tau_j - tau_k|,

    where the minimum runs over ALL four cluster labels, not the real
    ones alone.  On the two-image side of a caustic a near-critical real
    image's true cluster mates are the parked virtual labels it is about
    to spawn (or that have just annihilated); measuring the separation
    against real channels only misses that coincidence and lets a
    still-merged channel ramp its switch to one, handing it to the
    divergent stationary-phase kernel.  Measuring against every other
    label instead keeps a channel that is still merged with ANY cluster
    member (small separation) in the artificial gauge, while a fully
    resolved channel is handed to its physical target.  Virtual channels
    never switch (``S_j = 0`` for a virtual label, Eq. switch).

    Parameters
    ----------
    w : np.ndarray
        Dimensionless frequency grid.
    delays : np.ndarray
        Per-channel delays, indexed by cluster label ``0 .. _N_CHANNELS - 1``.
    real_mask : np.ndarray
        Boolean mask of real channels.

    Returns
    -------
    np.ndarray
        Shape ``(n_w, _N_CHANNELS)`` switch in ``[0, 1]``.
    """
    switch = np.zeros((w.shape[0], _N_CHANNELS), dtype=float)
    real_ids = np.flatnonzero(real_mask)
    for channel in real_ids:
        others = np.delete(np.arange(_N_CHANNELS), channel)
        separation = float(
            np.min(np.abs(delays[channel] - delays[others])))
        switch[:, channel] = smootherstep(w * separation,
                                          RHO_START, RHO_END)
    return switch


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
    contour-free wave operator `operator.F_op` and the stationary-phase
    `operator.geometric_amplification`.  The result is shifted by
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
        Operator-series order cap forwarded to `operator.F_op`.

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
    for index in range(n_w):
        frequency = float(w[index])
        exponent = cancellation_exponent(frequency, source, gamma, kappa)
        if select_branch(frequency, delta_min, exponent) == 'geometric':
            value = complex(geometric_amplification(
                frequency, source, gamma, beta=beta, kappa=kappa))
            orders[index] = _GEOMETRIC_ORDER
            converged[index] = True
        else:
            value, diagnostics = F_op(
                frequency, source, gamma, beta=beta, kappa=kappa,
                max_order=max_order)
            orders[index] = diagnostics.order_used
            converged[index] = diagnostics.converged
        total[index] = value * np.exp(-1j * frequency * t_min)
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
        If ``1 - kappa <= abs(gamma)`` (outside the positive-parity
        regime), raised by `geometry.macro_matrix` exactly as in
        `ChangRefsdalChannels.evaluate` and on the brute-force strain
        path, so the two paths refuse symmetrically.
    """
    source = np.asarray(y, dtype=float)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(source, matrix)
    absolute_delays = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    return np.sort(absolute_delays - absolute_delays.min())


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
        Shape ``(n_w, 4)`` channel kernels ``K_a(w)``.
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
            If ``1 - kappa <= abs(gamma)`` (outside the positive-parity
            regime).
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
        delays, real_mask = _labeled_delays(
            assignment, relative_delays, virtual_delay - t_min)

        physical = _physical_kernels(self._w, assignment, images, matrix)
        switch = _channel_switch(self._w, delays, real_mask)
        delta_min = _min_delay_separation(delays, real_mask)

        exact_total, orders, converged = _exact_total(
            self._w, source, gamma, beta, kappa, t_min, delta_min,
            self._max_order)

        kernels = exact_transition_channels(
            self._w, exact_total, float(np.mean(delays)), delays,
            physical, switch)

        return ChangRefsdalPartition(
            w=self._w,
            source=source,
            gamma=float(gamma),
            beta=float(beta),
            kappa=float(kappa),
            delays=delays,
            kernels=kernels,
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
