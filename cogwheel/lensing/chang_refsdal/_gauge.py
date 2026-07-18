"""
Exact gauge partitions for unresolved Chang--Refsdal image clusters.

WHAT
----
The primitives that choose *how* a Chang--Refsdal amplification is
split into channels, and nothing else.  Given a total amplification
supplied by the caller, these functions return kernels ``K_j`` such
that

    F = sum_j exp(1j * w * tau_j) * K_j

holds identically, plus the C2 switch used to hand a channel from its
unresolved-cluster gauge to its stationary-phase target.  There is no
lens physics here: no image solving, no wave-optics evaluation, no
contour integral.  The total is an input.

WHY
---
Near a fold or a cusp the members of an image cluster are not
individually resolved: their stationary-phase kernels diverge like
``sqrt(abs(mu_a))`` and their delays merge, so a decomposition built
from per-image asymptotics is both inaccurate and discontinuous across
the crossing.  The fix is to stop insisting that a channel *be* an
image.  These functions never approximate the supplied total; they only
move exact field between channels, so the channels can be made smooth
functions of the lens parameters while their coherent sum stays equal
to the input at every frequency.  Interpolating the smooth ``K_j`` is
then the only approximation the downstream likelihood makes.

The residual projection is what buys the exactness.  Given any trial
gauge ``T_j`` -- any blend at all of an artificial split and a physical
target -- the returned kernels are

    K_j = T_j + alpha_j * exp(-1j * w * tau_j) * R,
    R   = F - sum_j exp(1j * w * tau_j) * T_j,

with weights ``alpha_j`` summing to one.  Because ``w`` and ``tau_j``
are real, ``exp(1j * w * tau_j) * exp(-1j * w * tau_j) == 1`` exactly in
the algebra, so the coherent sum telescopes back to ``F`` for *any*
trial gauge:

    sum_j exp(1j*w*tau_j) * K_j = sum_j exp(1j*w*tau_j)*T_j
                                  + R * sum_j alpha_j
                                = F.

The identity therefore holds independently of how good ``T_j`` is; a
poor trial gauge costs smoothness, never exactness.  The correction
vanishes as the physical target becomes exact, so nothing is paid in
the resolved limit.

Because the identity is algebraic, the only error is floating-point
roundoff, and it scales with the *largest intermediate* --
``sum_j abs(K_j)``, which diverges like ``sqrt(abs(mu_a))`` at a
critical point -- not with ``abs(F)``.  Callers checking the identity
must use a scale-aware bound of the form
``C * eps * (abs(F) + sum_j abs(K_j) + abs(R))``; a flat relative gate
is not achievable near a fold and its failure would not indicate a bug.

Conventions
-----------
``w`` is the dimensionless frequency and ``tau_j`` the dimensionless
Fermat delays of `geometry`; both are real, which is exactly what makes
the carriers unimodular and the projection exact.  Every function
accepts ``w`` either as a scalar or as a 1-D grid, and returns kernels
of shape ``(n_members,)`` or ``(n_w, n_members)`` respectively.

All arithmetic is float64.  These are O(1) algebraic rearrangements
with no cancellation channel of their own; see the module docstring of
`_dd` for where double-double precision *is* required.

Weights are normalized to sum to one; equal weights are the safest
fold/cusp gauge.  A 1-D ``switch`` is interpreted as *per frequency*;
pass a 2-D ``(n_w, n_members)`` array for a per-channel switch.

This module is private to `lensing.chang_refsdal`.  It is imported by
both the production channel tracker and the crossing-scenario builders,
which is why it holds no state and knows nothing about either.
"""

from __future__ import annotations

import numpy as np


def smootherstep(x: np.ndarray | float, x0: float, x1: float
                 ) -> np.ndarray:
    """
    Return the C2 switch rising from zero at ``x <= x0`` to one at
    ``x >= x1``.

    This is Perlin's smootherstep, ``6*u**5 - 15*u**4 + 10*u**3``.  Both
    its first and second derivatives vanish at either join, which is why
    it is used here in preference to the C1 smoothstep: the channel
    kernels are interpolated downstream, and a discontinuous second
    derivative at the hand-over would show up as an interpolation error
    exactly where the switch is meant to be invisible.

    Parameters
    ----------
    x : float or array_like
        Switch argument, typically ``w * delta_j`` with ``delta_j`` the
        delay separation of channel ``j`` from its nearest neighbour.
    x0, x1 : float
        Start and end of the transition; ``x1`` must exceed ``x0``.

    Returns
    -------
    numpy.ndarray
        Switch value in ``[0, 1]``, with the shape of ``x``.

    Raises
    ------
    ValueError
        If ``x1 <= x0`` (or either is not finite), which would make the
        transition ill-defined rather than merely sharp.
    """
    if not x1 > x0:
        raise ValueError(
            'Cannot build a switch: smootherstep needs a transition '
            f'window with x1 > x0, got x0={x0!r}, x1={x1!r}. Pass an '
            'ordered, finite pair (e.g. rho_start=0.5, rho_end=4.0).')

    x = np.asarray(x, dtype=float)
    u = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return u**3 * (10.0 - 15.0*u + 6.0*u*u)


def exact_cluster_kernel(w: np.ndarray | float,
                         total_amplification: np.ndarray | complex,
                         persistent_total: np.ndarray | complex,
                         tau_cluster: float) -> np.ndarray:
    """
    Return the demodulated exact residual carried by an unresolved
    cluster.

    The cluster kernel is *defined* by the split

        total = persistent_total + exp(1j*w*tau_cluster) * K_cluster,

    i.e. it is whatever the resolved, persistent images do not account
    for.  It is exact by construction: no approximation is made to
    ``total``, and the divergent per-image asymptotics of the cluster
    members never appear.

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    total_amplification : complex or array_like
        The full amplification ``F``, broadcastable against ``w``.
    persistent_total : complex or array_like
        Coherent sum over the resolved images that are *not* in the
        cluster, broadcastable against ``w``.
    tau_cluster : float
        Dimensionless delay of the cluster's carrier.

    Returns
    -------
    numpy.ndarray
        ``K_cluster``, with the shape of ``w``.
    """
    w = _as_frequency(w)
    total = np.asarray(total_amplification, dtype=complex)
    persistent = np.asarray(persistent_total, dtype=complex)
    return np.exp(-1j * w * tau_cluster) * (total - persistent)


def unresolved_member_channels(w: np.ndarray | float,
                               cluster_kernel: np.ndarray | complex,
                               tau_cluster: float,
                               member_delays: np.ndarray,
                               weights: np.ndarray | None = None
                               ) -> np.ndarray:
    """
    Split one exact unresolved cluster among its member carriers.

    The split is artificial -- the members are unresolved, so there is
    no physically preferred way to apportion the field -- but it is
    exact:

        sum_j exp(1j*w*tau_j) * K_j == exp(1j*w*tau_cluster) * K_cluster

    holds identically for any weights.  Equal weights (the default) are
    the safest fold/cusp gauge, since they introduce no artificial
    asymmetry between members that the geometry does not have.

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    cluster_kernel : complex or array_like
        ``K_cluster``, with the shape of ``w``.
    tau_cluster : float
        Dimensionless delay of the cluster's carrier.
    member_delays : array_like
        Dimensionless delays ``tau_j`` of the cluster members, 1-D.
    weights : array_like, optional
        Non-negative apportionment weights, one per member.  Normalized
        internally; equal weights are used if omitted.

    Returns
    -------
    numpy.ndarray
        Member kernels, shape ``(n_members,)`` for scalar ``w`` or
        ``(n_w, n_members)`` for a 1-D grid.
    """
    w = _as_frequency(w)
    kernel = np.asarray(cluster_kernel, dtype=complex)
    tau = _as_delays(member_delays)
    alpha = _normalized_weights(weights, tau.size)
    return _member_split(w, kernel, tau_cluster, tau, alpha)


def exact_transition_channels(w: np.ndarray | float,
                              total_cluster: np.ndarray | complex,
                              tau_cluster: float,
                              member_delays: np.ndarray,
                              physical_kernels: np.ndarray,
                              switch: np.ndarray | float,
                              weights: np.ndarray | None = None
                              ) -> np.ndarray:
    """
    Blend artificial unresolved channels into physical image channels,
    exactly.

    ``physical_kernels[..., j]`` is the stationary-phase target ``H_j``
    that channel ``j`` must approach once it is resolved.  The blend
    alone would not sum to ``total_cluster``; a residual projection is
    added so that the channel sum equals the supplied total at every
    ``w``, *including through the transition*, where neither the
    artificial split nor the physical target is on its own adequate.
    The correction vanishes as ``H_j`` becomes exact, so the resolved
    limit is untouched.

    See the module docstring for why this is exact for any ``switch``:
    the switch buys smoothness, not accuracy, and injects no error into
    the total.

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    total_cluster : complex or array_like
        Exact cluster total to be reproduced, with the shape of ``w``.
    tau_cluster : float
        Dimensionless delay of the cluster's carrier.
    member_delays : array_like
        Dimensionless delays ``tau_j`` of the cluster members, 1-D.
    physical_kernels : array_like
        Stationary-phase targets ``H_j``, shaped like the return value.
    switch : float or array_like
        Blend fraction: zero selects the artificial split, one selects
        the physical targets.  A 1-D array is per frequency; pass a 2-D
        ``(n_w, n_members)`` array for a per-channel switch.
    weights : array_like, optional
        Non-negative projection weights, one per member.  Normalized
        internally; equal weights are used if omitted.

    Returns
    -------
    numpy.ndarray
        Channel kernels, shape ``(n_members,)`` for scalar ``w`` or
        ``(n_w, n_members)`` for a 1-D grid.

    Raises
    ------
    ValueError
        If ``physical_kernels`` or ``switch`` cannot be matched to the
        channel shape implied by ``w`` and ``member_delays``.
    """
    w = _as_frequency(w)
    total = np.asarray(total_cluster, dtype=complex)
    tau = _as_delays(member_delays)
    alpha = _normalized_weights(weights, tau.size)

    cluster_kernel = np.exp(-1j * w * tau_cluster) * total
    artificial = _member_split(w, cluster_kernel, tau_cluster, tau,
                               alpha)

    targets = np.asarray(physical_kernels, dtype=complex)
    if targets.shape != artificial.shape:
        raise ValueError(
            'Cannot blend physical kernels into the cluster gauge: '
            f'expected physical_kernels of shape {artificial.shape} '
            f'for {tau.size} members on a frequency argument of shape '
            f'{w.shape}, got {targets.shape}.')
    blend = _broadcast_switch(switch, artificial.shape)

    trial = (1.0 - blend)*artificial + blend*targets
    carrier = np.exp(1j * np.multiply.outer(w, tau))
    if w.ndim == 0:
        residual = total - np.sum(carrier * trial)
        return trial + alpha * np.conj(carrier) * residual
    residual = total - np.sum(carrier * trial, axis=1)
    return trial + alpha[None, :] * np.conj(carrier) * residual[:, None]


def reconstructed_total(w: np.ndarray | float,
                        member_delays: np.ndarray,
                        kernels: np.ndarray) -> np.ndarray:
    """
    Reconstruct ``sum_j exp(1j*w*tau_j) * K_j``.

    This is the single authoritative implementation of the coherent
    channel sum; every exactness check in the subpackage goes through
    it.  The result is invariant under relabelling the channels: the
    sum is symmetric in ``j``, so only the *smoothness* of individual
    kernels needs labels to be continued consistently, never the total.

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    member_delays : array_like
        Dimensionless delays ``tau_j``, 1-D.
    kernels : array_like
        Channel kernels, shape ``(n_members,)`` for scalar ``w`` or
        ``(n_w, n_members)`` for a 1-D grid.

    Returns
    -------
    numpy.ndarray
        The coherent total, with the shape of ``w``.
    """
    w = _as_frequency(w)
    tau = _as_delays(member_delays)
    kernel_values = np.asarray(kernels, dtype=complex)

    carrier = np.exp(1j * np.multiply.outer(w, tau))
    if w.ndim == 0:
        return np.sum(carrier * kernel_values)
    return np.sum(carrier * kernel_values, axis=1)


def switched_analytic_channels(w: np.ndarray | float,
                               total_amplification: np.ndarray | complex,
                               member_delays: np.ndarray,
                               saddle_kernels: np.ndarray,
                               switch: np.ndarray,
                               critical_delay: float,
                               weights: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    SACR-C projection: switched analytic trials plus one demodulated
    envelope, reconstructing the supplied total exactly.

    Each channel's trial gauge is the *switched analytic saddle kernel*
    ``T_j = S_j * H_j`` (no artificial cluster split): a resolved image
    carries its stationary-phase kernel ``H_j`` weighted by its switch
    ``S_j``, and an unresolved or virtual channel (``S_j -> 0``)
    contributes nothing to the trial.  Whatever the trials do not
    account for is carried by a SINGLE object -- the transition envelope

        E(w) := exp(-1j*w*tau_c) * (F - sum_j exp(1j*w*tau_j) * S_j*H_j),

    demodulated at the critical (parked) carrier ``tau_c``.  The channel
    kernels returned are

        K_j = S_j*H_j + alpha_j(w) * exp(-1j*w*(tau_j - tau_c)) * E,

    with per-frequency weights ``alpha_j(w)`` summing to one across
    channels, so the coherent sum telescopes back to ``F`` exactly for
    ANY weights (see the module docstring): the ``tau_c`` carrier cancels
    against the demodulation, ``exp(1j*w*tau_j) * exp(-1j*w*tau_j) == 1``,
    and ``sum_j alpha_j == 1``.  The switch and the weights buy
    smoothness of the individual channels, never accuracy of the total.

    Unlike `exact_transition_channels`, which splits the whole total
    among the members through an artificial cluster gauge, this
    projection injects only the *residual* the analytic saddles miss,
    demodulated at the critical carrier where -- by construction -- its
    phase against ``tau_c`` is bounded by the switch scale
    ``w * |tau_j - tau_c|`` (the beat-free property the microlensed
    likelihood interpolates).

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    total_amplification : complex or array_like
        The full amplification ``F``, with the shape of ``w``.
    member_delays : array_like
        Dimensionless delays ``tau_j`` of the channels, 1-D.
    saddle_kernels : array_like
        Analytic saddle kernels ``H_j``, shaped like the return kernels
        (``(n_members,)`` for scalar ``w`` or ``(n_w, n_members)`` for a
        grid); zero for virtual channels.
    switch : array_like
        Per-channel switch ``S_j`` in ``[0, 1]``, same shape as
        ``saddle_kernels``.
    critical_delay : float
        Dimensionless delay ``tau_c`` of the parked critical carrier.
    weights : array_like
        Non-negative per-frequency apportionment weights, same shape as
        ``saddle_kernels``; normalized to sum to one across channels.

    Returns
    -------
    kernels : numpy.ndarray
        Channel kernels ``K_j``, shape ``(n_members,)`` for scalar ``w``
        or ``(n_w, n_members)`` for a grid.
    envelope : numpy.ndarray
        The transition envelope ``E(w)``, with the shape of ``w``.

    Raises
    ------
    ValueError
        If ``saddle_kernels``, ``switch`` or ``weights`` cannot be
        matched to the channel shape implied by ``w`` and
        ``member_delays``, or the weights are not normalizable.
    """
    w, carrier, carrier_c, trial, alpha = _switched_setup(
        w, member_delays, saddle_kernels, switch, weights, critical_delay)
    total = np.asarray(total_amplification, dtype=complex)

    if w.ndim == 0:
        residual = total - np.sum(carrier * trial)
        kernels = trial + alpha * np.conj(carrier) * residual
    else:
        residual = total - np.sum(carrier * trial, axis=1)
        kernels = trial + alpha * np.conj(carrier) * residual[:, None]
    envelope = np.conj(carrier_c) * residual
    return kernels, envelope


def channels_from_envelope(w: np.ndarray | float,
                           envelope: np.ndarray | complex,
                           member_delays: np.ndarray,
                           saddle_kernels: np.ndarray,
                           switch: np.ndarray,
                           critical_delay: float,
                           weights: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward SACR-C reconstruction: rebuild channel kernels and the total
    from a (typically interpolated) envelope.

    The inverse of `switched_analytic_channels`: given the transition
    envelope ``E`` -- for instance a value interpolated onto a dense
    frequency from coarse engine nodes -- and the analytic saddle
    kernels evaluated at the same frequencies, return

        K_j = S_j*H_j + alpha_j(w) * exp(-1j*w*(tau_j - tau_c)) * E,
        F   = sum_j exp(1j*w*tau_j) * S_j*H_j + exp(1j*w*tau_c) * E,

    which satisfy ``F = sum_j exp(1j*w*tau_j) * K_j`` identically.  The
    approximation is entirely in ``E``; the algebra is exact.

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    envelope : complex or array_like
        The transition envelope ``E(w)``, with the shape of ``w``.
    member_delays : array_like
        Dimensionless delays ``tau_j`` of the channels, 1-D.
    saddle_kernels : array_like
        Analytic saddle kernels ``H_j``, shaped like the return kernels.
    switch : array_like
        Per-channel switch ``S_j`` in ``[0, 1]``, same shape as
        ``saddle_kernels``.
    critical_delay : float
        Dimensionless delay ``tau_c`` of the parked critical carrier.
    weights : array_like
        Non-negative per-frequency apportionment weights, same shape as
        ``saddle_kernels``; normalized to sum to one across channels.

    Returns
    -------
    kernels : numpy.ndarray
        Channel kernels ``K_j``.
    total : numpy.ndarray
        The reconstructed amplification total ``F``, with the shape of
        ``w``.

    Raises
    ------
    ValueError
        If the array shapes cannot be matched, or the weights are not
        normalizable.
    """
    w, carrier, carrier_c, trial, alpha = _switched_setup(
        w, member_delays, saddle_kernels, switch, weights, critical_delay)
    envelope = np.asarray(envelope, dtype=complex)
    residual = carrier_c * envelope

    if w.ndim == 0:
        kernels = trial + alpha * np.conj(carrier) * residual
        total = np.sum(carrier * trial) + residual
    else:
        kernels = trial + alpha * np.conj(carrier) * residual[:, None]
        total = np.sum(carrier * trial, axis=1) + residual
    return kernels, total


def envelope_total(w: np.ndarray | float,
                   member_delays: np.ndarray,
                   saddle_kernels: np.ndarray,
                   switch: np.ndarray,
                   critical_delay: float,
                   envelope: np.ndarray | complex) -> np.ndarray:
    """
    Reconstruct ``F`` from the SACR-C envelope identity.

    The single authoritative evaluation of

        F = sum_j exp(1j*w*tau_j) * S_j*H_j + exp(1j*w*tau_c) * E,

    the object every SACR-C exactness check compares against the exact
    total.  It reuses the same range-reduced carriers as the projection,
    so the carrier and its demodulation cancel to machine precision even
    where ``w * tau`` is large.

    Parameters
    ----------
    w : float or array_like
        Dimensionless frequency, scalar or 1-D grid.
    member_delays : array_like
        Dimensionless delays ``tau_j``, 1-D.
    saddle_kernels : array_like
        Analytic saddle kernels ``H_j``, shaped like the channel kernels.
    switch : array_like
        Per-channel switch ``S_j``, same shape as ``saddle_kernels``.
    critical_delay : float
        Dimensionless delay ``tau_c`` of the parked critical carrier.
    envelope : complex or array_like
        The transition envelope ``E(w)``, with the shape of ``w``.

    Returns
    -------
    numpy.ndarray
        The reconstructed total, with the shape of ``w``.
    """
    w = _as_frequency(w)
    tau = _as_delays(member_delays)
    saddle = np.asarray(saddle_kernels, dtype=complex)
    blend = np.asarray(switch, dtype=float)
    trial = blend * saddle
    envelope = np.asarray(envelope, dtype=complex)

    carrier = _unit_carrier(np.multiply.outer(w, tau))
    carrier_c = _unit_carrier(w * float(critical_delay))
    if w.ndim == 0:
        return np.sum(carrier * trial) + carrier_c * envelope
    return np.sum(carrier * trial, axis=1) + carrier_c * envelope


def _member_split(w: np.ndarray,
                  cluster_kernel: np.ndarray,
                  tau_cluster: float,
                  tau: np.ndarray,
                  alpha: np.ndarray) -> np.ndarray:
    """
    Apportion a cluster kernel among member carriers.

    Assumes validated inputs and weights that already sum to one, which
    is why it is private: it is the shared body of
    `unresolved_member_channels` and `exact_transition_channels`, and
    exists so that the weights are normalized exactly once per call.
    Normalizing twice is idempotent in exact arithmetic but not in
    float64, and would perturb the split at the 1e-16 level for no
    reason.
    """
    phase = np.exp(-1j * np.multiply.outer(w, tau - tau_cluster))
    if w.ndim == 0:
        return alpha * phase * cluster_kernel
    return phase * alpha[None, :] * cluster_kernel[:, None]


def _unit_carrier(phase: np.ndarray) -> np.ndarray:
    """
    Return ``exp(1j * phase)`` with ``phase`` range-reduced modulo
    ``2*pi``.

    Subtracting the nearest multiple of ``2*pi`` before the complex
    exponential (FINDINGS F001) keeps the exponent argument O(1) even
    where ``w * tau`` reaches many radians, so the carrier stays
    accurate to ~1 ulp; because the reduction is applied once and the
    same array is conjugated for the inverse carrier, the residual
    projection telescopes to machine precision regardless of ``w * tau``.
    """
    phase = np.asarray(phase, dtype=float)
    reduced = phase - 2.0 * np.pi * np.round(phase / (2.0 * np.pi))
    return np.exp(1j * reduced)


def _per_frequency_weights(weights: np.ndarray,
                           shape: tuple[int, ...]) -> np.ndarray:
    """
    Validate per-frequency apportionment weights and normalize them to
    sum to one across the channel (last) axis.

    Unlike `_normalized_weights`, which normalizes a single static
    weight vector, this accepts a full ``(n_w, n_members)`` (or
    ``(n_members,)`` for scalar ``w``) array so the SACR-C projection
    weights may vary with frequency, and normalizes each frequency's
    row exactly once.  The unit row sum is what keeps the residual
    projection exact, so it is enforced here rather than trusted.
    """
    alpha = np.asarray(weights, dtype=float)
    if alpha.shape != shape:
        raise ValueError(
            'Cannot apportion the SACR-C envelope: expected weights of '
            f'shape {shape}, got {alpha.shape}.')
    if not np.all(np.isfinite(alpha)):
        raise ValueError(
            'Cannot apportion the SACR-C envelope: weights must be '
            'finite, got non-finite entries.')
    if np.any(alpha < 0.0):
        raise ValueError(
            'Cannot apportion the SACR-C envelope: weights must be '
            'non-negative; a negative weight would amplify the residual '
            'projection instead of sharing it.')
    row_sum = alpha.sum(axis=-1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError(
            'Cannot apportion the SACR-C envelope: the weights at every '
            'frequency must have a positive sum to be normalizable.')
    return alpha / row_sum


def _switched_setup(w: np.ndarray | float,
                    member_delays: np.ndarray,
                    saddle_kernels: np.ndarray,
                    switch: np.ndarray,
                    weights: np.ndarray,
                    critical_delay: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
    """
    Shared validation and carrier setup for the SACR-C projection and
    its forward reconstruction.

    Validates the frequency grid and delays, checks that the saddle
    kernels, switch and weights all match the channel shape implied by
    ``w`` and ``member_delays``, builds the range-reduced per-channel
    carrier ``exp(1j*w*tau_j)`` and critical carrier ``exp(1j*w*tau_c)``,
    forms the switched trial gauge ``S_j*H_j``, and normalizes the
    weights across channels.
    """
    w = _as_frequency(w)
    tau = _as_delays(member_delays)
    saddle = np.asarray(saddle_kernels, dtype=complex)
    blend = np.asarray(switch, dtype=float)

    channel_shape = (tau.size,) if w.ndim == 0 else (w.size, tau.size)
    for name, array in (('saddle_kernels', saddle), ('switch', blend)):
        if array.shape != channel_shape:
            raise ValueError(
                f'Cannot assemble SACR-C channels: expected {name} of '
                f'shape {channel_shape} for {tau.size} channels on a '
                f'frequency argument of shape {w.shape}, got '
                f'{array.shape}.')

    alpha = _per_frequency_weights(weights, channel_shape)
    carrier = _unit_carrier(np.multiply.outer(w, tau))
    carrier_c = _unit_carrier(w * float(critical_delay))
    trial = blend * saddle
    return w, carrier, carrier_c, trial, alpha


def _as_frequency(w: np.ndarray | float) -> np.ndarray:
    """
    Validate and return the dimensionless frequency as a float array.

    Accepts a scalar or a 1-D grid.  Unlike the channel tracker's own
    grid validation, no ordering or positivity is imposed here: these
    primitives are pure algebra and are exercised at single frequencies
    and on unsorted grids by the tests.
    """
    arr = np.asarray(w, dtype=float)
    if arr.ndim > 1:
        raise ValueError(
            'Cannot form channel carriers: w must be a scalar or a 1-D '
            f'grid, got an array of shape {arr.shape}.')
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            'Cannot form channel carriers: w must be finite, got '
            'non-finite entries.')
    return arr


def _as_delays(member_delays: np.ndarray) -> np.ndarray:
    """Validate and return the member delays as a 1-D float array."""
    tau = np.asarray(member_delays, dtype=float)
    if tau.ndim != 1 or tau.size == 0:
        raise ValueError(
            'Cannot apportion a cluster: member_delays must be a '
            f'non-empty 1-D array, got shape {tau.shape}.')
    if not np.all(np.isfinite(tau)):
        raise ValueError(
            'Cannot apportion a cluster: member_delays must be finite, '
            'got non-finite entries.')
    return tau


def _normalized_weights(weights: np.ndarray | None,
                        n_members: int) -> np.ndarray:
    """
    Validate the apportionment weights and normalize them to sum to one.

    The unit sum is what makes the residual projection exact, so it is
    enforced here rather than trusted from the caller.
    """
    if weights is None:
        return np.full(n_members, 1.0 / n_members)

    alpha = np.asarray(weights, dtype=float)
    if alpha.shape != (n_members,):
        raise ValueError(
            'Cannot apportion a cluster: expected one weight per '
            f'member, i.e. shape ({n_members},), got {alpha.shape}.')
    if not np.all(np.isfinite(alpha)):
        raise ValueError(
            'Cannot apportion a cluster: weights must be finite, got '
            f'{alpha!r}.')
    if np.any(alpha < 0.0):
        raise ValueError(
            'Cannot apportion a cluster: weights must be non-negative, '
            f'got {alpha!r}. A negative weight would amplify the '
            'residual projection instead of sharing it.')

    total = alpha.sum()
    if total <= 0.0:
        raise ValueError(
            'Cannot apportion a cluster: weights must have a positive '
            f'sum to be normalizable, got {alpha!r}.')
    return alpha / total


def _broadcast_switch(switch: np.ndarray | float,
                      shape: tuple[int, ...]) -> np.ndarray:
    """
    Broadcast the switch to the channel shape.

    A switch whose shape matches the leading (frequency) axes is taken
    to be per frequency and is broadcast across channels; anything else
    must already broadcast against ``shape``.  The switch range is not
    policed: any value leaves the total exact (see the module
    docstring), so a value outside ``[0, 1]`` is a smoothness question
    for the caller, not a correctness failure here.
    """
    blend = np.asarray(switch, dtype=float)
    if blend.ndim == len(shape) - 1 and blend.shape == shape[:-1]:
        blend = blend[..., None]
    try:
        return np.broadcast_to(blend, shape)
    except ValueError as exc:
        raise ValueError(
            'Cannot blend the cluster gauge: switch of shape '
            f'{np.shape(switch)} does not broadcast to the channel '
            f'shape {shape}. Pass a scalar, a per-frequency array of '
            f'shape {shape[:-1]}, or a per-channel array of shape '
            f'{shape}.') from exc
