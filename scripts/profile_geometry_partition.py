#!/usr/bin/env python
"""Micro-benchmark for the ``geometry_partition`` residual sub-terms.

WHAT
----
Times the four sub-terms of
``ChangRefsdalChannels.geometry_partition`` that make up the ~2 ms
"residual" left after the Build-8b caustic-search optimization:

1. ``find_images`` -- the depressed-quartic root solve plus the
   deterministic Newton polish (``geometry.find_images``);
2. per-image Fermat delays -- ``geometry.delay`` over the image set
   plus the parked critical-carrier delay;
3. the analytic saddle kernels -- ``channels._physical_kernels``
   (``geometry.image_kernel`` per real channel over the ``w`` grid);
4. the criticality switch -- ``channels._channel_switch``
   (``smootherstep`` per real channel over the ``w`` grid).

For context it also reports the already-optimized nearest-caustic
search (``geometry.nearest_caustic_point``, Build 8b, NOT part of the
optimizable residual) and the cheap label continuation
(``channels._assign_labels``).

WHY
---
This script is the committed provenance for WP1 (Build 8f, lever 1):
it identifies the single *measured-dominant* residual sub-term so the
optimization touches only that term.  Run it on ``HEAD`` and on the
optimized tree to obtain the pre/post split.  It is a profiler -- it
measures wall time only and makes NO correctness claim; value
preservation is verified separately by the test suite.

The configuration sweep is drawn from a physically representative
box and is deliberately seeded (deterministic).  It includes explicit
near-caustic ``eta = +/- 0.002`` crossings, where the quartic acquires
a near-double root and the Newton polish iterates hardest.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    _assign_labels, _channel_switch, _labeled_delays, _physical_kernels)

#: Half-width of the explicit near-caustic offset (dimensionless source
#: separation) used to place sources just inside/outside the caustic.
_NEAR_CAUSTIC_ETA = 2e-3

#: Terms reported as the optimizable residual, in evaluation order.
_RESIDUAL_TERMS = ('find_images', 'per_image_delays',
                   'physical_kernels', 'channel_switch')

#: Terms reported for context but NOT part of the optimizable residual.
_CONTEXT_TERMS = ('nearest_caustic_point', 'assign_labels')


@dataclass(frozen=True)
class LensConfig:
    """One benchmark point in lens-parameter space."""

    gamma: float
    y: tuple[float, float]
    beta: float
    kappa: float


@dataclass
class TermTiming:
    """Accumulated wall time and call count for one sub-term."""

    total_seconds: float = 0.0
    calls: int = 0

    def add(self, seconds: float, calls: int) -> None:
        self.total_seconds += seconds
        self.calls += calls

    @property
    def microseconds_per_call(self) -> float:
        if self.calls == 0:
            return float('nan')
        return 1e6 * self.total_seconds / self.calls


def _draw_random_configs(n_configs: int,
                         rng: np.random.Generator) -> list[LensConfig]:
    """Draw ``n_configs`` representative lens configurations.

    The source radius spans both sides of the astroid caustic so the
    sweep naturally exercises two- and four-image topologies; ``kappa``
    is pinned to zero to match production.
    """
    configs: list[LensConfig] = []
    for _ in range(n_configs):
        gamma = float(rng.uniform(0.05, 0.8))
        radius = float(rng.uniform(0.02, 1.5))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        beta = float(rng.uniform(0.0, np.pi))
        configs.append(LensConfig(
            gamma=gamma,
            y=(radius * np.cos(phi), radius * np.sin(phi)),
            beta=beta,
            kappa=0.0))
    return configs


def _draw_near_caustic_configs(n_configs: int,
                               rng: np.random.Generator
                               ) -> list[LensConfig]:
    """Place sources at ``eta = +/- _NEAR_CAUSTIC_ETA`` from the caustic.

    The near-caustic pair is the ill-conditioned regime: the image
    quartic has a near-double root and the Newton polish iterates
    hardest there, so these points make the ``find_images`` cost
    honest.  A configuration whose caustic geometry is refused
    (``LensDomainError``) is skipped.
    """
    configs: list[LensConfig] = []
    attempts = 0
    while len(configs) < n_configs and attempts < 20 * n_configs:
        attempts += 1
        gamma = float(rng.uniform(0.1, 0.7))
        beta = float(rng.uniform(0.0, np.pi))
        # Seed source somewhere near the caustic scale, then snap it to
        # the nearest critical point and offset radially by +/- eta.
        seed = rng.uniform(-0.8, 0.8, size=2)
        try:
            caustic = geometry.nearest_caustic_point(
                gamma, beta, np.asarray(seed, dtype=float), kappa=0.0)
        except geometry.LensDomainError:
            continue
        source_caustic = np.asarray(caustic.source, dtype=float)
        norm = float(np.linalg.norm(source_caustic))
        if norm <= 1e-6:
            continue
        radial = source_caustic / norm
        for sign in (+1.0, -1.0):
            offset = source_caustic + sign * _NEAR_CAUSTIC_ETA * radial
            configs.append(LensConfig(
                gamma=gamma, y=(float(offset[0]), float(offset[1])),
                beta=beta, kappa=0.0))
            if len(configs) >= n_configs:
                break
    return configs


@dataclass
class PreparedPoint:
    """Geometry solved once for a config, reused to time each sub-term."""

    source: np.ndarray
    matrix: np.ndarray
    caustic: geometry.NearestCausticPoint
    images: list[np.ndarray]
    assignment: np.ndarray
    relative_delays: np.ndarray
    delays: np.ndarray
    real_mask: np.ndarray
    critical_delay: float


def _prepare_point(config: LensConfig, w: np.ndarray) -> PreparedPoint | None:
    """Reproduce ``geometry_partition`` once; ``None`` if the point is
    refused by a named domain error."""
    source = np.asarray(config.y, dtype=float)
    try:
        matrix = geometry.macro_matrix(config.gamma, config.beta, config.kappa)
        caustic = geometry.nearest_caustic_point(
            config.gamma, config.beta, source, kappa=config.kappa)
        images = geometry.find_images(source, matrix)
    except geometry.LensDomainError:
        return None

    absolute_delays = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    t_min = float(absolute_delays.min())
    relative_delays = absolute_delays - t_min

    assignment, _markers = _assign_labels(None, images, caustic.image)
    virtual_delay = geometry.delay(caustic.image, source, matrix)
    critical_delay = virtual_delay - t_min
    delays, real_mask = _labeled_delays(
        assignment, relative_delays, critical_delay)

    return PreparedPoint(
        source=source, matrix=matrix, caustic=caustic, images=images,
        assignment=assignment, relative_delays=relative_delays,
        delays=delays, real_mask=real_mask, critical_delay=critical_delay)


def _time_call(func: Callable[[], object], repeats: int) -> float:
    """Return the total wall time of ``repeats`` calls to ``func``."""
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return time.perf_counter() - start


def _subterm_callables(point: PreparedPoint,
                       config: LensConfig,
                       w: np.ndarray) -> dict[str, Callable[[], object]]:
    """Map each timed sub-term name to a zero-argument callable that
    recomputes exactly that term from the prepared geometry."""

    def find_images() -> object:
        return geometry.find_images(point.source, point.matrix)

    def per_image_delays() -> object:
        delays = [geometry.delay(image, point.source, point.matrix)
                  for image in point.images]
        delays.append(geometry.delay(
            point.caustic.image, point.source, point.matrix))
        return delays

    def physical_kernels() -> object:
        return _physical_kernels(
            w, point.assignment, point.images, point.matrix)

    def channel_switch() -> object:
        return _channel_switch(
            w, point.delays, point.real_mask, point.critical_delay)

    def nearest_caustic_point() -> object:
        return geometry.nearest_caustic_point(
            config.gamma, config.beta, point.source, kappa=config.kappa)

    def assign_labels() -> object:
        return _assign_labels(None, point.images, point.caustic.image)

    return {
        'find_images': find_images,
        'per_image_delays': per_image_delays,
        'physical_kernels': physical_kernels,
        'channel_switch': channel_switch,
        'nearest_caustic_point': nearest_caustic_point,
        'assign_labels': assign_labels,
    }


def profile(configs: list[LensConfig], w: np.ndarray, *,
            repeats: int) -> dict[str, TermTiming]:
    """Time every sub-term across the configuration sweep.

    Parameters
    ----------
    configs : list of LensConfig
        The benchmark sweep.
    w : np.ndarray
        Dimensionless frequency grid the kernels/switch broadcast over.
    repeats : int
        Timed repetitions per sub-term per configuration (a single
        warm-up call precedes the timed loop).

    Returns
    -------
    dict
        Sub-term name -> accumulated :class:`TermTiming`.
    """
    timings = {name: TermTiming()
               for name in _RESIDUAL_TERMS + _CONTEXT_TERMS}
    n_used = 0
    for config in configs:
        point = _prepare_point(config, w)
        if point is None:
            continue
        n_used += 1
        callables = _subterm_callables(point, config, w)
        for name, func in callables.items():
            func()  # warm-up, excluded from timing
            seconds = _time_call(func, repeats)
            timings[name].add(seconds, repeats)
    if n_used == 0:
        raise RuntimeError(
            'No configuration in the sweep survived the domain checks; '
            'nothing to profile.')
    return timings


def _report(timings: dict[str, TermTiming]) -> dict[str, object]:
    """Build the pre/post split report from accumulated timings."""
    residual_total = sum(
        timings[name].microseconds_per_call for name in _RESIDUAL_TERMS)
    terms = {}
    for name in _RESIDUAL_TERMS:
        per_call = timings[name].microseconds_per_call
        terms[name] = {
            'microseconds_per_call': per_call,
            'fraction_of_residual': (per_call / residual_total
                                     if residual_total > 0.0 else float('nan')),
        }
    context = {
        name: {'microseconds_per_call': timings[name].microseconds_per_call}
        for name in _CONTEXT_TERMS
    }
    dominant = max(_RESIDUAL_TERMS,
                   key=lambda name: timings[name].microseconds_per_call)
    return {
        'residual_microseconds_per_call': residual_total,
        'dominant_term': dominant,
        'terms': terms,
        'context': context,
    }


def _print_report(report: dict[str, object], *, n_configs: int,
                  n_w: int, repeats: int) -> None:
    """Human-readable split to stdout."""
    print('geometry_partition residual profile')
    print(f'  configs={n_configs}  n_w={n_w}  repeats={repeats}')
    print(f'  residual total: '
          f'{report["residual_microseconds_per_call"]:.2f} us/call')
    print(f'  dominant term : {report["dominant_term"]}')
    print('  --- optimizable residual sub-terms ---')
    for name in _RESIDUAL_TERMS:
        entry = report['terms'][name]
        print(f'    {name:<20s} {entry["microseconds_per_call"]:8.2f} us '
              f'({100.0 * entry["fraction_of_residual"]:5.1f} %)')
    print('  --- context (not optimized) ---')
    for name in _CONTEXT_TERMS:
        entry = report['context'][name]
        print(f'    {name:<20s} {entry["microseconds_per_call"]:8.2f} us')


def _build_configs(n_configs: int, seed: int) -> list[LensConfig]:
    """Assemble the deterministic sweep: random box plus near-caustic
    crossings."""
    rng = np.random.default_rng(seed)
    n_near = max(2, n_configs // 4)
    random_configs = _draw_random_configs(n_configs - n_near, rng)
    near_configs = _draw_near_caustic_configs(n_near, rng)
    return random_configs + near_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-configs', type=int, default=200,
                        help='Number of lens configurations in the sweep.')
    parser.add_argument('--n-w', type=int, default=128,
                        help='Size of the dimensionless frequency grid.')
    parser.add_argument('--w-min', type=float, default=5.0,
                        help='Smallest dimensionless frequency.')
    parser.add_argument('--w-max', type=float, default=400.0,
                        help='Largest dimensionless frequency.')
    parser.add_argument('--repeats', type=int, default=50,
                        help='Timed repetitions per sub-term per config.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the deterministic configuration sweep.')
    parser.add_argument('--json', type=str, default=None,
                        help='Optional path to write the split as JSON.')
    args = parser.parse_args()

    w = np.geomspace(args.w_min, args.w_max, args.n_w)
    configs = _build_configs(args.n_configs, args.seed)
    timings = profile(configs, w, repeats=args.repeats)
    report = _report(timings)
    _print_report(report, n_configs=len(configs), n_w=args.n_w,
                  repeats=args.repeats)
    if args.json is not None:
        with open(args.json, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(f'  wrote {args.json}')


if __name__ == '__main__':
    main()
