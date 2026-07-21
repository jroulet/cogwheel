#!/usr/bin/env python
"""Micro-benchmark for the lensed-likelihood moment contraction.

Times the two per-evaluation reductions of
``cogwheel.lensing.likelihood`` -- ``_data_term`` (``(d|h_L)``) and
``_norm_term`` (``(h_L|h_L)``) -- in isolation on synthetic inputs of a
representative ``(n_modes, n_det, n_bins, n_img)`` shape, and reports how the
per-call cost splits between them.

This is the committed provenance behind Build 8f lever 2: it documents that
``_norm_term`` -- whose mode-pair reduction is *quadratic* in the mode count
via the ``(n_m, n_m, n_det, n_bins)`` moment tensors -- dominates the
``~2.3 ms`` contraction, motivating the fusion of that reduction.  The
synthetic arrays reproduce the shapes and dtypes documented in the
``_data_term`` / ``_norm_term`` docstrings; the absolute values are random,
so this measures the contraction cost only (it is a timing diagnostic, never
a correctness gate -- value preservation is checked by the test suite).

Timing is inherently machine- and load-dependent; treat the reported numbers
as an ordering diagnostic (which term dominates), not a fixed benchmark.
"""
from __future__ import annotations

import argparse
import timeit
from dataclasses import dataclass

import numpy as np

from cogwheel.lensing import likelihood


@dataclass(frozen=True)
class ContractionShape:
    """Array dimensions for one synthetic contraction workload."""

    n_modes: int
    n_det: int
    n_bins: int
    n_img: int


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]
                    ) -> np.ndarray:
    """Unit-scale complex Gaussian array of the requested shape."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def build_data_term_inputs(shape: ContractionShape,
                           rng: np.random.Generator) -> tuple:
    """Synthetic positional arguments for ``likelihood._data_term``.

    Shapes follow the ``_data_term`` docstring: three mode moments
    ``A^(0..2)`` of shape ``(n_m, n_det, n_bins)``, conjugate ratio
    center/slope of the same shape, conjugate kernel center/slope of shape
    ``(n_img, n_bins)``, image delays ``(n_img,)`` and bin centers
    ``(n_bins,)``.
    """
    mode_shape = (shape.n_modes, shape.n_det, shape.n_bins)
    img_shape = (shape.n_img, shape.n_bins)
    a_moments = [_complex_normal(rng, mode_shape) for _ in range(3)]
    rho0 = _complex_normal(rng, mode_shape)
    rho1 = _complex_normal(rng, mode_shape)
    kbar0 = _complex_normal(rng, img_shape)
    kbar1 = _complex_normal(rng, img_shape)
    tau = rng.standard_normal(shape.n_img) * 1e-3  # seconds
    f_center = np.linspace(20.0, 1024.0, shape.n_bins)  # Hz
    return a_moments, rho0, rho1, kbar0, kbar1, tau, f_center


def build_norm_term_inputs(shape: ContractionShape,
                           rng: np.random.Generator) -> tuple:
    """Synthetic positional arguments for ``likelihood._norm_term``.

    Shapes follow the ``_norm_term`` docstring: four mode-pair moments
    ``B^(0..3)`` of shape ``(n_m, n_m, n_det, n_bins)``, ratio and conjugate
    center/slope of shape ``(n_m, n_det, n_bins)``, kernel and conjugate
    center/slope of shape ``(n_img, n_bins)``, image delays ``(n_img,)`` and
    bin centers ``(n_bins,)``.
    """
    pair_shape = (shape.n_modes, shape.n_modes, shape.n_det, shape.n_bins)
    mode_shape = (shape.n_modes, shape.n_det, shape.n_bins)
    img_shape = (shape.n_img, shape.n_bins)
    b_moments = [_complex_normal(rng, pair_shape) for _ in range(4)]
    r0 = _complex_normal(rng, mode_shape)
    r1 = _complex_normal(rng, mode_shape)
    rho0, rho1 = r0.conj(), r1.conj()
    k0 = _complex_normal(rng, img_shape)
    k1 = _complex_normal(rng, img_shape)
    kbar0, kbar1 = k0.conj(), k1.conj()
    delays = rng.standard_normal(shape.n_img) * 1e-3  # seconds
    f_center = np.linspace(20.0, 1024.0, shape.n_bins)  # Hz
    return (b_moments, r0, r1, rho0, rho1, k0, k1, kbar0, kbar1, delays,
            f_center)


def time_callable(func, args: tuple, repeats: int) -> float:
    """Return the mean per-call wall time [s] over ``repeats`` calls."""
    timer = timeit.Timer(lambda: func(*args))
    return timer.timeit(number=repeats) / repeats


def profile(shape: ContractionShape, repeats: int, seed: int) -> dict:
    """Time both contraction terms and return a summary dict [seconds]."""
    rng = np.random.default_rng(seed)
    data_args = build_data_term_inputs(shape, rng)
    norm_args = build_norm_term_inputs(shape, rng)

    # One warm-up call each so einsum path selection / allocation caches do
    # not contaminate the first timed sample.
    likelihood._data_term(*data_args)
    likelihood._norm_term(*norm_args)

    data_seconds = time_callable(likelihood._data_term, data_args, repeats)
    norm_seconds = time_callable(likelihood._norm_term, norm_args, repeats)
    return {'data_term_s': data_seconds, 'norm_term_s': norm_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-modes', type=int, default=5,
                        help='Harmonic mode count n_m (default: 5).')
    parser.add_argument('--n-det', type=int, default=3,
                        help='Detector count n_det (default: 3).')
    parser.add_argument('--n-bins', type=int, default=256,
                        help='Relative-binning bin count n_bins '
                        '(default: 256).')
    parser.add_argument('--n-img', type=int, default=4,
                        help='Macro-image count n_img (default: 4).')
    parser.add_argument('--repeats', type=int, default=200,
                        help='Timed calls per term (default: 200).')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for synthetic inputs (default: 0).')
    args = parser.parse_args()

    shape = ContractionShape(n_modes=args.n_modes, n_det=args.n_det,
                             n_bins=args.n_bins, n_img=args.n_img)
    result = profile(shape, args.repeats, args.seed)

    data_ms = result['data_term_s'] * 1e3
    norm_ms = result['norm_term_s'] * 1e3
    total_ms = data_ms + norm_ms
    print(f'shape: n_modes={shape.n_modes} n_det={shape.n_det} '
          f'n_bins={shape.n_bins} n_img={shape.n_img} '
          f'(repeats={args.repeats}, seed={args.seed})')
    print(f'  _data_term : {data_ms:8.4f} ms/call  '
          f'({100 * data_ms / total_ms:5.1f}% of contraction)')
    print(f'  _norm_term : {norm_ms:8.4f} ms/call  '
          f'({100 * norm_ms / total_ms:5.1f}% of contraction)')
    print(f'  total      : {total_ms:8.4f} ms/call')
    dominant = 'norm_term' if norm_ms >= data_ms else 'data_term'
    print(f'  dominant   : {dominant}')


if __name__ == '__main__':
    main()
