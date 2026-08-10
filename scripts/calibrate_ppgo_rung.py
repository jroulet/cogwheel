#!/usr/bin/env python
"""Calibrate ppGO rung gates for ``cusp_amplification``.

Sweeps ppGO accuracy against the certified Pearcey arm across w for
representative cusp-window configs (both parities).  Outputs calibrated
``_W_PPGO_FLOOR`` and ``_R_PPGO_ERROR_CONST`` for use in
``_pearcey_cusp.py``.

Method
------
For each parity (astroid γ=0.5, saddle γ=1.2):
  - Get cusp vertices.
  - Generate source positions within ``_CUSP_ARM_COVERAGE`` rad of the
    vertex by stepping along the critical curve (positive and negative
    delta_theta) — this probes the soft-axis direction — and by offsetting
    the source along the hard axis.
  - At each source position, sweep w ∈ [3, 30] log-spaced.
  - Compute ``fold_ppgo_correction`` (ppGO) and ``cusp_amplification``
    (oracle).  Treat ``None`` from either side as a refusal.
  - For each direction, find the minimum w where relative error
    < ``envelope_bar / _PPGO_BAR_DIVISOR`` = 0.005.
  - The binding w_threshold = max of those minima across directions.

Then:
  ``_W_PPGO_FLOOR = w_threshold * 0.7``
  ``_R_PPGO_ERROR_CONST = (R^(3/2) * bar_ppgo / _UNIFORM_ERROR_CONST) * 0.5``
  where R is the minimum Pearcey control radius among directions that pass.

Usage
-----
    python scripts/calibrate_ppgo_rung.py
"""
from __future__ import annotations

import math

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
    cusp_amplification,
    use_pearcey_table,
    _UNIFORM_ERROR_CONST,
    _PPGO_BAR_DIVISOR,
    _DEFAULT_ENVELOPE_BAR,
    _soft_normal_form,
)
from cogwheel.lensing.chang_refsdal._airy_fold import fold_ppgo_correction
from cogwheel.lensing.surrogate import _CUSP_ARM_COVERAGE

_BAR_PPGO = _DEFAULT_ENVELOPE_BAR / _PPGO_BAR_DIVISOR  # 0.005

_N_W = 15
_W_RANGE = (3.0, 30.0)

_N_DELTA_THETA = 5


def _log_spaced_w(n):
    return np.logspace(math.log10(_W_RANGE[0]),
                       math.log10(_W_RANGE[1]), n)


def _astroid_cusp_vertices(gamma: float) -> list[tuple[float, int,
                                                       geometry.CriticalPoint]]:
    vertices = []
    for theta_cusp in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        try:
            vertex = geometry.critical_point(
                gamma, theta_cusp, beta=0.0, kappa=0.0, branch=1)
            vertices.append((theta_cusp, 1, vertex))
        except geometry.LensDomainError:
            pass
    return vertices


def _saddle_cusp_vertices(gamma: float) -> list[tuple[float, int,
                                                      geometry.CriticalPoint]]:
    vertices: list[tuple[float, int, geometry.CriticalPoint]] = []
    for theta_cusp in np.linspace(0, 2.0 * math.pi, 6, endpoint=False):
        for branch in (1, -1):
            try:
                vertex = geometry.critical_point(
                    gamma, theta_cusp, beta=0.0, kappa=0.0, branch=branch)
            except geometry.LensDomainError:
                continue
            normal_form = _soft_normal_form(
                vertex.image, geometry.macro_matrix(gamma, 0.0, 0.0),
                vertex.soft_axis, vertex.hard_axis, vertex.hard_eigenvalue)
            if normal_form is None:
                continue
            vertices.append((theta_cusp, branch, vertex))
    return vertices


def _pearcey_controls_at_vertex(
    vertex: geometry.CriticalPoint,
    source: np.ndarray,
    w: float,
    gamma: float,
) -> tuple[float, float] | None:
    offset = source - vertex.source
    delta_parallel = float(offset @ vertex.soft_axis)
    delta_perp = float(offset @ vertex.hard_axis)
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    normal_form = _soft_normal_form(vertex.image, matrix,
                                    vertex.soft_axis, vertex.hard_axis,
                                    vertex.hard_eigenvalue)
    if normal_form is None:
        return None
    c4, _phi_ssr = normal_form
    abs_c4 = abs(c4)
    x = delta_parallel * math.sqrt(w) / math.sqrt(abs_c4)
    y = delta_perp * w ** 0.75 / abs_c4 ** 0.25
    return x, y


def _sweep_source(w_values, source, gamma):
    errors = []
    for w in w_values:
        try:
            exact = cusp_amplification(w, source, gamma)
        except Exception:
            exact = None
        try:
            ppgo_raw = fold_ppgo_correction(w, source, gamma)
            ppgo = complex(ppgo_raw)
        except Exception:
            ppgo = None
        if exact is None or ppgo is None:
            errors.append(None)
            continue
        denom = abs(exact)
        if denom == 0.0:
            errors.append(None)
            continue
        err = abs(complex(ppgo) - exact) / denom
        errors.append(err)
    return errors


def _min_w_passing(errors, w_values):
    passing_w = [w_values[i] for i, e in enumerate(errors)
                 if e is not None and e < _BAR_PPGO]
    if not passing_w:
        return None
    return min(passing_w)


def _calibrate_parity(
    gamma: float,
    vertices: list[tuple[float, int, geometry.CriticalPoint]],
    label: str,
) -> dict:
    w_values = _log_spaced_w(_N_W)
    all_direction_results: list[tuple[str, float | None, float | None]] = []

    for theta_cusp, branch, vertex in vertices:
        for sign in (+1, -1):
            for i in range(1, _N_DELTA_THETA + 1):
                delta = _CUSP_ARM_COVERAGE * i / _N_DELTA_THETA

                # --- Soft-axis probing: step along critical curve ---
                try:
                    pt = geometry.critical_point(
                        gamma, theta_cusp + sign * delta,
                        beta=0.0, kappa=0.0, branch=branch)
                except geometry.LensDomainError:
                    continue
                source = pt.source
                errors = _sweep_source(w_values, source, gamma)
                min_w = _min_w_passing(errors, w_values)
                if min_w is not None:
                    controls = _pearcey_controls_at_vertex(
                        vertex, source, min_w, gamma)
                    radius = math.hypot(*controls) if controls else None
                else:
                    radius = None
                all_direction_results.append(
                    (f'soft_axis_sign={sign}_dtheta={delta:.4f}',
                     min_w, radius))

                # --- Hard-axis probing: offset from cusp source ---
                for h_sign in (+1, -1):
                    step_size = delta * 0.1
                    hard_source = (
                        vertex.source + h_sign * step_size
                        * vertex.hard_axis)
                    errors = _sweep_source(w_values, hard_source, gamma)
                    min_w = _min_w_passing(errors, w_values)
                    if min_w is not None:
                        controls = _pearcey_controls_at_vertex(
                            vertex, hard_source, min_w, gamma)
                        radius = math.hypot(*controls) if controls else None
                    else:
                        radius = None
                    all_direction_results.append(
                        (f'hard_axis_sign={h_sign}_step={step_size:.4f}',
                         min_w, radius))

    passing = [(lbl, mw, r)
               for lbl, mw, r in all_direction_results
               if mw is not None and r is not None]
    if not passing:
        return {'w_threshold': None, 'R': None,
                'n_directions': len(all_direction_results),
                'n_passing': 0}

    binding_w = max(mw for _, mw, _ in passing)
    binding_directions = [(lbl, mw, r)
                          for lbl, mw, r in passing
                          if mw == binding_w]
    min_r_at_binding = min(r for _, _, r in binding_directions)
    return {'w_threshold': binding_w, 'R': min_r_at_binding,
            'n_directions': len(all_direction_results),
            'n_passing': len(passing),
            'binding_directions': binding_directions}


def main():
    print(f'bar_ppgo = {_DEFAULT_ENVELOPE_BAR} / {_PPGO_BAR_DIVISOR} '
          f'= {_BAR_PPGO}')
    print(f'CUSP_ARM_COVERAGE = {_CUSP_ARM_COVERAGE} rad')
    print(f'w range = ({_W_RANGE[0]}, {_W_RANGE[1]}), n = {_N_W}')
    print()

    table_ok = use_pearcey_table()
    if not table_ok:
        print('WARNING: Pearcey table not loaded; using live quadrature '
              '(slower but equivalent).')

    # --- Astroid parity ---
    print('=== Astroid parity (gamma=0.5) ===')
    astroid_vertices = _astroid_cusp_vertices(0.5)
    print(f'  Cusp vertices: {len(astroid_vertices)}')
    result_astroid = _calibrate_parity(0.5, astroid_vertices, 'astroid')
    if result_astroid['w_threshold'] is None:
        print('  ERROR: No direction passed for astroid parity.')
    else:
        print(f'  w_threshold = {result_astroid["w_threshold"]:.2f}')
        print(f'  R = {result_astroid["R"]:.2f}')
        print(f'  directions probed = {result_astroid["n_directions"]}, '
              f'passing = {result_astroid["n_passing"]}')
    print()

    # --- Saddle parity ---
    print('=== Saddle parity (gamma=1.2) ===')
    saddle_vertices = _saddle_cusp_vertices(1.2)
    print(f'  Cusp vertices: {len(saddle_vertices)}')
    result_saddle = _calibrate_parity(1.2, saddle_vertices, 'saddle')
    if result_saddle['w_threshold'] is None:
        print('  WARNING: No direction passed for saddle parity.')
    else:
        print(f'  w_threshold = {result_saddle["w_threshold"]:.2f}')
        print(f'  R = {result_saddle["R"]:.2f}')
        print(f'  directions probed = {result_saddle["n_directions"]}, '
              f'passing = {result_saddle["n_passing"]}')
    print()

    # --- Combine ---
    results = [r for r in (result_astroid, result_saddle)
               if r['w_threshold'] is not None]
    if not results:
        print('ERROR: No parities produced a result.  Cannot calibrate.')
        return

    w_threshold = max(r['w_threshold'] for r in results)
    R_candidates = [r['R'] for r in results
                    if r['w_threshold'] == w_threshold]
    R = min(R_candidates) if R_candidates else results[0]['R']

    W_FLOOR = w_threshold * 0.7
    R_ERROR_CONST = (R ** (3.0 / 2.0) * _BAR_PPGO
                     / _UNIFORM_ERROR_CONST) * 0.5

    print('========================================')
    print('CALIBRATED CONSTANTS')
    print('========================================')
    print(f'_W_PPGO_FLOOR       = {W_FLOOR:.6f}')
    print(f'_R_PPGO_ERROR_CONST = {R_ERROR_CONST:.6f}')
    print()
    print('Derivation:')
    print(f'  binding w_threshold = {w_threshold:.2f}')
    print(f'  binding R = {R:.2f}')
    print(f'  _W_PPGO_FLOOR = {w_threshold:.2f} * 0.7 = {W_FLOOR:.2f}')
    print(f'  _R_PPGO_ERROR_CONST = ({R:.2f}^(3/2) * {_BAR_PPGO:.3f} '
          f'/ {_UNIFORM_ERROR_CONST}) * 0.5 = {R_ERROR_CONST:.2f}')


if __name__ == '__main__':
    main()
