#!/usr/bin/env python
"""
Train a Born residual chart for the far exterior (rho > 2).

Computes R(w; gamma, rho) = F_exact_demod(w) - F_carrier_demod(w) on a
sparse (gamma, rho, log_w) tensor-product grid and stores the result as
a BornResidualChart artifact at cogwheel/data/born_residual_chart.npz.

The exact total comes from ChangRefsdalChannels.evaluate (already in the
min-relative-delay frame), and the carrier from born_lead_carrier
(absolute frame, demodulated here by exp(-1j*w*t_min)).

Grid:
    gamma : 7 values log-spaced in (0.05, 0.9), positive parity
    rho   : 5 values in (2.0, 4.0), far exterior
    w     : 10 values log-spaced in (5, 60)
    theta : pi/4 fixed (source-plane polar angle)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is importable.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from cogwheel.lensing.born_residual_chart import (
    _SCHEMA, BornResidualChart, _content_hash)
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal._born import born_lead_carrier
from cogwheel.lensing.chang_refsdal.geometry import r_caustic


def main() -> None:
    """Build, evaluate, and save the Born residual chart to disk.

    Constructs the tensor-product grid, calls ``ChangRefsdalChannels`` for
    the exact total and ``born_lead_carrier`` for the carrier at every grid
    point, computes the residual ``R = F_exact - F_carrier_demod``, assembles
    a ``BornResidualChart``, runs a self-check, and writes the result to
    ``cogwheel/data/born_residual_chart.npz``.
    """
    # --- Grid definition ---
    gamma_grid = np.geomspace(0.05, 0.9, 7)
    rho_grid = np.linspace(2.0, 4.0, 5)
    w_grid = np.geomspace(5.0, 60.0, 10)
    log_w_grid = np.log(w_grid)

    theta = np.pi / 4.0  # fixed source-plane polar angle

    n_gamma = len(gamma_grid)
    n_rho = len(rho_grid)
    n_w = len(w_grid)

    real_coeffs = np.empty((n_gamma, n_rho, n_w), dtype=float)
    imag_coeffs = np.empty((n_gamma, n_rho, n_w), dtype=float)

    print(f'Training Born residual chart: '
          f'{n_gamma} gamma x {n_rho} rho x {n_w} w = '
          f'{n_gamma * n_rho} evaluations')
    print(f'  gamma in [{gamma_grid[0]:.4f}, {gamma_grid[-1]:.4f}]')
    print(f'  rho   in [{rho_grid[0]:.2f}, {rho_grid[-1]:.2f}]')
    print(f'  w     in [{w_grid[0]:.1f}, {w_grid[-1]:.1f}]')
    print()

    t0 = time.perf_counter()

    engine = ChangRefsdalChannels(w_grid)

    for i_g, gamma in enumerate(gamma_grid):
        for i_r, rho in enumerate(rho_grid):
            # Source position: y = rho * r_caustic(gamma, theta) * direction
            rc = r_caustic(gamma, theta)
            r_source = rho * rc
            source = np.array([r_source * np.cos(theta),
                               r_source * np.sin(theta)])

            # Exact total in min-relative-delay frame.
            engine.reset()
            partition = engine.evaluate(gamma=gamma, y=source)
            F_exact = partition.exact_total  # shape (n_w,), demodulated
            t_min = partition.t_min

            # Carrier in absolute frame -> demodulate.
            F_carrier = np.array([
                born_lead_carrier(float(w), source[0], source[1], gamma)
                for w in w_grid
            ], dtype=complex)
            F_carrier_demod = F_carrier * np.exp(-1j * w_grid * t_min)

            # Residual.
            R = F_exact - F_carrier_demod

            real_coeffs[i_g, i_r, :] = R.real
            imag_coeffs[i_g, i_r, :] = R.imag

            elapsed = time.perf_counter() - t0
            print(f'  gamma={gamma:.4f}  rho={rho:.2f}  '
                  f'|R|_max={np.max(np.abs(R)):.4e}  '
                  f'[{elapsed:.1f}s]')

    elapsed = time.perf_counter() - t0
    print(f'\nAll grid points computed in {elapsed:.1f}s')

    # --- Build and save the chart ---
    provenance = {
        'driver': 'scripts/train_born_residual.py',
        'theta': theta,
        'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'grid_shape': [n_gamma, n_rho, n_w],
        'max_abs_residual': float(np.max(np.abs(
            real_coeffs + 1j * imag_coeffs))),
    }

    chart = BornResidualChart(
        gamma_grid=gamma_grid,
        rho_grid=rho_grid,
        log_w_grid=log_w_grid,
        real_coeffs=real_coeffs,
        imag_coeffs=imag_coeffs,
        provenance=provenance,
    )

    # Verify the chart is functional.
    test_R = chart.evaluate(w_grid, gamma_grid[0], rho_grid[0])
    print(f'Self-check: chart.evaluate at corner -> |R|_max = '
          f'{np.max(np.abs(test_R)):.4e}')

    # Save to disk.  `content_hash` pins the numeric payload and `schema`
    # tags the artifact framing so `BornResidualChart.load` can hard-refuse
    # a stale/corrupt file (mirrors CertifiedPpgoMap in ppgo_map.py).
    out_path = _project_root / 'cogwheel' / 'data' / 'born_residual_chart.npz'
    content_hash = _content_hash(chart.gamma_grid, chart.rho_grid,
                                 chart.log_w_grid, chart.real_coeffs,
                                 chart.imag_coeffs)
    np.savez(
        out_path,
        gamma_grid=chart.gamma_grid,
        rho_grid=chart.rho_grid,
        log_w_grid=chart.log_w_grid,
        real_coeffs=chart.real_coeffs,
        imag_coeffs=chart.imag_coeffs,
        provenance=np.array(json.dumps(chart.provenance)),
        content_hash=np.array(content_hash),
        schema=np.array(_SCHEMA),
    )
    print(f'\nSaved chart to {out_path}')
    print(f'  File size: {out_path.stat().st_size} bytes')
    print(f'  Provenance: {provenance}')


if __name__ == '__main__':
    main()
