#!/usr/bin/env python
"""Step 4 driver measurement: far-zone crossover rho*.

Sweeps Born carrier relative error inward in |y| from the box corner,
per gamma (positive parity, gamma < 3/4), to find rho* where the carrier
becomes accurate enough.

Output: rho* per gamma, suitable for inlining into the step 5 (C8) brief.

COST ESTIMATE:
  - 5 gamma values × 20 |y| values × 4 theta angles = 400 evaluations
  - Each: one engine.evaluate + born_carrier_from_partition (~0.5-1s)
  - Sequential: ~5-7 minutes

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/measure_far_zone_crossover.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cogwheel.lensing.chang_refsdal import channels
from cogwheel.lensing.ppgo_map import caustic_rho

# Configuration
GAMMAS = np.array([0.05, 0.10, 0.15, 0.20, 0.25])  # positive parity only (gamma < 3/4)
Y_MAGNITUDES = np.linspace(2.0, 4.5, 20)  # |y| in Einstein-radius units
THETAS = np.array([0.0, np.pi/4, np.pi/2, np.pi])  # azimuthal angles
W_GRID = np.geomspace(0.01, 8.0, 200)
EPS_BAR = 0.01  # 1% relative error bar
KAPPA = 0.0


def main():
    print("=" * 70)
    print("FAR-ZONE CROSSOVER MEASUREMENT (Step 4)")
    print("=" * 70)
    print(f"  Gammas: {GAMMAS}")
    print(f"  |y| range: [{Y_MAGNITUDES[0]:.1f}, {Y_MAGNITUDES[-1]:.1f}] ({len(Y_MAGNITUDES)} points)")
    print(f"  Theta angles: {len(THETAS)}")
    print(f"  w grid: {len(W_GRID)} points in [{W_GRID[0]:.3f}, {W_GRID[-1]:.1f}]")
    print(f"  eps bar: {EPS_BAR}")
    print(flush=True)

    engine = channels.ChangRefsdalChannels(W_GRID)
    t_start = time.time()
    results = {}  # gamma -> list of (rho, max_err_over_theta)

    for gamma in GAMMAS:
        results[gamma] = []
        print(f"\n  gamma={gamma:.2f}:", flush=True)

        for y_mag in Y_MAGNITUDES:
            # Worst-case error over all theta angles
            max_err = 0.0
            for theta in THETAS:
                source = (y_mag * np.cos(theta), y_mag * np.sin(theta))
                try:
                    engine.reset()
                    partition = engine.evaluate(
                        gamma=gamma, y=source, beta=0.0, kappa=KAPPA)
                    carrier = channels.born_carrier_from_partition(partition)

                    exact = partition.exact_total
                    denom = np.max(np.abs(exact))
                    if denom < 1e-15:
                        continue
                    rel_err = np.max(np.abs(exact - carrier)) / denom
                    max_err = max(max_err, rel_err)
                except Exception as e:
                    pass  # skip failed points

            rho = caustic_rho(gamma, y_mag, kappa=KAPPA)
            results[gamma].append((rho, max_err))
            marker = " *" if max_err < EPS_BAR else ""
            print(f"    |y|={y_mag:.2f} rho={rho:.2f} err={max_err:.4f}{marker}",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # Find rho* per gamma
    print()
    print("=" * 70)
    print("CROSSOVER SUMMARY (rho* = smallest rho where err < bar)")
    print("=" * 70)

    rho_stars = []
    for gamma in GAMMAS:
        pairs = results[gamma]
        # Find smallest rho where error < bar (scanning from far to near)
        passing = [(rho, err) for rho, err in pairs if err < EPS_BAR and err > 0]
        if passing:
            rho_star = min(rho for rho, _ in passing)
            rho_stars.append(rho_star)
            print(f"  gamma={gamma:.2f}: rho* = {rho_star:.2f}")
        else:
            print(f"  gamma={gamma:.2f}: carrier never accurate enough (min err="
                  f"{min(err for _, err in pairs if err > 0):.4f})")

    if rho_stars:
        print(f"\n  OVERALL: rho* = {max(rho_stars):.2f} "
              f"(worst-case gamma, use this for C8)")


if __name__ == "__main__":
    main()
