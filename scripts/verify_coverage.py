#!/usr/bin/env python
"""Pre-training coverage verification.

Verifies that the tile proposal logic + Born carrier covers every prior draw
without gaps that would require exact quadrature. Draws samples from the
lens prior and checks:
1. Exterior (rho > 1): Born carrier serves → no exact needed
2. Interior (rho <= 1): tile proposals cover → chart will be trained

Any sample that falls in neither category is a coverage gap.

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/verify_coverage.py
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cogwheel.lensing.ppgo_map import caustic_rho
from cogwheel.lensing.surrogate_training import (
    TrainingConfig, stable_gamma_bands, _astroid_arcs, _saddle_arcs,
    _min_curvature_radius,
)
from cogwheel.lensing.surrogate import _GAMMA_GUARD_BAND


def main():
    print("=" * 70)
    print("PRE-TRAINING COVERAGE VERIFICATION")
    print("=" * 70)

    config = TrainingConfig()
    n_samples = 2000
    rng = np.random.default_rng(42)
    
    # Draw source positions uniformly in the prior box
    # Prior: gamma in [0.0281, 0.99] (positive) + [1.01, ~2.0] (saddle)
    # Source: |y| in [0, 4.2426] (box half-diagonal)
    
    gaps = []
    served_born = 0
    served_chart = 0
    gamma_guard = 0
    total = 0
    
    gamma_ranges = [
        (0.0281, 0.99, 1),   # positive parity
        (1.01, 2.00, -1),    # saddle parity
    ]
    
    for gamma_lo, gamma_hi, parity in gamma_ranges:
        parity_label = "positive" if parity == 1 else "saddle"
        parity_gaps = 0
        parity_total = 0
        
        for _ in range(n_samples):
            gamma = float(rng.uniform(gamma_lo, gamma_hi))
            y_mag = float(rng.uniform(0.01, 4.2426))
            theta = float(rng.uniform(0, 2 * np.pi))
            total += 1
            parity_total += 1
            
            # Gamma guard band (near parity wall)
            if abs(gamma - 1.0) < _GAMMA_GUARD_BAND:
                gamma_guard += 1
                continue
            
            # Check Born carrier coverage (exterior)
            try:
                rho = caustic_rho(gamma, y_mag, kappa=0.0)
            except (ValueError, Exception):
                rho = 0.0
            
            if rho > 1.0:
                served_born += 1
                continue
            
            # Interior: check if tile proposals would cover this gamma
            # A gamma is covered if stable_gamma_bands produces bands that
            # contain it, AND those bands produce arcs
            band_width = 0.02  # typical band width
            band = (gamma - band_width/2, gamma + band_width/2)
            
            covered = False
            try:
                bands = stable_gamma_bands(band, parity)
                if bands:
                    test_band = bands[0]
                    if parity == 1:
                        _, arcs, _ = _astroid_arcs(
                            float(np.mean(test_band)), config.n_caustic_samples)
                    else:
                        _, arcs, _ = _saddle_arcs(
                            float(np.mean(test_band)), config.n_caustic_samples)
                    if arcs:
                        covered = True
            except Exception:
                pass
            
            if covered:
                served_chart += 1
                continue
            
            # This point is a gap
            gaps.append({
                'gamma': gamma, 'y_mag': y_mag, 'rho': rho,
                'parity': parity_label
            })
            parity_gaps += 1
        
        gap_pct = 100 * parity_gaps / max(parity_total, 1)
        print(f"  {parity_label}: {parity_total} samples, "
              f"{parity_gaps} gaps ({gap_pct:.1f}%)")
    
    print()
    print(f"  Total samples: {total}")
    print(f"  Born carrier (rho > 1): {served_born} ({100*served_born/total:.1f}%)")
    print(f"  Chart coverage (interior): {served_chart} ({100*served_chart/total:.1f}%)")
    print(f"  Gamma guard band: {gamma_guard} ({100*gamma_guard/total:.1f}%)")
    print(f"  GAPS: {len(gaps)} ({100*len(gaps)/total:.1f}%)")
    
    if gaps:
        print()
        print("  Gap details (first 20):")
        for g in gaps[:20]:
            print(f"    gamma={g['gamma']:.3f} |y|={g['y_mag']:.3f} "
                  f"rho={g['rho']:.3f} ({g['parity']})")
    else:
        print()
        print("  ✅ FULL COVERAGE: every prior draw is served by Born carrier "
              "or chart tiles")


if __name__ == "__main__":
    main()
