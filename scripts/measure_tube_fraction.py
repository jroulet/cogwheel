#!/usr/bin/env python
"""Step 2 driver measurement: tube fraction sweep.

Sweeps held-out envelope eps against the dimensionless fraction eta/R_c
across gamma, both parities, to find f_max (where eps crosses tube_eps_max
= 0.05) and f_floor (the inner boundary).

Output: prints f_max and f_floor suitable for inlining into the step 3 brief.

COST ESTIMATE:
  - 14 gamma values x 8 fractions = 112 chart builds
  - Sequential (numba uses all cores internally per chart)
  - First chart: ~30s (JIT warmup); subsequent: ~3-5s each
  - Total wall time: ~30s + 111*4s ≈ 8 minutes

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/measure_tube_fraction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from cogwheel.lensing.surrogate_training import (
    PriorBox, TrainingConfig, _min_curvature_radius, _astroid_arcs,
    _saddle_arcs, _build_tube_chart, _heldout_eps, _tube_heldout_samples,
    _capped_w_range,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAMMA_POSITIVE = np.linspace(0.03, 0.28, 7)
GAMMA_SADDLE = np.linspace(1.05, 1.80, 7)
FRACTION_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
EPS_BAR = 0.05
N_CAUSTIC_SAMPLES = 200

#: Provenance dict for tube chart held-out eps (matches production).
_TUBE_PROV = {'envelope_definition': 'kernel_sum'}

#: Base training config (smoke-scale grids, fast builds).
_BASE_CONFIG = TrainingConfig(
    n_gamma=5, n_u=5, n_theta=5, w_nodes_per_decade=3,
    n_heldout=30, n_caustic_samples=N_CAUSTIC_SAMPLES,
)


def _get_arc_and_reach(gamma: float, parity: int, n: int):
    """Get the first fold arc and caustic reach for the given gamma/parity."""
    if parity == 1:
        _cusps, arcs, reach = _astroid_arcs(gamma, n)
    else:
        _cusps, arcs, reach = _saddle_arcs(gamma, n)
    if not arcs:
        return None, None
    return arcs[0], reach


def measure_one(gamma: float, parity: int, f: float):
    """Measure held-out eps for one (gamma, parity, fraction) triple.

    Returns (gamma, parity, f, eps_or_None, error_string_or_None).
    """
    try:
        n = N_CAUSTIC_SAMPLES
        arc, reach = _get_arc_and_reach(gamma, parity, n)
        if arc is None:
            return gamma, parity, f, None, "no arcs"

        # Narrow gamma band centred on the test gamma (thin enough that
        # curvature varies little, wide enough that n_gamma>=5 fits).
        band = (gamma - 0.01, gamma + 0.01)

        # Minimum curvature radius over this arc/band.
        r_c = _min_curvature_radius(band, arc, n)

        # Tube eta_max from fraction; floor fixed at 20% of eta_max.
        eta_max = f * r_c
        eta_floor = 0.2 * eta_max

        # Sanity bounds: skip if eta is unphysically large or tiny.
        if eta_max < 0.002 or eta_max > 1.0:
            return gamma, parity, f, None, (
                f"eta_max={eta_max:.4f} out of range (R_c={r_c:.4f})")

        config = _BASE_CONFIG

        # Compute w_range using a physical prior box (same as the test file).
        box = PriorBox.from_prior_classes()
        y_max = reach + eta_max
        w_range = _capped_w_range(box, parity, y_max)

        # Gamma grid over the narrow band.
        gamma_grid = np.linspace(*band, config.n_gamma)

        # Build the tube chart.
        chart, _calls, _refused = _build_tube_chart(
            gamma_grid=gamma_grid, arc=arc, parity=parity,
            w_range=w_range, config=config,
            eta_max=eta_max, eta_floor=eta_floor)

        # Generate held-out samples and compute eps.
        rng = np.random.default_rng(42)
        samples = _tube_heldout_samples(band, arc, config, rng,
                                         eta_max=eta_max, eta_floor=eta_floor)
        eps = _heldout_eps(chart, samples, _TUBE_PROV)

        return gamma, parity, f, eps, None
    except Exception as e:
        return gamma, parity, f, None, str(e)[:120]


def main():
    print("=" * 70)
    print("TUBE FRACTION MEASUREMENT SWEEP")
    print("=" * 70)
    print(f"  Positive-parity gammas: {GAMMA_POSITIVE}")
    print(f"  Saddle-parity gammas:   {GAMMA_SADDLE}")
    print(f"  Fraction grid (eta/R_c): {FRACTION_GRID}")
    print(f"  Sequential (numba uses all cores internally)")
    print(f"  eps bar: {EPS_BAR}")
    print()

    # Build task list.
    tasks = []
    for gamma in GAMMA_POSITIVE:
        for f in FRACTION_GRID:
            tasks.append((float(gamma), 1, float(f)))
    for gamma in GAMMA_SADDLE:
        for f in FRACTION_GRID:
            tasks.append((float(gamma), -1, float(f)))

    print(f"  Total tasks: {len(tasks)}")
    print(flush=True)

    # Run sequentially — numba parallelizes internally per chart build.
    results = []
    t_start = time.time()
    for i, (gamma, parity, f) in enumerate(tasks):
        t0 = time.time()
        gamma_val, parity_val, f_val, eps, err = measure_one(gamma, parity, f)
        dt = time.time() - t0
        if eps is not None and not np.isnan(eps):
            results.append({'gamma': gamma_val, 'parity': parity_val,
                            'f': f_val, 'eps': eps})
            marker = " <<<OVER BAR" if eps > EPS_BAR else ""
            print(f"  [{i+1:3d}/{len(tasks)}] gamma={gamma:.3f} p={parity:+d} "
                  f"f={f:.1f} eps={eps:.4f} ({dt:.1f}s){marker}", flush=True)
        else:
            print(f"  [{i+1:3d}/{len(tasks)}] gamma={gamma:.3f} p={parity:+d} "
                  f"f={f:.1f} SKIP: {err} ({dt:.1f}s)", flush=True)
    
    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if not results:
        print("No results — all configurations failed.")
        sys.exit(1)

    # f_max: largest f where ALL gammas (both parities) have eps < EPS_BAR.
    f_max = None
    for f in reversed(sorted(FRACTION_GRID)):
        eps_at_f = [r['eps'] for r in results
                    if r['f'] == f and not np.isnan(r['eps'])]
        if eps_at_f and max(eps_at_f) < EPS_BAR:
            f_max = f
            print(f"  f_max = {f:.2f}  "
                  f"(largest f with max(eps)={max(eps_at_f):.4f} < {EPS_BAR})")
            break
    if f_max is None:
        print("  f_max: NO fraction keeps all gammas below bar!")
        for f in sorted(FRACTION_GRID):
            eps_at_f = [r['eps'] for r in results
                        if r['f'] == f and not np.isnan(r['eps'])]
            if eps_at_f:
                print(f"    f={f:.2f}: max(eps)={max(eps_at_f):.4f}  "
                      f"n_valid={len(eps_at_f)}")

    # f_floor: smallest f where the tube still has meaningful content
    # (min eps > 1e-3 means even the cleanest gamma is non-trivial).
    f_floor = None
    for f in sorted(FRACTION_GRID):
        eps_at_f = [r['eps'] for r in results
                    if r['f'] == f and not np.isnan(r['eps'])]
        if eps_at_f:
            f_floor = f
            print(f"  f_floor = {f:.2f}  "
                  f"(smallest measured f, min(eps)={min(eps_at_f):.4f})")
            break
    if f_floor is None:
        print("  f_floor: could not determine (no valid measurements)")

    print()
    print("-" * 70)
    print("PER-GAMMA DETAIL")
    print("-" * 70)
    for parity in [1, -1]:
        p_label = "positive (astroid)" if parity == 1 else "saddle (deltoid)"
        print(f"\n  --- {p_label} ---")
        gammas = sorted(set(r['gamma'] for r in results
                            if r['parity'] == parity))
        for gamma in gammas:
            subset = [r for r in results
                      if r['gamma'] == gamma and r['parity'] == parity]
            line = f"  gamma={gamma:.4f}:"
            for r in sorted(subset, key=lambda x: x['f']):
                marker = "*" if r['eps'] > EPS_BAR else " "
                line += f"  {r['f']:.1f}->{r['eps']:.4f}{marker}"
            print(line)

    print()
    print("=" * 70)
    if f_max is not None:
        print(f"CONCLUSION: f_max={f_max:.2f}, f_floor={f_floor:.2f}")
    else:
        print("CONCLUSION: No safe fraction found (all exceed eps bar)")
    print("=" * 70)


if __name__ == "__main__":
    main()
