#!/usr/bin/env python
"""Measure coverage hole from dropped gamma slivers (region 10).

Calls ``stable_gamma_bands`` over the full prior sub-range for each
parity with production ``min_width=1e-6``, sums dropped-sliver widths,
and reports the fraction relative to the full prior range [0, 1.6].

Professor domain assessment: zero topology metamorphoses exist in the
Chang-Refsdal model across the prior range, so the expected result is
dropped=[] for both parities (fraction=0.0) and region 10 closes.

A lightweight stability advisory reruns at n_samples=400 and warns if
the dropped list changes between resolutions.

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/measure_dropped_slivers.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogwheel.lensing.surrogate_training import stable_gamma_bands

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Prior sub-range for positive parity (gamma < 1).
BAND_POSITIVE: tuple[float, float] = (0.0, 0.999)

#: Prior sub-range for saddle parity (gamma > 1).
BAND_SADDLE: tuple[float, float] = (1.001, 1.6)

#: Production minimum sliver width (matching TrainingConfig default).
MIN_WIDTH: float = 1e-6

#: Full prior range denominator [0, 1.6].
PRIOR_RANGE: float = 1.6

#: Threshold for closing region 10.
CLOSE_THRESHOLD: float = 1e-3


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    """Measure and report dropped-sliver prior mass for both parities."""
    print("=" * 70)
    print("DROPPED-SLIVER COVERAGE MEASUREMENT (region 10)")
    print("=" * 70)
    print(f"  Positive parity band: {BAND_POSITIVE}")
    print(f"  Saddle parity band:   {BAND_SADDLE}")
    print(f"  min_width:            {MIN_WIDTH}")
    print(f"  Prior range:          [0, {PRIOR_RANGE}]")
    print(flush=True)

    total_dropped_width: float = 0.0
    all_dropped: list[tuple[str, tuple[float, float]]] = []

    for parity, band, label in [
        (1, BAND_POSITIVE, "positive"),
        (-1, BAND_SADDLE, "saddle"),
    ]:
        print(f"\n{'─' * 70}")
        print(f"  PARITY: {label}  band={band}")
        print(f"{'─' * 70}")

        # Primary measurement (n_samples=200)
        stable_200, dropped_200 = stable_gamma_bands(
            band, parity, n_samples=200, min_width=MIN_WIDTH)

        dropped_width = sum(hi - lo for lo, hi in dropped_200)
        total_dropped_width += dropped_width

        print(f"  n_stable_bands:     {len(stable_200)}")
        print(f"  n_dropped_slivers:  {len(dropped_200)}")
        print(f"  total_dropped_width:{dropped_width:.6f}")
        if dropped_200:
            print("  dropped slivers:")
            for lo, hi in dropped_200:
                print(f"    ({lo:.6f}, {hi:.6f})  width={hi - lo:.6f}")
        else:
            print("  dropped slivers: none")

        for sliver in dropped_200:
            all_dropped.append((label, sliver))

        # Stability advisory: rerun at n_samples=400
        _stable_400, dropped_400 = stable_gamma_bands(
            band, parity, n_samples=400, min_width=MIN_WIDTH)

        if dropped_400 != dropped_200:
            print(f"  *** STABILITY WARNING: dropped list differs at n=400 ***")
            print(f"      n=200: {dropped_200}")
            print(f"      n=400: {dropped_400}")
        else:
            print(f"  stability advisory (n=400): unchanged ✓")

    # Global summary
    fraction = total_dropped_width / PRIOR_RANGE

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print(f"  Total dropped width: {total_dropped_width:.6f}")
    print(f"  Prior range:         {PRIOR_RANGE}")
    print(f"  Dropped fraction:    {fraction:.2e}")
    print()

    if fraction < CLOSE_THRESHOLD:
        print(
            f"REGION 10 CLOSED: dropped-sliver prior mass = {fraction:.2e} < 1e-3"
        )
    else:
        print(
            f"REGION 10 OPEN: dropped-sliver prior mass = {fraction:.2e} >= 1e-3"
        )
        print("  Affected gamma intervals:")
        for label, (lo, hi) in all_dropped:
            midpoint = 0.5 * (lo + hi)
            print(f"    parity={label}  gamma in ({lo:.6f}, {hi:.6f})"
                  f"  midpoint={midpoint:.6f}")
        print("  Proposed fix: [reduce min_gamma_band | explicit sliver handling]")


if __name__ == "__main__":
    main()
