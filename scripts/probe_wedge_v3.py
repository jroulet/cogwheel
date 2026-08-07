#!/usr/bin/env python
"""Re-measure the interior wedge charts under schema v3 (post-build driver probe).

The 2026-08-06 coordinate probes measured an 18-chart / median 5.47e-4 /
~10.5 min interior result against v2 charts, which hard-refuse under v3.
This probe re-runs the interior wedge training against the CURRENT code
(v3 schema) via the production `regions=('wedge_interior',)` path, and
reports the chart count, held-out eps distribution, and runtime so the
numbers can be re-quoted with a v3 baseline.

Usage (driver, post-build — slow tier enabled):
    COGWHEEL_TRAIN_TIER=1 python scripts/probe_wedge_v3.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogwheel.lensing.surrogate_training import TrainingConfig, train
from dataclasses import replace


def main():
    # Production-scale config (matches the 2026-08-06 probes' interior grids).
    config = replace(
        TrainingConfig(),
        n_gamma=7, n_u=7, n_theta=7,
        n_rho=7, n_theta_c=7,
        w_nodes_per_decade=15,
        interior_w_nodes_per_decade=15,
        engine_budget=2000,
        gamma_band_halfwidth=0.48,  # wide band: interior probe spans the astroid
        n_heldout=30,
        max_farfield_regions=None,  # far-field excluded via regions= filter
    )
    outdir = "/tmp/probe_wedge_v3"
    os.makedirs(outdir, exist_ok=True)

    print(f"Probe: wedge_interior, config n_gamma={config.n_gamma}, "
          f"w_nodes_per_decade={config.w_nodes_per_decade}", flush=True)
    t0 = time.time()
    surrogate, report = train(
        outdir=outdir,
        config=config,
        report_path=f"{outdir}/report.json",
        regions=("wedge_interior",),
    )
    elapsed = time.time() - t0

    n_charts = len(surrogate.charts)
    eps_values = []
    for ch in surrogate.charts:
        prov = getattr(ch, "provenance", {}) or {}
        eps = prov.get("heldout_eps")
        if eps is not None:
            eps_values.append(eps)
    eps_values.sort()
    import statistics
    median = statistics.median(eps_values) if eps_values else float("nan")

    print(f"\n=== WEDGE v3 PROBE RESULT ===")
    print(f"  charts:        {n_charts}")
    print(f"  held-out eps:  median={median:.3e}  n={len(eps_values)}")
    print(f"  eps range:     [{eps_values[0]:.3e}, {eps_values[-1]:.3e}]"
          if eps_values else "  eps range:     (none)")
    print(f"  runtime:       {elapsed/60:.1f} min")
    print(f"  artifact:      {outdir}")
    print(f"\n  Baseline (v2, 2026-08-06): 18 charts, median 5.47e-4, ~10.5 min")
    if n_charts == 18:
        print(f"  v3 chart count matches v2 baseline: 18")
    else:
        print(f"  v3 chart count DIFFERS from v2 baseline (18): {n_charts}")


if __name__ == "__main__":
    main()
