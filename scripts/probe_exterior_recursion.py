#!/usr/bin/env python
"""Measure exterior recursion effectiveness (post-build driver probe).

Baseline (2026-08-06): 84% of exterior charts were subdivision children and
35 of 57 failed the 1e-3 bar. Hypothesis: every marginal tile got one
halving and was then abandoned. This probe trains ONE exterior band with
recursion live and reports (i) how many of the previously-failing charts
now clear 1e-3, (ii) the achieved-depth histogram, (iii) whether any tile
hits the depth-3 cap (a cap hit = coordinate problem, not cap too low).

Usage (driver, post-build — slow tier enabled):
    COGWHEEL_TRAIN_TIER=1 python scripts/probe_exterior_recursion.py
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogwheel.lensing.surrogate_training import TrainingConfig, train
from dataclasses import replace


def main():
    config = replace(
        TrainingConfig(),
        n_gamma=7, n_u=7, n_theta=7,
        n_rho=7, n_theta_c=7,
        w_nodes_per_decade=15,
        engine_budget=2000,
        gamma_band_halfwidth=0.04,  # one exterior band (matches 2026-08-06)
        n_heldout=30,
    )
    outdir = "/tmp/probe_exterior_recursion"
    os.makedirs(outdir, exist_ok=True)

    print(f"Probe: exterior (one band), recursion live", flush=True)

    # Progress stream: count chart files as they are written.
    import threading

    def _watch(outdir, stop):
        last = -1
        while not stop.is_set():
            try:
                n = len([f for f in os.listdir(outdir)
                         if f.endswith('.npz')])
            except FileNotFoundError:
                n = 0
            if n != last:
                print(f"  [beat] {n} chart(s) written", flush=True)
                last = n
            stop.wait(15)

    _stop = threading.Event()
    _w = threading.Thread(target=_watch, args=(outdir, _stop), daemon=True)
    _w.start()

    t0 = time.time()
    surrogate, report = train(
        outdir=outdir,
        config=config,
        report_path=f"{outdir}/report.json",
        regions=("exterior",),
    )
    elapsed = time.time() - t0
    _stop.set()
    _w.join(timeout=1)

    # Report: charts, eps distribution, achieved depth histogram.
    n_charts = len(surrogate.charts)
    eps_values = []
    depths = Counter()
    cap_hits = 0
    for ch in surrogate.charts:
        prov = getattr(ch, "provenance", {}) or {}
        eps = prov.get("heldout_eps")
        if eps is not None:
            eps_values.append(eps)
        depth = prov.get("subdivision_depth")
        if depth is not None:
            depths[depth] += 1
        if depth is not None and depth >= 3:
            cap_hits += 1

    eps_values.sort()
    import statistics
    n_pass = sum(1 for e in eps_values if e <= 1e-3)
    median = statistics.median(eps_values) if eps_values else float("nan")

    print(f"\n=== EXTERIOR RECURSION PROBE RESULT ===")
    print(f"  charts:            {n_charts}")
    print(f"  pass 1e-3 bar:     {n_pass}/{len(eps_values)}")
    print(f"  median eps:        {median:.3e}")
    print(f"  achieved depths:   {dict(depths)}")
    print(f"  depth-3 cap hits:  {cap_hits}")
    print(f"  runtime:           {elapsed/60:.1f} min")
    print(f"\n  Baseline (2026-08-06): 57 charts, 35 failed 1e-3, "
          f"84% subdivision children")
    if cap_hits:
        print(f"  WARNING: depth-3 cap hits — evidence the (s,d) coordinate is "
              f"wrong, route to polar re-chart.")


if __name__ == "__main__":
    main()
