#!/usr/bin/env python
"""Measure exterior recursion effectiveness (post-build driver probe).

Post-polar-rechart probe. Trains ONE exterior band with recursion live
in (rho, theta_c) coordinates. Reports (i) how many charts clear 1e-3,
(ii) achieved-depth histogram, (iii) whether any tile hits the depth-3
cap (cap hit = coordinate / subdivision problem).

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
        n_gamma=4, n_u=4, n_theta=4,
        n_rho=4, n_theta_c=4,
        w_nodes_per_decade=8,
        engine_budget=200,
        gamma_band_halfwidth=0.04,
        n_heldout=100,
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
    if cap_hits:
        print(f"  WARNING: depth-3 cap hits — coordinate or subdivision "
              f"problem in polar (rho, theta_c).")


if __name__ == "__main__":
    main()
