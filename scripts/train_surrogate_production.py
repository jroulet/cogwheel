#!/usr/bin/env python
"""Production surrogate training — DD band (w <= 60).

Trains the full-prior surrogate with production-scale grids on the DD
frequency band only (w <= 60, both parities). The mpmath band (w > 60,
saddle only) is deferred to Phase 2.

Run on neso (64 uncontended cores):
    ssh neso "nohup bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && \
      conda activate cogwheel-newlal && \
      cd ~/Work/cogwheel-claude-dev && \
      python scripts/train_surrogate_production.py' \
      > /tmp/train_production.log 2>&1 &"

ETA: ~2.5 hours on 64 cores (6.7M evals @ 90ms/eval, parallelized).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogwheel.lensing.surrogate_training import train, TrainingConfig

OUTDIR = "/tmp/surrogate_production_dd"


def main():
    print("=" * 70)
    print("PRODUCTION SURROGATE TRAINING — DD band (w <= 60)")
    print("=" * 70)

    config = TrainingConfig(
        # Production grid sizing
        n_gamma=7,
        n_u=7,
        n_theta=7,
        n_rho=7,
        n_theta_c=7,
        w_nodes_per_decade=15,
        interior_w_nodes_per_decade=15,
        # Tube shell (curvature-relative, C6)
        f_floor=0.16,
        f_max=0.40,
        # Far-field
        farfield_overlap=0.05,
        n_farfield_tiles_per_side=5,
        max_farfield_regions=None,  # uncapped
        # Gamma bands
        gamma_band_halfwidth=0.02,
        min_gamma_band=1e-6,
        # Registration gates
        tube_eps_max=5e-2,
        farfield_eps_max=1e-3,
        interior_eps_max=5e-2,
        # Near-parity-wall refinement
        gamma_refine_near_one_window=0.15,
        gamma_refine_near_one_width=0.05,
        # Budget per chart (production: generous)
        engine_budget=2000,
        # Held-out
        n_heldout=50,
        n_caustic_samples=500,
        seed=2026,
    )

    print(f"  n_gamma={config.n_gamma}, n_u={config.n_u}, "
          f"n_theta={config.n_theta}, n_rho={config.n_rho}")
    print(f"  w_nodes_per_decade={config.w_nodes_per_decade}")
    print(f"  gamma_band_halfwidth={config.gamma_band_halfwidth}")
    print(f"  engine_budget={config.engine_budget}")
    print(f"  Output: {OUTDIR}")
    print(flush=True)

    t0 = time.time()

    # Progress callback: print each chart as it's built.
    import functools
    from pathlib import Path

    _chart_count = [0]
    _orig_train_band = None

    def _progress_wrapper(original_fn):
        """Wrap _train_band_charts to print progress per band."""
        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            _chart_count[0] += 1
            band = args[0] if args else kwargs.get('band', '?')
            parity = args[1] if len(args) > 1 else kwargs.get('parity', '?')
            elapsed_min = (time.time() - t0) / 60
            print(f"  [{elapsed_min:6.1f}m] Band {_chart_count[0]}: "
                  f"gamma={band}, parity={parity}", flush=True)
            return original_fn(*args, **kwargs)
        return wrapper

    # Monkey-patch for progress visibility
    import cogwheel.lensing.surrogate_training as _st
    if hasattr(_st, '_train_band_charts'):
        _st._train_band_charts = _progress_wrapper(_st._train_band_charts)

    surrogate, report = train(
        outdir=OUTDIR,
        config=config,
        report_path=f"{OUTDIR}/training_report.json",
    )
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"TRAINING COMPLETE in {elapsed/3600:.2f} hours")
    print(f"  Charts: {len(surrogate.charts)}")
    print(f"  Artifact: {OUTDIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
