#!/usr/bin/env python
"""Step 9: Train the full surrogate (smoke-scale proof-of-concept).

Runs the training pipeline with default TrainingConfig (smoke-scale grids)
to verify the pipeline works end-to-end in final coordinates. The artifact
is saved to /tmp/surrogate_smoke/.

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/train_surrogate.py
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogwheel.lensing.surrogate_training import train, TrainingConfig

OUTDIR = "/tmp/surrogate_smoke"

def main():
    print("=" * 70)
    print("SURROGATE TRAINING (smoke-scale proof-of-concept)")
    print("=" * 70)
    config = TrainingConfig()
    print(f"  Config: n_gamma={config.n_gamma}, n_u={config.n_u}, "
          f"n_theta={config.n_theta}")
    print(f"  gamma_band_halfwidth={config.gamma_band_halfwidth}")
    print(f"  Output: {OUTDIR}")
    print(flush=True)

    t0 = time.time()
    surrogate, report = train(
        outdir=OUTDIR,
        config=config,
        report_path=f"{OUTDIR}/training_report.json",
    )
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Charts: {len(surrogate.charts)}")
    print(f"  Artifact: {OUTDIR}/lens_amplification_surrogate.npz")
    print()

    # Summary from report
    for label, info in report.get('parities', {}).items():
        print(f"  {label}:")
        print(f"    bands: {info.get('n_stable_sub_bands')}")
        print(f"    dropped slivers: {len(info.get('dropped_gamma_slivers', []))}")

    print(flush=True)


if __name__ == "__main__":
    main()
