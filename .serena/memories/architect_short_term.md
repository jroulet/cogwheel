# Architect Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-07)

2026-08-07 fix_tree_gate_hang: 2-WP housekeeping build. The 4 originally-diagnosed offenders are already fixed (mass-capped or tier-gated). Two remaining issues: (1) agents.py missing COGWHEEL_TRAIN_TIER="" in build env → train-tier tests un-skip if env leaks; (2) pytest-timeout not installed → conftest's 900s timeout a no-op. WP1 installs pytest-timeout + pins the env var. WP2 adds _f_schwinger_mpmath sentinel guard to conftest.py. No domain changes, no new tests.

2026-08-07 remeasure-v3-recursion: Single-WP build. Adding `regions` filter to `_train_band_charts`/`train()`/CLI — the TODO fragment `lensing_training_path_cannot_be_run_per_region.md` explicitly asks for this. No domain changes. Post-build: driver runs wedge-v3 and exterior-recursion probes using the now-filterable production path, records numbers; doc-sync writes completed.d fragment.
