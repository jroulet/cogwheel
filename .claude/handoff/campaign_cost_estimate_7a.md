# 7a training campaign — recorded cost estimate (2026-08-15, pre-launch)

Standing rule satisfied: NO engine-run launch without a recorded estimate.

Two INDEPENDENT estimates, agreeing to 0.4%:

1. Engine-free tiling census at production config, post-F081/trim (HEAD
   c661d62 era; JSON at `.claude/handoff/tiling_census_production_postF081
   .json`): 2,016,000 engine labels, modeled 181,440 s serial.
   Per (region x parity): tube:+1 10,290 nodes / exterior:+1 205,800 /
   wedge_interior:+1 6,300 / tube:-1 20,580 / lobe_interior:-1 2,730 /
   lobe_exterior:-1 6,300 (x8 labels/node). Astroid exterior = 82% of
   the budget. Deltoid far-field EXCLUDED (standing Q2 redesign verdict,
   fragment `lensing_deltoid_farfield_coordinate_redesign`).
2. Smoke-run currency (this morning, post lobe-cusp-edge fix, end-to-end
   green: 248 packed charts, 586 built, 36,672 engine calls in 3,311.8 s
   = 0.0903 s/call): 2,016,000 x 0.0903 = 182,000 s.

VERDICT: ~50.5 h single-process serial (the trainer has no parallelism;
grep confirms no Pool/n_jobs). Launched as ONE monitored process
(monitored-not-unattended: terminal Monitor + stale alarm + push on
completion), OUTDIR /tmp/surrogate_production_dd per the production
script. Post-campaign driver steps: attach + commit the artifact,
retrain certified_ppgo_map --production with the F080 edge-biased
binding, post_build_sweeps.sh, then 7b.
