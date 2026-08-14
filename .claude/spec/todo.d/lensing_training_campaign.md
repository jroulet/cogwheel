---
section: Backlog
depends_on: [lensing_find_cusps_wrap_bug, lensing_tiling_census_node_budget, lensing_wire_serving_artifacts, lensing_saddle_admission_c3]
---

- **NEXT-SESSION ORDER 7a/7 — THE TRAINING CAMPAIGN (cost estimate FIRST,
  always)** `[→ spec]` — the low-w tables that close every remaining
  chart-owned cell of the coverage map. Standing rule: NO engine-run launch
  without a recorded cost estimate. Method: run the SMOKE config
  (`scripts/train_lens_surrogate.py`, TrainingConfig defaults,
  engine_budget=400/chart, minutes-scale) to a scratch outdir, read
  training_report.json's engine-call counts, extrapolate to the production
  config via the tiling-census numbers, RECORD the estimate, then launch
  with a monitor + stale alarm (monitored-not-unattended).

  Production config content, all measured 2026-08-13/14: regions must
  include lobe_interior + lobe_exterior (the deltoid annulus rho 0.5-1.5
  and lobe interiors are the genuine table needs) and the near-lobe /
  near-cusp annuli; near-cusp tile axes per the tiling-census verdict
  (F074 control coordinates are the candidate); labels ONLY from the safe
  oracle band (w <= 60 DD; SADDLE_WALL = 58) — labels above 60 were the
  F075 contamination vector, now fixed but the band cap stays as
  defense-in-depth. Also in this campaign: (a) RETRAIN
  certified_ppgo_map.npz (`train_ppgo_map.py --production`) — 32
  positive-parity exterior cells were measured against the contaminated
  fold-arm oracle (over-conservative direction; re-measure, re-hash);
  (b) replace the saddle cusp-arm coverage placeholder 0.0
  (`measure_saddle_cusp_arm_coverage.py` or the retirement path from the
  coverage sweep — see FINDINGS F079 body: the angular-coverage concept
  may be dead post-F074, decide from the same evidence); (c)
  born_residual_chart.npz is CLEAN, no action. Post-training: attach the
  new artifact (wiring already landed), run `post_build_sweeps.sh`
  (driver-side, never in-build).
