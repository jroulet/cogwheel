---
section: Backlog
depends_on: [lensing_tiling_census_node_budget]
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
  BINDING (F080, measured 2026-08-14): the retrain MUST replace the
  one-center-config-per-cell certification with edge-biased,
  worst-over-cell sampling (gamma near band lo, rho near the caustic
  side, transverse angles) — a shipped CERTIFIED saddle cell was 3.5
  orders over the bar at its band-edge corner while its center passed;
  and the F080 fan asymmetry (mirrored fan angles 2.4x apart under exact
  D2) must be resolved before the retrain trusts the fan;
  (b) DONE via retirement, not measurement — F079 (2026-08-14,
  `find_cusps_wrap_fix` build) measured `_SADDLE_CUSP_ARM_COVERAGE` INERT
  (0 differing serve decisions over 64 production windows) and retired it
  (`retired_concepts.json`); the saddle cusp-arm coverage placeholder no
  longer exists to replace — the tube gate excludes on the full cusp
  window, real structure is the F074/F075 w-floor 49; (c)
  born_residual_chart.npz is CLEAN, no action. Post-training: attach the
  new artifact (wiring already landed), run `post_build_sweeps.sh`
  (driver-side, never in-build).
