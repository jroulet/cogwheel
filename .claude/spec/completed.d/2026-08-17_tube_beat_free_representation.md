---
date: 2026-08-17
section: Surrogate
---

- **Beat-free tube representation SHIPPED — the tube chart stores
  r = E/F_ref and the node count collapsed 48 → 10 with a 5.6× accuracy
  margin** `[→ spec]` — the F083 falsification, measured 2026-08-17 on
  the shipped test (`TubeF083AccuracySweepTestCase`, gamma=0.4 astroid,
  8 off-node held-out at eta=0.5·eta_max, w ∈ [40,60], refused_count=0):
  **n_theta=10, eps=4.2652e-03** against the 0.0237 bar (the 48-node
  brute baseline) — the old beating representation needed ~48 nodes to
  meet the same bar (F083: 24 → 0.146, 48 → 0.0237). Per-point held-out
  eps monotone, worst 4.3e-3. The representation: `_tube_f_ref`
  (surrogate.py) builds a non-vanishing q=p uniform-Airy two-carrier
  reference (Airy Wronskian guarantees no zeros) in the SAME
  tau_c = virtual_delay − t_min frame as the envelope; the chart stores
  the beat-free residual; serve re-modulates E = r·F_ref at the RAW
  eigenframe source (D2-invariant F_ref; theta-fold only on the residual
  interpolation). Delta_tau DRY-imported from `_merging_fold_pair`.
  `TUBE_BEAT_FREE_AIRY = 'tube_beat_free_airy_v1'` envelope-definition
  tag + stale-artifact hard-refusal live. gamma=0.045 stays a benign
  build-side refusal band (coalescing fold pair). Build history: five
  launches; launch-5 DAG ran Inspector PASS (2 driver escalation rulings:
  caller-threading fixes in surrogate_census + cusp-window fixtures;
  1 accept with driver-side verification) and Professor PASS; the tree
  gate then caught the legacy-fixture lag (9F+5E, all fixture staleness
  — none production), hand-finished driver-side: fixtures re-pointed
  with golden literals bit-identical, the 2 overlap-band precedence
  tests made structural probes (`require_fref=False`;
  [[lensing_tube_exterior_double_match_dead_branch]] files the
  dead-branch question). Driver full-gate re-run: ALL GREEN
  (2408 collected, 2158 passed / 235 skipped / 4 xfailed + timing
  guards). Unblocks: tube training, the (f_max, f_floor) sweep, the
  demand-sized campaign.
