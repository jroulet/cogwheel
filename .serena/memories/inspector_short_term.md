# Inspector Short-Term Observations

(consolidated by Dreamer on 2026-08-13 — durable lessons promoted to
`mem:inspector_knowledge`. Only OPEN carry-forwards are kept below.)

## Open carry-forwards (not code defects)

- DRIVER: retrain `certified_ppgo_map.npz`. The WP-4 advisory bounds the
  contamination at 32 positive-parity exterior cells (w in {66,80,97,117,141}),
  direction is over-conservative (coverage/perf loss, never over-certification).
  `born_residual_chart.npz` is CLEAN (its w grid tops out at 60.0).
- LIBRARIAN (doc staleness lineage, unchanged across several passes): SPEC.md +
  DATA_CONTRACTS.yaml still cite `exterior_polar_rho_log_carrier_v1` as the
  "ONLY tag" since the V5 2D carrier shipped, and neither surface names the
  region vocabulary (`lobe_exterior` etc.). The arm-ladder order change
  (fold -> ghost -> cusp) is internal to operator.py and SPEC.md does not
  document the ladder order, so it is NOT a spec divergence.
