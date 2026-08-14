---
date: 2026-08-14
section: Lensing serving
---

**Serving artifacts wired (F077 closed: the chart layer is no longer dead
code)** `[→ spec]` — build `wire_serving_artifacts` + driver completion.
`born_residual_chart.npz` and `certified_ppgo_map.npz` are now reachable
from the production entry WITHOUT a surrogate: a first-class Born
intercept in `_amplification_coefficients` (after the saddle-certificate
rung, before the seed engine; gate kappa==0 AND beta==0 AND gamma != 0
AND rho > 2 AND `covers(gamma, rho, chart_w)`; astroid-parity artifact),
band-split against the certified map (`w_trust` / cell ceiling), with a
byte-exact null-split identity. `BornResidualChart.load()` added (schema
+ content-hash hard-refusals naming the regen script; npz re-saved with
byte-identical numeric payload); auto-attach at construction with
explicit-None opt-out (refuse-to-None + warning on a bad artifact); JSON
round-trip on construction INTENT (default omitted -> re-auto-loads;
None verbatim; in-memory chart raises naming the limitation);
`born_residual_chart` threaded through the marginalized class; census
mirror re-keyed. New suite `test_lensing_born_analytic_reachability.py`
(serve-path trace, band-split premises, null-split identity, off-path
battery, loader refusals, auto-attach fallback, JSON round-trips,
self-falsification teeth).

Plan gate: driver REJECTED v1 for an inverted kappa/beta guard (it would
have refused every standard draw and false-admitted untrained kappa!=0
configs); v2 corrected throughout. In-build escalation INS-3 (contracts
"both parities" claim vs the astroid-only artifact) fixed by
text-narrowing. The build survived its full 25-min tree gate under load —
first live success of the orchestrator gate heartbeat — and the gate
caught a real defect its targeted tests could not: the auto-attached
chart exposed the new rung to the F -> 1 zero-noise anchors, where
`caustic_rho` raises a raw ZeroDivisionError at exactly gamma == 0
(no caustic). Driver fix: an explicit unlensed-limit guard (gamma == 0
declines before any caustic-frame computation) + a gate-miss test row;
both anchor classes and the new suite green (10/10); remaining tree green
by union (the guard is additive and reachable only at gamma == 0.0,
which previously crashed).
