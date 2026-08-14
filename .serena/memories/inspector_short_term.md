# Inspector Short-Term Observations

## 2026-08-14 (pass-4) — saddle tier-1 eta floor RAISED 0.5 -> 0.9, still under-protective

Scope: re-review of uncommitted working-tree diff. WP1 raised
`_SADDLE_ETA_FLOOR` 0.5 -> 0.9 (likelihood.py L229) via rule
`boundary(0.784)*1.15 = 0.9016 -> 0.9`; withdrew the old `min(0.5,
boundary*2)` cap (documented in provenance block L214-228). WP2 census
mirror faithful (surrogate_census.py L508-523 reuses shared predicate with
eta=geom.caustic_distance; served==counted). Files: likelihood.py (+205),
surrogate_census.py (+17), test_lensing_saddle_tier1_accuracy.py (+1864),
scripts/measure_saddle_eta_floor.py (+237).

VERDICT: ISSUES. INS-1-001 / INS-2-001 / INS-3-001 NOT RESOLVED ->
re-issued INS-4-001. Two NEW findings INS-4-002, INS-4-003.

### INS-4-001 (bug) — floor=0.9 STILL breaches production bar. NOT RESOLVED.
Floor is now used correctly in the gate (L633 `eta < _SADDLE_ETA_FLOOR ->
False`). BUT gate-admitted near-floor witnesses at eta in [0.90, 0.97]
STILL breach the production accuracy contract. MEASURED (ran the class'
own helpers, floor=0.9):
- gamma=1.257 eta=0.9000 p90=2.56e-3 max=4.11e-3
- gamma=1.793 eta=0.9562 p90=5.14e-3 max=1.02e-2
- POPULATION p90=3.70e-3 (3.7x over P90_TOL=1e-3), max=1.02e-2 (> OUTLIER_TOL=1e-2)
- only the eta=1.2868 (gamma=1.150) witness passes; every eta~0.90-0.97 breaches.
ROOT CAUSE: the boundary (0.784) was measured by
scripts/measure_saddle_eta_floor.py against a **1e-4-rel-|F|-at-band-max**
metric over w in [8,60], which is NOT the production **p90<=1e-3 / max<=1e-2**
contract over w in [8, W_CEILING_SCHWINGER] that the accuracy tests (and the
far-from-caustic cert) enforce. A fixed 1.15x safety factor over a
mis-calibrated boundary does not certify the production bar. Classic
"which object's error does the gate bound" — term eta>=0.9 does NOT bound
the served zero-envelope error at the production bar in [0.9, ~1.0). Per
Professor asymmetry (false-admit = silent lnL bias), floor must rise
further. Empirically eta must exceed ~0.97 (likely ~1.0+) — which begins
eroding the transverse-cone win (audited real-use eta 1.0-2.5), so this is
a genuine tension for Professor to adjudicate: raise the floor toward ~1.0,
OR re-derive the measurement script against the ACTUAL production p90/max
bar and set the floor from that.

### INS-4-002 (bug) — suite ships RED (deterministic).
`SaddleTier1EtaFloorNonRegressionTestCase.test_floor_still_below_inspector_
flagged_edge` asserts `assertLess(_SADDLE_ETA_FLOOR, _INSPECTOR_FLAGGED_
WORST_EDGE=0.784)`. With floor=0.9 this is `assertLess(0.9, 0.784)` ->
FAILS. CONFIRMED by direct run (1 failed, 1 passed in 22s). The governance
test was designed to "flip to failing" precisely when the floor clears
0.784, on the (wrong) assumption that clearing the script edge == resolving
the breach. Clearing 0.784 and resolving the p90 breach are DECOUPLED
(INS-4-001). This test + the frozen anchor logic (L1810-1874) must be
reconciled; the suite must not ship red.

### INS-4-003 (design/masked-red) — @expectedFailure masks a live breach.
`SaddleTier1NearFloorEtaAccuracyTestCase.test_p90_within_production_
tolerance` and `test_max_within_outlier_guard` are still `@expectedFailure`
and still xfail (ran: 4 passed, 2 xfailed). They keep the near-floor breach
GREEN so the commit gate won't catch INS-4-001. INS-3-001 item-2 explicitly
forbade this pattern ("an @expectedFailure on a production-reachable
accuracy claim is a masked red -- never ship; if a bar cannot be met, the
GATE moves, not the test"). Class docstring is stale: says "Measured eta
range [0.5035, 0.5339]", "the 0.5 floor is under-protective", "until a
Coder raises _SADDLE_ETA_FLOOR past it" — floor is now 0.9, measured range
is [0.90, 1.29]. Resolution is coupled to INS-4-001: once the floor truly
certifies the band, promote these to plain assertions; do NOT promote while
they still fail.

### INS-4-004 (trivial) — stale docs contradict shipped 0.9.
- scripts/measure_saddle_eta_floor.py L24-25, L230 still document AND PRINT
  the WITHDRAWN `_SADDLE_ETA_FLOOR = min(0.5, boundary*2)` rule; the script's
  own output would claim 0.5 while the shipped constant is 0.9. Update to the
  boundary*safety rule (and ideally re-derive against the production p90/max
  bar per INS-4-001).
- test file stale "(0.5)" annotations: L19 ``>= _SADDLE_ETA_FLOOR`` (0.5)`,
  L973 `eta < _SADDLE_ETA_FLOOR (0.5)`.

### Verified OK
- Gate wiring: L633 uses the floor; live rung L2075+ and census mirror both
  route through the same `_saddle_farfield_analytic_serves`.
- WP2 census mirror faithful (served==counted); zero threshold literals in
  census (single authoritative predicate).
- `_SADDLE_TIE_EPS=1e-12` mirror-pair tie discipline unchanged/correct.
- provenance block (L214-228) correctly documents the withdrawn cap and the
  0.784 per-gamma edges (1.2->0.601, 1.5->0.705, 2.0->0.784).

### Carry-forward -> Librarian (doc staleness, NOT code defects)
- SPEC.md + DATA_CONTRACTS.yaml still cite exterior_polar_rho_log_carrier_v1
  as "ONLY tag" (stale since V5 2D carrier); region vocab (lobe_exterior,
  wedge_interior) undocumented; SPEC may still describe retired rho reach
  gauge for the saddle tier-1 rung (now eta-keyed at floor 0.9).
