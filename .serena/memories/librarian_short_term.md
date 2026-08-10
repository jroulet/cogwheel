# Librarian Short-Term Observations

## 2026-08-10 — Doc sync: 2D (rho, u) fold-carrier build (lensing_exterior_2d_fold_carrier)

### Scope
Doc-only sync for the 2D fold-carrier build (surrogate.py + tests, Inspector PASS).
Code grounded independently: `_EXTERIOR_POLAR_AXIS_SCHEMA_V5 = 'exterior_polar_rho_u_carrier_v2'`
(current write tag), V4 `'exterior_polar_rho_log_carrier_v1'` retained in known set;
`_compute_rho_u_carrier` → `(n_rho, n_theta_c)` per-spline-node `Re(tau_c(rho, u))`;
serve re-modulation is bilinear (rho, u) interpolation at the query u-coordinate
(surrogate.py ~2933-2952); load broadcasts legacy 1-D `rho_carrier` to 2-D (line 4427-4431).
`_needs_fold_carrier` (surrogate_training.py) unchanged; `_compute_rho_carrier` is GONE.

### What changed
- SPEC.md: exterior-polar key-abstractions paragraph — both known tags (V4 retained,
  V5 current write tag), `rho_carrier` → 2-D `rho_u_carrier` (+1-D-broadcast note).
- SPEC.md line 56 FOLD-CARRIER DEMODULATION sentence: `_compute_rho_carrier` →
  `_compute_rho_u_carrier`, per-rho median → 2-D, `rho_carrier` array → `rho_u_carrier`,
  added the u-winding flattening (11.66 rad → <= 1.63 rad). THIS second spot was stale
  silently — the task brief only named the key-abstractions paragraph; grep for the dead
  symbol name caught it (librarian dead-reference rule).
- DATA_CONTRACTS.yaml line 199: both known tags + 2-D `rho_u_carrier` + broadcast note.
- Fragments: spec_changelog.d (bump: minor), contracts_changelog.d (bump: minor),
  completed.d (section: lensing-surrogate), deleted todo.d/lensing_exterior_2d_fold_carrier,
  root changelog.d entry. Left completed.d/2026-08-10_exterior_fold_carrier_demodulation.md
  in place.
- render_fragments.py clean ("All surfaces up to date" on re-run). SPEC.md → 0.37.4,
  DATA_CONTRACTS schema_version → 3.1.3 (minor bump stacks alphabetically after the
  three 2026-08-10 patch fragments in the same date bucket — known rendering quirk, don't fix).

### Fragile cross-references (next build)
- Both surfaces cite `_EXTERIOR_POLAR_AXIS_SCHEMA_V4` AND `_EXTERIOR_POLAR_AXIS_SCHEMA_V5`
  plus the two literal tags; and SPEC.md line 56 cites `_compute_rho_u_carrier` — any
  rename/bump touches both surfaces simultaneously.
- The "Old 1-D rho_carrier artifacts load by broadcasting to 2-D" sentence in BOTH
  surfaces is paired with the V4-retained claim: if a future build drops V4 or the
  broadcast, all three sentences go stale together.

### Surprise / observation (NOT librarian scope — Inspector-owned)
`test_lensing_surrogate_training.py` DT-10 tests STILL access `chart.rho_carrier`
(assertIsNotNone, `len(chart.rho_carrier) == len(chart.rho_grid)`, isfinite checks)
but `ExteriorPolarChart` has NO `rho_carrier` attribute — the field is `rho_u_carrier`
(grep: `rho_carrier` appears in surrogate.py ONLY at the legacy-load line 4429). Either
the tests mock charts or they are red despite "Inspector PASS". If a failure surfaces,
that is the first place to look; do not "fix" from the doc side.

### sync_derived_docs.py
Ran clean; only the known recurring `lens_amplification_surrogate` test-consumer
warnings (tracked by open escalation fragment todo.d/surrogate_contract_test_consumer_warning.md —
do NOT create a duplicate). No stray diffs in tidy_advisory.json/foreman_lite.json this run.
