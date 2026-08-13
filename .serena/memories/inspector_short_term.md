# Inspector Short-Term Observations

## 2026-08-13 (pass 3) — Review: saddle_above_ceiling_serving (INS-2-001/002 re-check)

Scope: uncommitted working tree (3 untracked test files + likelihood.py,
surrogate_census.py, _gauge.py). VERDICT: ISSUES. BOTH mandatory re-check
findings STILL UNRESOLVED. Build NOT shippable — its own named acceptance
gates are RED.

### INS-2-001 — STILL RED (NOT resolved)
- The three named test files are UNTRACKED and byte-unchanged since pass-2.
  pytest: `10 failed, 24 passed, 19 errors` — IDENTICAL profile to pass-2.
- Root cause unchanged: gate `_saddle_farfield_analytic_serves(real_delays,
  w_lo, rho)` requires 3 args; tests still call 2-arg form -> TypeError at
  accuracy L276 (and L437/481/500), refusal L434. Confirmed live:
  `TypeError: _saddle_farfield_analytic_serves() missing 1 required
  positional argument: 'rho'`.
- Fixture skew also unchanged: refusal test_admitted_control_returns_tuple
  -> `_serve(AD_GAMMA, AD_Y, AD_ADMIT_W_LO)` returns None (AD_Y at rho<2.0,
  refused by rho-floor); anti-vacuity self-falsification checks make 0 gate
  comparisons. Docstrings still describe the old 2-term gate.
- The Coder/Test-Dev did NOT touch the test files this pass. Carry forward.

### INS-2-002 — STILL PRESENT (NOT resolved) — likelihood.py L2098-2103
- `_saddle_farfield_analytic` still builds geometry at `kappa=lens['kappa']`
  but computes gate rho via `caustic_rho(gamma, |y|, kappa=0.0)` HARDCODED,
  and the false comment "kappa == 0 (the dispatch refused kappa != 0)"
  persists. Verified NO dispatch guard refuses kappa!=0 before this rung:
  dispatch at L2217 gates only on `lens['gamma'] > 1.0`; surrogate intercept
  returns None (falls through) on kappa!=0; ppGO only fires w_max>150. So a
  general-API kappa!=0, gamma>1, w_max<=150 candidate reaches this rung with
  a mis-gauged rho -> may serve wrongly (never-serve-where-wrong class).
  Sibling `_ppgo_above_ceiling` L1753 correctly passes `lens['kappa']`.
  Fix: pass `lens['kappa']`; correct the false comment. Census stays
  kappa=0 (sampled space) -> no census skew.

### INS-3-001 (NEW, trivial/design) — _gauge.py
- New pure helpers `_saddle_switch_delay`/`_saddle_phase_delay` are
  referenced ONLY by test_lensing_saddle_gauge.py — zero production callers.
  The live tier-1 rung does NOT use them (resolvability is computed
  independently inside `_saddle_farfield_analytic_serves`). Docstring claims
  they are "the SINGLE authoritative gauge for BOTH saddle tiers" but tier-2
  chart doesn't exist and tier-1 doesn't call them, so the DRY/single-source
  claim is aspirational and _RHO_END=4.0 is now a 3rd independent copy of
  the RHO_END resolvability boundary (operator.RHO_END, the predicate's
  `w_lo*min_delta_tau>=RHO_END`, and this switch-delay). Non-blocking.

### WP-2 census wiring — OK (no finding)
- surrogate_census.characterize_sample correctly passes `rho` (3-arg) to the
  shared predicate, kappa=0.0 (census space is kappa=0 -> correct there),
  reuses the same geom partition, labels served records
  'saddle-farfield-analytic'. fallthrough_breakdown adds served_breakdown
  over _SERVED_CATEGORIES with an unknown-cause guard. Consistent.

### Carried forward -> Librarian (doc staleness, unchanged)
- exterior_polar_rho_log_carrier_v1 "only tag" + region-vocabulary staleness
  in SPEC.md / DATA_CONTRACTS.yaml. Not touched by this diff.

### PATTERN (carry forward)
- A "Files actually changed" manifest that LISTS the test files does NOT mean
  they were edited — they can be untracked and byte-identical to a prior
  red state. ALWAYS re-run the named acceptance files; never trust the
  manifest or the Coder's word that a signature-bump test skew is fixed.
