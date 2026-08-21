# Inspector Short-Term — 2026-08-21 (low_w_chart_rho_hierarchy, INS pass 3 FINAL)

Scope: re-review of uncommitted changes for build "low_w_chart_rho_hierarchy"
(rho-partitioned carrier hierarchy). Files: low_w_diffractive_chart.py,
train_low_w_diffractive_chart.py, likelihood.py,
test_lensing_low_w_diffractive_chart.py.

Test state (fresh run): 65 passed, 0 failed, 0 errors (was 4 failed + 1 error).

## Resolved (re-verified in code + fresh run)
- INS-2-001: RESOLVED. `_pearcey_cusp_reference` now applies
  `_renormalize_macro_lead` (delta_tau = `_reduced_min_delay_separation`), so
  fold and cusp forms asymptote to the SAME sqrt_mu. `test_fref_continuous_
  across_handoff` green (CONTINUITY_FREF_RATIO_TOL=5.0).
- INS-2-002: RESOLVED. `MacroCarrierReferenceTestCase` uses `_MACRO_FREF_W_GRID
  = geomspace(0.02, 0.4, 8)` confined below w_split~0.49.
- INS-2-003: RESOLVED. `test_sweep_straddles_the_carrier_partition` asserts on
  the single unresolved `CONTINUITY_W` frequency.
- INS-2-004: RESOLVED. `test_geometric_far_exterior_serve` uses NODE_EXACT_TOL;
  GEO_SERVE_TOL_* constants removed; docstring rewritten.

## Findings (2, both non-blocking)
- INS-3-001 (trivial): 5 stale "authoring-time state" docstrings in the test
  file describe now-fixed bugs: lines 178, 1169, 1571, 1658, 2221 ("RED until
  the cusp-transition detection is fixed" / "currently DECLINED ... keyed on
  the wrong refusal" / "RED until the INS-2-002 production fix lands"). All
  green now; stale notes mislead future readers. Test Dev/Tidier cleanup.
- INS-2-005 (design, Librarian, CARRIED FORWARD = INS-1-004): SPEC.md line 54 +
  DATA_CONTRACTS.yaml line 389 still describe `fold_cusp_reference`,
  `_NON_VANISHING_MIN_RATIO`, schema v1/v2 — all stale. Plan listed both as
  expected-to-change; neither changed.

## Verified-safe (not findings)
- `_NON_VANISHING_MIN_RATIO` removal from `partitioned_reference` is safe: the
  Airy Wronskian never vanishes (math guarantee); the cusp form's P->0 collapse
  is guarded INSIDE `cusp_uniform_reference_grid` (all-or-nothing per-node None,
  non-finite check) — so a serve-time collapse still declines -> engine.
- `_renormalize_macro_lead` cannot inject a pole: |F_ref| -> h|F_ref| +
  (1-h)sqrt_mu is a magnitude convex combination bounded by max(|F_ref|,
  sqrt_mu); nan from 0*inf is impossible since the cusp form is pre-guarded
  nonzero and the fold Wronskian is strictly positive.
- trainer per_kind_stats indexing is consistent: in the non-error path
  (n_refused==0 raises SystemExit earlier) `declined`(=unbuildable|sentinel) ==
  unbuildable_mask|guard_declined_mask, so kind_mask (nested i_gp/i_rho/i_theta
  filtering those two masks) aligns with ratios[:grid_n] order exactly.
- `_cell_w_split`'s empty-domain-admit path remains unreachable via
  `_residual_at` (kind=='macro' implies w_split > max(w_grid) or inf) — harmless.

## New patterns
- RENORMALIZATION SAFETY BY MAGNITUDE-CONVEXITY: a phase-preserving
  magnitude renormalization `f * (h + (1-h) k/|f|)` can never blow up or NaN if
  `f` is guaranteed nonzero (cusp pre-guard / Airy Wronskian); the only NaN
  risk would be 0*inf, precluded by the nonzero guarantee. Audit the GUARD
  location (moved inside `cusp_uniform_reference_grid`) rather than assuming a
  removed ratio-guard left a hole.
- STALE AUTHORING-TIME DOCSTRING SWEEP: a multi-pass fix cycle leaves "NOTE
  (authoring-time state): RED until ..." docstrings behind on EVERY test that
  was red-then-fixed. Grep for "authoring-time state" + "currently DECLINED"
  after any resolution pass and strip them — they describe the pre-fix state
  and read as live bugs.
