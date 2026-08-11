# Librarian Short-Term Observations

## 2026-08-11 -- ppGO fold-pair/resolution gate build sync (SPEC change + fragment retirement)

Scope: In-build doc sync for work package "Gate ppGO rung on fold-pair existence or resolution" (working-tree code: `_pearcey_cusp.py` dual gate + `test_lensing_airy_fold.py` self-falsification test).

### What went stale / was fixed

- SPEC.md was stale, NOT a no-op: the ppGO fast rung sentence enumerated firing conditions but the code added a fourth gate. Fixed by adding the fold-pair-existence-or-resolution gate (`_merging_fold_pair(...) is not None` OR `w*delta_min >= _PPGO_RESOLUTION_GATE = 4.0`, mirroring `RHO_END`) to the SPEC row. Spec bumped 0.37.7 -> 0.37.8 (patch) via `spec_changelog.d/2026-08-11_ppgo_fold_pair_resolution_gate.md`.
- Retired TWO todo.d fragments: `lensing_one_home_routing_disagreement.md` (the tracked work package — implemented option (a), the resolution guard; one-home pin now green, verified by running `test_thresholds_have_one_home`) and `lensing_serving_ladder_guards_are_red.md` (all eleven items now resolved; moved to `completed.d/2026-08-11_serving_ladder_guards_are_red.md` with STILL RED section rewritten to RESOLVED).
- New `changelog.d/2026-08-11_ppgo_fold_pair_resolution_gate.md` + `completed.d/2026-08-11_ppgo_fold_pair_resolution_gate.md`.
- FINDINGS.md stale path ref fixed: `todo.d/lensing_serving_ladder_guards_are_red.md` -> `completed.d/2026-08-11_serving_ladder_guards_are_red.md` (F061 consequence text, line ~3482).
- `sync_derived_docs.py`: no file changes; only the recurring `lens_amplification_surrogate` test-only-consumer warning (escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md` still open — no duplicate).

### Patterns / gotchas (new)

- A gate-criterion change to a SPEC-described rung IS staleness even when the code comment/constant docstring explains the new condition — SPEC.md enumerates firing conditions, so adding a gate condition requires the SPEC sentence update. This is the same family as the `_PPGO_BAR_DIVISOR`/`_R_PPGO_ERROR_CONST`/`_W_PPGO_FLOOR` fragile cluster; `_PPGO_RESOLUTION_GATE` is now a fourth constant in that cluster (and it's a mirror of `RHO_END` — if `RHO_END` changes, the mirror note breaks).
- The new gate keeps the "returns before any table or quadrature lookup" SPEC phrase TRUE (the gate computes `geometry.delay` per image, no table consult) — verified before deciding the phrase could stay.
- Retiring a multi-item program fragment: mark the last STILL RED section RESOLVED inline (matching the file's existing inline RESOLVED pattern), add `date:` to frontmatter, then `mv` to `completed.d/<date>_<slug>.md`. Its prose name appears in FINDINGS.md and other completed fragments — plain-text refs to a retired todo path must be swept manually (the dangling-link checker only sees `[[...]]`).
- `replace_content` literal mode failed because a NEWLINE sat between "behind" and "this one" in the FINDINGS.md sentence ("sitting behind\nthis one (see...") — when a literal needle reports no match, verify the raw bytes for line wraps before assuming a unicode issue.

### Cross-references to watch (carried forward)

- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER STILL STALE (INS-1-002/003): SPEC.md ~line 63 and DATA_CONTRACTS.yaml ~line 199 still describe `exterior_polar_rho_log_carrier_v1` as the only known tag. Pending.
- Lobe axis-schema contract (INS-4-002/F050): DATA_CONTRACTS.yaml still describes old lobe axis schemas; production ships `lobe_caustic_relative_v1`. Pending.
- Surrogate escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md`: open, do NOT duplicate.
- Constant-name clusters in SPEC.md/completed.d: `_MP_PANEL_ORDER = 32`, `_PANEL_ORDER = 24`, `_PPGO_BAR_DIVISOR`, `_R_PPGO_ERROR_CONST`, `_W_PPGO_FLOOR`, NEW `_PPGO_RESOLUTION_GATE = 4.0` (mirrors `RHO_END`).
