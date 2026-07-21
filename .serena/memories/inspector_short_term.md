# Inspector Short-Term Observations

## 2026-07-20 — Build 8d homogenization (WP1 reroute + WP3 census)

Scope: uncommitted tree, worktree /home/tejaswi/Work/cogwheel-claude-dev
(main-tree `cd` hook-blocked; use `git -C <worktree>`). Full python:
/home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.

### What landed & verified CORRECT
- operator.py WP1: `_positive_parity_grid_with_fallback` renamed ->
  `_positive_parity_grid`; positive-parity dispatch now keys on
  `gamma_prime = _mass_sheet_map(y,gamma,kappa)[2]`. gamma'>0 -> per-node
  `_schwinger.f_schwinger` (reduce/rotate/reconstruct copied VERBATIM from
  `_saddle_grid`, verified byte-for-byte identical modulo
  `_mass_sheet_map` vs `_saddle_mass_sheet_map`); gamma'==0 (`not
  gamma_prime>0.0`) -> legacy `_grid_certified` (sole legacy production
  exit). try/except-CancellationError fallback DELETED. Added
  w_array.ndim!=1 guard. Runtime-verified: gamma'>0 order_used=0 &
  grid==scalar bit-identical; gamma'==0 order_used=9; w=68 gamma'>0 raises
  SchwingerCertificationError. `legacy_operator_oracle = _grid_certified`
  alias present, NOT in __all__. No geometric branch inside
  `_positive_parity_grid` is CORRECT — positive-parity geometric routing
  lives in channels.py select_branch (F_op is wave-entry only); matches
  old `_grid_certified` which was pure-wave too. No stale
  `_positive_parity_grid_with_fallback` refs anywhere.
- test_lensing_surrogate.py CrownContractFlipWitnessTestCase (8 tests):
  RAN GREEN (34s). NEW-Schwinger vs OLD-legacy_operator_oracle on
  legacy-certified overlap, max-normalized 1e-10; dispatch spies; F010
  mutation reds. Correct re-baseline.
- scripts/census_homogenization_corners.py: geometry-only reporting tool,
  no engine edits. All imported constants/symbols resolve. Smoke n=200
  seed=1 GREEN (gamma'==0=0, corner 0.26, xcheck 0.0, max_w 433.8<500).

### FINDINGS (introduced by this change, actionable)
- INS-1-001 (implementation): plan-listed test suites NOT re-baselined ->
  lensing suite RED. CONFIRMED red:
  test_lensing_schwinger.py::PositiveParityBitFreezeTestCase (asserts
  order_used>0 + frozen literals for gamma=0.2 positive parity — both now
  false); test_lensing_operator.py 6 fail + 4 err (OperatorOracle /
  ContractionCertification / CancellationRefusal — old positive-parity
  operator-series refusal contract rerouted to Schwinger). UNVERIFIED but
  plan-listed & likely affected: test_lensing_fast_path.py,
  test_lensing_ratio_layer.py, test_lensing_batched_operator.py (value +
  timing pins on the positive-parity operator path; Schwinger ~58ms/node
  is far slower). Coder+Test-Dev memories both OWE these re-baselines to
  a separate Test Dev dispatch — not yet done. Acceptance criterion
  "full lensing-suite green at fixture scale" NOT met as delivered.
- INS-1-002 (design/spec): SPEC.md line 54 spec-code divergence. Two now-
  FALSE claims: (a) parity dispatch "whose positive-parity arm is
  bit-frozen"; (b) Build-7a "(2) ... the all-certified hot path and every
  certified output stay byte-identical". Post-WP1 the positive-parity
  gamma'>0 hot path IS Schwinger and values changed ~1e-15. SPEC was in
  the plan's expected-change list but untouched. Route to Librarian/spec-
  owner (I own accuracy, they own the edit). Interpretation: spec needs
  updating (WP1 is approved homogenization), not code wrong.
- INS-1-003 (trivial): census node_classification_totals split — a
  positive-parity node with w<=60 AND resolved AND L>48 is 'geometric' in
  production (select_branch is w-independent) but the census labels all
  w<=60 as served_by_schwinger. Minor undercount of served_by_geometric in
  the owner-facing report; documented simplification; HEADLINE fractions
  (gamma'=0, refusal corner) unaffected.

### Carry-forward
Once the operator/schwinger/fast-path/ratio/batched suites are re-baselined
(order_used>0->==0 for gamma'>0; frozen literals->Schwinger values;
CancellationError->SchwingerCertificationError above ceiling; timing gates
retuned for Schwinger cost), re-run full lensing suite to confirm green
before commit. INS-1-001 (census test suite from 8c) — check if still open
in a later review; not in scope this build.
