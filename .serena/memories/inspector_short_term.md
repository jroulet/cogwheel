# Inspector Short-Term — 2026-08-07: regions filter + slow-operation guard audit (e4b7b80, 7b63b7f, 3aedd41)

## Scope
Post-strand audit of the remeasure_v3 build's WP1 (regions filter) + driver
slow-operation admission judge.  Coders' work was stranded when the test_dev
agent crashed.  Reviewed `surrogate_training.py` (regions in train/`_train_band_charts`,
`guard_slow_operation`, `_self_estimate`), `train_lens_surrogate.py` (`--regions` CLI),
and 52 new tests (28 regions + 24 guard).

## Results
PASS — all 52 tests green (1 engine-backed skip), 64 existing surrogate
training tests green, no regressions.

## Key invariants verified
- `regions=None` preserves byte-identical all-regions behavior (astroid + saddle).
- `regions=('wedge_interior',)` builds only wedge charts; exclusivity pinned
  structurally via counting stub (no re-derivation of filter logic).
- `guard_slow_operation`: refuses over-budget without slow-tier env, admits
  with one; `_self_estimate` correctly classifies smoke (under), production
  (over), wedge-probe (under).
- Tests are value-asserting with reachable-red: both self-falsification
  classes corrupt their contracts and prove the checks trip.
- The 5 fast-tier slow-vars match conftest; `regions` is keyword-only with
  `None` default; all 5 `train()` callers are forward-compatible.
- `_self_estimate(..., ())` is deliberately conservative (falls back to
  full estimate, documented) while `_train_band_charts(regions=())` builds
  nothing — a deliberate asymmetry (guard overestimates, never undercharges).

## Findings
None.  No correctness, design, or trivial concerns.
