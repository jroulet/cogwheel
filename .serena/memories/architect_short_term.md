# Architect Short-Term Observations

(empty — last consolidated by Dreamer on 2026-07-28)

## Build: positive-parity-resolved-first (2026-07-28)
Unify geometric-vs-wave predicate on `select_branch` in operator.py's two grids.
- POS-PARITY (_positive_parity_grid): add geometric branch. For w>ceiling nodes:
  L=w*|y'| (cache |y'| from top-of-grid _mass_sheet_map y_scaled; do NOT re-call
  cancellation_exponent per node), delta_min via _real_delay_min_separation(
  physical source, macro_matrix) once, guarded by any(w>ceiling). geometric ->
  geometric_amplification (physical frame!), else existing arms-then-refuse.
- SADDLE (_saddle_grid): Professor OVERRULED my pi*w/4 idea AND Simplifier's
  rubber-stamp of it. pi*w/4 (DD-mantissa depth) vs L_MAX=48 (1F1 onset proxy) =
  unit-mismatch + opens dead band (60,61.115]. RULING: pass cancellation_exp=
  math.inf so only the resolution leg routes through select_branch -> byte-
  identical w>60 AND resolved boundary; ceiling stays enclosing branch.
- Residual: ~1% O(1) tail survives 2-condition gate (p99 7.1e-1, max 74); NEVER
  "certified/exact"; add FINDINGS entry.
- All tests -> Test Developer (new + re-point ~8 blast-radius files). Docs ->
  post-gate doc-sync/Librarian.
