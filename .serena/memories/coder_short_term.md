# Coder Short-Term Observations

- INS-2-001 DECLINED ON ROLE GROUNDS (2026-08-13): finding routed to Coder
  but it is Test Developer scope. The three RED files
  (test_lensing_saddle_tier1_accuracy/refusal/gauge) are the ACCEPTANCE
  GATES that bless the INS-1-001 rho-floor fix I authored; re-calibrating
  their pin/positive-control fixtures to rho>=2.0 and flipping the
  leak-witness tests (test_pin_admitted_at_floor_eight_serves_wrongly,
  test_far_source_accurate_near_source_inaccurate) from assert-leak to
  assert-refusal is authoring the tests that judge my own production
  change — the prohibited "grading own homework" circularity. Left the
  test files UNTOUCHED. Precise handoff spec given to Test Developer:
  (1) add rho arg to every `_saddle_farfield_analytic_serves` call
  (accuracy L275/L437/L481/L500, refusal L434) computed as
  caustic_rho(gamma, hypot(y1,y2), kappa=0.0) — SAME as production;
  (2) re-calibrate admitted/pin fixtures to rho>=2.0 (or intentionally
  <2.0 for refusal witnesses); (3) flip the two leak-witness tests to
  assert refusal (rung returns None / predicate False at rho<2.0);
  (4) update module docstrings (accuracy L9-11, refusal L14-16) to the
  TWO-term gate (resolvability AND rho>=_SADDLE_FARFIELD_RHO_FLOOR).
  Production surface confirmed intact this turn (likelihood.py +152,
  surrogate_census.py +59, tests unchanged).

- INS-1-001 FIX (2026-08-13): the tier-1 saddle-analytic gate
  `_saddle_farfield_analytic_serves` (likelihood.py) keyed ONLY on
  resolvability (n_real>=2 AND w_lo*min_delta_tau>=RHO_END) with NO
  error-bounding term, so near-caustic (tier-2, ~23% beyond-shell)
  resolvable saddles were served with a ZERO envelope -> O(1)-wrong
  likelihoods (a production regression vs HEAD, where they fell through
  to the exact engine). FIX: added a REQUIRED third arg `rho` to the
  shared predicate + a leading proximity gate `if rho is None or not
  np.isfinite(rho) or rho < _SADDLE_FARFIELD_RHO_FLOOR: return False`
  BEFORE the resolvability terms. New module constant
  `_SADDLE_FARFIELD_RHO_FLOOR = 2.0` (== RHO_FAR in
  test_lensing_saddle_tier1_accuracy) placed after
  `_STRONG_SHEAR_STOP_THRESHOLD`. BOTH callers compute rho the SAME way
  so served-set==counted-set can't skew: live rung
  `_saddle_farfield_analytic` and census `characterize_sample` tier-1
  block each do `caustic_rho(gamma, float(np.hypot(y1,y2)), kappa=0.0)`
  in try/except (ValueError, LensDomainError) -> live rung returns None
  on failure, census sets rho=None (predicate then declines). kappa=0.0
  pinned on BOTH (dispatch refuses kappa!=0). Predicate + method
  docstrings updated to the two-term gate. NOTE: the two witness tests
  that PASS by asserting the leak
  (test_pin_admitted_at_floor_eight_serves_wrongly,
  test_far_source_accurate_near_source_inaccurate) and the 2-arg
  predicate call sites in test_lensing_saddle_gauge.py /
  test_lensing_saddle_tier1_refusal.py now need Test Developer updates
  (signature change + refusal-not-leak assertions) — NOT Coder scope.
  Verified: ast.parse both files, import both, predicate truth table
  (far/resolvable->True; near/rho<2->False; rho None/nan->False;
  far/unresolvable->False; single-image->False), 3-arg sig visible from
  census. UNVERIFIED: full lensing/census suite not run (Coder does not
  run suites).

- WP-2 census tier-1 wiring (2026-08-13): surrogate_census.py now imports
  the SHARED `_saddle_farfield_analytic_serves` from likelihood.py (no
  re-derived gate math) — safe: census already pulls likelihood
  transitively via lensing.prior, likelihood does NOT import the census.
  In `characterize_sample`, after the ppgo_fold (image_count==4) handoff
  and BEFORE `classify_fallthrough`, a new `if gamma > 1.0:` block extracts
  real delays from the SAME geom partition already built
  (`geom.delays[geom.real_mask]`, no 2nd partition), w_lo=float(w_grid.min())
  (== exp(log_w_min)); on predicate True -> record.served=True,
  category='saddle-farfield-analytic', return. classify_fallthrough's
  `gamma>1 and image_count==2 -> 'born'` branch is UNCHANGED (only the
  gate-admitted subset is intercepted upstream and served; unresolvable
  saddle sources still fall to 'born'). New module constant
  `_SERVED_CATEGORIES=('chart','ppgo_fold','saddle-farfield-analytic')`
  after `_FALLTHROUGH_CATEGORIES`. `fallthrough_breakdown` gains a
  `served_breakdown` dict: served records tallied by category (None->
  'chart'), unknown served cause raises CensusError; partition invariant
  unchanged (served up, born down by same count). This ALSO surfaces the
  previously-invisible 'ppgo_fold' served sub-type. Verified: ast.parse,
  import, predicate module==likelihood, synthetic fallthrough_breakdown
  smoke (served=3, served_breakdown correct, unknown-cause raises,
  partition holds). UNVERIFIED: full census/lensing suite not run (Coder
  does not run suites) — Shard B unit test is the fast-tier check.

- WP-1 saddle_above_ceiling_serving (2026-08-13): added tier-1 far-from-
  caustic macro-saddle analytic rung. Module-level pure predicate
  `_saddle_farfield_analytic_serves(real_delays, w_lo)` in likelihood.py
  (after `_loo_stop_for_lens`) = the EXACT inline resolvability test from
  `_ppgo_above_ceiling` (n_real>=2 AND min positive delta_tau AND
  w_lo*min_delta_tau>=RHO_END), factored out as the SINGLE source of truth
  for WP-2 census to import. New method `_saddle_farfield_analytic` (after
  `_ppgo_above_ceiling`): geometry_partition -> gate on shared predicate ->
  zero complex envelope -> reconstruct_farfield(FARFIELD_KERNEL_SUM) -> no
  engine/fold_ppgo, no geom.switch/critical_delay; LensDomainError
  propagates. Dispatched in `_amplification_coefficients` immediately after
  the ppGO-above-ceiling block, guarded `if lens['gamma']>1.0` (astroid
  byte-identical). _gauge.py: added `_RHO_END=4.0` (mirrors operator.RHO_END,
  local constant to dodge channels<-_gauge->operator->__init__->channels
  cycle, same pattern as _pearcey_cusp._PPGO_RESOLUTION_GATE) + pure
  `_saddle_switch_delay(tau_min, w_min)=tau_min-_RHO_END/w_min` and
  `_saddle_phase_delay(tau_min)=tau_min` — w_min is a PASSED band-floor arg,
  never a live serve-time w (ghost-gate skew bug class). Verified: ast.parse
  both files, import both, predicate truth table (resolved/unresolved/
  single/degenerate), gauge arithmetic. UNVERIFIED: full lensing test suite
  not run (Coder does not run suites).
