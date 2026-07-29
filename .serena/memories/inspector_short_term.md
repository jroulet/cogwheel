# Inspector Short-Term Observations

## 2026-07-29 — Build 8f (F028) select_branch routing — RE-REVIEW (v3, RESOLVING pass)

Scope: uncommitted diff. Production edit UNCHANGED from v1/v2 (operator.py
`_positive_parity_grid` geometric branch via `select_branch(w,dmin,w*|y'|)`,
`_saddle_grid` via `select_branch(w,dmin,inf)`). THIS pass finally re-pointed
the three blast-radius test files the prior two passes left untouched.

### THE THREE CARRIED FINDINGS ARE NOW RESOLVED (ran each test by name, PASS)
- INS-1-001/INS-2-001 (airy_fold): fold ladder node radius 0.14->0.06 so
  L=500*0.06=30<L_MAX -> stays WAVE, fold-served. Docstring cites F028.
  test_fixed_priority_fold_tried_before_cusp + _served_value_equals_labelled
  _rung_bitwise PASS.
- INS-1-002/INS-2-002 (fast_path): FOP_REFUSALS (63,0.9,0.2)->(63,0.3,0.2)
  (L=18.9 wave, both arms decline -> hard refusal). Flip-witness test made
  THREE-way (hard/arm/geometric) with new `served_geometric>0` non-vacuous
  assert. All 3 (refuses_uncertifiable, grid+scalar arm_flip_witness) PASS.
- INS-1-003/INS-2-003 (operator): sheared refusal fixture y=[1,0]->[0.08,0]
  (small on-axis, unresolved, both arms decline). assertRaises intact. PASS.
Assertions NOT weakened; docstrings cite F028; fixtures legitimately route to
the claimed branch.

### FULL BLAST RADIUS GREEN (ran to completion)
airy_fold+operator+fast_path 96 passed/11 skip/1 xfail; saddle_geometry 28;
surrogate+waveform 82 passed/1 skip; schwinger 48. Zero failures.

### PRODUCTION CODE: CORRECT (re-verified diff)
`w*y_prime_norm` (y_prime_norm=sqrt(y_scaled@y_scaled)) reproduces
cancellation_exponent==w*|y'| exactly; saddle inf makes L_MAX leg vacuous so
boundary == old (w>60 AND resolved). delta_min guarded by np.any(w>ceiling).
Frame discipline OK (physical source). math.inf resolves (green saddle tests
prove `import math` present).

### OPEN — SPEC/doc work the brief scoped was NOT done (flag to Librarian)
git diff/status on .claude/spec/ is EMPTY. None of the mandated spec edits landed:
- SPEC.md line 54 (Build 8e row) still reads "uniform arms (certified)" and
  "the uniform arms fire ONLY at the previously-refusing sites" — now DIVERGENT:
  this build routes resolved+cancelling above-ceiling positive-parity nodes to
  GEOMETRIC via select_branch, not the arm; and F028 measured the arm 60-267%
  wrong (NOT certified). Brief explicitly required this correction.
- No NEW FINDINGS entry for the ~1% O(1) geometric-tail residual (existing F028
  @1381 documents the ARM error, not the geometric tail the brief flagged).
- todo.d/lensing_fold_arm_serves_wrong_values.md NOT rewritten to carry only
  q/b4; no completed.d / spec_changelog.d fragment.
Carried forward from v1/v2 memory; still open.

### Pattern (reinforced, now closed on code side)
The plan's expected-to-change list IS the checklist; it took 3 build cycles to
re-point all three files. Distinguish arm-vs-geometric via w*|y'| vs L_MAX=48
and w*delta_min vs RHO_END=4. Code+tests done; spec story still owed.
