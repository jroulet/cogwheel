# Coder Short-Term Observations

- INS-2-001/002/003 RESOLVED (2nd bounce; pipeline forced Coder to edit
  the stale test fixtures after Test Dev didn't). Probed real production
  fns to pick verified wave/refusing re-points, then ran the 6 named tests
  GREEN + all 3 full files (96 passed/11 skip/1 xfail, no fail):
  * airy_fold `_LADDER_NODES` fold radius 0.14->0.06 (L=30<L_MAX stays
    WAVE, fold arm serves; verified fold-not-cusp + F_op order=0). +F028
    comment. Sibling ladder tests (_ladder_route mirror is is_saddle-only
    so pos-parity wave nodes still label 'fold') unaffected.
  * fast_path FOP_REFUSALS[-1] (63,0.9,0.2)->(63,0.3,0.2): hard-core WAVE
    (unresolved dmin=0, both arms decline) -> RAISE
    SchwingerCertificationError. Fixes refusal test + scalar flip witness
    (both get refused>0 from this entry). +F028 comment.
  * fast_path grid flip witness: FOP_GRID_SQRT_S is shared by 6 tests so
    could NOT drop 0.9; above ceiling ANY |y|~0.9 is unavoidably geometric
    (L=w*0.9>54, resolved). So added a THIRD dispatch branch per Inspector's
    explicit sanction: on-axis supra outcomes refused=2(ss0.3 b0)/arm=2(ss0.3
    b0.7)/geometric=4(ss0.9). Geometric branch is DISPATCH-parity only
    (node_value == geometric_amplification byte-exact, order 0) NOT an
    accuracy gate -- independent geometric-accuracy gate STILL OWED to Test
    Dev. +served_geometric>0 anti-vacuity.
  * operator sheared test y=[1.0,0.0]->[0.08,0.0] (hard-core wave refuse,
    RAISE SCE); kept [0.05,0.0]. +F028 docstring.
  Key routing facts (w=63, gamma=0.2): on-axis ss<=0.3 -> wave (dmin=0
  unresolved); ss=0.9 -> geometric. To keep an above-ceiling node WAVE:
  w*|y'|<48 OR w*delta_min<4. geometric-served nodes report order 0.

- OWED->Test Dev (INS-1-001/002/003, positive-parity-resolved-first):
  Inspector routed 3 TEST-file findings to Coder after WP1 gave
  _positive_parity_grid its select_branch geometric branch. CONFIRMED
  production is CORRECT (WP1 routes above-ceiling nodes through
  select_branch(w, delta_min, w*|y'|); WP2 saddle via inf-cancel leg) —
  NOT a production defect. Findings are stale fixtures encoding the OLD
  'every above-ceiling pos-parity node hits arm/refuses' contract. Did
  NOT edit the tests: selecting fixtures + flipping assertions that
  certify my own WP1 change is self-grading (role: Coder never authors
  gates for own code; Test Dev re-points). Precise work order:
  * test_lensing_airy_fold.py: _LADDER_NODES 'fold' entry (line ~1838:
    ('fold',500.0,0.14,_RAY_ANGLE,_GAMMA=0.3,0,0)) now routes geometric
    (delta_min~0.134, w*delta_min~67>=RHO_END=4, L=w*|y'|=70>L_MAX=48).
    Re-point to STAY 'wave'/fold-served: drop into wave regime via
    w*|y'|<48 (smaller |y|/gamma) OR unresolved w*delta_min<4, so the
    fold arm is still consulted. _UNIFORM_LADDER_NODES derives from it.
    Keeps test_fixed_priority_fold_tried_before_cusp +
    test_served_value_equals_labelled_rung_bitwise meaningful. Add F028
    docstring note.
  * test_lensing_fast_path.py (~1506): 3 tests —
    test_fop_refuses_uncertifiable_contractions needs a genuinely
    hard-core node (unresolved w*delta_min<4 AND both arms decline) so
    refused>0; test_fop_grid/scalar_schwinger_arm_flip_witness need
    above-ceiling fixtures that stay wave/refusing (w*|y'|<48 or
    w*delta_min<4), else update witness expectation to geometric.
  * test_lensing_operator.py OperatorOracleTestCase::
    test_sheared_host_above_ceiling_refuses_schwinger: pick a sheared
    above-ceiling fixture that is genuinely wave-routed + hard-core
    (unresolved, no arm certifies) so SchwingerCertificationError still
    fires, OR split into geometric-served vs still-refusing. Cite F028.
  SEPARATE OWED: no test yet exercises the NEW geometric serve on
  pos-parity above-ceiling nodes — Test Dev to ADD that positive gate.

- WP2 (operator.py _saddle_grid): replaced hand-rolled
  `w>ceiling and w*delta_min>=RHO_END` geometric test with
  `select_branch(w_node, delta_min, math.inf)=='geometric'`. Per
  Professor ruling passed cancellation_exp=inf so strongly_cancelling
  leg is vacuously true and ONLY the resolution leg is live ->
  algebraically byte-identical to old `resolved` (verified:
  select_branch(70,1,inf)=geometric, (70,0.001,inf)=wave). Restructured
  cascade: ceiling is now the ENCLOSING `if w_node>ceiling`, inner
  if/else on select_branch (geometric branch) vs arm/ceiling_refusers
  (unchanged), else=batch tail (unchanged). math already imported
  (line 170). Docstring + inline pre-pass comment updated: states inf
  cancellation exp -> only resolution leg live, PRESERVES w>60 AND
  resolved boundary EXACTLY (boundary did not move), and that a saddle
  geometric-onset gate (L>L_MAX accuracy leg) is OPEN/UNMEASURED (F028
  sweeps positive-parity only; ceiling exhaustion explains wave
  unavailability not geometric accuracy). Untouched: delta_min
  compute-once guard, batch tail, refusal reduction. parse+import OK.

- WP1 (Build re: F028): `_positive_parity_grid` above-ceiling nodes now
  route via `select_branch(w, delta_min, w*y_prime_norm)`. delta_min +
  macro_matrix guarded behind `np.any(w>W_CEILING)` (skip quartic below
  ceiling, accept #6); y_prime_norm=sqrt(y_scaled@y_scaled) reuses the
  norm already computed (== cancellation_exponent/w, no per-node
  _mass_sheet_map). 'geometric'->geometric_amplification(physical y),
  'wave'->existing _uniform_arm_value fold/cusp then named refusal. w<=ceiling
  untouched/byte-identical. Frame discipline: physical y/beta/matrix only.
  Smoke: below-ceiling finite; gamma0.9/w500 geometric serves finite;
  wave-branch refusals still raise SchwingerCertificationError.
- OWED (other roles, per brief): _saddle_grid still uses resolved-only rule
  (w>ceiling AND w*delta_min>=RHO_END, no L>L_MAX) — likely separate WP.
  SPEC serving-ladder + FINDINGS(F028 ~1% O(1) tail) + todo.d/completed.d/
  spec_changelog.d fragments NOT done here (Inspector/Librarian own those).
  Existing tests encode old 'every above-ceiling node hits the arm' contract
  (test_lensing_levers refusal tests + brief's blast-radius list) — Test Dev
  to re-point.
