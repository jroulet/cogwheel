# Architect Short-Term Observations

## tube_d2_fold build (2026-08-14, Architect)
- Professor (code-grounded) settled the core design: caustic_theta is a LENS-plane
  gauge angle; under D2 source reflections at eigenframe (beta=0) it maps by EXACT
  arithmetic: s1=sign(y1_eig)<0 -> t->pi-t; s2=sign(y2_eig)<0 -> t->-t; both->t->pi+t.
  Reflect the gauge angle (subtraction + %2pi via existing _theta_into_frame) — do
  NOT recompute caustic_theta from the abs-folded source (nearest_caustic_point is a
  Newton/argsort search: only ~1e-13, and costs an engine call). Reflection is
  parity-agnostic (same formula astroid+saddle; saddle interpretation = lobe-swap
  0<->pi for s1, branch-swap for s2).
- Professor CORRECTED handoff fact 2: select_chart runs tube FIRST, so saddle tube is
  live+unfolded (6 arcs reachable) — F079 half-ring hole exists on saddle too.
  Decision: SERVE fold both parities (closes F079 both, enables equality pin both);
  TRAINING reduce astroid 4->1 only (fundamental arc = caustic_theta in (pi/4,3pi/4),
  the arc bracketing pi/2). Saddle training left at 6 (F079 closes via serve fold).
- Tolerances (Professor): D2 equality pin across the 4 sign octants = BIT-EXACT ==
  (identical float64 inputs to np.interp/spline). Fundamental-domain query (s1=s2=+)
  vs unfolded incumbent = bit-exact too (fold is identity there). Folded-arc vs the
  OLD reflected-arc serve = only rtol 1e-6 (independently integrated theta_to_s) —
  do NOT bit-pin that; it is transitional, not a durable test.
- Simplifier: fold tube-local via production functions the census reuses -> census
  auto-current, NO census code change (Item E lean). Do NOT consolidate the abs()
  idiom at serve() (Item B2 trim, scope creep). Keep _EXPECTED_ARCS {1:4,-1:6}
  UNCHANGED (topology guard runs in detect_caustic_structure, separate gate from the
  training slice). Check whether max_tube_arcs default already restricts to 1.
- ONE Coder WP (surrogate.py serve fold + surrogate_training.py fundamental-arc
  selection) — one head owns the "which arc is fundamental" convention. Count
  reduction = acceptance evidence quoted in completed.d, not a permanent test.

# tube_d2_fold triage (2026-08-14)
- INS-1-001: classify_fallthrough's tube _tube_serves call (line 325) omits
  the new y1_eig/y2_eig fold args that the SIBLING exterior-polar call in the
  same function (line 340) already passes -> internal inconsistency (coder
  missed one of two call sites), not a design ambiguity. coder_fix: thread
  y1_eig,y2_eig into the line-325 call to match line 340; fundamental-octant
  draws fold to identity so byte-identical there. Also remove the
  deliberately-excluded equality-pin exclusion and fix the misleading
  'identical across sign images' figure title once the category is actually
  fold-consistent.
</content>
