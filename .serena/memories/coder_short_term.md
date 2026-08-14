# Coder Short-Term Observations

## 2026-08-14 (WP1 revision — INS-1-001 census diagnostic fold-consistency)

- classify_fallthrough (surrogate_census.py) HAS y1_eig/y2_eig params and
  characterize_sample passes the REAL eigenframe source, but the
  cusp-window probe called `_tube_serves(..., image_count)` with RAW theta,
  OMITTING y1_eig/y2_eig — so the cusp-window fall-through CATEGORY was not
  D2-equivariant (mirror-image draws misclassified, per-category gap counts
  deflated up to 4x). The sibling exterior-polar refusal-ball probe already
  passed them. FIX: append `y1_eig, y2_eig` to that `_tube_serves` call
  (positional, after image_count — matches the nan-default signature). Served
  flag / chart_index / aggregate serve fraction were ALWAYS fold-invariant
  (production serving folds via select_chart); this is a DIAGNOSTIC-ACCURACY
  defect only, no over-certification.
- Closed the deliberate category exclusion in test_lensing_tube_d2_fold.py
  `_assert_route_equality`: added `categories=[r.category...]` to the D2
  equality pin + corrected the stale "category deliberately NOT asserted"
  comment. Done at Inspector's explicit direction (INS-1-001 suggested fix);
  pre-existing test scaffolding for already-landed D2 physics, and the class
  docstring already claimed "same fall-through category" so the pin now
  matches its own contract. Diagnostic test figure-title ('census route is
  identical across D2 sign images') is now accurate with no change.
- LESSON: "census inherits the fold via the NaN default" holds ONLY for a
  caller WITHOUT the source in scope; any caller that DOES have y1_eig/y2_eig
  must thread them or it silently runs the unfolded (identity) diagnostic.

## 2026-08-14 (INS-2-001/002 — census mirror + dead max_tube_arcs knob)

- INS-2-001: scripts/census_dry_run.py saddle path computed arc_r_min over
  `structure.arcs[:cfg.max_tube_arcs]` — STALE MIRROR after WP1 routed
  production tube-arc selection through `_tube_training_arcs` (saddle => ALL
  arcs). Fix: `tube_arcs = _st._tube_training_arcs(structure, _SADDLE_PARITY)`
  (already importable via the `_st` = surrogate_training alias, no new logic),
  iterate arc_r_min over `tube_arcs`. Now the dry-run's reported max_eta_max
  bound == what the trainer builds (served == counted). Diagnostic/mirror
  accuracy only; production serving + aggregate serve fraction were already
  correct.
- INS-2-001 cont.: `config.max_tube_arcs` no longer consumed by
  `_train_band_charts`. Removed the dead `max_tube_arcs=20` assignment AND
  its banner print line from scripts/train_surrogate_production.py (grep
  clean, 0 residual). Added a 3-line comment on `TrainingConfig.max_tube_arcs`
  (surrogate_training.py L305) stating it no longer governs production tube
  training (superseded by `_tube_training_arcs`) and is retained only for
  tests that set it explicitly. Did NOT touch test_lensing_caustic_cusps.py
  slices (unrelated test-local uses) per Inspector direction.
- INS-2-002: NO ACTION NEEDED — the `_evaluate_chart` docstring
  (surrogate.py L3327-3333) was ALREADY reworded during the WP1 serve-fold
  edit; it now says "A tube chart also consumes them, to fold theta into the
  D2 fundamental domain via _fold_caustic_theta ... (the same fold applied at
  the _tube_serves gate)". No "ignored for a tube chart" text remains anywhere
  (grep empty). Finding was based on a pre-WP1 snapshot. All 4 files
  py_compile OK.

## 2026-08-14 (INS-1-001 — census cusp-window fold consistency)

- surrogate_census.py `classify_fallthrough`: the cusp-window probe
  `_tube_serves(relaxed, gamma, log_w_min, log_w_max, eta, theta,
  image_count)` was RAW-theta (unfolded) while its SIBLING exterior-polar
  refusal-ball probe already threaded y1_eig/y2_eig. Fix: pass
  `y1_eig, y2_eig` (positional, matching `_tube_serves` nan-default
  signature added by WP1). Now the cusp-window fall-through category is
  D2-equivariant — a mirror-image (negative-eigenframe) draw classifies
  identically to its first-quadrant counterpart; no 4x-deflated per-category
  gap counts under unfolded census sampling. Serving (served flag +
  chart_index + aggregate fraction) was ALREADY fold-invariant, so this is a
  DIAGNOSTIC-ACCURACY fix only, never over-certification.
- CORRECTION to my earlier "CENSUS NEEDS ZERO EDITS" WP1 claim: true for the
  SERVE path (fold-invariant), FALSE for the classify_fallthrough diagnostic
  category — a caller that HAS the eigenframe source in scope must thread it;
  the nan-default identity only covers callers WITHOUT a source.
- test_lensing_tube_d2_fold.py `_assert_route_equality`: closed the
  deliberate category exclusion at Inspector's explicit direction — added
  `categories = [r.category for r in records]` + a single-set equality pin,
  replaced the stale "category NOT asserted" comment. Coder editing a test is
  permitted here per inspector_knowledge precedent (PRE-EXISTING test pinning
  already-landed physics, Inspector-directed). Both files py_compile OK.

## 2026-08-14 (WP1 — D2 tube serve-fold + astroid fundamental-arc training)

- surrogate.py serve fold: new private `_fold_caustic_theta(theta, y1_eig,
  y2_eig)` before `_theta_into_frame` — exact D2 reflection (y1_eig<0 ->
  pi-theta; y2_eig<0 -> -theta), PARITY-AGNOSTIC (identical arithmetic
  astroid+saddle, no parity special-case). Applied at BOTH `_tube_serves`
  gate (threaded y1_eig,y2_eig params, NaN defaults) and `_evaluate_chart`
  tube branch. `serve()` public signature UNCHANGED; y1_eig/y2_eig already
  computed there via `_rotate_to_eigenframe`. eta (caustic_distance) is
  D2-invariant, passes UNFOLDED. NaN default => identity => surrogate_census
  (`classify_fallthrough` calls `_tube_serves` w/o source) inherits
  no-op; `characterize_sample` passes real y1_eig/y2_eig -> inherits real
  fold. CENSUS-NEEDS-ZERO-EDITS was WRONG for the DIAGNOSTIC path — see INS-1-001
  correction below. (Serving IS fold-invariant; the classify_fallthrough
  cusp-window CATEGORY was not.) Original (partly-wrong) claim: zero census
  edits, single-source the convention — single-source
  the convention, no second fold in the tree.
- FRAME SLIP CORRECTED (deviation from brief, documented in code): brief
  said train the astroid arc "bracketing pi/2" with "cusps on the DIAGONALS
  {pi/4,3pi/4,...}". That is the SOURCE-PLANE frame. In THIS code's caustic
  gauge angle the astroid cusps (caustic-speed minima from `_find_cusps`)
  sit on the AXES {0, pi/2, pi, 3pi/2} — measured deterministically via
  `detect_caustic_structure(g,1)` across gamma {0.2..0.9}; arc0=first
  quadrant [~0.14,~1.48]. pi/2 is a CUSP, not an arc interior: selecting on
  pi/2 returns ZERO arcs (a serve regression). Correct predicate brackets
  **pi/4**. `_tube_training_arcs(structure, parity)` (new helper before
  `_train_band_charts`): parity==1 -> `[arc for arc in structure.arcs if
  arc.theta_lo <= 0.25*math.pi <= arc.theta_hi]`; parity==-1 -> all arcs
  unchanged (saddle F079 closes via SAME serve fold; deltoid lobes are fold
  images handled by wedge/lobe). Deterministic ID from FoldArc fields (NOT
  a new empirical measurement) so no escalation. `_train_band_charts` uses
  `tube_arcs = _tube_training_arcs(...)` at BOTH sites (arc_r_min comp +
  enumerate loop), replacing `structure.arcs[:config.max_tube_arcs]`.
- ACCEPTANCE EVIDENCE (read off deterministic structure, no campaign):
  astroid tube charts per gamma band 4 -> 1 (arc count 4->1); engine calls
  scale linearly with tube-chart count (identical per-arc node grid,
  unchanged) => ~4x reduction. Saddle 6 -> 6 (unchanged). Verified
  `_tube_training_arcs` returns 1 for astroid g{0.2,0.5,0.9}, 6 for saddle
  g{1.1,1.5,2.0}. _EXPECTED_ARCS / detect_caustic_structure UNTOUCHED
  (topology guard still detects 4/6). Fold identity on fundamental domain
  (y1_eig>=0,y2_eig>=0) => byte-identical to unfolded incumbent there.
  py_compile OK both files; D2 fold arithmetic unit-checked incl NaN->id.

## 2026-08-14 (WP2 — retire cusp-arm coverage constants in surrogate.py + census note)

- surrogate.py: DELETED `_SADDLE_CUSP_ARM_COVERAGE = 0.0` /
  `_CUSP_ARM_COVERAGE = 0.07` (~L295-313) and both preceding comment
  blocks; kept `_MACRO_SADDLE_EXTERIOR_IMAGE_COUNT = 2` and the
  `_DEFAULT_ARTIFACT_NAME` block. In `_tube_serves` (~L2886) dropped the
  `coverage = (_SADDLE... if parity==-1 else _CUSP...)` +
  `residual = max(0, delta_theta - coverage)` -> `residual = delta_theta`
  (full-window exclusion); rewrote the comment shrink-free (no
  `_CUSP_ARM_COVERAGE` token, notes post-F074 no angular serve boundary).
- surrogate_census.py `classify_fallthrough`: KEPT the `cusp-window`
  category (detection = relax cusp_windows to empty + re-call
  `_tube_serves`, untouched, still valid); corrected item-4 note to state
  WHY kept (tube cusp-window exclusion real+unchanged over full window)
  and per F074/F079 cusp losses now surface as eta-floor/w-cap, no angular
  arm boundary. No `_CUSP_ARM_COVERAGE` literal.
- VERIFY: grep clean (0 tokens) in both files; py_compile OK on both.
  Scope was surrogate.py + surrogate_census.py ONLY — the WP1
  surrogate_training.py wrap fix, the test-suite retirements, and the
  scripts/ deletions (census_dry_run.py, calibrate_ppgo_rung.py,
  measure_*_cusp_arm_*.py) are OTHER WPs in this build, not touched here.

## 2026-08-14 (WP3 — delete dead cusp-arm measurement scripts + census re-express)

- `git rm` scripts/measure_cusp_arm_reach.py, measure_cusp_arm_actual_boundary.py,
  measure_saddle_cusp_arm_coverage.py, calibrate_ppgo_rung.py. Confirmed no
  production/test import references them (only docs: FINDINGS/COMPLETED/TODO/
  todo.d/changelog.d + one provenance comment — all Librarian/Inspector scope).
  measure_cusp_exclusion.py is a DIFFERENT script, correctly retained.
- scripts/census_dry_run.py: deleted mirrored `_CUSP_ARM_COVERAGE=0.07`;
  added `_CUSP_ARM_W_FLOOR=49.0` (no importable production constant — F074
  w-floor confirmed 49 in FINDINGS ~L4356). cusp_arm route now
  `if is_near and w >= _CUSP_ARM_W_FLOOR` (w IS in classify_draw scope) —
  angular `delta_cusp` no longer gates. Tube residual arithmetic
  (`residual = max(0, _TYPICAL_CUSP_HALF_WINDOW - _CUSP_ARM_COVERAGE)`)
  replaced by full-window exclusion `delta_cusp < _TYPICAL_CUSP_HALF_WINDOW`
  (mirrors WP2's `_tube_serves` full-window change). Banner prints w-floor
  not coverage. py_compile OK; grep clean (0 tokens) across scripts/ incl. pyc.
- FLAG -> Inspector/Librarian: cogwheel/lensing/chang_refsdal/_pearcey_cusp.py
  ~L447 has a live provenance comment "Measured: scripts/calibrate_ppgo_rung.py
  sweep..." pointing at a now-deleted script (documents _W_PPGO_FLOOR=8.0
  origin). Left untouched — out of WP3's census-only edit scope + historical
  provenance like a changelog. Adjudicate whether to reword.
</content>
