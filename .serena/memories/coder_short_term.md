# Coder Short-Term Observations

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
