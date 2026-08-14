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

(empty — last consolidated by Dreamer on 2026-08-14)
</content>
