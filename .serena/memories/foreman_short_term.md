# Foreman-Lite Short-Term Observations (2026-08-09)

## Session: INS-14-001 trivial control-flow fix

- INS-14-001: in `_exclude_near_cusp` (cogwheel/lensing/surrogate_training.py
  ~1798), the `for g in gammas` band-edge loop bailed the WHOLE function with
  `return False` when a single gamma (e.g. gamma_lo) resolved no cusp
  positions — silently skipping gamma_mid/gamma_hi. Changed to `continue`
  (a gamma with no cusps contributes no exclusion constraint; the check
  proceeds to the remaining gammas). Verified semantics match the docstring:
  "tile excluded if ANY corner within d_exclude of ANY cusp at ANY of the
  three gammas".
- Verified with targeted read_file (DIFF TRAP: the edit sits INSIDE a parallel
  build's uncommitted exterior-cusp-exclusion block, so git diff shows the
  whole block as '+' and cannot isolate it — confirmed the exact new text
  `if not cusp_positions:\n            continue` via source read).
- Test suite: cogwheel/tests/test_lensing_exterior_admission.py — 53 passed,
  1 skipped (no regression). ast.parse + import probe green.
- Note: `_exclude_near_cusp` docstring "A domain refusal from
  `geometry.r_caustic` is treated conservatively as excluded" is now MORE
  consistent with code after the fix (per-angle LensDomainError `continue`
  means that cusp contributes nothing; a fully-empty gamma contributes no
  constraint). Pre-existing nuance, not introduced here, not in scope.

## Session: INS-17-002 + INS-18-002 trivial doc/dead-code fixes

- INS-17-002: removed dead `n_w = log_w_grid.size` in the k-chart estimation loop of `train_exterior_polar_surrogate` (surrogate.py ~3032). Verified via exact-string assert (count==1) — the DIFF TRAP (parallel-build uncommitted schema-v3 block) shows the whole region as '+', so byte-string grep, not git diff, is the isolation method. `n_w` had zero other references (grep count 0 after removal).
- INS-18-002: `_validate_exterior_polar_axis_schema` docstring now lists ALL retired schemas (`exterior_polar_rho_theta_c`, `exterior_polar_rho_u_v1`, `exterior_polar_carrier_demod_v2`), not just the one. Confirmed the retired names are historical: `exterior_polar_carrier_demod_v2` retired in f4652e7, `exterior_polar_rho_u_v1` in 1a97bbd, `exterior_polar_rho_theta_c` in 4d59a6d.
- Tooling: Serena MCP tools were NOT exposed in the OpenCode backend session (only built-in bash/skill/task available) — memory write + file edits done via built-ins. ast.parse + import probe via full conda python path green. No Serena fallback-to-builtin transitions needed beyond this.
