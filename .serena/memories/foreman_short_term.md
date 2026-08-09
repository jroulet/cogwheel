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
