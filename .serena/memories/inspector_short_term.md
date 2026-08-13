# Inspector Short-Term Observations

## 2026-08-13 — Build ppgo_interior_certificate (v2 re-review, working-tree diff)

Scope: uncommitted working-tree diff, worktree cogwheel-claude-dev / branch
claude-dev. Code files changed: chang_refsdal/geometry.py (+218: poly-algebra
Isserlis/Wick c3 series `_series_coefficients`, `_c3_coefficient`, public
`ppgo_error_estimate`), chang_refsdal/__init__.py (+1 export, import first, no
cycle), likelihood.py (interior handoff rung re-gated), ppgo_map.py
(docstring-only, rho-is-not-a-predicate clarification). New test file
test_lensing_ppgo_certificate.py.

### Re-gate summary (likelihood.py ~1785-1865)
OLD 3-leg fold gate (rho<=1 + _merging_fold_pair/xi_min>=_XI_FOLD_THRESHOLD +
_uniform_error_estimate<=BAR) REPLACED by exact `int(geom.real_mask.sum())==4`
predicate + c3 certificate `est*_PPGO_INTERIOR_SAFETY(=2.0) <=
CERTIFICATION_BAR(1e-4)`. Serves raw ppGO via geometric_amplification ->
reconstruct_farfield(FARFIELD_KERNEL_SUM). except
(LensDomainError,ValueError,ZeroDivisionError): pass. _XI_FOLD_THRESHOLD=4.0
RETAINED (still imported by surrogate_census + test_lensing_fold_ppgo_handoff).

### Verification (all GREEN)
- test_lensing_ppgo_certificate.py: 16 pass (11s). c1/c2 vs saddle_coefficients
  1e-12; c3 purely imaginary; exact w**-3 ratio; conservativeness vs
  f_schwinger true_err<=cert; near-caustic self-refusal; self-falsification.
- test_lensing_fold_ppgo_handoff + test_lensing_likelihood: 31 pass / 15 skip /
  1 xfail (27s). Handoff tests exercise the RETAINED _airy_fold helpers
  directly (not the removed production leg) — NOT stale. TRAIN_TIER classes
  skipped test fold helper properties, unaffected by the re-gate.
- Import chain clean; ppgo_error_estimate exported; likelihood +
  surrogate_census import OK.
- INS-1-002 (empty real_images -> 0.0) RESOLVED: guard
  `if w_min <= 0.0 or len(real_images) == 0: return None` present at top of
  ppgo_error_estimate; test_none_for_nonpositive_w_min green.
- Behavioral equivalence for non-served interior: 4-image ⟹ interior ⟹ rho<=1
  (scalar reach is max), so the removed `rho<=1` outer guard cannot admit a
  new config; rho>1 & 4-image only at the boundary where cert self-refuses.
  != 4 images falls through to the same terminal `return None` as before.

### Findings
- INS-2-001 (design, REPORT-ONLY per approved plan; = carried prior INS-1-001):
  surrogate_census.py characterize_sample (~L468-505) STILL uses the OLD
  xi-based fold gate (_merging_fold_pair / xi_min>=_XI_FOLD_THRESHOLD /
  _uniform_error_estimate) to classify `ppgo_fold`, with a comment claiming to
  "Mirror _surrogate_coefficients". The likelihood rung it mirrored is now the
  exact-4-image + c3 certificate -> census ppgo_fold counts skew vs what
  likelihood serves. Plan scoped this consumer report-only (do NOT re-gate this
  build). Non-crash classification skew.

### Resolved
- INS-1-002 (empty real_images -> 0.0 read as admit-zero) resolved by the
  explicit len==0 guard.

### Patterns (carried)
- MIRROR-BREAK BY RE-GATE: swapping a serve gate in likelihood.py silently
  diverges any census "dry-run" that claims to MIRROR it, even though the
  census file is untouched. grep census for OLD gate symbols after any
  likelihood serve re-gate.
- CERTIFICATE SOUNDNESS: sum_a sqrt|mu_a||c3_a|/w^3 is a triangle-inequality
  UPPER bound (conservative) ONLY where every image is real + ghost==0 = the
  4-real-image interior. The exact image-count predicate makes the ghost-free
  certificate valid; the rho<=1 scalar gauge does not.
