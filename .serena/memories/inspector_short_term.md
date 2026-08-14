# Inspector Short-Term Observations

## 2026-08-14 (saddle_farfield_serve_gate rewrite — RE-REVIEW) — VERDICT: PASS

Scope: uncommitted working-tree diff (c3-led saddle far-field serve gate).
Re-review of prior-pass findings INS-1-001/002/003. ALL THREE RESOLVED,
NO new findings. Files: likelihood.py, surrogate_census.py,
calibrate_saddle_exterior_certificate.py; 3 orphaned test files DELETED;
NEW untracked test_lensing_saddle_serve_gate.py.

### INS-1-001 (double-mask crash) — RESOLVED
Both saddle caller sites now build `real_images = np.asarray(geom.images)`
directly (no `[real_mask]` re-index):
- likelihood.py `_saddle_farfield_analytic` L2153.
- surrogate_census.py saddle block (gamma>1) L523 — diff confirms the old
  `real_delays = np.asarray(geom.delays)[real_mask]` + 3-arg
  `(real_delays, w_lo, rho)` call is GONE, replaced by the 4-arg
  `(real_images, source, matrix, w_lo)` with `matrix = macro_matrix(gamma,
  0.0,0.0)` (census kappa=0 contract, faithful mirror).
- calibrate script L272 also migrated; uses a defensive `_real_images(geom)`
  helper that handles both length regimes.
- Predicate docstring reworded: "geom.images is already the real-only
  array; pass it directly -- do NOT index it with the length-4 channel
  mask". 
NOTE: the interior `image_count==4` ppgo_fold block in census STILL uses
`np.asarray(geom.images)[real_mask]` — CORRECT and harmless there (len-4
images, all-True mask = no-op); pre-existing (Build ppgo_interior_certificate),
not this diff.

### INS-1-002 (orphaned siblings error at collection) — RESOLVED
The 3 offending files are git-DELETED (status `D`):
test_lensing_saddle_gauge.py, _tier1_accuracy.py, _tier1_refusal.py.
Search for `_SADDLE_FARFIELD_RHO_FLOOR` across cogwheel/ = EMPTY. Search
for `_saddle_farfield_analytic_serves(` across cogwheel/tests = only the
new serve-gate file (new signature). rho-floor contract fully superseded.

### INS-1-003 (masked-red tripwire) — RESOLVED
`test_census_crashes_reproducing_production_args` DELETED; NO
`@expectedFailure` and NO `assertRaises(IndexError)` anywhere in the file.
`test_census_served_matches_production_gate` is now a LIVE, undecorated,
non-vacuous assertion comparing the production gate boolean to the census
`saddle-farfield-analytic` verdict on both a serve and a refuse config.
RAN the file: 33 passed / 0 xfail / 5.4s.

### Verified OK
- All 4 production/test callers use the new 4-arg signature (find_referencing_symbols).
- `caustic_rho`/`LensDomainError` still used in census (L305/416/495) — no orphaned imports.
- `import cogwheel.lensing.likelihood, surrogate_census` OK.
- New module constants present: _SADDLE_FARFIELD_SAFETY=20.0 (L211),
  _CERT_BAR=1e-3 (L212), _MIN_IMAGE_SEP=0.05 (L223); RHO_FLOOR retired.

### Carry-forward
- New test file test_lensing_saddle_serve_gate.py is UNTRACKED (`??`) — driver must `git add` before commit.
- MANIFEST TRUST TRAP reconfirmed: task manifest listed the 3 test files as
  "changed" but git showed them DELETED and the new file untracked — always
  git-status + run, never trust the manifest.
