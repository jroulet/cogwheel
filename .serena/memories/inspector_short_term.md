# Inspector Short-Term Observations

## 2026-07-22 (Build 8g re-review #4) — uncommitted tree, code GREEN

Worktree /home/tejaswi/Work/cogwheel-claude-dev. Full python:
/home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
`git diff --stat HEAD` on SPEC.md + DATA_CONTRACTS.yaml = EMPTY (untouched).
Code file: cogwheel/lensing/surrogate_training.py (+536/-... ). Tests:
test_lensing_surrogate_training.py (new), test_lensing_levers.py (8f fixture adapt).

### VERDICT: ISSUES — INS-1-003 still open (Librarian/spec-sync) + one new trivial.

### WP verification (all correct)
- WP1 eps gate: `_chart_gated`/`_gate_chart`/`_load_or_build` legacy_no_eps
  pass-through correct. Gated charts appended with marker + `continue`, NOT
  added to `charts` (additive-serve F005 intact). `train` builds `gated_charts`
  census (per-parity nan_eps/eps_above_bar). Solid, well-tested.
- WP2 tiling: `_mass_strata` (log strata, R=sqrt(f_hi/f_lo)), `_stratum_w_range`
  (INS-1-001 sqrt(2) corner factor INTACT at line 871 `y_corner = Y*sqrt(2)`),
  `_farfield_tiles` (min-L2 disk-exclusion `hypot(max(0,|cx|-h),max(0,|cy|-h))`).
  y_extent=`_source_scale(m_lo)` = per-axis half-width (u_max=1), tiling
  `[-Y,Y]^2` matches prior y=u*Y. Whole-band containment [w(f_lo,m_lo),
  w(f_hi,m_hi)] correct (w monotone in f,m). beyond_w_cap/truncated/zero-tile
  strata all recorded LOUDLY, never silent-dropped.
- WP3 saddle cusp: `_find_cusps` gained keyword-only width_safety/min_halfwidth.
  Astroid path (line 476) uses DEFAULTS -> byte-identical (AstroidByteIdentity
  green). Saddle path (513) passes _SADDLE_CUSP_* + wedge-edge guard windows in
  `_saddle_arcs`. Correctly isolated.

### NEW behavioral addition beyond the 3 WPs (benign, owner-mandated)
- `_min_curvature_radius` (line ~742) + tube-skip in `_train_band_charts`
  (`if config.eta_max > 0.5*r_min: skip foot_of_normal`). NOT in HEAD
  (`git grep` confirms). Circumradius formula abc/(2*area2) with area2=2*Area
  => R=abc/(4A) CORRECT. Refusal-conservative (skip->ladder serves, never wrong
  serve). geometry.critical_point(...).source signature matches 4 pre-existing
  call sites. Brief lists foot-of-normal as owner-mandated. Benign plan
  deviation, not a defect. BUT SPEC training narrative doesn't mention it ->
  folded into INS-1-003 scope for Librarian.

### NEW finding
- INS-4-001 (trivial/design): TrainingConfig.max_farfield_regions default=1.
  Under the new tiling its meaning changed from "1 full box (adequate legacy
  coverage)" to "1 tile of ~16 admitted" => severe truncation at defaults.
  Production/driver overrides it (brief), tests set it, truncation recorded
  loudly -> not a correctness bug. Recommend bumping default or documenting
  "production MUST set max_farfield_regions".

### Prior-findings status
- INS-1-003 STILL OPEN (Librarian). SPEC.md + DATA_CONTRACTS untouched
  (git diff --stat empty). Training paragraph still describes single-box
  raw-coordinate far-field; omits mass-stratified tiling, eps registration
  gate, foot-of-normal runtime skip, beyond_w_cap report. todo.d fragment
  surrogate_farfield-tiling-eps-gate.md tags it [→ spec]. spec_version bump owed.
- INS-1-001 / INS-2-001 / INS-2-002 confirmed-resolved in re-review #3 (sqrt2,
  non-circular oracle, legacy resume test) — not re-listed this task.

### Tests run this session (subset, GREEN)
- AstroidByteIdentityTestCase + 2 TilingRecord geometry tests: 6 passed (5s)
- EpsGateResume legacy + WholeBandContainment recompute/beyond-ceiling: rc=0
- import surrogate_training OK.
Full suite (~200s/class) too slow for serena 120s shell cap; re-review #3
already ran all classes green on this identical tree (git status matches).

### OPEN carried
- INS-1-003 (Librarian): SPEC far-field narrative + eps gate + foot-of-normal
  + beyond_w_cap; bump spec_version.
- INS-3-001 (Librarian, older): SPEC omits registered pearcey_table product.
