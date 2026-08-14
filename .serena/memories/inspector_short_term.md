# Inspector Short-Term Observations

## 2026-08-14 — F079 re-review pass 2 (VERDICT: PASS, INS-1-001 resolved)

Scope: git working tree, /home/tejaswi/Work/cogwheel-claude-dev.
Files: _pearcey_cusp.py, surrogate.py, surrogate_census.py,
surrogate_training.py, census_dry_run.py, 4 test files.

### INS-1-001 RESOLVED
`_pearcey_cusp.py:447-449`: provenance comment for `_W_PPGO_FLOOR = 8.0`
reworded — drops the `scripts/calibrate_ppgo_rung.py` live path, now reads
"The retired calibration sweep observed sub-percent agreement for w >= 5 ...
floor set to 8 (1.6x safety margin over that measured region)". No dangling
script reference in shipping code. Confirmed no other refs to the 4 deleted
scripts anywhere in cogwheel/ (grep clean).

### Verified correct this pass
- WP1 wrap-aware span (`_find_cusps` :602-611): periodic-gated house idiom
  `abs((a-b+pi)%2pi-pi)` on BOTH sub-spans; `periodic=False` (saddle) takes
  ELSE branch → byte-identical saddle path. All 7 `_find_cusps` call sites
  checked: astroid uses periodic=True (line 643,1691), saddle periodic=False.
- Arc-count check (`detect_caustic_structure`): `_EXPECTED_ARCS={1:4,-1:6}`.
  RE-PROBED full production box this pass: astroid gamma∈linspace(0.05,0.99,25)
  → all 4 arcs; saddle gamma∈linspace(1.01,1.6,20) → all 6 arcs; ZERO spurious
  raises. (Out-of-box gamma=3.0 saddle drops to 2 → RAISE, latent only if box
  ever extends >~1.6; not actionable.)
- surrogate.py: `_CUSP_ARM_COVERAGE`/`_SADDLE_CUSP_ARM_COVERAGE` deleted;
  `_tube_serves` now `residual = delta_theta` (full-window), `two_pi` wrap
  intact. Only surviving ref = intentional docstring in
  TubeCuspWindowExclusionTestCase (:5064) documenting the retirement.
- surrogate_census.py `classify_fallthrough` cusp-window note: category KEPT
  with sound justification (real structural gate), F074/F079 wording correct.
- census_dry_run.py: `_CUSP_ARM_W_FLOOR=49.0` mirror replaces coverage;
  residual/banner/route re-expressed on w-floor; ast.parse OK.

### Scripts
The 4 WP3-target scripts are NEITHER in HEAD tree NOR index NOR on disk —
already untracked/removed, so no git `deleted:` trace. Nothing to flag.

### Tests
362 collect clean across all 4 files (269-line deletion in
surrogate_training left no dangling refs). Ran & PASS:
AstroidArcSurvivalTopology + Theta0CuspWindowValue (5), TubeCuspWindowExclusion
(3). Fixtures derive window edges from chart.cusp_windows (schema-following).

### Non-finding note (accepted, not flagged)
census_dry_run `_CUSP_ARM_W_FLOOR=49.0` is a pinned literal in a DIAGNOSTIC
dry-run script (no artifact consumer, not a gate, not shipping serve logic).
Coder comment states no importable production constant exists to mirror.
Low blast radius; accepted per prior review. Not a test fixture → checklist#9
derive-don't-pin does not bite.

### Carried to Librarian (doc-sync, NOT code defects)
- TODO.md / todo.d fragments + FINDINGS F079 resolution pointer reference the
  now-deleted scripts; historical records (COMPLETED/CHANGELOG) leave as-is.
