# Inspector Short-Term Observations

## 2026-08-13 — Build F075 fold_exterior_ghost (FINAL re-review, working tree) — VERDICT: PASS

Scope: uncommitted working tree, worktree cogwheel-claude-dev / branch claude-dev.
Re-check of carried finding INS-1-001 + full re-audit of WP-1..WP-5.

### INS-1-001 RESOLVED (was the sole blocker)
- cogwheel/tests/test_lensing_airy_fold.py IS modified this pass (` M`; prior
  passes it was untouched -> the written-but-not-applied handoff was finally
  executed). Full run: 128 passed, 7 skipped, 2 xfailed, 0 FAILED (was 3 failed).
- Three tests fixed exactly as prescribed: (a) `_CUSP_TIE_SOURCE_OFF_AXIS`
  swapped [0.7,0.05] (2-image exterior) -> [0.15,0.14] (verified 4-image
  interior; docstring documents find_images==4, gap 0.255, serves @ w=500);
  (b) `OnAxisServingLadderDeterminismTestCase` renamed/inverted to
  fold-refuses/cusp-serves contract (assertIsNone(fold), ladder==cusp bytes,
  order==['fold','cusp']); (c) 'interior' docstrings corrected to 'exterior
  (2-image)'. The [0.15,0.14] pinned literal is self-guarding: its own
  test_fold_amplification_serves_off_axis goes RED if it ever leaves the
  4-image domain (fold refuses -> assertIsNotNone fails), so check-#9
  silent-strand risk does not apply (domain boundary is the caustic = physics,
  not a movable constant). Not flagged.

### WP audit (all faithful)
- WP-1: `len(images) != 4` refusal added at 3 sites — _airy_fold.fold_amplification
  (~L470), _airy_fold.fold_ppgo_correction (~L618, falls back to raw ppGO),
  channels.born_carrier_from_partition (~L1611, pair=None). Correct.
- WP-2: operator._ghost_ppgo_amplification new rung inserted in _uniform_arm_value
  BETWEEN fold and cusp. Contract-faithful vs geometry.ghost_kernel([w],src,mtx)
  -> GhostContribution(kernel,delay,position): uses delay.imag decay gate,
  min complex-Euclid separation, cmath.exp(1j*w*delay) carrier (+ sign,
  non-conjugated), np.atleast_1d(kernel)[0]. Catches GhostAbsentError(4-image
  interior)->None (interior serve byte-identical), GhostDomainError(undecayed/
  on-axis)->None (refuse, never zero), LensDomainError->None. INS-1-002
  empty-guard `if len(real_images)==0: return None` present BEFORE min(). Gates
  single-sourced from geometry (_GHOST_DECAY_IM_THRESHOLD=0.4,
  _GHOST_SEPARATION_MIN=0.7).
- geometry: two new module constants (0.4, 0.7) are the authoritative home;
  channels re-references them (value-preserving, byte-identical to old 0.7 and
  _FARFIELD_WINDOW_RADIANS/5.0=0.4). DRY, correct.
- WP-3: surrogate_census.characterize_sample interior rung re-gated from the
  RETIRED xi-fold+uniform-error gate to the c3-certificate rung. Now a FAITHFUL
  MIRROR of production likelihood.py L1815 rung: image_count==int(geom.real_mask
  .sum())==4, real_images=geom.images[real_mask], ppgo_error_estimate(real_images,
  source,matrix,w_min), est*_PPGO_INTERIOR_SAFETY(=2.0)<=CERTIFICATION_BAR.
  _PPGO_INTERIOR_SAFETY bound from likelihood (not re-typed). Old machinery
  (_merging_fold_pair,_uniform_error_estimate,_XI_FOLD_THRESHOLD) fully removed
  from census (0 dangling refs). Note: census uses macro_matrix(gamma,0,0)
  consistent with its own geom built at beta=kappa=0. likelihood.py NOT changed
  this build (re-gate shipped in prior ppgo_interior_certificate build); census
  was the laggard now synced. surrogate_training.py + train_lens_surrogate.py
  planned in WP-3 but correctly NOT changed — they carry NO exterior fold gate
  (only dataclass image_count fields). Not a finding.
- WP-4 (report only): certified_ppgo_map.npz CONTAMINATED at 32 exterior
  positive-parity cells @ w in {66,80,97,117,141} but OVER-conservative
  direction (coverage/perf loss, never over-certifies) -> DRIVER RETRAINING
  ADVISORY, not a correctness defect. born_residual_chart.npz CLEAN (w-grid
  tops at 60.0, below the 60<w band). No code/artifact edits.
- WP-5 (report only): probe P2 + ghost-rung error sweep handoff present.

### Suites run green
- test_lensing_airy_fold: 128p/7s/2xf. test_lensing_fold_ghost_exterior: 17p.
- ghost_gate+operator+surrogate_census: 67p/13s. channels: 16p. imports OK.

### Carry-forward (NOT code defects, for driver/Librarian)
- DRIVER: retrain certified_ppgo_map.npz post-commit (WP-4 32-cell advisory).
- LIBRARIAN lineage (doc-staleness, unchanged from prior passes): SPEC.md +
  DATA_CONTRACTS.yaml still cite exterior_polar_rho_log_carrier_v1 "ONLY tag"
  since V5 2D carrier; region vocabulary (lobe_exterior etc.). Arm-ladder order
  change (fold->ghost->cusp) is internal to operator.py; SPEC.md does not
  document the ladder order, so no spec-divergence finding.
