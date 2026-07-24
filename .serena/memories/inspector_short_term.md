# Inspector Short-Term Observations

## 2026-07-23 — Build 8h-b3-fin FINAL re-review (caustic-fixed core complete)

Scope: uncommitted worktree cogwheel-claude-dev, branch claude-dev. This
build COMPLETED the caustic-fixed migration + S1-2/S1-3/S2-1/S2-2/S2-3 WPs
(per plan). Prod code changed: channels.py, geometry.py, likelihood.py,
surrogate.py, surrogate_training.py. One NEW test file added
(test_lensing_exterior_windows.py). NO existing suites ported.

### Verdict: ISSUES (2 carried findings BOTH still open; 0 resolved)

### Carried-finding dispositions
- INS-2-001 STILL OPEN (unchanged root cause, broad scope). Existing
  far-field regression battery left RED because the (y1,y2)->(rho,theta_c)
  from_engine/from_values migration was NOT propagated to tests:
    * test_lensing_surrogate.py: 21 failed, 18 passed, 1 skipped, 9 errors.
      Confirmed TypeError: from_engine() got unexpected kwarg 'y1_range'
      (L458). Dark certs: DomainGate, Serialization(single+MultiChart),
      ChartSelection, EnvelopeReconstruction, RefusalPreservation
      (never-serves-refused/finite-no-NaN/nonzero-kappa-never-served),
      LnlikeAccuracy (served-lnlike-tracks-engine).
    * test_lensing_ppgo_bandsplit.py + test_lensing_surrogate_census.py:
      19 failed, 66 passed, 15 errors (143s). Same stale-API root cause.
  NEW test_lensing_exterior_windows.py: 68 passed, 2 xfailed (283s) — green
  but has NO served-lnlike-tracks-engine and NO never-serves-refused cert;
  serve-path accuracy/refusal certs remain dark. Build ships regression
  battery RED. ACTIONABLE (bug-severity blocker).
- INS-3-001 STILL OPEN (flag to Librarian). git diff --stat HEAD shows
  SPEC.md and DATA_CONTRACTS.yaml BOTH UNCHANGED; plan has_spec_update=True.
  Far-field chart schema materially changed: axes now caustic-fixed
  (rho,theta_c) not (y1,y2); each chart carries envelope_definition tag
  (FARFIELD_KERNEL_SUM / FARFIELD_DIFFRACTIVE / FARFIELD_KERNEL_SUM_MINUS_
  GHOST / INTERIOR_SACR_C) with train/serve tag-symmetry contract. SPEC
  narrative still describes old axes. Bidirectional divergence. Inspector
  does not edit canonical surfaces.

### Production verified SOUND this pass (no new bug)
- All 5 lensing modules import cleanly (IMPORTS_OK).
- Only TEST call sites use stale from_engine(y1_range=...); production
  surrogate_training.py L2153 uses rho_range/theta_c_range; producer
  scripts/train_lens_surrogate.py goes through TrainingConfig (unaffected).
- Serve mirror (likelihood.py _surrogate_coefficients) tag-symmetric with
  training label: definition in KNOWN_FARFIELD_DEFINITIONS ->
  reconstruct_farfield(...,definition) mirroring farfield_envelope_from_
  partition (switch=_farfield_switch(definition), tau_c=0). MINUS_GHOST
  re-adds farfield_ghost_term(chart_w, source, macro_matrix(gamma,beta,
  kappa)) over below_mask before reconstruct; GhostDomainError -> return
  None (symmetric refusal). Diffractive band-split refused (telescoping
  identity holds only for kernel-sum switch=1 family). Interior/tube ->
  reconstruct_from_envelope with geom switch+critical_delay, asserted
  definition is None or in KNOWN_INTERIOR_DEFINITIONS. Well-constructed.

### Carry-forward
- INS-1-001 (ghost_kernel double _ghost_delay compute, geometry.py) — not
  re-examined this pass; still presumed open non-blocking micro-DRY.
- After suites ported, RE-RUN full lensing battery; watch on-axis ghost
  xfail and RB delta_t_max edge-margin fixtures under larger-separation
  caustic-fixed configs (INS long-term note).
