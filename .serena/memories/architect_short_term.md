# Architect Short-Term Observations

## fold_exterior_ghost build (F075, 2026-08-13, planning)
- Two-sided fix. (A) FOLD REFUSAL: guard `<4 real images -> refuse` at the
  fold-correction ENTRY POINTS, NOT inside `_merging_fold_pair` (Professor
  Q1: shared primitive has 5 consumers; tightening it flips the
  `_pearcey_cusp` L862 OR-disjunct — F074-owned, out of scope). Sites:
  `_airy_fold.fold_amplification` (required) + `fold_ppgo_correction`
  (consistency, INS-c8-003 pairs them) + `channels.born_carrier_from_partition`
  L1601 (CO-BUG: applies Airy fold to the far pair on 2-image exterior — same
  bug, label path). Guard = `if len(images) != 4: return None/refuse`. Leave
  `_merging_fold_pair`, `_pearcey_cusp` L862, `surrogate_census` L484 UNTOUCHED
  (last is interior-gated image_count==4, no-op).
- (B) GHOST RUNG: new `_ghost_ppgo_amplification` helper (like the fold/cusp
  arms, `complex|None`) inserted in `_uniform_arm_value` BETWEEN fold and cusp.
  Serves `geometric_amplification + ghost.kernel*exp(1j*w*ghost.delay)`
  (ABSOLUTE carrier; call `geometry.ghost_kernel` DIRECTLY, NOT
  `farfield_ghost_term` — that's min-subtracted t_min frame). GATE = the two
  FREQUENCY-INDEPENDENT config gates `Im(tau_c)>=0.4` AND `min|x_a-x_c|>=0.7`
  (reuse channels' `_GHOST_DECAY_IM_THRESHOLD`/`_GHOST_SEPARATION_MIN`; single-
  source). NO new w-dependent floor — Professor: a w-floor re-opens the
  train/serve skew that build 8h-d1 retired, and this rung is a LABEL oracle so
  frequency-independence is the precondition. Professor PREDICTS the two config
  gates already partition the caveat band (|y|/rc=1.05); probe P2 confirms
  (report). Exceptions: `GhostAbsentError`->decline (fall through, interior
  4-image byte-identical); bare `GhostDomainError`->refuse (return None) — copy
  `_pearcey_cusp` L889-891 discrimination.
- WPs: WP1 fold refusal (Coder). WP2 ghost rung (Coder, dep WP1). WP3 mirror
  audit+re-gate surrogate_census.characterize_sample/surrogate_training/
  train_lens_surrogate.py (Coder, dep WP1,WP2). WP4 retroactive label check
  train_ppgo_map.py/train_born_residual.py REPORT-ONLY (Foreman-Lite). WP5
  acceptance sweep probe P2 + 1e-2 bar + caveat refusal count REPORT
  (Foreman-Lite, dep WP1,WP2). Oracle = f_schwinger reconstruction ONLY, never
  F_op (self-oracle >60), never exact_total w/o t_min pairing.
- Tests (Test Dev, 3 cheap synthetic invariants): census refusal + interior
  byte-identity; gate-predicate serve/decline/refuse w/ boundary flip; ghost
  sign +exp(1j*w*tau_c) machine-precision internal-consistency pin. Value sweep
  is WP5 REPORT, not a permanent test.


## ppgo_interior_certificate build (2026-08-13, planning)
- Handoff: re-gate interior fold-ppGO rung in likelihood.py (~L1782).
  Leg1 rho<=1 -> EXACT interior = 4 real images (geom.real_mask.sum()==4,
  both parities); replaces the current rho<=1 + saddle-only !=4 guard.
  Leg3 _uniform_error_estimate -> new c3-based ppgo_error_estimate.
  On TRUE interior ghost is exactly ZERO (fact5) -> NO ghost term.
- New fn ppgo_error_estimate(real_images, source, matrix, w_min) in
  chang_refsdal layer = sum_a sqrt|mu_a|*|c3_a|/w_min**3. c3 from ported
  reference series_coefficients (validated vs shipped _c1/_c2 to 2.4e-15/
  5.8e-14). Cost 6.27ms/4img. Assert GhostAbsentError on interior.
- Leg2 (_merging_fold_pair/xi_min) likely DROPPED — cert doesn't need a
  fold pair. CONFIRM with Professor. Safety factor: fact3 says 1.0 already
  suffices on TRUE interior (max ratio 0.980); modest margin ok, 10x not.
- Do NOT change caustic_rho; report other consumers (_ppgo_cell_coords,
  surrogate_training._train_band_charts). No surrogate retrain, no slow tiers.
- Reference: .claude/handoff/ppgo_c3_reference.py, ppgo_cert_sweep.json.
