# Architect Short-Term Observations

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
