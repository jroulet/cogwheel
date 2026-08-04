# Architect Short-Term Observations

- 2026-08-XX: brief_remove_min_gamma_band — set min_gamma_band=0.0 in THREE
  sites: TrainingConfig.min_gamma_band (L274), stable_gamma_bands default
  (L851), scripts/measure_dropped_slivers.py MIN_WIDTH (L36). Also update
  the call-site comment (L3476) noting no slivers are dropped with 0.0.
  Simplifier: lean one-WP Foreman-Lite, no infinite-bisection risk (topology
  converges at sub-eps widths). Existing tests unaffected (use explicit
  min_width= args). Test-file naming staleness (_NEW_DEFAULT_MIN_WIDTH) is
  cosmetic, not gating.


(last consolidated by Dreamer on 2026-08-05)

- 2026-08-XX: brief_analytic_cusp_serving.md is a STALE handoff — build 1c already
  shipped in commit b9c3ed6 (2026-07-30), ancestor of HEAD. _cusp_vertex uses brentq
  on analytic derivatives (no FD/scan/golden); caustic_third_derivative exists with
  full cascade. Both targets marked DONE in lensing_analytic_derivatives.md. Escalated
  as zero-WP plan.

- 2026-08-XX: brief_fix_dropped_slivers — one-liner config change 0.02→0.005
  in THREE sites: TrainingConfig.min_gamma_band (L274), stable_gamma_bands
  default (L851), and scripts/measure_dropped_slivers.py MIN_WIDTH. Fix two
  misleading comments (L864 docstring + L3476 call-site). No artifact version
  bump needed. Existing F041 test _F041_MIN_WIDTH=0.02 stays (tests the arc-
  guard fix, not the threshold). Simplifier: lean one-WP, found missing 3rd
  site (function default).

- 2026-08-XX: brief_ppgo_interior_handoff — fold_ppgo_correction serve path
  for interior draws (rho<1) above wedge chart w-ceiling. Gate: ξ_min >= 4.0
  (cheap pre-filter) + error-estimate fine gate (c_A*ξ^{-3/2} < CERT_BAR).
  Professor: ξ=4 gives ~2% max pair error, fine gate pins sub-1e-4;
  DO-NOTHING property means fold_ppgo_correction can't be WORSE than raw ppGO.
  Reconstruction: demod to minrel, extract far-field envelope via
  reconstruct_farfield(FARFIELD_KERNEL_SUM) — same pattern as Born rung.
  Census: served=True with serve_method indicator. Simplifier concerns about
  regime (interior vs exterior) were wrong — fold_ppgo_correction works on ALL
  regimes (it calls geometric_amplification for all 4 images), and the ξ≥4
  gate ensures all images are well-resolved. One WP (likelihood.py +
  surrogate_census.py).

- 2026-08-XX: brief_cusp_arm_table — Ship pearcey_table.npz + enable _CUSP_ARM_COVERAGE.
  Already implemented: PearceyTable, build_table, derive_box, save_table in _pearcey_table.py;
  cusp_amplification in _pearcey_cusp.py; train_pearcey_table.py script. Missing: the actual
  .npz artifact, coverage constant is 0.0, no measurement script. Professor: R-gate is the
  binding constraint for reach; calibration passes generically at large R; oracle is F_op;
  reach at minimum w (5-10) is the worst case; expect 0.02-0.04 rad. Simplifier: formula-
  based reach is sufficient (R_min inversion through geometry), but brief explicitly asks for
  a comparison script. Compromise: write script that computes reach analytically + verifies
  ~20 boundary points vs F_op. One merged WP (generate table + measure + set constant).

- 2026-08-XX: brief_interlobe_corridor — Pure measurement script (no code changes).
  Task: write scripts/probe_interlobe_corridor.py that computes corridor geometry for
  saddle gammas and reports whether the inter-lobe gap is negligible. Simplifier: drop
  select_chart (no surrogate artifact on disk), compute geometry directly from
  _lobe_caustic_points + _INTERLOBE_CORRIDOR_ETA_SCALE + f_max * R_c. Professor: corridor
  is geometrically thin (3-10% of centroid sep), O(1-5%) of lobe prior mass near gamma=1,
  NO accuracy concern (exact engine fallback), purely an efficiency issue. Minimal
  quantities: centroid_sep, eta_max, corridor_width/sep, area_fraction (MC). One WP ~25 turns.
