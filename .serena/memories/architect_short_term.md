# Architect Short-Term Observations

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
