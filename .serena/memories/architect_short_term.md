# Architect Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-05)

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
