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

- 2026-08-XX: brief_schwinger_qd — extend Schwinger evaluator above w=60 via
  mpmath quadrature. TWO WPs: (1) engine extension in _schwinger.py (add
  _f_schwinger_mpmath, paired N/2N cert, W_CEILING_SCHWINGER_QD=150, lazy
  mpmath import), (2) training pipeline wiring in operator.py + surrogate_training.py
  (raise _SADDLE_W_CEILING, adjust _saddle_grid/_positive_parity_grid routing to
  handle w∈(60,150] via sequential f_schwinger calls). Professor confirmed:
  dps=30+ceil(w) sufficient, practical ceiling ~150 (runtime-limited), paired N/2N
  cert recommended, no mathematical ill-posedness at high w. Simplifier: split into
  two WPs (engine vs training pipeline), lazy import pattern, must measure ceiling
  empirically (hardcoding 150 analytically is borderline — formula ceiling ~139).

- 2026-08-XX: brief_saddle_born_carrier.md is a STALE handoff — all five in-scope
  items shipped in commits 31ee133 (2026-07-28, Born carrier+band split+saddle) and
  65eebcb (2026-08-02, C8 caustic-relative admission replacing gamma fences). The
  saddle fence (1.0502342 < gamma < 3) was added then architecturally superseded by
  per-point `caustic_rho > 1` which is physically exact (Professor confirmed). All
  53 tests pass in test_lensing_born.py including comprehensive saddle test classes.
  Escalated as zero-WP plan.

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

- 2026-08-XX: INS-1-005 triage (schwinger_qd build): CODER_FIX. ONEHOME_WS=(5,40,59,61,70,150,500) includes w=61,70,150 in (60,150] → mpmath path → ~120s/call for wave nodes. _observed_branch catches SchwingerCertificationError to identify wave; with mpmath succeeding instead, F_op actually evaluates (correctly, but slowly). Fix: update ONEHOME_WS to replace {61,70,150} with values that stay ≤60 or ≥W_CEILING_SCHWINGER_QD+epsilon (>150) so wave nodes above the old ceiling still raise SchwingerCertificationError rather than invoking mpmath. w=500 already does this (>150). Equivalent for XOR_BAND_LS: top is L=59.4, so max w=59.4/0.9=66—within mpmath band. Fix: cap XOR_BAND_LS top at L=54 (w=60) or use CERT_SQRT_S such that max w ≤ 60, OR mock the mpmath path. Routing-purpose option (b) is Test-Dev scope. Best fix: adjust ONEHOME_WS to remove 61,70,150 or replace with w=500+ alternatives that stay above W_CEILING_SCHWINGER_QD; and adjust XOR_BAND_LS top to stay ≤ W_CEILING_SCHWINGER.

- 2026-08-XX: INS-2-001/INS-1-002 triage (schwinger_qd build): CODER_FIX (both).
  BAND_EDGE.w_probes=(30.0,40.0,60.5) routes w=60.5 through slow mpmath (~240s
  F_op + ~120s diagnostic_scatter = ~360-480s), busting the 5-min fast-tier ceiling.
  Fix: change 60.5→59.9 in the BAND_EDGE _LensConfig constant and update the
  fixture docstring (remove the "w=60.5 served by mpmath" sentence). w=59.9 stays
  on the DD path (<1s), same wave branch, same isfinite assertions pass identically.
  Same pattern as INS-1-005 (ONEHOME_WS/XOR_BAND_LS). INS-1-002 is fully subsumed
  by INS-2-001 — single fix resolves both.

- 2026-08-XX: brief_interlobe_corridor — Pure measurement script (no code changes).
  Task: write scripts/probe_interlobe_corridor.py that computes corridor geometry for
  saddle gammas and reports whether the inter-lobe gap is negligible. Simplifier: drop
  select_chart (no surrogate artifact on disk), compute geometry directly from
  _lobe_caustic_points + _INTERLOBE_CORRIDOR_ETA_SCALE + f_max * R_c. Professor: corridor
  is geometrically thin (3-10% of centroid sep), O(1-5%) of lobe prior mass near gamma=1,
  NO accuracy concern (exact engine fallback), purely an efficiency issue. Minimal
  quantities: centroid_sep, eta_max, corridor_width/sep, area_fraction (MC). One WP ~25 turns.
