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

- 2026-08-XX: brief_low_w_extrapolation — serve draws with w < chart.w_min via
  flat extrapolation (clamp log_w_query to log_w_grid[0]) instead of falling
  through to exact engine. ONE WP: add _log_w_band_serveable function (only
  checks high end), replace _log_w_band_inside in all 5 call sites, clamp
  log_w_query inside _evaluate_chart before spline eval. Professor confirmed:
  KERNEL_SUM envelope → 0 as w→0, DIFFRACTIVE/INTERIOR → sqrt(mu_macro); flat
  clamp is O(w_min²) error for diffractive/interior; kernel-sum charts are not
  the main beneficiary (they hand off to diffractive below w_floor anyway).
  Simplifier: one WP correct, new function not mutate old, clamp BEFORE spline
  (scipy BSpline extrapolates polynomially), use np.clip not np.maximum, keep
  high-end guard strict. Test: w_min/2 accuracy within 3e-3 of max|F|.

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

- 2026-08-XX: brief_cusp_arm_boundary — Write measurement script
  scripts/measure_cusp_arm_actual_boundary.py that calls cusp_amplification
  directly (not the R-gate formula) to binary-search for the actual
  serve/refuse boundary. Then set _CUSP_ARM_COVERAGE in surrogate.py to
  the measured min (floor to 2 dp, conservative). ONE WP. Professor:
  search direction correct (arm serves AWAY from cusp), use plain bisection
  (xtol=1e-4), monotonicity mostly holds but verify post-hoc, extend grid
  to gamma near parity boundary and high w. Simplifier: LEAN on one WP +
  direct calls; WATCH on monotonicity (dense sweep or post-hoc check);
  safety margin = floor to 2dp (conservative direction); TRIM F_op cross-
  check from this script (existing script already has it). TRIM coarse
  scan from 200 to 50 pts; use max(all_refused) not last-consecutive.
  Coordinate confirmed: delta_theta (theta-radians on critical curve) is
  consumed directly in _tube_serves, so script outputs in same units.

- 2026-08-XX: brief_census_dry_run — standalone script scripts/census_dry_run.py,
  structural coverage audit without trained charts. ONE WP. Samples 10K draws
  from (gamma∈[0,1.6], |y|∈[0,4.2426], θ∈[0,2π], w log-uniform [5,148]).
  Uses geometry_partition (cheap quartic, no engine) with minimal w_grid (2 pts).
  Classification: born (rho>1), tube_feasible (eta in f_floor*Rc..f_max*Rc),
  wedge_feasible (rho<1, w*|y|≤58), ppgo_fold (rho<1, 4 images, ξ≥4), cusp_arm
  (within cusp window above coverage threshold), exact_engine (residual).
  Professor: ALL exterior served by Born; interior gap at DD cap with ξ<4 is
  the main structural hole; gamma guard band and near-cusp vertex core are the
  other gaps. Simplifier: collapse farfield_possible into born; use actual
  _merging_fold_pair for ppGO (cheap, O(16)); single WP, fresh ChangRefsdal
  per draw for label-continuation safety; n_freq=2 for structural-only.
