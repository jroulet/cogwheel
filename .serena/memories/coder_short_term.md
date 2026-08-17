# Coder Short-Term Observations

## 2026-08-17 build (serve_route_census WP1 finalize + acceptance report)

- WP1 = confirm-and-run, ZERO code edits. Audited all 14 thin callers in
  serve_route_census.py against live production via find_symbol: every
  signature/arg-order matched (macro_matrix(gamma,beta,kappa);
  caustic_rho(gamma,y_mag,kappa); _gamma_band_edges();
  select_branch(w,delta_min,cancellation_exp,eta);
  _uniform_arm_value(w,y,gamma,*); _real_delay_min_separation(source,matrix);
  cancellation_exponent(w,y,gamma,kappa);
  _saddle_farfield_analytic_serves(real_images,source,matrix,w_lo);
  fold_amplification/_ghost_ppgo_amplification/cusp_amplification(w,src,gamma,*);
  may_serve(gamma,lo,hi); serve(w,*,gamma,y1,y2,beta,eta,theta,image_count);
  dimensionless_frequency(f,m,z); draw_samples(config) duck-types n_samples+seed;
  geometry_partition(*,gamma,y,beta,kappa) keyword-only). ChangRefsdal
  GeometryPartition exposes .images/.caustic_distance/.caustic_theta/
  .real_mask — all four census reads valid. Top imports engine-free
  (ppgo_map + chang_refsdal.geometry only). NO defect found → no edit.
- Serena execute_shell_command has a HARD 240s cap; Bash is whitelisted
  ONLY for git/gh/conda/ps/cat/sleep-in-loop etc. Long runs: launch detached
  via serena `nohup ... </dev/null &`, then poll with a `for i in seq;
  do ...; sleep 5; done` loop kept UNDER 240s per serena call.
- HEAD 10k demand acceptance (seed 0, n_freq=128, ~40 min wall):
  engine_residual 83.37% (8337), born_analytic 15.40% (1540),
  analytics_engine_hosted 0.91% (91), saddle_c3 0.32% (32),
  ppgo_above_ceiling 0%, surrogate 0%, engine_refused 0%. Residual 3-way
  (on caustic_rho): interior 76.98% (6418), near_caustic_tube 23.02%
  (1919), born_chart_demand 0%, undetermined 0%.
- DESIGN OBSERVATIONS (not defects — routes reachable in principle, engine
  demand correctly split): (1) engine_refused=0% at HEAD — the classify_draw
  intercept-1 docstring calls it "the largest single population", empirically
  FALSE for the kappa=beta=0 full-reach prior (geometry_partition never hard-
  refuses on this sample; also 0 at 300-draw smoke). The task's "~59%" is the
  F073 rho<=1-at-exterior figure (58.7%), NOT an engine_refused prediction.
  → flag to Inspector/Librarian as doc-prose accuracy, did NOT silently edit.
  (2) residual born_chart_demand bucket (rho>2) is 0 BY CONSTRUCTION: born_
  analytic intercept (gamma!=0 & rho>2) skims all rho>2 draws before the per-
  node pass, so no engine_residual draw can carry rho>2 unless gamma==0 (→
  undetermined). Redundant-but-harmless bucket, left as-is per YAGNI.
- Structural smoke (300 draws, n_freq=32) proved engine-free: mpmath never
  entered sys.modules after classifying real draws. schema/keys/sum/residual-
  sum all pass. Both files staged (git add explicit paths; new files A/A).

## 2026-08-15 build (serve_route_census WP1)

- Created cogwheel/lensing/serve_route_census.py — engine-free serve-route
  DEMAND census. 7-label MECE waterfall SERVE_ROUTES =
  (surrogate, ppgo_above_ceiling, saddle_c3, born_analytic,
  analytics_engine_hosted, engine_residual, engine_refused). Decision order
  differs from label order: engine_refused FIRST (geometry_partition raises a
  refusal), then surrogate (artifact mode only — asserted never in demand),
  saddle_c3 (thin likelihood._saddle_farfield_analytic_serves at band-floor
  w_lo), born_analytic (geom predicate: gamma!=0 and caustic_rho>2), then a
  per-node arm pass resolves the last three. Per-node uses
  operator.select_branch + operator._uniform_arm_value on REAL geometry;
  except tuple = _REFUSAL_ERRORS + (ValueError, ZeroDivisionError) — the
  ZeroDivisionError is load-bearing (caustic_rho raises it at gamma==0).
  route-KIND vector (geometric/fold/ghost_ppgo/pearcey/exact_wave) is the
  D2-reflection-invariant object. residual_demand splits engine_residual by
  caustic_rho (NEVER rho_lobe): >2 born_chart, (1,2] near_caustic_tube, <=1
  interior, None undetermined.
- CancellationError DOES NOT EXIST in cogwheel/lensing — the WP brief named it
  for the engine_refused intercept + per-node except tuple. Reconciled to
  surrogate._REFUSAL_ERRORS (= LensDomainError, SchwingerCertificationError)
  for refusals, plus (ValueError, ZeroDivisionError) for per-node degeneracy.
  Recorded so a future reader doesn't re-add a phantom CancellationError.
- cancellation_exponent RAISES LensDomainError for gamma>=1 (1-kappa<=|gamma|),
  so the per-node pass gates it: saddle_host = (1.0 <= abs(gamma)) passes
  math.inf (mirrors operator _saddle_grid/_op_grid), gamma<1 passes the
  measured L=w|y'|. Do NOT call cancellation_exponent unconditionally.
- draw_samples(config) DUCK-TYPES: it only reads config.n_samples and
  config.seed, so ServeRouteCensusConfig (my frozen dataclass) works unchanged
  as its argument — no dependence on surrogate_census.CensusConfig type.
- DOUBLE-MASK avoided: real_images = np.asarray(geom.images) with NO re-mask
  (geom.images is already real-only; length-4 real_mask is CHANNEL-array only).
- Verified end-to-end: run(n_samples=8) returns schema serve_route_census_v1,
  one label/draw, JSON-serializable, and mpmath NEVER enters sys.modules even
  after classifying real draws (the engine-free no-CALL proof). Module-top
  namespace binds no engine door (ChangRefsdalChannels/_schwinger/mpmath all
  behind _load_production_modules). NO CLI (WP2), NO tests (Test Developer).
