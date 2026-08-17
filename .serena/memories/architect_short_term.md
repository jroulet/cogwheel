# Architect Short-Term Observations

2026-08-17 serve_route_census RE-PLAN: the implementation was ALREADY
PRE-PLACED in the tree — cogwheel/lensing/serve_route_census.py (full
7-label MECE classifier: engine_refused-first waterfall -> surrogate ->
saddle_c3 -> born_analytic -> per-node ppgo_above_ceiling/analytics_engine_
hosted/engine_residual; lazy engine-free imports via _ProductionModules;
aggregate_cells + residual_demand 3-way caustic_rho split) AND
scripts/serve_route_census.py (CLI --with-artifact) are COMPLETE and
signature-correct (verified serve/may_serve/macro_matrix/caustic_rho/
_gamma_band_edges/_real_delay_min_separation/_saddle_farfield_analytic_
serves/_uniform_arm_value/select_branch/cancellation_exponent against live
prod). A .pyc exists (imports OK). Professor cross-checked design vs wired
gates = PASS; flagged (a) add "finite-but-huge est -> refuse-not-serve"
invariant, (b) don't hardcode 59%, report it. Simplifier: new module LEAN;
weight/prior_mass_fraction TRIM (already equal-weight, kept as note);
region-threshold dup vs tiling_census WATCH (leave per YAGNI). So build =
1 lean Coder WP (finalize/verify + acceptance smoke + HEAD report) + Test
Developer authors 5 disjoint invariant pins (NO test file exists yet):
MECE/surrogate-empty, residual-partition disjointness, engine-free
mock-to-raise sentinel, D2 sign-flip route+kind equality (both parities,
bit-exact via negation not rotation), saddle finite-huge-est refuse. NOT
test-only (net-new tests + report deliverable + must land pre-placed code
in diff). has_domain_changes=true (taxonomy domain-sensitive).

2026-08-15 serve_route_census plan: NEW engine-free module
cogwheel/lensing/serve_route_census.py (+ scripts/serve_route_census.py CLI,
--with-artifact) modelled on tiling_census.py (lazy imports) NOT
surrogate_census.py (imports engine at module load). 7-label MECE taxonomy
(Professor): {surrogate, ppgo_above_ceiling, saddle_c3, born_analytic,
analytics_engine_hosted, engine_residual, engine_refused}, first-admitting,
ordering assertion. Two granularities: draw-level intercepts (saddle_c3 via
_saddle_farfield_analytic_serves thin call; born by geometric predicate;
ppGO DERIVED from per-node coverage not explicit mirror) -> per-node arm
coverage via select_branch/_uniform_arm_value on real geometry; except
(LensDomainError,CancellationError,ValueError,ZeroDivisionError) -> exact-wave.
Residual split 3-way by CAUSTIC-relative rho (never rho_lobe): born_chart
(rho>2), near-caustic/tube (1,2], interior (<=1). engine_refused = lnL=-inf
(~59%), MUST be its own label. Reuse draw_samples/_LensSampledPrior (FULL w
reach, drop census_dry_run's w cap), classify_fallthrough, _REGIONS_BY_PARITY;
drop per-draw weight col (prior-distributed => %mass==%count). WP1 Coder ~90
(module core), WP2 Coder ~50 (CLI+smoke, route breakdown quoted in completion
record as acceptance evidence NOT a test). D2 route-equality pins on route
KIND vector not lobe index. has_domain_changes=true (taxonomy/gates domain-
sensitive though NO serving change).
