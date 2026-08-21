# Inspector Short-Term — 2026-08-21 (low_w_shell_born_extension, INS pass 5 FINAL)

Scope: full review of the low-w shell-chart build (kill the failed bespoke
quotient LowWDiffractiveChart; new LowWShellChart macro-lead demodulated-
difference; Born rho floor 2.0 -> 1.4; census mirror; trainers). Re-checked
INS-4-001 against the live uncommitted test file.

## Re-check result
- INS-4-001 STILL OPEN (confirmed, not resolved): the overclaiming prose in
  test_lensing_low_w_shell_chart.py survives — module docstring item 3
  ("has no gap and no step: the two boundaries are the SAME constant (1.4)"),
  ShellBornBoundaryContinuityTestCase class docstring ("Both boundaries are
  1.4 -- the SAME float -- so there is no gap and no overlap"), and
  test_shell_outer_boundary_equals_born_floor docstring ("... the SAME
  constant (no gap/overlap)"). The production comment (likelihood.py:136-156)
  is honest: Born floor is SCALAR-reach gauge (ppgo_map.caustic_rho =
  |y|/caustic_reach), shell RHO_HI is DIRECTIONAL gauge (_caustic_rho =
  sqrt(s)/|y_c(theta)|), scalar<=directional always, so a theta-dependent
  coverage GAP exists (exact-engine tie-breaker). The value pin
  assertEqual(RHO_HI, _BORN_RHO_FLOOR) is correct; only the prose is false.

## New findings (INS-5-xxx) — spec/contracts divergence (Librarian scope)
- The plan listed SPEC.md and DATA_CONTRACTS.yaml as expected-to-change, but
  NEITHER is in the changed-file list. Both now contradict the shipped code:
  * SPEC.md:55 Born intercept gate still "rho > 2" (code: _BORN_RHO_FLOOR=1.4);
    "zero-quadrature exterior serve path for rho > 2" stale.
  * SPEC.md:54 LOW-W DIFFRACTIVE RUNGS still describes the DELETED quotient
    LowWDiffractiveChart (r_new = f_pure*sqrt(1-g'^2)/F_ref, derate,
    declined_mask, covers union band shell OR wall) and SERVE_ROUTES
    "... low_w_diffractive_chart ...".
  * DATA_CONTRACTS.yaml:376 born_residual_chart still "7 x 5 x 10 nodes
    covering rho > 2 ... gate caustic_rho(...) > 2.0" (code: 7x8x13, down to
    1.4; gate _BORN_RHO_FLOOR=1.4).
  * DATA_CONTRACTS.yaml:388 low_w_diffractive_chart entry still describes the
    deleted artifact (schema low_w_diffractive_v2, derate, declined_mask,
    F_ref, LowWDiffractiveChart, train_low_w_diffractive_chart.py); NO
    low_w_shell_chart entry exists.

## Verified-correct (no finding)
- Fresh runs all green: test_lensing_low_w_shell_chart.py 22 passed;
  born_analytic_reachability + born_certificate 75 passed;
  serve_route_census 42 passed; likelihood + ratio_layer 28 passed/18 skip/1
  xfail; whole-suite collect-only 2636, no collection errors.
- Smoke trainer scripts/train_low_w_shell_chart.py --scale smoke: 768 nodes,
  round-trip LowWShellChart.load() verified (schema low_w_shell_v1, hash match).
- Serve `_low_w_shell_chart_serve` gates in order: chart None -> _reduced_shear
  -> gamma_prime==0 -> rho=_caustic_rho -> RHO_LO<=rho<=RHO_HI ->
  gamma_prime box (chart.gamma_prime_grid) -> delta_min/w_shell -> w_shell<=w_lo
  -> below=below_split&in_log_w (any?) -> below composition + above engine host
  (above=~below includes above-w_shell AND outside-log-w nodes, both correct).
  below reconstruction mass_sheet_phase*(carrier+R)/lam == oracle
  mass_sheet_phase*f_pure/lam (carrier+R=f_pure by construction); gauge
  FARFIELD_DIFFRACTIVE consistent with _engine_farfield_total host.
  reduced_source/_caustic_rho round-trip exact (|y'|=rho*|caustic_point|).
- `_born_residual_analytic` uses ppgo_map.caustic_rho (SCALAR) for the
  rho <= _BORN_RHO_FLOOR gate (line 3277) and docstring (3161-62); no stale
  `rho <= 2.0`/`rho > 2.0` literals remain in likelihood.py.
- Census mirror: born_rho_floor=float(_BORN_RHO_FLOOR) single-sourced (line
  313), rho_lo/rho_hi=float(RHO_LO)/float(RHO_HI), reduced_source/
  _reduced_min_delay_separation bound as modules; route gate rho_lo<=rho_dir
  <=rho_hi then band-split w_shell>w_lo + below.any() + covers(...w_grid[below]).
  _region_of residual_demand buckets still rho>2/1<rho<=2 (aggregation bands,
  correctly NOT switched to _BORN_RHO_FLOOR).
- Test fixtures derived: _BELOW_FLOOR_RHO = 0.5*(1.0+_BORN_RHO_FLOOR) in
  born_analytic_reachability; born_certificate re-pointed to
  likelihood._BORN_RHO_FLOOR. rho>2.0 pin kept only as a "deep far-exterior"
  fixture premise (stronger than floor, valid).
- Content-hash field order identical across module/trainer/loader/test (6
  fields: gamma_prime_grid, rho_grid, theta_grid, log_w_grid, real_coeffs,
  imag_coeffs); provenance excluded from hash (pinned by test).
- No stale refs to LowWDiffractiveChart/_AUTO_LOW_W_CHART/partitioned_reference/
  declined_mask/.derate; `_low_w_diffractive_serve` (w_low_fit split method) is
  the distinct still-live method, correctly retained and mocked.

## Advisory (driver, not a defect)
- cogwheel/data/low_w_shell_chart.npz + re-baked born_residual_chart.npz still
  absent; full bake is a DRIVER post-build step. train_born_residual.py records
  a driver_prerequisite (azimuthal sweep at rho=1.4 for N(theta)<=8). Shell
  chart has NO de-rate/declined-mask by design — the 1e-4 served-error bar is a
  driver full-bake acceptance, verified off-grid post-bake, not an in-build gate.

## Patterns carried forward
- Cross-gauge single-sourcing (INS-3-001 -> INS-4-001 lineage): a "same
  constant (no gap/overlap)" claim spanning two rungs is FALSE whenever the two
  boundaries are in DIFFERENT rho gauges (scalar reach vs directional). Verify
  the GAUGE of each "rho" before accepting "no gap/no overlap" prose in ANY
  surface (production comment is now honest; test docstrings lag).
- Spec/contracts were in the plan's expected-change list but never edited — when
  a build renames a data-product tag/route and changes a gate constant, sweep
  SPEC.md + DATA_CONTRACTS.yaml explicitly; a green code diff does not certify
  the doc surfaces (recurring INS-1-xxx lineage, now INS-5-xxx).
