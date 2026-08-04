# Build Brief: Census serve-fraction audit (dry run — no trained charts)

## Mission

Verify that the current architecture WOULD achieve 100% serve fraction
once charts are trained, by running a census that accounts for ALL serve
paths now wired:

1. Tube charts (eta ∈ [f_floor*R_c, f_max*R_c]) — WOULD serve if trained
2. Far-field charts (rho > rho_exterior_min) — WOULD serve if trained
3. Interior wedge charts (rho < 1, w < DD cap) — WOULD serve if trained
4. Born carrier (rho > 1, exterior) — ALREADY serves (analytic)
5. ppGO fold handoff (rho < 1, w > DD cap, ξ ≥ 4) — ALREADY serves
6. Cusp arm (delta_theta > _CUSP_ARM_COVERAGE) — serves once table loaded
7. Schwinger qd (saddle w ∈ (60, 148)) — engine available for training

## Task

Write `scripts/census_dry_run.py` that:

1. Sample N=10000 draws from the full prior:
   - gamma uniform in (0, 1.6)
   - |y| uniform in (0, 4.2426) (prior box)
   - theta uniform in (0, 2π)
   - w log-uniform in (5, 148)

2. For each draw, classify which serve path WOULD handle it:
   - `select_chart` returns non-None → "chart" (trained chart serves)
   - `select_chart` returns None but Born carrier covers → "born"
   - ppGO fold handoff gate passes (ξ ≥ 4, error < bar) → "ppgo_fold"
   - Cusp arm serves (cusp_amplification non-None) → "cusp_arm"
   - None of the above → "exact_engine" (fallthrough)

3. For the "chart" category, further classify by chart type:
   - Would a tube chart cover it? (eta ∈ tube range, gamma in a band)
   - Would a far-field chart cover it? (rho > exterior boundary)
   - Would an interior wedge chart cover it? (rho < 1, w < DD cap)

4. Report:
   - Total fraction that WOULD be served (all non-"exact_engine")
   - Per-category breakdown
   - The "exact_engine" residual: what draws have NO serve path?
   - For each residual draw: print (gamma, |y|, w) to identify the gap

## Key insight

This is NOT about whether charts ARE trained — it's about whether the
architecture has a serve path for every draw. If the residual is > 0,
it means there's a structural coverage gap that no amount of training
will fix.

## Acceptance

- Script runs in < 2 minutes (10K draws, geometry only, no engine).
- Reports the structural coverage fraction.
- If < 100%: identifies which draws have no serve path and why.
- Target: > 99% structural coverage (the residual should only be
  pathological corners like gamma=0 exact or w above all ceilings).

## Constraints

- No trained chart artifacts needed (the script tests ARCHITECTURE, not
  the presence of trained data).
- No engine calls (expensive). Use geometry + gate logic only.
- Follow AGENTS.md and the spec/TODO workflow.
