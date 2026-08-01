# Build Brief: Step 3 (C6) — Tube Shell Becomes Curvature-Relative

## Mission

Replace the fixed `_DEFAULT_ETA_MAX = 0.05` and `_DEFAULT_ETA_FLOOR = 0.02`
with `f_max * R_c` and `f_floor * R_c` where `R_c` is the minimum caustic
curvature radius over the arc. Then DELETE the foot-of-normal skip guard
(`eta_max > 0.5 * r_min` → skip), since with `eta_max = f * R_c` and `f < 0.5`
it is vacuous by construction.

## Measured facts (driver measurement, step 2)

The minimum curvature radius `R_c` varies from 0.016 (gamma=0.03) to 0.157
(gamma=0.28) on the positive parity, and 0.197 to 0.440 on the saddle parity.
The current fixed `eta_max = 0.05` corresponds to `eta_max/R_c` ranging from
2.58 (gamma=0.03, far above the 0.5 guard!) down to 0.11 (gamma=1.05).

This is why the small-gamma collar exists: at gamma < 0.17 (positive parity),
`eta_max > 0.5 * R_c` and the chart is SKIPPED. Making eta_max curvature-
relative eliminates this.

The guard threshold is `f = 0.5`. The working range is `f < 0.5`. The exact
optimal `f_max` and `f_floor` should be determined empirically as part of this
build: build charts at several f values in [0.2, 0.48] at representative gammas
where charts currently succeed (gamma >= 0.17 positive, all saddle), measure
held-out eps, and find where it crosses the 0.05 bar.

## In scope

- Replace `_DEFAULT_ETA_MAX` with `f_max * R_c(band, arc)` in `_build_tube_chart`
  and `from_engine`'s tube-chart loop.
- Replace `_DEFAULT_ETA_FLOOR` with `f_floor * R_c` (proportional to f_max).
- DELETE the `if config.eta_max > 0.5 * r_min: skip` guard — it's vacuous when
  eta_max is already `f * R_c` with `f < 0.5`.
- Convert the guard into an assertion (belt: verify `f < 0.5` at construction).
- Determine `f_max` and `f_floor` empirically by sweeping eps at representative
  gammas. Use the TrainingConfig defaults for grid sizes (n_gamma=4, n_u=4,
  n_theta=4) — NOT smoke-scale; raise `engine_budget` if needed.
- Tests verifying: no chart skipped for curvature at any gamma in the prior;
  held-out eps under bar at both gamma extremes; the same `f` serves every gamma.

## Out of scope

- Far-field or lobe coordinate changes (already done).
- The small-gamma collar's other two causes (those were fixed in step 1b).
- Training any artifacts.
- The interior-admission distance optimization (step 4).

## Acceptance (from TODO)

- No chart skipped for curvature at any gamma in the prior.
- Held-out eps under bar at both gamma extremes.
- The same `f` serves every gamma.
- The small-gamma collar (coverage-map region 3) closes HERE — specifically the
  foot-of-normal skips at gamma in [0.0281, 0.0462] and [0.0644, 0.1550].

## Constraints

- Fast tests only (no COGWHEEL_BRUTE_ACCURACY, no COGWHEEL_STRICT_TIMING).
- Do not train any chart artifacts.
- Follow AGENTS.md and the spec/TODO workflow.
- The `f` measurement IS in scope for this build (the Architect should plan a
  WP that measures it before the WP that implements the replacement).
