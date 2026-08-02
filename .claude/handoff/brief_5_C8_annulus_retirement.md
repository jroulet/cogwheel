# Build Brief: Step 5 (C8) — Far Zone Becomes Caustic-Relative

## Mission

Replace `ANNULUS_INNER_RADIUS = 3.0` (a prior-box artifact) with a
gamma-dependent caustic-relative boundary. DELETE `GAMMA_FENCE = 3/4` and
the saddle fence `1.0502342` — both are consequences of the annulus radius,
not independent physics. Re-express `surrogate_census`'s breakdown in the
caustic-relative coordinate.

## Measured facts (driver measurement, step 4)

The Born carrier crosses the 5% accuracy bar at `|y|*/reach`:
- gamma=0.03: 64× reach
- gamma=0.05: 40× reach
- gamma=0.10: 21× reach
- gamma=0.15: 15× reach
- gamma=0.20: 16× reach
- gamma=0.25: 13× reach
- gamma=0.28: 12× reach

The crossover is GAMMA-DEPENDENT. A single constant cannot replace
ANNULUS_INNER_RADIUS. The boundary must be `rho_crossover(gamma) * reach`.

At mid-to-high gamma (0.15-0.28), `rho* ≈ 12-16`. At small gamma
(0.03-0.05), it balloons to 40-64. The current fixed 3.0 / reach gives
rho = 3.0/0.28 = 10.7 at gamma=0.28 (too aggressive — 5% error there)
up to 3.0/0.06 = 50 at gamma=0.03 (actually fine).

The production system currently falls through to exact evaluation in the
annulus — it doesn't use the carrier there. So this boundary is about
where the far-field CHART takes over from exact, not about carrier accuracy.
The chart needs fewer nodes far from the caustic (smoother envelope),
so the real crossover is: at what `rho` does a chart become cheaper than
exact evaluation?

## In scope

- Replace `ANNULUS_INNER_RADIUS = 3.0` with a gamma-dependent boundary
  computed from `reach` (the caustic reach at each gamma band).
- DELETE `GAMMA_FENCE = 3/4` and the saddle fence `1.0502342`.
- Re-express `surrogate_census`'s MECE breakdown in the caustic-relative
  coordinate (regime count becomes TWO per parity: caustic-attached and exterior).
- RENAME `ppgo_map.annulus_rho` (a public symbol named for a retired concept).

## Out of scope

- Training any artifacts.
- The Born residual chart (blocked on this step per the TODO).
- Ghost decay gate (step 6, independent).
- 1e-gamma (runs after C8 per the ordering).

## Acceptance (from TODO)

- No gamma in the prior yields a "boundary cuts the caustic" region.
- Regime count is TWO per parity (caustic-attached + exterior).
- Coverage-map regions 6-9 collapse to one row.

## Constraints

- Fast tests only.
- Do not train any chart artifacts.
- Follow AGENTS.md and the spec/TODO workflow.
