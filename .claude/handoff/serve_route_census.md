# Build: serve-route demand census (7a's demand map = 7b's acceptance tool)

## Mission

Owner doctrine: analytics serve first; charts exist ONLY where the
analytic ladder cannot serve. The instrument: a census that classifies a
dense prior sample by the draw's ACTUAL serve route through the
production ladder — engine-free — with no surrogate attached. The
residual (every analytic rung refuses) is the chart demand map that
sizes the training campaign. The SAME tool, re-run with a trained
artifact attached, is the 7b acceptance census (owner's bar: every
region of (y1, y2, w, reduced gamma), both parities, serves from a table
or analytic rung, NEVER the engine; report zero engine-served or the
survivor list with owners).

## Facts

1. The ladder at HEAD (serving order in
   `likelihood._amplification_coefficients`): surrogate intercept (None
   when no artifact attached) -> ppGO above-ceiling (w > 150) -> tier-1
   saddle c3 certificate (gamma > 1) -> first-class Born intercept
   (kappa==0, beta==0, gamma!=0, rho>2, covers) -> exact seed engine.
   Below the likelihood layer, per-node ladders exist in the engine
   (geometric / uniform arms fold->ppGO+ghost->Pearcey / exact wave) —
   the census classifies at BOTH granularities: the draw-level intercept
   route, and for engine-fall-through draws the per-node arm coverage
   (a draw whose every node is arm/geometric-served is analytics-served
   even though the "engine" evaluator hosts the arms; only nodes that
   reach the exact wave evaluator (_schwinger / mpmath) count as ENGINE
   demand). This distinction is load-bearing — do not conflate the
   evaluator with the exact-wave rung.
2. Existing machinery to reuse as thin callers (NEVER a parallel
   reimplementation — the standing rule): `scripts/census_dry_run.py`
   (structural classification, 10k draws ~1 min),
   `surrogate_census.classify_fallthrough` / `characterize_sample`,
   the rungs' own gates (`_saddle_farfield_analytic_serves`,
   `_born_residual_analytic`'s gate, `CertifiedPpgoMap.w_cert`,
   the arms' admission functions). Each rung's own gate is its accuracy
   warrant (they were calibrated all week); the census records ROUTE,
   not error. Driver-side oracle spot-checks are separate and NOT this
   build's scope.
3. Sampling: the PRIOR's full physical reach — w up to ~444 (mass and
   frequency range top), NOT the old training wall 148
   (`lensing_saddle_above_the_training_cap_is_unmeasured`); both
   parities; gamma over the full sampled range including near-wall.
   Draws carry prior mass weights so 7b can report "% of prior mass"
   per route, not just draw counts.
4. Invariants to pin (fast synthetic tests; the full census is a
   report): route-equality across D2 fold images (owner-confirmed
   census invariant — a draw and its three mirrors report the SAME
   route; serve fractions invariant to folded vs unfolded sampling);
   the engine-free guarantee for the classification path (mock the
   exact-wave evaluator to raise — the tiling-census pattern); the
   route classifier's MECE property (every draw gets exactly one route).
5. Output schema: JSON — per-draw records (coords, w-band, route,
   refusal chain for residuals) + aggregation to (region x gamma-band x
   w-band) cells with counts and prior mass + a `residual_demand`
   section the tiling design consumes directly. Mode flag:
   `--with-artifact PATH` attaches a surrogate for the 7b acceptance
   run (route `surrogate` then preempts; zero `exact_wave` is the pass
   bar).

## Scope

IN: the census module + CLI (new `cogwheel/lensing/serve_route_census.py`
or an extension of the census machinery — Architect decides against the
one-tiling-machine DRY rule and the engine-free import constraint that
motivated tiling_census's separation); the two-granularity route
classifier; the demand-map JSON; the invariant pins; a small-sample
smoke report in-build (structural, ~1 min).
OUT: any training; any tiling design (next step, consumes the JSON);
oracle error sweeps; the deltoid redesign; serving changes of any kind.

## Acceptance

- The census runs engine-free (mock-to-raise pinned) on a 10k-draw
  sample in ~minutes, both parities, w to the prior's reach; MECE
  routes; D2 route-equality pinned.
- The demand JSON exists with the residual section, prior-mass weighted.
- A written REPORT: the route breakdown at HEAD (no surrogate) — % of
  prior mass per analytic rung, % residual (= the campaign's true
  demand), the residual's (gamma, w, region) shape. This report is the
  input to the demand-sized tiling design and the owner's next decision.

## Constraints

Branch claude-dev; fragments (`[→ spec]`); thin callers of production
gates only; values-not-paths; test parsimony (one pin per invariant);
escalate rather than iterate on any surprise.
