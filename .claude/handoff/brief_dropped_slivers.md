# Build Brief: Measure dropped gamma slivers (coverage region 10)

## Mission

Determine whether dropped gamma slivers are a real coverage hole or a
rounding error. `stable_gamma_bands(..., min_width=config.min_gamma_band)`
discards topology-stable sub-bands narrower than 0.02. Draws landing in
a dropped sliver get NO chart and fall through to exact quadrature.

## Task

Write `scripts/measure_dropped_slivers.py` that:

1. For both parities (branch=+1, branch=-1), call `stable_gamma_bands`
   over the full prior range with the production `min_gamma_band = 0.02`.
2. Compute the TOTAL gamma width dropped (sum of all sub-bands narrower
   than 0.02 that `stable_gamma_bands` would have returned without the
   min_width filter).
3. Express as a fraction of the total prior gamma range.
4. If the fraction exceeds 1e-3: report which gamma values are affected
   and propose a fix (reduce min_gamma_band, or handle slivers explicitly).
5. If the fraction is < 1e-3: document as acceptable (measure-zero in
   practice) and mark region 10 as CLOSED in the coverage map.

## Acceptance

- The measurement runs and produces a number.
- If slivers are significant (>1e-3): a concrete fix is proposed (not
  implemented — that's a follow-up build).
- If negligible: region 10 marked closed with the measured number.

## Constraints

- Pure measurement script, no code changes to the training pipeline.
- Fast (no engine calls — `stable_gamma_bands` is pure geometry).
- Follow AGENTS.md and the spec/TODO workflow.
