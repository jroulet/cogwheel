# Build Brief: Fix dropped gamma slivers at prior edges

## Mission

The measurement script (`scripts/measure_dropped_slivers.py`) found 2.15%
of prior mass in dropped slivers:
- Positive parity: gamma ∈ (0, 0.0156) — the low-gamma edge
- Saddle parity: gamma ∈ (1.001, 1.0197) — the near-unity edge

Both are simply the first sub-band from `stable_gamma_bands` being narrower
than `min_gamma_band = 0.02`. These are NOT topology slivers — the geometry
is perfectly well-behaved there. They're just below the width threshold.

## Fix

Reduce `min_gamma_band` from 0.02 to 0.005 in `TrainingConfig`. This is
safe because:
1. The threshold exists to avoid training a chart on a band so narrow that
   gamma interpolation has no room. At 0.005 width with n_gamma=4, the
   spacing is 0.00125 — still resolvable.
2. The analytic cusp detection (build 1b) means `stable_gamma_bands` no
   longer depends on sampling resolution for cusp finding — the 0.02
   floor was a guard for the OLD numerical detector's resolution limit.
3. Both edge bands have constant topology (no metamorphoses), so there's
   no reason to exclude them.

## Implementation

1. Change `_DEFAULT_MIN_GAMMA_BAND` (or wherever `min_gamma_band = 0.02`
   is defined) to 0.005.
2. Re-run `scripts/measure_dropped_slivers.py` and confirm:
   - Both edge slivers are now captured (dropped fraction < 1e-3)
   - `stable_gamma_bands` still returns sensible bands (no pathological
     fragmentation)
3. Update any tests that pin `min_gamma_band = 0.02`.

## Acceptance

- `scripts/measure_dropped_slivers.py` prints "REGION 10 CLOSED" with
  dropped fraction < 1e-3.
- No regression in existing training tests.
- Step 8 Part 0 mechanical test still passes.

## Constraints

- One-line production code change + test updates.
- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
