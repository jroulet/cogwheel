# Build Brief: Remove min_gamma_band threshold (log-reach makes it redundant)

## Mission

The residual dropped slivers (gamma ∈ (0, 0.0039) and (1.006, 1.010))
fall through because `stable_gamma_bands` drops bands narrower than
`min_gamma_band = 0.005` in raw gamma. But the gamma axis WITHIN each
band already uses log-reach spacing (`_log_reach_gamma_axis`), which
places n_gamma nodes by the caustic-relative coordinate regardless of
the band's raw-gamma width.

A band of raw width 0.004 at gamma~0.004 gets 4 log-reach-spaced nodes
that resolve the geometry just as well as a band of width 0.1 at gamma~0.5
— because what matters is the INTRA-band coordinate, not the raw width.

The `min_gamma_band` threshold was a guard for the OLD uniform-gamma axis
where a narrow band meant tightly-spaced nodes with no interpolation room.
With log-reach spacing that concern is gone.

## Fix

1. Set `min_gamma_band = 0.0` (or remove the threshold entirely).
   `stable_gamma_bands` already handles the degenerate case (a band
   narrower than machine epsilon) via its topology-split logic.

2. Alternatively, set the threshold in LOG-REACH units rather than raw
   gamma: `min_log_reach_span = 0.01` or similar, so that a band is only
   dropped if its caustic geometry has genuinely zero variation across it
   (which can't happen for a topology-stable band by definition).

3. Re-run `scripts/measure_dropped_slivers.py` to confirm REGION 10 CLOSED.

## Key insight

The `min_gamma_band` floor was never about topology — it was about the
gamma AXIS not being able to resolve a narrow band. Since 1e-gamma switched
to log-reach internally, that failure mode is gone. Any topology-stable
band, no matter how narrow in raw gamma, has resolvable caustic structure
in the log-reach coordinate.

## Acceptance

- `scripts/measure_dropped_slivers.py` prints "REGION 10 CLOSED".
- No band with topology-stable geometry is dropped.
- Existing tests still pass (the F041 test uses its own explicit threshold).

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
