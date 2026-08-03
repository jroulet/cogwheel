---
date: 2026-08-03
section: Backlog
---

### Dropped metamorphosis gamma slivers — closed by setting `min_gamma_band = 1e-6`

`TrainingConfig.min_gamma_band` and `stable_gamma_bands(min_width=...)` default
changed to `1e-6` (commit `70affbb`). With the log-reach gamma axis (`1e-gamma`),
nodes within any band are well-placed regardless of raw-gamma width, so the
0.005 floor that guarded the old uniform-gamma axis is redundant. Bisection now
continues to near-float resolution: total dropped prior mass ~1.5e-6 (fraction
~1e-6 of the prior range), negligible. Region 10 ("DROPPED GAMMA SLIVERS")
closes. The `dropped-sliver` census category remains in
`_FALLTHROUGH_CATEGORIES` and reports the negligible residual.

Items from `lensing_dropped_gamma_slivers.md`:
1. Mass measurement — measured: ~1.5e-6 total dropped width, ~1e-6 fraction.
2. Comment fix — done in the same commit (train() docblock updated).
3. Treatment decision — not needed; residual mass is negligible at 1e-6 floor.
