---
bump: minor
---
### Build 8g — far-field tiling + eps registration gate

The far-field training stage replaces the legacy single hard-coded box
per image-count region with a mass-stratified exterior tiling: each
parity's reachable lens-mass range is partitioned into log strata of
fixed ratio `R = sqrt(f_hi/f_lo)`, whole-band w-contained per stratum,
and tiled into square tiles admitted only wholly outside the
`caustic_reach + eta_max` disk; mass beyond the parity's w-ceiling and
tile-count truncation (`max_farfield_regions`, now defaulting to
`None` = uncapped) are both recorded loudly rather than dropped
silently. A held-out max-normalized envelope-eps registration gate
(`TrainingConfig.tube_eps_max = 5e-2`, `farfield_eps_max = 3e-3`)
excludes charts above bar or with NaN eps from the packed artifact,
recording them gated; the decision round-trips through per-chart
provenance on resume. A runtime foot-of-normal guard skips a tube
chart whose `eta_max` exceeds half the band's minimum caustic
curvature radius, recorded loudly.
