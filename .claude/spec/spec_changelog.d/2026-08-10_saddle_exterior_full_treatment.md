---
date: 2026-08-10
bump: patch
---

Update SPEC.md following build saddle_exterior_full_treatment (238d21e):

1. GLOBAL MULTI-CHART ARTIFACT: macro-saddle exterior charts now conditionally
   apply `_deltoid_cusp_axis_map` when a deltoid cusp ray falls inside the
   tile's theta_c range on one side (straddle or no cusp in
   range falls back to raw theta_c). Was incorrectly stated as always using
   raw theta_c.

2. FOLD-CARRIER DEMODULATION: parity label updated from "positive parity" to
   "both parities" -- _needs_fold_carrier and _exclude_ghost_dominated were
   extended to handle both parities (ghost exists for astroid and near-saddle
   exterior tiles).

3. Key abstractions exterior chart coordinate contract: macro-saddle exterior
   charts carry a conditional cusp-adapted theta_to_u map via
   _deltoid_cusp_axis_map, not always raw theta_c.
