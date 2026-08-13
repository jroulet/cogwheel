---
date: 2026-08-13
bump: patch
---

- Documented the new tier-1 far-from-caustic macro-saddle analytic intercept
  (`LensedRelativeBinningLikelihood._saddle_farfield_analytic`, gated by
  `_saddle_farfield_analytic_serves`) in the "Microlensed waveform &
  likelihood" pipeline row: full dispatch order (surrogate intercept ->
  ppGO above-ceiling -> tier-1 saddle analytic -> exact seed engine),
  the two-term gate (`rho >= _SADDLE_FARFIELD_RHO_FLOOR = 2.0` AND
  resolvability), measured accuracy, and the deferred `_gauge.py` tier-2
  helpers (no production caller yet). Explicitly noted that the tier-1
  domain is structurally disjoint from the measured saddle serving gap, so
  this rung does not move structural coverage — do not read the new prose
  as a coverage improvement.
