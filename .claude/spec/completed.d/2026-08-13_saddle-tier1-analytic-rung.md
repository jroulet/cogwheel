---
date: 2026-08-13
section: Lensing
---

- **Tier-1 far-from-caustic macro-saddle analytic serve rung** — a resolvable
  `gamma > 1` saddle far from the caustic is served from the switched
  analytic channels with a ZERO envelope: no engine call, no `fold_ppgo`
  correction. Gate `_saddle_farfield_analytic_serves(real_delays, w_lo, rho)`
  is the single source of truth and has TWO required terms — resolvability
  (`n_real >= 2` and `w_lo * min_delta_tau >= RHO_END`) AND caustic proximity
  (`rho >= _SADDLE_FARFIELD_RHO_FLOOR = 2.0`). Measured accuracy on the
  gate-admitted population: pointwise relative error p90 ~5e-5, max ~7e-4 at
  the floor, against a 1e-3 p90 bar. Production dispatch is surrogate ->
  ppGO(`w_max > 150`) -> tier-1 -> exact seed engine.

  **The rho term is load-bearing, not cosmetic.** The pinned witness
  (`gamma = 1.5859`, `y = (-1.1208, -0.9002)`, `rho ~ 0.73`) is resolvable at
  `w_lo = 8` and the OLD resolvability-only gate would have served it with
  err ~9e-2. Resolvability says the image pair is separable; it says nothing
  about whether the dropped envelope is negligible.

  **Coverage did NOT move: 87.61%, unchanged.** The tier-1 domain and the
  measured saddle gap are disjoint by construction — the census routes
  `rho > 1` to the Born rung before the saddle path, so all 1742 saddle draws
  have `rho <= 1.0` while tier-1 demands `rho >= 2.0`. Full measurement and
  the consequence for tier 2 in
  [[lensing_saddle_tier1_cannot_reach_the_census_gap]]. WP-2's census wiring
  landed in `surrogate_census.characterize_sample`, which the census script
  never calls, and is NOT delivered.

  Also landed: `_saddle_switch_delay` / `_saddle_phase_delay` in `_gauge.py`
  as the gauge for the deferred tier-2 chart (no production caller yet, and
  the docstrings say so rather than claiming to be the authoritative gauge
  for both tiers).

  Three suites, 46 tests: `test_lensing_saddle_gauge.py` (gauge identities,
  handover continuity at the serve floor, outward-accuracy monotonicity),
  `test_lensing_saddle_tier1_accuracy.py` (22-source seeded population,
  leaky-gate witness, self-falsification), `test_lensing_saddle_tier1_refusal.py`
  (refusal + census attribution). Mutation-verified: dropping the rho term is
  caught by the accuracy shard, dropping resolvability by the refusal and
  gauge shards.
