---
bump: minor
---

### Low-w flat extrapolation for all surrogate chart types

SPEC.md "Microlensing engine" surrogate row updated to describe the new
low-w flat extrapolation behavior in `surrogate.py` (commit `afff8e7`):

- `_log_w_band_serveable` now uses a one-sided high-end check — a query
  band whose *lower* edge falls below `chart.w_min` is admitted (open
  low end); only the upper edge is strict.
- `_evaluate_chart` applies `np.clip(log_w_query, chart.log_w_grid[0], ...)`
  before the spline call so the grid is never under-shot.
- Five call sites updated: `_tube_serves`, `_farfield_serves`,
  `_lobe_serves`, `_wedge_serves`, `may_serve`.

Physics rationale (Professor-confirmed): the envelope is smooth and nearly
constant below the first Airy fringe — `O(w_min^2)` correction to the
geometric limit — so the clamped value is accurate.

Certified by `cogwheel/tests/test_lensing_low_w_extrapolation.py`
(220 lines, synthetic charts, no engine dependency).
