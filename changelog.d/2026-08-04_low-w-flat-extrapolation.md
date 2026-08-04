## 2026-08-04

### Lensing surrogate: low-w flat extrapolation — clamp below chart w_min

`LensAmplificationSurrogate` now serves draws whose `w` lies **below** a
chart's trained `w_min` by clamping `log_w_query` to `log_w_grid[0]` before
the spline evaluation (flat extrapolation).  The envelope is smooth and
nearly constant below the first Airy fringe — the correction is
`O(w_min^2)` relative to the geometric limit (Professor-confirmed), so the
clamped value is accurate.

**Changes in `cogwheel/lensing/surrogate.py`:**

- `_log_w_band_serveable(chart, log_w_min, log_w_max)`: the low-end check
  is now **open** (a query band whose *upper* edge exceeds `chart.w_max`
  still fails; a query band whose lower edge is below `chart.w_min` is
  admitted).
- `_evaluate_chart`: applies `np.clip(log_w_query, chart.log_w_grid[0], ...)` 
  before the spline call so the grid is never under-shot.
- Five call sites updated: `_tube_serves`, `_farfield_serves`,
  `_lobe_serves`, `_wedge_serves`, and `may_serve`.

**New test file:** `cogwheel/tests/test_lensing_low_w_extrapolation.py`
(220 lines; synthetic charts, no engine dependency).

Commit: `afff8e7`
