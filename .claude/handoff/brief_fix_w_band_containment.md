# Build Brief: Fix select_chart w-band Containment for Band-Split Serving

## Mission

`select_chart` currently requires the query's ENTIRE `[log_w_min, log_w_max]`
band to fit inside the chart's `log_w_grid` range. This rejects charts that
could serve the LOW-w portion while ppGO handles the HIGH-w tail (the band-
split architecture from Build 8h-a).

Result: 50% of census draws that have high `log_w_max` (low lens mass → high
dimensionless frequency) are rejected by `select_chart` even though the
chart + ppGO band-split would serve them correctly. They fall through to
expensive exact quadrature unnecessarily.

## The fix

`select_chart` should accept a chart when:
- `log_w_min >= chart.log_w_grid[0]` (the LOW end must be in the chart), AND
- `log_w_max <= chart.log_w_grid[-1]` OR a ppGO band-split is available
  above the chart ceiling

The ppGO availability check is: the certified ppGO map covers this
(gamma, rho) cell with a `w_trust` floor below `chart.log_w_grid[-1]`.
If so, the band-split serves everything above `w_trust` via ppGO, and
the chart only needs to cover `[log_w_min, min(log_w_max, chart_ceiling)]`.

Alternatively (simpler): relax `select_chart` to only require `log_w_min`
containment. The high-w excess is handled downstream by `_surrogate_coefficients`
which already has the band-split logic. If no band-split is available, the
serve returns None there (graceful degradation to exact).

## In scope

- Modify `select_chart`'s w-containment check to allow partial coverage
  (low-w inside chart, high-w handled by band-split)
- Verify that `_surrogate_coefficients` correctly handles the case where
  the chart covers only part of the w-band
- Update the census `characterize_sample` to reflect the relaxed containment
- Tests verifying: a sample with `log_w_max > chart ceiling` IS served when
  the chart covers its `log_w_min`

## Out of scope

- Training (running in background)
- The Born residual chart (already wired)
- Changing ppGO map or carrier logic

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
- The fix must not regress any currently-passing test.
