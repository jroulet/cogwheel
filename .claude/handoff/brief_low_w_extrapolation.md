# Build Brief: Low-w chart extrapolation toward geometric limit

## Mission

Every chart (tube, far-field, wedge, lobe) has a finite w_min on its
log_w_grid. Draws with w < w_min currently fall through to exact engine.
At low w the amplification approaches sqrt(mu_macro) * exp(i*w*phi_geo)
(the geometric optics limit) — a KNOWN analytic target. Extrapolating
the chart's envelope downward from its lowest w-nodes toward this limit
is safe, cheap, and closes the low-w coverage gap.

## Physics

As w → 0:
- The amplification F(w) → sqrt(mu_macro) (geometric magnification)
- The envelope (after carrier demodulation) → a smooth constant
- The chart's spline at w_min already captures the envelope there
- Between w=0 and w_min, the envelope varies slowly (it's BELOW the
  first Airy fringe, so no oscillation)

## Implementation

In the chart evaluation path (`_evaluate_chart` in surrogate.py or
`_surrogate_coefficients` in likelihood.py):

1. When a query has `w < chart.w_min` (i.e. `log_w < chart.log_w_grid[0]`):
   - Evaluate the chart's spline at `w_min` (the lowest grid point) to get
     `envelope_at_wmin`
   - Compute the geometric limit: `envelope_at_w0 = sqrt(mu_macro)` in the
     chart's demodulated frame
   - Linearly interpolate in log_w between `(0, envelope_at_w0)` and
     `(log_w_min, envelope_at_wmin)`:
     ```
     frac = log_w / log_w_min  # 0 at w→0, 1 at w_min
     envelope = (1-frac) * envelope_at_w0 + frac * envelope_at_wmin
     ```
   - Reconstruct and return

2. Alternatively (simpler): just clamp `log_w` to `log_w_grid[0]` (flat
   extrapolation from the lowest w-slice). This is less accurate but
   trivial to implement and already better than falling through to exact.

3. The choice between linear-in-log-w interpolation toward the geometric
   limit vs flat extrapolation should be Professor-decided.

## Acceptance

- Draws with w < w_min are served (not exact engine fallthrough)
- Served values agree with exact engine within the eps bar at w = w_min/2
- No regression at w >= w_min (chart evaluation unchanged)

## Constraints

- No training needed — this is a serve-path code change only.
- Fast tests.
- Follow AGENTS.md and the spec/TODO workflow.
