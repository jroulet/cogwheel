# Build Brief: Re-chart exterior in polar (rho, theta_c), retire (s,d) bridge

## Mission

The exterior `(s,d)` coordinate is the obstacle — measured foot `tie_ratio`
reaches 1.000 inside the charted domain, and the probe confirmed 13 depth-3
cap-fails with eps up to 3.6. The object `E(y)` is real-analytic on the
open exterior away from cusps, so the failure is purely that `(s,d)` is not
single-valued. Chart in the tiler's OWN polar frame `(rho, theta_c)` instead;
it is single-valued, well-conditioned, respects both reflection symmetries,
and every window's object is analytic in it.

## Work

1. Re-chart the exterior bulk in `(rho, theta_c)`; delete the
   `_farfield_box_to_smooth` bridge for the bulk. New `axis_schema` tag;
   stale `(s,d)` artifacts hard-refuse.
2. Keep `(s, d)` for the thin near-fold tube only.
3. Put tile edges ON the principal axes (kink-free by symmetry).
4. Add a cusp carve-out (~0.2 y-units, sized by the separation-gate
   contour). The exterior tiler currently has NO cusp-ball exclusion.
5. Do NOT move any label boundary. Document the MINUS_GHOST gap.

## Acceptance
- Exterior charts per band fall well below the current ~57 at the same 1e-3 bar.
- No chart's eps is dominated by a coordinate discontinuity.
- A query at a former foot-tie location serves to tolerance.

## Measured facts (SHA ab48b25)
- Exterior recursion probe: 13 depth-3 cap-fails, eps 1.2e-3..3.6
- Wedge v3 confirmed: 9/9 charts pass 5e-2, median 6.0e-3
- `FARFIELD_KERNEL_SUM_MINUS_GHOST` label exists in code but is never stamped by the tiler

## Constraints
- Fast tests only. Follow AGENTS.md and spec/TODO workflow.
