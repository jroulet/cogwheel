---
section: Backlog
---

- **THE TUBE MAP IS BUILT AT ONE GAMMA AND USED ACROSS A BAND** `[→ spec]` —
  measured 2026-07-30. `_build_tube_chart` builds `theta_to_s` once at
  `rep_gamma = float(np.median(gamma_grid))` and stores it on the chart;
  `_evaluate_chart` reads that same map at serve. But `s(theta) = int |y'| dtheta`
  is gamma-dependent, and the caustic's extent varies 28x across the prior and
  diverges at the parity wall (F036).

  Measured normalized coordinate drift, `max |s/s_total(gamma_edge) -
  s/s_total(rep)|`, N_map=2001, arcs inset inside the wedge:

  | band | width | parity | drift | x the map's own 1e-6 tol |
  |---|---|---|---|---|
  | [0.50, 0.70] | 0.20 | astroid | 2.19e-2 | 21925 |
  | [0.80, 0.99] | 0.19 | astroid | **1.39e-1** | 138958 |
  | [0.90, 0.95] | 0.05 | astroid | 2.60e-2 | 25988 |
  | [1.05, 1.25] | 0.20 | saddle | 2.71e-2 | 27144 |
  | [3.00, 3.20] | 0.20 | saddle | 7.88e-3 | 7879 |

  Band refinement does NOT rescue it: a 0.05-wide band still drifts 2.6e-2.

  ## This is a degradation, not a correctness bug

  Train and serve share the SAME stored map, so there is no train/serve skew,
  and `s`-at-`rep_gamma` is still a legitimate monotone reparametrization of
  `theta` that removes most of the F042 pathology. What it is not is the RIGHT
  coordinate away from `rep_gamma` — resolution goes where the caustic turns
  fastest at the median gamma, not at the query's gamma. 1e-tube's bound-shift
  acceptance may well still pass.

  ## Why it matters anyway: consistency

  1e-farfield's Professor called gamma-resolution FIRM on the same magnitude
  (measured O(10-25%) at a 0.2-wide band edge) three hours after 1e-tube
  shipped without it. Two sibling builds answered one design question two ways,
  and [[lensing_collocation_from_local_scales]]'s whole premise is one
  authoritative representation per coordinate.

  ## Fix

  Transplant 1e-farfield's gamma-resolved map: a 2-D `s(theta, gamma)` table on
  the spline's OWN gamma nodes. No new solve and no new engine calls, since
  gamma is already a spline axis. NOTE the serve cost: `_evaluate_chart`
  currently does a 1-D `np.interp` on `theta_to_s`; this makes it 2-D.

  Cheapest home is 1e-lobe, which already touches the same map machinery.

  ACCEPTANCE: drift at both band edges falls to the map's own round-trip
  tolerance; tube held-out eps at fixed node count does not regress; the serve
  cost delta is measured and stated.
