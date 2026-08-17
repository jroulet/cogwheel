---
section: Backlog
---

- **TUBE THETA NODE DENSITY LAW — derive the count from the envelope's
  own structure scale; then re-derive (f_max, f_floor) on resolved
  charts** `[→ spec]` — F083: the theta axis is in the RIGHT coordinate
  (arc length, 1e-tube) but carries a bare count (`n_theta = 7`) that
  under-resolves the envelope's measured w-driven structure (~0.17 rad
  at w ~ 52; ~36 nodes/rad needed; 48 nodes take gamma=0.4 astroid eps
  from 0.40 to 0.0237). The collocation doctrine's second half: placement
  AND density from local scales. Build: (1) derive the density law —
  candidate: nodes per radian proportional to the demodulated envelope's
  angular frequency, computable from the caustic geometry's delay
  variation along the arc times w_max (closed form via the step-1
  cascade; MEASURE the constant against the F083 ladder), with adaptive
  refinement against the held-out bar as the fallback if the closed form
  under-predicts; (2) raise engine_budget to match (the 24-node build
  already trips 400); (3) fix `_heldout_eps`'s silent-skip blind spot
  (unserved held-out points must be REPORTED as coverage, never
  silently dropped) and record the ~40% arc-end shell that cannot serve
  (nearest-point crosses the cusp — decide: shrink the constructed
  shell to the servable region, or route those queries to the adjacent
  arc's chart via the fold machinery); (4) THEN re-run the joint
  (f_max, f_floor) sweep (runner ready at /tmp/f_fraction_sweep.py,
  priced ~2.6-2.8 h at production density, w capped 60) on resolved
  charts — `_DEFAULT_F_MAX = 0.40` has no valid measurement behind it
  (F083) and `f_floor = 0.16` is already measured unsupported. Blocks
  tube training in the demand-sized campaign; independent of the demand
  census and the deltoid redesign.
