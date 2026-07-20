# Architect Short-Term Observations

## Build 8b-levers — FINAL plan emitted 2026-07-20

Two Coder WPs (Simplifier: keep split). WP-A geometry Newton caustic
shortcut, WP-B operator contraction fusion. Professor rulings encoded:
(Q1) HEAD_NEAREST_CAUSTIC_PINS bit-exact theta -> <=1e-10 value-
preservation gate (legit re-cert; distance stays assertEqual/places=14);
routed to Test Dev. (Q2) 1-D scalar Newton on g'(theta)=0, analytic
g'/g'', 32-pt coarse seed, 2 best cells, MANDATORY single-cell Brent
fallback, g''>0 guard, wedge-clamp per lobe/branch, seed-per-lobe
take-min. (Q3) 9 saddle branch/lobe configs at 1e-10 both parities.
(Q4/Q5) fusion = dispatch-only njit merge preserving accumulation order,
NO reassociation, byte-exact re-cert, half_sum stays arg +
_SERIES_TOLERANCE module-global for F010. has_domain_changes=true,
has_spec_update=true.

(empty — last consolidated by Dreamer on 2026-07-20)
