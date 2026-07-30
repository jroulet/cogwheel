# Architect Short-Term Observations

Build 1e-tube (2026-07-30): TubeChart splines in arc length s=∫|y'|dθ, not
raw theta. Design: ONE new dataclass field `theta_to_s` = (2,N_map) table
[theta_fine, s_fine]; from_values/_assemble gain OPTIONAL map (default =
identity s=theta-theta_lo → existing synthetic fixtures byte-identical);
_evaluate_chart serves v2 = np.interp(theta_inframe, theta_fine, s_fine);
membership+cusp windows STAY in theta. Build path (_build_tube_chart) builds
map at rep_gamma = median(gamma_grid) [Professor: band midpoint minimizes
worst-case eff excursion; single-gamma adequate for topology-stable bands,
degrades only near parity wall eff→1 which existing foot-of-normal skip +
gamma-refine-near-1 already bound]. N_map=2001 (Professor h² bound: coord
err ~3e-8 « round-trip tol 1e-6). F016 bar=0.05 COMPLEX. Knife-edge gate:
swing<5% under ±0.01 rad bound shift (incumbent ±23%). Reuse existing
_wp3_fixture / _wp3_build_and_measure / _heldout_eps scaffolding at
_WP3_GAMMA(=1.55). Simplifier: (2,N) map lean; omit s_grid field (knots
encode it) — correct; identity default a documented seam not footgun.
