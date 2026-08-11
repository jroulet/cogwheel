# Architect Short-Term Observations

## Build: zero_quadrature_pearcey (2026-08-11)

Key design decisions (Professor + Simplifier):
- Residual tabulation: R(x,y) = P(x,y) - P_asymp(x,y) replaces demodulated tabulation.
- Schema bump to 0.2.0; hard-refuse old artifacts.
- demodulate/remodulate/_carrier_phase/_dominant_stationary_point deleted.

## Build: operator_routing_one_home (2026-08-11)

- Fix: ppGO rung in cusp_amplification serves geometric_amplification via fold_ppgo_correction fallback at unresolved saddle nodes. Gate on (_merging_fold_pair is not None OR w*delta_min >= RHO_END). Local constant _PPGO_RESOLUTION_GATE=4.0 mirrors operator.RHO_END. Professor + Simplifier: lean. ONE Coder WP.

## Build: interior_cusp_serving_barrier (2026-08-11)

- Fix: skip _calibration_certified for interior (3 stationary points) — uniform error gate.

## Build: revert_residual_table_fix_routing (2026-08-11)

- _cusp_vertex routing: probe all nearby cusps by source-plane distance.
- ONE Coder WP (not two), no NPZ regeneration needed.

## ## Build: mpmath_fixed_panel_rule (2026-08-11, relaunched)

Plan decisions (second launch, post brief-correction):
- ONE Coder WP: replace mp.quad with fixed-order composite GL in _raw_integral_mp
- mp.gauss(24) for nodes/weights (no Newton fallback — Simplifier trim)
- lru_cache on (order, dps) at module level
- N/2N certification stays on reconstructed F (unchanged)
- Professor: optional belt-and-suspenders N/2N in mpmath before complex()
- CANCELLATION_LENS likely certifies as-is (w_max > 150 via factor-4 mass scale)
- One domain_test_description: overlap-band DD-vs-mpmath cross-agreement (test_lenschwinger.py)
- Simplifier: all lean except Newton fallback trimmed, overlap-band watch addressed

Professor rulings (from prior round, confirmed by relitigation):
- Order-24 GL sufficient (12 nodes/wavelength; mpmath dps >> dd)
- Spot grid: w∈{61,80,100,120,150} × gamma'∈{0.3,0.7,1.5} × y_eig∈{(0.1,0.1),(0.4,0.3),(0.8,0.5)} = 45pts
- Tolerance = _CERTIFICATION_TOL = 3e-10 on reconstructed F
- Certification must stay on reconstructed F (raw I would underflow at w=150)
- complex(raw_n) safe (F011-audited: relative ~eps64, common-mode cancels)
- _reconstruct needs no changes; oracle _oracle_1d is ground truth
- Runtime ~65-130s @ w=150, ~22-44s @ w=80 — O(seconds), deterministic, no hang
- Optional: γ'=1.05 edge case at w=80 for near-parity boundary stress

Simplifier:
- One Coder WP — single-unit change, splitting is artificial
- Reuse _PANEL_ORDER=24; bump if fails, don't preempt
- Cache _mp_gl_rule on (order, dps) with lru_cache at module level
- TRIM Newton-refinement fallback (mp.gauss stable 12+ years)
- WATCH overlap-band re-validation — addressed via domain_test_description
- Train-tier oracle gate mapped to verification

Professor rulings:
- Order-24 GL sufficient (same oscillation physics, 12 nodes/wavelength; mpmath dps >> dd)
- Spot grid: w∈{61,80,100,120,150} × gamma'∈{0.3,0.7,1.5} × y_eig∈{(0.1,0.1),(0.4,0.3),(0.8,0.5)} = 45pts
- Tolerance = _CERTIFICATION_TOL = 3e-10 on reconstructed F
- Certification must stay on reconstructed F (raw I would underflow at w=150)
- complex(raw_n) safe (F011-audited: relative ~eps64, common-mode cancels)
- CANCELLATION_LENS likely certifies now → Test Developer re-point at gamma=0.9, y=(0.1,0.1)
- Runtime ~65-130s @ w=150, ~22-44s @ w=80 — O(seconds), deterministic, no hang
- _reconstruct needs no changes; oracle _oracle_1d is ground truth

Simplifier:
- One Coder WP — single-unit change, splitting is artificial
- Reuse _PANEL_ORDER=24; bump if fails, don't preempt
- Cache _mp_gl_rule on (order, dps) with lru_cache at module level
- Spike mp.gauss API first (Newton-refinement fallback if unavailable)
- CANCELLATION_LENS re-point/retract is Test Developer contingency
- Train-tier oracle gate promoted to verification