# Professor short-term (session 2026-07-27): Build 8h-b5 test-spec consult

Task: authoritative acceptance-test spec for the repaired exterior-tile admission
`_InteriorAdmission.admits_exterior` (cogwheel/lensing/surrogate_training.py:1520-1581),
Chang-Refsdal positive-parity amplification surrogate. Test-completion build, no redesign.

## Code-path facts verified in-tree (feed to professor_code_observations later)
- `admits_exterior(center=(rho_center,theta_c), half=(half_rho,half_theta), source_magnitude_max)`:
  probes the INNER rho edge (rho_inner = rho_center - half_rho) — the point CLOSEST to
  the caustic — at `_INTERIOR_EDGE_SAMPLES` polar angles across the tile theta span.
  Reconstructs y_mag = r_caustic(gamma,theta_c) + rho_inner - 1 (additive rho>1 arm of
  surrogate._from_caustic_fixed) at EVERY band gamma. Admits iff every probe: rho_inner>1,
  min-distance to per-gamma caustic-CLOUD >= eta_max, y_mag <= source_magnitude_max.
- eta_max = 0.05 (class attr). Cloud = per-gamma (K,2) eigenframe point array; test is a
  discrete `min` over that cloud.
- ORACLE `geometry.nearest_caustic_point(gamma,beta,source,kappa,n_grid)`: solves the
  CONTINUOUS stationarity g'(theta)=0 of angular squared distance along the critical curve
  by analytic-Newton (+ minimize_scalar fallback). GENUINELY different computation from the
  discrete-cloud min in admits_exterior. Independence for the coverage test holds, with the
  caveat that the cloud is a discrete sampling of the SAME caustic curve the oracle searches
  — so the test grades a discrete approximation against the continuous truth (honest, and the
  intended thing under test). NOT circular: admits_exterior never calls nearest_caustic_point.

## Rulings issued (full text in plan file
## /home/tejaswi/.claude/plans/cached-bubbling-crab-agent-a7f82a14b90f986a9.md)
1. Coverage = fraction of ORACLE-truly-exterior sample points ({nearest-caustic-dist >= eta_max}
   AND inside prior box) that land inside SOME admitted tile. Oracle independent. Confirmed.
2. Zero-false-admit: 5x5 grid INSUFFICIENT as stated — must PIN the inner rho edge and the two
   theta extremes explicitly (worst case = inner edge; interior sampling can miss it). Spec: a
   grid that INCLUDES the inner-rho face at both theta extrema + midpoints, oracle-checked.
3. Reachable-red (restore scalar -> (0.80,0.90) collapses to zero tiles): valid falsification.
   exclusion circle 5.74 > box 4.24 => whole box excluded => zero tiles. Real teeth. Also add a
   POSITIVE red on the fixed code (>=1 admitted tile in that band) so the test can't pass by
   admitting nothing everywhere.
4. 0.95 too tight if measured over full band. Fix: measure coverage over the region ACTUALLY
   inside the prior box (union extent), not full sampled band. Then 0.95 defensible. Recommend
   a per-band floor with a modest discretization allowance and assert MONOTONE improvement vs
   the scalar ground truth (0.944/0.632/0.271/0.000) as the load-bearing non-circular teeth.
5a. w_min*Im(tau_c)=0.72 gate fixture: almost certainly a FIXTURE problem (source too on-axis /
    w_min too low to construct a gate-PASSING source). Fix the fixture (farther off-axis / higher
    w_min) AND keep a separate test that the gate RAISES GhostDomainError below 2.0. Only suspect
    a production bug if a source that SHOULD pass (analytically w_min*Im(tau_c) >= 2) still raises.
5b. Single held-out eps=2.6e-1 vs 0.2 budget, 7x median: do NOT widen tolerance. Leave failing,
    report, root-cause (locate the point: near-cusp? box corner? interpolation-node gap?). 7x
    median outlier at ONE interior point smells like a local defect / node-placement gap, not
    global miscalibration.

No blockers. This is a validation obligation (vi/PP-adjacent) under the CR memory's
verification list.
