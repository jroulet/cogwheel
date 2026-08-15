# Professor short-term (latest review)

## 2026-08-15 — lobe cusp-coincident-edge tolerance review (verdict PASS)
Reviewed `_lobe_cusp_axis_map` edge-coincidence fix (surrogate.py L617-728,
`_CUSP_EDGE_COINCIDENCE_ULPS=8`). Physics sound: u=d**(2/3) is the A3 cusp
caustic-reach scaling; a tile edge landing on a cusp ray clamps d->0 exactly
(`max(theta_lo-cusp,0)` left / `max(cusp-theta_hi,0)` right) and anchors u
there; `np.clip(base_lo-u_fine,0,None)**1.5` on the right guards the power
against FP-negative base (would NaN). Symmetric tol on BOTH edges.
Fast tests (test_lensing_surrogate_lobe.py): 19 pass in 4.4s
(LobeCuspAxisMap*, EdgeCoincidence*, LobeChildBoxesCoincidentEdge*).
test_lensing_lobe_subdivision.py: 49 pass in 112s. Independent numeric check:
- Pin A: endpoints bit-exact, u_max==0.5**(2/3) to 0 rel-err, strict incr.
- Pin B (7a sliver tl=0, th=3.5527e-16, ca=3.2703e-16): ca<th holds (real
  straddle premise), no raise, endpoints bit-exact, NON-decreasing with 0
  strictly-negative diffs (linspace nodes collide at ~3.5e-16 -> >= is the
  correct invariant, as specified).
- Trichotomy: tol band 8*eps*max(1,0.9)=1.776e-15; 2e-17 admits, 1e-3
  raises. exterior/on-edge/hair-inside->map, genuine straddle->ValueError,
  both sides.
- Caller path `_lobe_child_boxes`: coincident lower edge -> side='left',
  4 children, split in [theta_lo,theta_hi]; interior cusp still propagates
  ValueError (guard not vacuous).
Heavy full-engine training + sampling (COGWHEEL_TRAIN_TIER=1) operator-deferred.
