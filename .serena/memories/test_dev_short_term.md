# Test Dev Short-Term Observations

2026-08-15 WP1 edge-coincidence FOLLOW-ON shard (trichotomy + caller-path):
extended test_lensing_surrogate_lobe.py. (A) Added
test_boundary_trichotomy_at_edge to LobeCuspAxisMapEdgeCoincidenceTestCase:
subTest sweep BOTH sides x {exterior +1e-3, on_edge, hair_inside -/+2e-17,
straddle -/+1e-3} at theta_lo=0.4/theta_hi=0.9 — pins </<= flip guard;
exterior/edge/hair build strictly-increasing maps w/ bit-exact endpoints,
straddle raises. NOTE hair 2e-17 < half-ULP(0.9)=5.55e-17 so rounds to
edge exactly — still exercises tol branch, assertions hold. (B) New
LobeChildBoxesCoincidentEdgeTestCase (training_module._lobe_child_boxes,
engine-free): coincident lower edge (center theta=3.5e-16, half=3.5e-16 ->
theta_lo=0.0, lobe_cusps=[0.0]) routes _lobe_nearest_cusp->side='left'->map,
returns 4 boxes + theta_split in [lo,hi]; teeth = interior cusp 0.6 in
[0.2,0.8] side='right' straddle propagates ValueError through the splitter.
Audit: all pre-existing assertRaises on _lobe_cusp_axis_map (subdivision
1006-1032, surrogate_lobe 2240-2264) use gross 0.1+ rad offsets >> 1.8e-15
band -> none moot. Full file 90 pass/10 skip 1m47s; subdivision cusp class
11 pass.

2026-08-15 WP1 `_lobe_cusp_axis_map` edge-coincidence tolerance (keep-map):
extended test_lensing_surrogate_lobe.py with
LobeCuspAxisMapEdgeCoincidenceTestCase (Pin A right/left cusp==edge keep-map;
Pin B logged-7a machine-precision sliver theta_hi=3.552713678800501e-16,
cusp=3.270275691376951e-16 -> non-decreasing map, no ValueError) +
self-falsification (cusp 1e-6 inside edge >> 8-ULP band STILL raises, both
sides). Audit: pre-existing "genuine straddle raises" tests (offsets 0.1-0.7
in BOTH surrogate_lobe and lobe_subdivision) are far above _CUSP_EDGE_
COINCIDENCE_ULPS(=8)*eps~1.8e-15, so the new tolerance does NOT flip them —
no regression. Pin B teeth: old guard `if not cusp_angle > theta_hi` tripped
because cusp<theta_hi (assert it). Full file 87 pass/10 skip in 1m51s.
