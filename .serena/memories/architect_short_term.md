# Architect Short-Term Observations

Build 1d (wedge standoff + tube normal), 2026-07-30: delete `_WEDGE_EPS`
(6 sites, all np.linspace wedge bounds) sample edge-to-edge; keep the
`_saddle_arcs` wall exclusion anchored at true edge (separate concern).
Replace `_tube_normal` finite diff with `y'/|y'|` from
`geometry.caustic_derivatives`; SIGN is the tripwire (F041). Part C: fix 4
false prose lines. All production+prose -> one Coder WP; all test authoring
+ test-file helper/prose fix -> Test Developer via domain_test_descriptions.
Gate 4 (inward_sign identical, both parities) is load-bearing; verify
analytic vs finite-diff orientation agreement per arc (non-circular).

VERIFIED (build 1d plan): _WEDGE_EPS at 6 linspace sites (_saddle_arcs
628-29, _lobe_cusp_source_angles 1417-18, _caustic_inradius 1533-34,
_lobe_caustic_points 1992-93, _lobe_winding_loop 2022-23, second cusp
sweep 2054-55). _tube_normal L391-409 dth=1e-6; y' from
caustic_derivatives returns (2,) for scalar theta. Walls anchor
edge_hw=_SADDLE_CUSP_MIN_HALFWIDTH -> anchor at true edge (drop eps),
KEEP the wall exclusion. Prose 'deltoid cusp' in geometry.py
caustic_derivatives docstring L1756-57 AND L1783-85 (fix both; NOT
L2137 surrogate = real interior cusps, correct). Test helper
_chosen_serve_theta L343-354 + SERVE_ALIGN_MIN prose L200 -> Test Dev.
One Coder WP (prod+docstring); all test authoring+helper -> Test Dev.
