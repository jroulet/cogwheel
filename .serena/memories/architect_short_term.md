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
