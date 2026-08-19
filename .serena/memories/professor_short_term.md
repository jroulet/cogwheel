# Professor short-term (tiling_plan inference review, 2026-08-19)

Reviewed `cogwheel/lensing/tiling_plan.py` (engine-free demand-sized chart
cost predictor) via `cogwheel/tests/test_lensing_tiling_plan.py`.

## Result: PASS. 39/39 tests green in ~24 s (env cogwheel-newlal).

Formulas verified against my 2026-08-18 rulings (professor_short_term prior):
- Law 1 gamma: `_gamma_resolution = _C_GAMMA(0.4) * r_caustic / |dr/dgamma|`,
  central FD on engine-free `_scalar_caustic_reach`; step clamped to stay on
  the band's side of the parity wall. Monotone-tightening toward gamma=1 on
  BOTH sides, cross-checked by an INDEPENDENT polar sweep of
  `geometry.r_caustic` (astroid on-axis, machine-exact ~2e-16). No built band
  straddles gamma=1 (structural test over real band contexts). ✓
- Law 3 w: `ceil(per_decade * log10(w_hi/w_lo))`, interior 15/dec vs
  exterior 4/dec — matches SACR-C fast-fringe vs beat-free-residual density.
  w-edges are measured demand edges (lobe_exterior 38, not blanket 60);
  above-ceiling residual CLIPS to DD ceiling 60 (INS-1-001 regression pin). ✓
- Annulus: explicit `gauge` field; astroid caustic_rho round-trips through the
  independent `ppgo_map.caustic_rho` to 1e-6 (teeth: 2x-wrong reach fails);
  saddle rho_lobe prior edge = real ~20 scale, above retired 2.40 cap. ✓
- Escalation: strict `>` on both caps (5e5 calls, 0.40 share), RECORDS not
  raises; currency pinned SECONDS_PER_CALL==0.0903, _LABELS_PER_NODE==8. ✓
- Engine-free whole-tool: real `run()` under 4 booby-trapped doors
  (ChangRefsdalChannels.evaluate, f_schwinger, _f_schwinger_mpmath,
  mpmath.gauss_quadrature) -> all call_count==0; doors proven live + disjoint
  from the caught-refusal tuple (no silent swallow). ✓

## Operator-deferred (do NOT run): production 10k-sample census cost/share.
`run(n_samples=2000)` exceeded 240 s; 10k is the heavy ship-gate path. My
n=100 probe: 2820 nodes / 22560 calls but max_region_share=1.0 (only
wedge_interior:+1 caught demand) -> escalation fired the share reason. This is
an UNDERSAMPLING artifact (predicate working as designed), NOT an explosion;
the real per-region distribution (expected tens-of-thousands nodes, ~1-3e5
calls, no region >40%) requires the full census and is operator-deferred.
No physics concern.
