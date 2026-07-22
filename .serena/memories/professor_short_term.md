# Professor short-term — Build 8g inference review (2026-07-22)

VERDICT: PASS. Ran the new WP1/WP2/WP3 domain shard
`cogwheel/tests/test_lensing_surrogate_training.py` in env `cogwheel-newlal`:
**59 passed, 0 failed in 452 s (~7.5 min)**. Independently re-derived the
physics-critical margins (script, since deleted):

- WP3 saddle tube-tail (Q6-iv): fix-ON heldout_eps = **0.0263** (< tube bar
  5e-2, < 0.1 headline); fix-OFF = **0.4335** (> pathology floor 0.09, matches
  Professor's measured ~0.43; coarse fixture so below the 1.15 headline). Fix-on
  arc is a genuine `_saddle_arcs` product carrying the wedge-edge exclusion
  window; fix-off faithfully reverts to astroid cusp-safety + no wedge window.
  Straddles the registration bar => reachable-red, bites the real pathology.
- Residue partition (Q5): N=3000 draws, seed 8080808, closes exactly
  (359 beyond_w_cap + 30 chart_served + 2611 residue = 3000), beyond_served=0
  (no beyond-ceiling draw served its whole band). residue_frac=0.870 is
  MEASURED-and-reported, NOT asserted zero (Build 8h north star). The ~1%
  whole-prior chart-served is a SMOKE-CONFIG artifact (n_gamma=n_u=n_theta=4),
  not production coverage — expected, not a defect.
- Tiling (Q6-i): strict disjointness (max-norm sep >= 2h, tol 1e-9 ULP guard),
  outside-disk, >=3 low-stratum tiles, dropped=admitted-cap, saddle beyond_w_cap
  starts m>400 (~458) with w_ceiling=58, astroid fully reachable (no beyond
  bucket) — all consistent with my 8g consult numeric geometry.
- Corner-cap fix (INS-1-001) LANDED: FarFieldCornerCap tests (designed red vs
  unfixed code) now green — corner product 82 -> 58 via y_max=Y*sqrt2.
- Eps gate reachable-red + resume: healthy registers, poisoned+NaN gated with
  reasons, reverting poison re-registers, persisted eps round-trips on resume
  (deterministic, no recompute), legacy-no-eps mixed-version resume passes
  ungated. Astroid byte-identity max diff 0.0. SelfFalsification suite (11
  mutations) all bite. Whole-band containment uses independent w oracle
  (1.2372e-4), distinct from production `dimensionless_frequency`.

Operator-deferred (out of my budget): heavy full-sampling PP/injection
recovery. Fresh diagnostic PNGs at cogwheel/tests/output/wp{1,2,3}_*.png,
q5_residue_bucket_over_lnm.png (07:06-07:11) — could not view (no image tool
in this mode); verified their embedded numbers via the assertions + rerun.
