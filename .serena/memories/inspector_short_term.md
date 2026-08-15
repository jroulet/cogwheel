# Inspector Short-Term Observations

## 2026-08-15 review: lobe_cusp_axis_edge_tolerance (WP1) — PASS

Scope: uncommitted diff in cogwheel/lensing/surrogate.py (`_lobe_cusp_axis_map`)
+ cogwheel/tests/test_lensing_surrogate_lobe.py (new edge-coincidence suites).
Fixes the 7a smoke crash: a lobe-exterior tile whose theta_hi edge lands on
the theta=0 deltoid cusp ray (cusp 3.27e-16 vs theta_hi 3.55e-16, 2.8e-17
apart) tripped the strict `cusp_angle > theta_hi` guard.

FIX VERIFIED CORRECT:
- New constant `_CUSP_EDGE_COINCIDENCE_ULPS = 8` (dimensionless ULP count,
  well-documented as float-representation noise, NOT a physical fudge — passes
  the F041/part0 discipline; ULPS suffix, not _EPS).
- Guard relaxed to admit cusp within `8*eps*max(1,|edge|,|cusp|)` of the
  side-appropriate edge; far-edge d clamped via `max(..., 0.0)` so the
  degenerate d=0 anchor is exact. Endpoints still forced
  (theta_fine[0]=theta_lo, [-1]=theta_hi). Genuine straddle (cusp interior
  beyond band) still raises. Keep-map semantics per brief option (a).
- Signature UNCHANGED (4 positional). Both production callers
  (from_lobe_engine:3884, from_lobe_exterior_engine:4081) and
  surrogate_training._lobe_child_boxes:4225 pass side via _lobe_nearest_cusp,
  which sets side from the tile CENTRE — so side='right' => cusp>center>theta_lo
  and side='left' => cusp<center<theta_hi, guaranteeing the far-edge distance
  (d_lo right / d_hi left) is > 0. The negative-base complex-power regime
  (would need tile width < ~1.4e-15 rad with cusp on the far side) is therefore
  UNREACHABLE via production callers. Noted, not a finding.

SIBLING AUDIT (brief-required):
- `_wedge_cusp_axis_map` (:531): cusp fixed at domain edge via `origin`,
  no cusp_angle straddle guard. Different shape. Not affected. Correctly
  untouched.
- `_deltoid_cusp_axis_map` (:730): has cusp_angle but handles the coincident/
  hair-inside case gracefully — `if theta_lo < cusp_angle < theta_hi: return
  None` (raw-theta fallback), and edge-exact goes to a valid d=0 branch. Does
  NOT crash on machine-precision coincidence. Different shape. Correctly
  untouched.

TESTS: new suites (LobeCuspAxisMapEdgeCoincidence*, LobeChildBoxesCoincidentEdge)
value-assert endpoints bit-exact, monotonicity, the exact 7a literals as a
regression pin, a boundary trichotomy (exterior/on-edge/hair-inside->map,
straddle->raise) on BOTH sides, and self-falsification (1e-6 straddle still
raises; caller-path interior cusp propagates ValueError). Pre-existing
"wrong-side raises" tests survive (offsets 0.1-0.7 rad >> 8-ULP band).
Ran test_lensing_surrogate_lobe.py + test_lensing_lobe_subdivision.py:
139 passed, 10 skipped (train-tier). Targeted subset 19 passed.

Plan noted test_lensing_lobe_subdivision.py as expected-to-change; it was NOT
edited (only surrogate.py + test_lensing_surrogate_lobe.py). No problem — the
subdivision file's existing suite still passes.

No SPEC/DATA_CONTRACTS impact (internal helper, no schema tag / data product).
Nothing for Librarian from this build.

Carried-forward doc-staleness for Librarian (NOT this build):
INS-1-002/003 exterior_polar_rho_log_carrier_v1 "ONLY known tag" staleness
since V5 2D carrier; SPEC region-vocabulary gaps (lobe_exterior etc.).
