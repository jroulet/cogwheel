# Test Dev Short-Term Observations

## Build 8h-b-2 ppGO outer-annulus cap + strata sweep (SAME suite, 55->66)
- Added 2 classes to test_lensing_ppgo_bandsplit.py: StrataTrimCeilingSweep
  (spec: strata trim respects ceiling, per-stratum action VECTOR view) and
  OuterAnnulusRhoCap (spec: outer rho band [4,inf) capped at rho_measured_max).
  No production edits. Baseline 55->66, green in ~15s.
- OuterAnnulusRhoCap: new helper `_finite_rho_map(rho_measured_max, w_cert,
  w_ceiling, gamma)` — rho_edges=[0,4,inf], certifies (0,0,1) outer positive
  cell with FINITE rho_measured_max. CertifiedPpgoMap._cell cuts on strict
  `rho > rho_measured_max_grid[cell]` -> None -> UNKNOWN. Boundary radius is
  INCLUSIVE; step tested with math.nextafter(6.0, inf). Reachable-red = twin
  built with rho_measured_max=inf certifies rho=50 (finite). Consumer routing:
  _stratum_ppgo_boundary/_ceiling -> None beyond measured -> _apply_ppgo_trim
  keeps whole band. w_trust oracle = max(1.5*wc, wc+2).
- Had to ADD certified_w_cert, certified_w_trust to the import block (only
  certified_w_ceiling was imported before) — global-accessor test NameError'd.
- StrataTrimCeilingSweep: mirrors _train_band_charts loop by calling REAL
  _apply_ppgo_trim per stratum over a heterogeneous set; asserts the action
  VECTOR (keep/cap/drop/keep/keep) and that KEPT beyond-ceiling strata retain
  full range (tail intact). Reachable-red = same sweep w/ ceiling=None (HEAD)
  gives (keep/cap/drop/cap/drop). boundary/ceiling sourced from _synthetic_map
  via real _stratum_ppgo_boundary/_ceiling. Overlaps existing CeilingAware
  (per-call isolation) but adds the one-pass vector + tail-preservation angle.
- Neighbor test_lensing_surrogate_training collects clean (59) but full run
  >10min (engine campaigns) — test-only change can't regress it; did NOT run.

## Build 8h-b ppGO w_ceiling suite (test_lensing_ppgo_bandsplit.py)
- Extended the sole ppGO suite (36->55 tests) with 4 classes:
  TruncationOnRefusal, CellCeilingBandSplitGuard, CeilingAwareStrataTrim,
  LoaderCeilingRefusal. No production edits.
- WP1 schema drift first broke the existing suite: from_arrays is now
  10-arg (added w_ceiling_grid + rho_measured_max_grid); _synthetic_map and
  the corrupt-artifact re-save both had to carry all 9 stored arrays or
  load() raises KeyError BEFORE the hash check (masks the intended
  ValueError). Fixed both.
- WP2 refactored inline cell coords into LensedRelativeBinningLikelihood.
  _ppgo_cell_coords, now called via self by _ppgo_band_split /
  _ppgo_cell_ceiling; the old `_ppgo_band_split(object(), lens)` breaks with
  AttributeError. Fix: a stateless `_DispatchProbe` class binding the three
  REAL methods as class attrs, called as `_DispatchProbe()._ppgo_band_split(lens)`.
- Truncation stub (spec 1): _measure_cell imports engine LOCALLY at call
  time, so patch cogwheel.lensing.chang_refsdal.channels.ChangRefsdalChannels,
  .operator.geometric_amplification, AND module-level ppgo_map.caustic_geometry.
  Stub evaluate raises the REAL CancellationError above a per-angle w*(angle)
  (tightest at pi/2); glue==exact so error 0 -> CERTIFIED, w_cert=node0,
  w_ceiling=min-over-angle endpoints. Oracle = largest _w_nodes(wall) node
  <= w*(angle), min over the 5 angles (0..pi/2), recomputed independently.
  Reachable-red: patch ppgo_map._max_accepted_prefix with a no-prefix variant
  (return 0,None on any top-node refusal) -> STATUS_INVALID.
- Band-split guard (spec 2): production decision lives inline in
  _surrogate_coefficients (too heavy to call). Reproduced faithfully in a
  test helper `_dispatch_band_splits(lens,dense_w,honor_ceiling)` sourcing
  w_trust/ceiling from the REAL probe methods; honor_ceiling=False = HEAD
  (parity-wall-only) = the reachable-red arm that wrongly splits a
  w_hi-in-(C,W) draw. eff_ceiling=min(wall,cell_ceiling).
- Strata trim ceiling arg: _apply_ppgo_trim(w_range,boundary,ceiling); when
  ceiling not None and w_max>ceiling -> 'keep' (else drop/cap); ceiling=None
  (HEAD) -> drops/caps beyond-ceiling strata = red witness.
  _stratum_ppgo_ceiling(parity_int, gamma, rho, map): parity 1=positive.
- Loader (spec 3): _saveable_ceiling_map (full provenance + _content_hash)
  survives save_map/load; re-save dropping 'w_ceiling' -> KeyError,
  mutating a w_ceiling value w/o re-hash -> ValueError; both make
  use_certified_ppgo_map return False + leave global None. UNKNOWN sentinel
  is `is`-comparable from ppgo_map.
- No neighbor suite imports ppgo_map/dispatch/strata-trim (grep clean) — this
  suite is the sole owner; surrogate_training suite collects clean (59) but
  is slow (engine campaigns), unrelated to this test-only change.