# Professor Short-Term Observations

## F080 CLEAN-cell saddle rho<1 relaxation review (2026-08-14) — PASS
Reviewed the ppGO-map allowlist relaxation build (uncommitted working tree:
ppgo_map.py, likelihood.py, surrogate_census.py + tests). Ran the 13 relaxed-cell
pins (ShippedMapSaddleRelaxedCell / BandScopedRelaxation / RelaxedCellGovernance /
CensusLikelihoodBandSplitMirror / RelaxedCellSelfFalsification) — all 13 PASS in 3s.
Independent first-principles check on `CertifiedPpgoMap.load()`:
- In-box Cell-1 (gamma mid of [1.15729,1.33933], rho=0.25): w_cert=19.164305537818887
  (exact, rtol1e-9); w_trust=28.74645830672833 == max(1.5*floor, floor+2)=1.5*floor
  (the 50% inflation dominates, 28.746 > 21.164); w_ceiling=58.0 finite & >= w_trust. ✓
- F073 preserved: cell2 MARGINAL, cell3 CONTAM, generic gamma=2.5, rho=0.5 edge,
  gamma±1e-3 neighbors ALL return UNKNOWN on all three methods. Band/edge-scoped. ✓
- Allowlist source `_SADDLE_RHO_RELAXED_CELLS` has exactly ONE active entry (Cell 1);
  Cell 2 recipe present but COMMENTED OUT (documentation-only) — matches governance pin.
- Census/likelihood mirror (served==counted) agree in-box and both refuse off-band;
  empty-allowlist self-falsification flips both to None. ✓
- Test-parsimony: +15 test defs / -3 retired (retired old-blanket guards
  test_corridor_source_0, test_ppgo_cell_coords_would_return_tuple_without_guard,
  test_site4_rho_none_is_load_bearing in test_lensing_saddle_rho_guards.py).
GOTCHA: `get_certified_ppgo_map()` returns the process-global singleton = None in a
fresh process; tests/oracles must use `CertifiedPpgoMap.load()`.
NOT run (operator-deferred): full test_lensing_ppgo_map.py / test_lensing_surrogate_census.py
time out (>240s) due to the known mpmath 60<w<=150 slow band, unrelated to this build.
