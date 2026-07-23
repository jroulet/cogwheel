# Coder Short-Term Observations

WP2 ppGO consumers cell-ceiling wiring (likelihood.py + surrogate_training.py):
- Chose ADDITIVE non-breaking design over tuple-return so existing HEAD-
  behavior tests survive as regression guards; Test Dev adds ceiling tests.
- likelihood: extracted LensedRelativeBinningLikelihood._ppgo_cell_coords
  (parity,gamma,rho | None; DRY the caustic_geometry derivation) shared by
  _ppgo_band_split (unchanged float|None w_trust contract) and NEW
  _ppgo_cell_ceiling (float|None). _surrogate_coefficients beyond-wall
  guard now eff_ceiling=min(wall, cell_ceiling) via _ppgo_cell_ceiling;
  cell_ceiling None -> wall alone = HEAD byte-identical.
- surrogate_training: NEW _stratum_ppgo_ceiling(parity,gamma,rho,map)
  mirrors _stratum_ppgo_boundary; _apply_ppgo_trim gained OPTIONAL
  ceiling=None 3rd arg (keeps existing 2-arg test calls green). Trim gate:
  boundary None->keep; ceiling not None AND w_max>ceiling->keep (tail stays
  charted/refused); else HEAD drop/cap/keep. Both callers (ext L~1927,
  int L~1986) query *_ceiling and pass it. Verified trim truth table +
  parse + import + no other certified_ppgo_map consumer (pipeline_graph).
- NOT owed but note: _apply_ppgo_trim mock-spy test in
  test_lensing_farfield_envelope.py unaffected (Mock swallows 3rd arg).
- OWED to Test Dev (all in test_lensing_ppgo_bandsplit.py): ceiling-aware
  gates — (a) draw w_hi>cell_ceiling but <parity_wall no longer band-splits
  (falls to whole-band refuse) while w_hi<=ceiling still splits; (b) strata
  trim drop/cap only when ceiling>=stratum top, else keep; (c) UNKNOWN cell
  / no map byte-identical to HEAD. Note the _synthetic_map from_arrays
  fix (old 8 positional args) from WP1 is ALSO still owed and blocks these.

WP1 ppGO map truncation-on-refusal (ppgo_map.py, schema 0.1.0->0.2.0):
- _measure_cell now returns a 4-tuple (status,w_cert,w_diag,w_ceiling);
  build_map caller updated. New _max_accepted_prefix bisects the w-node
  prefix INDEX (O(log n), monotone saddle refusal) per angle; best_k==0
  (refusal at lowest w) or caustic LensDomainError still -> INVALID.
  w_ceiling = min over angles of max accepted w; guard floor>w_ceiling ->
  BEYOND_WALL (empty certified interval). Verified vs real engine:
  wall=90 gamma=0.2 truncates ceiling~51 (BEYOND_WALL), gamma=0.3
  certifies w_cert~62 ceiling=90.
- rho cap: rho_measured_max_grid (band top edge; lo*1.5 for open outer inf
  band) added to the (2,n_gamma,n_rho) family. Cap enforced INSIDE _cell
  (rho>cap -> None) so ALL accessors fall through to UNKNOWN, no new
  sentinel.
- Both new grids hashed in _content_hash (added args) + read in load via
  data['w_ceiling']/data['rho_measured_max'] (KeyError => ceiling-less
  artifact hard-refused, use_certified returns False). save_map savez'd
  them. schema_version 0.2.0, provenance w_ceiling_rule +
  rho_measured_max_rule strings. New w_ceiling accessor + module
  certified_w_ceiling (added to __all__), mirrors w_cert exactly.
- OWED to Test Dev: test_lensing_ppgo_bandsplit.py _synthetic_map (L151)
  calls from_arrays with OLD 8 positional args -> now TypeError; needs
  w_ceiling_grid (after diag) + rho_measured_max_grid (after interp).
  That file is also where WP2 ceiling-aware dispatch/strata-trim tests go.
- from_arrays new params are REQUIRED positional (explicit schema); no
  defaults (a map without a real ceiling must not silently certify).