Last session: 2026-08-04 production batch. Clean.

2026-08-06 WP1 wire InteriorWedgeChart into training (surrogate_training.py ONLY):
- Retired ffin path: positive-parity astroid interior now tiled by
  _wedge_interior_tiles (ONE angular col theta_wedge in [0,pi/2], center pi/4
  half pi/4, uniform radial rows floored at _WEDGE_R_MIN=1e-2, r_extent capped
  <1) + _build_wedge_chart -> LensAmplificationSurrogate.from_wedge_engine
  (definition=INTERIOR_SACR_C; DD-cap + arc map applied INSIDE engine). Build
  loop region=='wedge_interior' branch: inline held-out via chart.wedge_map +
  _from_wedge_fixed; CarrierDiscontinuityError/gated -> ladder-served gap, NO
  subdivision (mirrors lobe).
- DELETED _farfield_interior_tiles (no prod caller). _build_farfield_chart +
  _subdivide_farfield_tile now exterior-only (dropped `definition` param;
  both subdivide call sites pass interior_admission=None). _heldout_eps:
  is_farfield_label = isinstance(chart, FarFieldChart) (wedge -> else-branch);
  annotation + _load_or_build annotation gained | InteriorWedgeChart.
- KEPT _interior_admission UNCHANGED (live exterior-tiler dep at ~L4014 via
  exterior_admission; brief was WRONG). admission var stays None on positive
  parity path (annotated assign L4182); only consumed in parity!=1 saddle path.
- chart_types provenance (L3462) LEFT coarse tube/farfield: InteriorWedgeChart
  -> 'farfield' same bucket as LobeInteriorChart (out-of-scope census label,
  no crash/dispatch impact). _gate_chart type-agnostic (kind='interior').
- Smoke: ast.parse OK + import OK. UNVERIFIED: no training run (per WP1
  no-exec constraint) -> wedge training path UNVERIFIED end-to-end.
- OWED TEST BREAKAGE (Test Dev): test_lensing_exterior_windows.py ~L2079 and
  test_lensing_ppgo_bandsplit.py ~L89/L620 import/use deleted
  _farfield_interior_tiles.
