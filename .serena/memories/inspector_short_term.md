# Inspector Short-Term Observations

## 2026-07-23 — Build 8h-b1 (WP1 map ceilings+schema, WP2 consumers)

Scope: uncommitted working-tree review. Code files: ppgo_map.py,
likelihood.py, surrogate_training.py, DATA_CONTRACTS.yaml, +test file.

Verdict: ISSUES (one minor design finding); all logic sound, 235+ tests green.

### Verified sound
- `_max_accepted_prefix` bisection: monotone-refusal assumption; non-monotone
  only shrinks k_max (conservative). best_result always from a prefix that
  fully evaluated w/o refusal (evaluate(k) runs engine on ALL of w_nodes[:k]).
- `_measure_cell` truncation-on-refusal: per-angle floor<=own ceiling always
  (floor is a w_prefix value). Degenerate check `if floor>w_ceiling ->
  BEYOND_WALL` correctly refuses empty certified interval [max floors, min
  ceilings]. Fully-accepted cell => ceiling=wall => eff_ceiling=wall =>
  byte-identical to HEAD.
- rho_measured_max cap: only bites in open outer annulus [4,inf) (inner bands
  have rho<hi=cap already). outer cap = _rho_center = lo*1.5 (=6.0), matches
  measured radius. `_cell` returns None (UNKNOWN) for rho>cap.
- Loader hard-refusal: load() does data['w_ceiling']/['rho_measured_max'] ->
  KeyError on pre-0.2.0 -> use_certified_ppgo_map catches (OSError,ValueError,
  KeyError) -> global stays None. Confirmed.
- Consumer guards: eff_ceiling=min(wall,cell_ceiling); ppGO served region
  [w_trust,w_hi] ⊆ [w_cert,w_ceiling]. UNKNOWN ceiling => wall alone =>
  byte-identical HEAD. w_ceiling accessor returns UNKNOWN for non-CERTIFIED.
- DATA_CONTRACTS certified_ppgo_map entry fully updated (schema 0.2.0,
  w_ceiling, rho_measured_max, truncation+cap prose). Producer
  scripts/train_ppgo_map.py only calls build_map/save_map -> unaffected
  (correctly unmodified).
- Tests include the mandated mutation/falsification cases
  (test_disabling_truncation_invalidates_the_cell,
  test_ignoring_ceiling_wrongly_splits_the_above_draw,
  test_dropping_the_ceiling_wrongly_trims_..., test_uncapped_twin_wrongly_...).

### Finding (design, INS-1-001)
`_surrogate_coefficients` calls `_ppgo_band_split(lens)` then
`_ppgo_cell_ceiling(lens)`; each independently re-derives `_ppgo_cell_coords`
-> `caustic_geometry` (a 720-angle x2-branch = ~1440 critical_point sweep).
So caustic_geometry runs TWICE per band-split draw. The shared `_ppgo_cell_coords`
helper dedupes CODE but not RUNTIME work. Correct, opt-in path only (surrogate+map
both off by default). Fix: derive coords once and pass to both queries (or a
single cell lookup returning w_trust+w_ceiling).

### Carry-forward
- Pattern confirmed: a fully-accepted cell must reduce to HEAD (ceiling=wall);
  the code does this by making angle ceiling = w_nodes[-1] on full accept.
