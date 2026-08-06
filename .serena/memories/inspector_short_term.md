# Inspector short-term

## 2026-08-06 — Build wire_interior_wedge_chart (re-review, findings resolved)
Scope: uncommitted worktree diff in cogwheel-claude-dev.
WP1: wire InteriorWedgeChart into positive-parity interior branch of
`_train_band_charts`; retire the ffin (far-field interior) path.

### Verdict: PASS

### Previously-open findings — BOTH RESOLVED (re-verified independently)
- INS-1-001 RESOLVED: `definition` param REMOVED from `_build_farfield_chart`
  signature (now `(*, gamma_band, parity, box_center, half, w_range, config,
  w_nodes_per_decade=None)`; hardcodes `definition=FARFIELD_KERNEL_SUM`
  internally). The `_reprovision_w_nodes._eps_for` call site (~L3170) no longer
  passes `definition=`. Import-probe + read confirm. The reprovision test
  (`test_reprovision_recommendation_forwarded_to_node_density`) now mocks
  `LensAmplificationSurrogate.from_engine` (NOT `_build_farfield_chart`), so the
  real `_build_farfield_chart` binding runs unmocked = the guard the prior
  finding asked for.
- INS-1-002 RESOLVED: `_farfield_interior_tiles` fully DELETED
  (hasattr==False). `_farfield_exterior_tiles` docstring (~L1860) rewritten to
  reference the cusp-alignment convention / `_cusp_aligned_theta_tiles`; no
  dangling reference to the deleted helper.

### New wedge wiring verified correct end-to-end
- `_wedge_interior_tiles(r_extent, n_per_side)` (~L2313): uniform radial rows
  over [_WEDGE_R_MIN=1e-2, r_extent], single angular col theta_center=pi/4,
  half_theta=pi/4, j=0; [] if r_extent <= _WEDGE_R_MIN.
- `_build_wedge_chart(*, gamma_band, parity, box_center, half, w_range, config,
  w_nodes_per_decade=None)` (~L2934): raises ValueError if parity!=1; calls
  `from_wedge_engine(gamma_range, r_range, theta_wedge_range, w_range,
  n_gamma=config.n_gamma, n_r=config.n_rho, n_theta_wedge=config.n_theta_c,
  w_nodes_per_decade, definition=INTERIOR_SACR_C)`; returns (chart, n, refused).
  Signature matches surrogate.py `from_wedge_engine` (L3714-3887); config fields
  n_gamma/n_rho/n_theta_c all exist.
- `_heldout_eps` type annotation extended `| InteriorWedgeChart`; gate uses
  `interior_eps_max` bar. Held-out probe uses `_from_wedge_fixed(gamma, r,
  theta_wedge, chart.wedge_map)` — same eigenframe the engine trains/serves on.
- `_train_band_charts` interior else branch: r_extent = min(grid_rho_extent,
  1.0 - max_eta_max/coordinate_radius_min); region label 'interior'→
  'wedge_interior'; gated/carrier-flip → ladder-served gap (no subdivision).
  `_coordinate_radius_bounds` (L3969) bit-identical to old
  np.min(admission.radius_grid).
- `_subdivide_farfield_tile` interior branch removed; both call sites pass
  interior_admission=None.
- InteriorWedgeChart (surrogate.py L2390-2574) has wedge_map, image_count,
  refused_points, log_w_grid, envelope_definition, theta_to_s.

### Non-blocking observations (NOT findings; harmless)
- Slightly stale comment ~L3721-3724 in `_subdivide_farfield_tile` and a dead
  `elif region in ('interior','lobe_interior')` branch survive but are
  unreachable (all callers pass exterior region).

### Spec / contracts
- SPEC.md already documents InteriorWedgeChart as a trained chart (ASTROID
  INTERIOR CHARTS paragraph: from_wedge_engine, _wedge_serves, DD-product
  ceiling, arc-length axis). NO spec-code divergence introduced.
- DATA_CONTRACTS: lens_amplification_surrogate is the shipped artifact; wedge
  chart already an accepted chart type. No new serialized-artifact change here.

### Tests (fast tier, this env) — all green
- test_lensing_interior_wedge_chart.py: 63 passed.
- test_lensing_exterior_windows + ppgo_bandsplit + census batch:
  146 passed / 4 skipped / 1 xfailed.
- Census slow mpmath test now @_TRAIN_TIER_SKIP (todo fragment
  lensing_census_tests_hit_mpmath_in_fast_tier.md added).

### Carry-forward
- This build completes TODO `lensing_interior_wedge_chart_unwired`
  (note for Foreman/Librarian).

### Pattern reinforced
- Removing a param from a helper: an unmocked binding test on the real callee
  is the only reliable guard — mocking the callee with **kwargs hides a stale
  kwarg. Coder correctly moved the mock UP to `from_engine` so the real
  `_build_farfield_chart` binds for real.
