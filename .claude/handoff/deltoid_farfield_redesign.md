# Build: deltoid far-field coordinate redesign (census Q2 to false)

## Mission

The saddle (parity -1) deltoid far-field is the last chartless region:
production packs NO origin-polar saddle exterior (the legacy tiler is
retired from packing; `_REGIONS_BY_PARITY[-1] = ('tube', 'lobe_interior',
'lobe_exterior')` in `tiling_census.py`), the region falls through to the
exact engine, and the census's standing Q2 verdict on the legacy gauge is
`redesign_needed = true` — "cusp ray strictly inside a tile angular span".
A cusp ray interior to a tile makes the 2/3-power directional reach
non-monotone across it: no node budget fixes it. Redesign the deltoid
far-field angular coordinate/tiling so CUSP RAYS ARE TILE BOUNDARIES,
mirroring the astroid far-field's own construction: `_farfield_exterior_tiles`
folds the cusp rays into the quadrant and feeds them to
`_cusp_aligned_theta_tiles` ("Partitions [theta_lo, theta_hi] into sectors
bounded by the cusp rays … so NO tile straddles a cusp-ray kink"), and the
chart's angular spline axis is the per-cusp `u = d**(2/3)` (SPEC: "d the
angular distance to the NEAR caustic cusp … absorbing the d**(-1/3)
near-cusp divergence in dE/dtheta_c"). The deltoid has THREE cusps per lobe
and TWO lobes in D2 relation: fold to one lobe-edge fundamental, one
representative per orbit, mirrors served by the gauge-image law. Owed after
7a (which excludes the region) and before 7b claims table coverage here.

## Facts (measured; SHA 8f58104 unless noted)

1. Q2 verdict (`.claude/handoff/tiling_census_production_postF081.json`,
   produced 2026-08-15, committed 5aedd5a; config n_farfield_tiles_per_side=5,
   n_caustic_samples=500, gamma_band_halfwidth=0.02, f_max=0.40):
   mis_alloc_ratio = 1.655, redesign_needed = true, reason "cusp ray strictly
   inside a tile angular span", n_tiles = 24, representative_band
   [1.285, 1.325]. `_Q2_MISALLOC_THRESHOLD = 2.5`: the ratio is UNDER the
   bar — the cusp-in-tile condition alone triggers.
2. Verdict machinery (`tiling_census.py::_q2_deltoid_redesign`): computes
   `st._deltoid_cusp_source_angles(gamma_mid, n)` (six cusps, D2-folded to
   [0, pi/2]), tiles the representative band with the LEGACY
   `st._farfield_tiles(ctx.exclusion_rho, ctx.rho_outer_region, n_per_side,
   cusp_angles=…, gamma=…, gamma_band=…)`, then measures mis_alloc_ratio =
   max/min outer rho edge and strict cusp-in-span containment (tol 1e-9).
   A candidate passes when NO tile contains a cusp ray and ratio <= 2.5.
3. The defect (`surrogate_training.py::_farfield_tiles`): uniform n x n grid
   over origin-polar rho in [rho_inner, rho_outer] x theta_c in [0, pi/2]
   (D2 fold); cusps enter only as an EXCLUSION filter (`_exclude_near_cusp`,
   default d_exclude 0.35) — no edge alignment, so rays land tile-interior.
   Production RETIRED it for parity != 1 (`_train_band_charts`: "origin-
   centred far-field-window machinery is POSITIVE-PARITY ONLY"); this build
   certifies a coordinate for a fall-through region, not a live serving path.
4. In-repo cusp-adapted precedents: astroid — `_farfield_exterior_tiles` +
   `_cusp_aligned_theta_tiles` + per-column `admits_exterior`; lobe (saddle)
   — `_lobe_interior_tiles`/`_lobe_exterior_tiles` cusp-align lobe-local
   theta via the same helper, `_lobe_cusp_axis_map` (surrogate.py) supplies
   the per-cusp u = d**(2/3) axis with the F082 edge-coincidence tolerance
   (`_CUSP_EDGE_COINCIDENCE_ULPS = 8`), and `_subdivide_lobe_tile` splits at
   the u-midpoint, never the theta midpoint across a cusp.
5. F081 wiring (resolved 2026-08-15): the deltoid far-field inner edge
   (`physical_exclusion_radius`) is sized per band by
   `min_eta_max = f_max * min(arc_r_min)` — the lobe-edge arcs' own shell.
6. f-constants ruling (f_constants_decision.md, measured at 77da2e6):
   f_max = 0.40, f_floor = 0.08, both parities. Density, not constants,
   closes the bar gap: n_theta sized per band on the served span, linear in
   span. Flagged saddle density band: gamma ~1.1 (deltoid transition,
   eps 0.076-0.14); the other saddle bands pass at 0.003-0.05.
7. Demand shape (`demand_census_post_c3_regate_10k.json`, 10k draws,
   2026-08-17, extracted at 8f58104): 3792 saddle draws. Saddle
   engine_residual 1720: lobe_interior 868 (caustic_rho 0.037-0.999) +
   tube-region 852 (caustic_rho 1.00-1.99, median 1.41). The far-field-
   classified lobe_exterior cell holds 66 draws TOTAL, ALL analytically
   served (saddle_c3 42, born_analytic 24) — ZERO residual there; the
   below-split table need reads from per-draw w_split: lobe_exterior
   median 3.3 / max 5.9, tube-region median 6.2 / max 21.5. Residual gamma
   is broad over 1.0-1.55 (gamma [1.1, 1.157]: 159 draws). Demand is
   MODEST — size tiles and nodes to it; no collocation explosion.
8. Census saddle tiles at production config (same JSON as Fact 1): tube 2,
   lobe_interior 13, lobe_exterior 30; the legacy Q2 probe emits 24.

## Scope

IN: an ANALYTIC cusp-aligned deltoid far-field coordinate + tiler — angular
edges on the (folded) cusp rays, per-cusp u = d**(2/3) axis, eta-adapted
radial coordinate off the lobe edge (the log(rho-1) fix's deltoid analogue),
all scales derived from the problem's own geometry (cusp angles, r_deltoid,
min_eta_max); D2 orbit fold (one lobe-edge fundamental; mirrors by the
gauge-image law); Q2 machinery re-pointed at the new tiler so the census
re-run adjudicates the candidate; demand-sized tile/node budgets (Fact 7,
density per band per Fact 6); fast synthetic tests.
OUT: tube trainer resolvable-subarc trim (separate fragment
`lensing_tube_trainer_resolvable_subarc_trim.md`; `_tube_training_arcs` /
`_build_tube_chart` untouched — the only interaction to RESPECT, not
modify, is that the far-field inner edge abuts the per-arc tube shell
`min_eta_max` and the near-cusp annuli stay the tube/Pearcey arm's domain,
F074 controls); training campaigns; serving-ladder changes; astroid-side
tiling (`_farfield_exterior_tiles` and its callers byte-identical); lobe
interior/exterior tiling and charts.

## Acceptance

- Tiling census Q2 re-run at production config (Fact 1 settings):
  `redesign_needed` reads FALSE; report the new mis_alloc_ratio (improves
  from 1.66 toward ~1).
- Value pin, per-lobe per-cusp: for every emitted tile and each of the six
  folded cusp rays, the ray lies on a tile edge or outside the tile's
  angular span — asserted on computed tile geometry, never on which branch
  emitted it.
- Existing lobe/wedge/astroid tiling byte-identical: one canonical equality
  pin on the incumbent tilers' outputs at a fixed config.
- Full fast suite green.
- Test parsimony: re-point the existing census Q2 pins rather than adding
  parallel ones; one canonical pin per invariant; report added-vs-retired.

## Constraints

Branch claude-dev only. The coordinate map is ANALYTIC — no fitted or
measured constants inside it (measured facts may size budgets, never the
map). Train/verify on the D2 orbit representative only. In-build tests
fast/synthetic; census re-runs at production config and bulk sweeps are
driver post-build steps. Escalate on surprise rather than iterate — in
particular if no edge-aligned candidate can satisfy the Q2 cusp-ray test,
or the u = d**(2/3) scaling demonstrably fails on a deltoid lobe edge
(that falsifies the design premise, not the plumbing). Closes
`todo.d/lensing_deltoid_farfield_coordinate_redesign.md` `[→ spec]`:
delete the fragment, add the completed.d record, spec_changelog.d fragment
with `bump:`, run `python scripts/render_fragments.py`.
