# Architect Short-Term Observations

## saddle_tube_fundamental_training build (planning 2026-08-14)
- Follow-on to tube_d2_fold. Trim SADDLE `_tube_training_arcs` from 6 arcs
  to one-rep-per-D2-orbit (expect ->3, DERIVE). Group = {id, pi-t, -t, pi+t}.
  Deltoid = 2 lobes(theta 0,pi) x 2 branches; lobes are D2 images -> orbits
  pair across lobes. KEY: r_min (caustic curvature radius) is D2-invariant,
  so one-rep-per-orbit PRESERVES max(arc_r_min)=max_eta_max (feeds lobe
  admissions + interior-skip) IFF orbits are r_min-homogeneous (outer ~3.5
  vs lobe-edge ~0.28 are separate orbits). Coverage preserved by group
  closure (h(gi(theta)) in D2). Census Q1 auto-follows via len(tube_arcs).
  Serve-coverage sweep (fact 2) is THE acceptance test, engine-free via
  `_tube_theta_inframe`. max_tube_arcs fate: retire-vs-cap TBD (cap is
  dangerous: [:1] could drop outer r_min arc, shrink shell).

(empty — last consolidated by Dreamer on 2026-08-14)

## saddle_tube_fundamental_training v2 (post-rejection, 2026-08-14)
- v1 rejected: it enshrined band-wide `max_eta_max=f_max*max(arc_r_min)`
  (F081 starvation defect) as a load-bearing invariant to preserve. REVERSAL:
  that max IS the defect (outer-arc r_min 3.80 balloons shell to 1.519 >
  lobe reach 1.033 -> 0 lobe_int/lobe_ext/deltoid-ff tiles; 1236-draw gap).
- v2 = ONE merged Coder WP (Simplifier: avoid committed starved intermediate;
  same fn `_train_band_charts`). Part A = v1 orbit-partition trim 6->3 +
  retire max_tube_arcs knob (unchanged). Part B = per-arc shell fix.
- Part B LEAN FORM (Professor+Simplifier convergent): NOT a general nearest-
  segment union machine. Per-arc tube charts (L4879 `arc_r_min[idx]`) are
  ALREADY per-arc-correct; only two SHARED admission scalars wrongly used the
  band-wide max. Fix = feed those two the region-adjacent lobe-edge r_min =
  `f_max*min(arc_r_min)`: (a) `_saddle_lobe_admissions(eta_max=)` +
  corridor_half; (b) far-field `exclusion_rho`. KEEP (e) tube w-cap on max.
  (c) interior-skip L5374 + (d) wedge r_extent are parity==1-only (DEAD for
  saddle) -> untouched; astroid single-arc => min==max => byte-identical.
  Mirror BOTH parts in tiling_census.py.
- Sharper invariant (Professor): synthetic 2-arc witness point outside every
  arc's own shell but inside old max-shell must be ADMITTED. Nonzero-tiles +
  tube:-1 node drop from 61,740 = ACCEPTANCE (verification), NOT permanent
  test. F081 confirms fundamental set retains an outer-arc rep, so max does
  NOT self-heal — Part B required.

## tiling_census_node_budget build — triage round 2 (2026-08-14)
- INS-1-001 re-raised (non-blocking, partially addressed): Test Developer
  added the structural independence guard but the shipped-surface
  disclosure (docstring caveat + `ppgo_trim_modeled: False` field) is still
  missing. coder_fix again, same option (b) scope as round 1 — add the
  docstring sentence + output field only, no ppGO-trim modeling in the
  counting loops.

## tiling_census_node_budget build — triage (2026-08-14)
- INS-1-001 (census omits loop-level ppGO trim, so aggregate_call_count is a
  conservative upper bound not the real campaign's exact count): coder_fix
  via Inspector's lighter option (b) — disclose, don't replicate. Full exact
  mirroring of `_apply_ppgo_trim` (option a) would require re-implementing
  production's per-stratum/window DROP DECISION logic inside the census, a
  second copy of decision logic beyond mere tile-counting — same drift risk
  class as the F-defect the census itself exists to avoid duplicating.
  Divergence is one-directional/conservative (over-count only), so
  documenting + flagging is proportionate; full fidelity is not worth the
  scope/duplication risk for a pre-campaign gate.

## tiling_census_node_budget build (planned 2026-08-14)
- ENGINE-FREE tiling census + node-budget predictor (pre-campaign gate). NEW
  module `cogwheel/lensing/tiling_census.py` (NOT surrogate_census.py — that
  imports ChangRefsdalChannels at module level, breaking the engine-free
  guard). ONE Coder WP + thin CLI scripts/tiling_census.py. Reuse the six
  engine-free tiler fns (detect_caustic_structure, _tube_training_arcs,
  _farfield_tiles, _farfield_exterior_tiles, _wedge_interior_tiles,
  _lobe_interior_tiles, _lobe_exterior_tiles) as THIN CALLERS — a parallel
  reimplementation is the F-class defect. Reuse `_self_estimate` as a
  cross-check JSON field, don't rebuild call-count.
- Simplifier: new module (not census.py); Q3 kink-check modest/deferred null;
  ONE WP not split. Professor: engine=amplitude eval only (find_images etc
  allowed); Q1 astroid det4/train1, saddle det6/train min(6,1); Q2 redesign
  iff mis_alloc_ratio>2-3 OR tile angular span contains cusp ray; Q4 c3
  floor=(2e4*K)^(1/3); mock-to-raise engine-free pin is load-bearing.
