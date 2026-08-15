# Architect Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-14)

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
