# Inspector Short-Term Observations

## 2026-08-17 (serve_route_census, WP1 final re-review — PASS, INS-1-001 RESOLVED)

Re-reviewed engine-free serve-route DEMAND census after Coder applied the
dead-binding fix. In-scope diff: cogwheel/lensing/serve_route_census.py
(2 deletions only), scripts/serve_route_census.py (staged, unchanged), untracked
test cogwheel/tests/test_lensing_serve_route_census.py. Rest = agent_state/memory
files (out of scope).

VERDICT: PASS.

- INS-1-001 (trivial, RESOLVED): `git diff` shows exactly the two removals —
  `cusp_amplification: Any` field decl (was ~line 178) and the
  `cusp_amplification=op._pearcey_cusp.cusp_amplification` assignment in
  `_load_production_modules` (was ~line 211). `search_for_pattern
  cusp_amplification` on the module now returns {} (zero refs). `_resolve_arm_kind`
  pearcey-by-elimination logic intact and correct: probes fold -> ghost_ppgo ->
  returns 'pearcey' by elimination (only reached after `_uniform_arm_value`
  returned non-None, so the winning arm is known not to raise). No behavior
  change; pure dead-binding cleanup.

- Test suite GREEN: 28 passed in 23.55s (fast-tier compliant). All prior-review
  invariants (engine-free booby-traps, D2 sign-flip invariance, saddle refusal,
  demand zero-surrogate assertion, split_gauge='caustic_rho', no SPEC/contract
  divergence — serve_route is a diagnostic tool, not a registered pipeline
  artifact) remain as verified in the prior pass; only the dead binding changed.

No open findings carried forward.
