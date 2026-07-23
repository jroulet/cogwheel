# Inspector Short-Term Observations

## 2026-07-23 — Build 8h-b2 RE-REVIEW #2 (ghost/complex-saddle kernel)

Scope: uncommitted worktree changes. Production file:
`cogwheel/lensing/chang_refsdal/geometry.py` (+451 lines, purely
additive block appended after `nearest_caustic_point`, L1489+) + new
untracked test `cogwheel/tests/test_lensing_ghost.py`. Diff is
BYTE-IDENTICAL to the previous re-review (still +451, same content).
Rest of diff = agent_state/handoff/memory bookkeeping (ignored).

### Verdict: PASS (1 trivial carry-forward, INS-1-001 — NOT resolved)

### Re-verified this pass (byte-identical diff, still re-ran everything)
- Import probe clean; all six symbols exported
  (ghost_kernel, _ghost_kernel, _ghost_candidates, _ghost_delay,
  GhostContribution, GhostDomainError).
- test_lensing_ghost.py: 36 passed, 1 xfailed (literal on-axis contract,
  documented UNREACHABLE @expectedFailure). 16.8s.
- Real-path regression: test_lensing_geometry + test_lensing_saddle_geometry
  = 41 passed. No real-image-path regression.
- Consumer scan (search_for_pattern over cogwheel code files):
  ghost_kernel/GhostContribution/GhostDomainError referenced ONLY in
  geometry.py + test_lensing_ghost.py. No production consumer yet — matches
  spec (additive primitive for 8h-b3). In-memory NamedTuple, no serialized
  artifact ⇒ no DATA_CONTRACTS entry required. No spec-code divergence.
- Physics identities (continuation faithfulness, bilinear Hessian,
  det==1 saddle frame, branch-pinned amplitude absorbing Morse index-1
  phase) previously re-derived by hand and unchanged — see prior entry in
  git history; content did not change since.

### Finding (trivial, non-blocking, carry-forward — STILL PRESENT)
- INS-1-001: `ghost_kernel` computes `_ghost_delay` twice for the
  selected candidate (once in the all-candidates `delays` comprehension
  for the Im-tau argmax at L1930, again inside `_ghost_kernel` at L1737).
  Cheap (few mults + 1 log, ≤2 candidates), negligible; not an
  over-refusal risk (conjugate pair shares Re(z)). Pure micro-DRY nit.
  Optional fix: pass pre-computed tau_c into `_ghost_kernel`.

### Carry-forward
- 8h-b3 will consume `ghost_kernel` and add the w·Im tau_c gate; watch
  for the on-axis xfail flipping to xpass then (would need a plan update).
