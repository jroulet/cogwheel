# Inspector Short-Term Observations

## 2026-07-18 — Build 3d re-review (RE-DISPATCH, SAME uncommitted diff) — VERDICT ISSUES

Scope: uncommitted diff on branch claude-dev, worktree
/home/tejaswi/Work/cogwheel-claude-dev. HEAD = 6a62eff (NO new commits since
the INS-2 review). Code files in diff: cogwheel/lensing/likelihood.py,
cogwheel/tests/test_lensing_fast_path.py. SPEC.md NOT in the diff.

### Headline: the working tree is BYTE-IDENTICAL to what INS-2 reviewed.
No Coder/Test-Dev fix landed. Every INS-2 finding re-confirmed by direct
re-measurement this session; NONE resolved.

Re-verified facts (fresh, this session):
1. FATAL collection error STILL present (INS-2-001 -> INS-3-001):
   `pytest --co cogwheel/tests/test_lensing_fast_path.py` =>
   `ImportError: cannot import name '_MAX_SEGMENT_NODES' from
   cogwheel.lensing.likelihood` at test line 122-123. likelihood.py defines
   `_LINEAR_NODES_PER_BEAT/_MIN_LINEAR_NODES/_MAX_LINEAR_NODES` +
   `_DEFAULT_KERNEL_NODES=32`; the deleted `_MAX_SEGMENT_NODES/
   _MIN_SEGMENT_NODES/_SMOOTH_BAND_NODES` are still imported. 0 tests
   collected, 1 collection error. Whole suite un-runnable.
2. Stale segmentation-era suite (INS-2-002 -> INS-3-002): file still has
   `_segmented_reconstruct`, "SEGMENTED grid" docstring, expectedFailure
   masks, and inverted node-reduction gates (`_DEFAULT_KERNEL_NODES/ncoarse
   >= 4`). Shipped code is a SINGLE global not-a-knot spline
   (boundaries=[w_min,w_max]). Even if the import were patched, the three
   accuracy @expectedFailure tests now PASS their inner assertion (=>
   unexpected-success failure) and the node-reduction gates fail (grid GREW).
3. Objective unmet & REGRESSED (INS-2-003 -> INS-3-003), re-measured:
   node grid crown 62, near-cusp 60, well-sep 91, near-fold 83, sheared 58
   (100/n = 1.1-1.7x — lever A INVERTED, grid ~2x the 32-node base). Warm
   crown lnlike best-of-7 = 27.4 ms (threads pinned) — vs 10 ms hard target,
   15 ms ceiling, and the prior build's 18.8 ms. No `_surrogate.py` (lever B),
   no DATA_CONTRACTS artifact.
4. SPEC.md divergence (INS-2-004 -> INS-3-004): `git diff HEAD .claude/spec/
   SPEC.md` is EMPTY. SPEC still says `_DEFAULT_KERNEL_NODES = 100`, old union
   scheme, "warm lnlike ~0.3 s". Librarian owns the edit; flagged as a
   spec-invariant divergence.

Code correctness note (unchanged from INS-2): the underlying accuracy is
FINE in the shipped global-spline code (INS-2 measured dense 1.4e-5..1.9e-4
< 1e-3, off-grid < 8.2e-5, near-fold RB 0.044 << 1.5). The problem is purely
(a) the suite can't run to prove it, and (b) speed regressed while node count
grew — the build's entire reason for existing (10 ms) is unmet.

### Pattern reinforced
- Re-dispatch with an unchanged tree: always re-run `pytest --co` + a cheap
  node-count/timing probe rather than trusting that "a fix must have landed."
  Here the diff was literally identical; confirming byte-identity via HEAD +
  empty SPEC diff + live ImportError is faster than re-deriving.
- The Coder's own `_coarse_w_node_grid` docstring now CONCEDES the design
  can't reach a few-node regime (~50-60 nodes needed) and that removing the
  in-kernel oscillation upstream (a surrogate / component reconstruction) is
  "out of scope for this interpolation layer" — i.e. lever A cannot hit 10 ms
  and lever B was declined. That is the escalation the brief's step-rule asks
  for, but it must be reported as UNMET-objective, not shipped green.

### Open issues carried forward
INS-3-001..004 all open (== INS-2-001..004). INS-1-001/003 subsumed.
