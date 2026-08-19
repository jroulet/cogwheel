# Inspector Short-Term Observations

## 2026-08-19 review: tiling_plan (7a step 2) — pass 4, PASS

Scope: uncommitted working-tree, branch claude-dev (Serena root = sibling
worktree /home/tejaswi/Work/cogwheel-claude-dev; run pytest with that cwd).
The 3 code files are UNTRACKED new files: cogwheel/lensing/tiling_plan.py,
scripts/tiling_plan.py, cogwheel/tests/test_lensing_tiling_plan.py. Only
agent_state + memories tracked-modified. SPEC.md / DATA_CONTRACTS.yaml
UNCHANGED (not in changed set). Suite: 39 passed / 23.66s.

Verdict: PASS (no NEW code defects). INS-3-001 CARRIED FORWARD (still open,
NOT a code defect): deliverable .claude/handoff/tiling_plan_and_cost_7a2.json
STILL absent (`ls` confirms). It is a DRIVER run step — run
`python scripts/tiling_plan.py` (engine-free, 10k default) — build_plan/run
correct and exercised end-to-end by the passing run() test. Do not block.

### Fresh verification this pass (did NOT trust prior-pass memory)
- Re-ran full suite green (test file mtime 00:19 later than module 00:11 —
  checked it wasn't a silent regression; it's not).
- Census header contract CONFIRMED by reading serve_route_census.py:
  header['w_band_edges']['w_ceiling_dd'] (line 1138) = mods.w_ceiling_dd
  (line 301) = float(_schwinger.W_CEILING_SCHWINGER). Same constant
  _resolve_dd_ceiling(None) lazily resolves to. build_plan reads it
  unconditionally — no KeyError risk (run() test would catch a missing key).
- CLI (scripts/tiling_plan.py): I/O-only, relative _DEFAULT_OUT, no secrets /
  absolute paths, writes JSON + prints escalation verdict. Clean.
- INS-1-001 (w clip both branches), INS-2-001 (_gamma_resolution step<=0
  raise, defensive-only) — unchanged, still correct.
- Cost currency SECONDS_PER_CALL=0.0903 vs tiling_census._SECONDS_PER_LABEL
  =0.09: DELIBERATE ~0.3% gap, reconciled in emitted cost_model note. Not DRY
  defect. _LABELS_PER_NODE single-sourced from tiling_census.

### Carry-forward (Librarian-owned, NOT a Coder defect in this diff)
- INS-5-001 lineage: region vocabulary lobe_exterior/lobe_interior/
  wedge_interior (used here as _INTERIOR_REGIONS/_FARFIELD_REGIONS literals)
  still absent from SPEC.md + DATA_CONTRACTS.yaml. Doc-staleness, Librarian.

### Reusable
- SERENA-ROOT-IS-SIBLING-WORKTREE: pytest cwd = cogwheel-claude-dev; use
  mcp__serena__execute_shell_command (Bash blocked for pytest by hook).
- UNTRACKED-FILE RE-REVIEW: files can be new/untracked (`?? ` in git status),
  so `git diff` shows nothing — re-verify by re-running the suite + re-reading,
  never by diff-stat. A later test-file mtime than the module warranted a
  fresh run (came back green).
