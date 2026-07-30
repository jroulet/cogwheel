---
section: Backlog
---

- **SDK GUARD: catch git-HEAD-relative tests at the build that introduces them**
  `[housekeeping]` — F043. A test that reconstructs the pre-change
  implementation via `git show HEAD:<path>` (to compare worktree-vs-HEAD) is
  valid ONLY while HEAD is still the pre-change commit — i.e. during its own
  build, before that build commits. The instant it commits, HEAD becomes the
  new version and the oracle either compares a version to itself or fails to
  reconstruct because the change deleted a symbol it needs. This BROKE THREE
  TESTS across two builds today (F043), and — critically — the tree gate
  cannot catch it: the gate runs BEFORE the commit, so the test sees the OLD
  HEAD and PASSES; it goes red only in the NEXT build's gate. No amount of
  reviewer/agent vigilance closes a hole that is green in its own gate.

  ## The fix (targeted, matches the existing drift-gate shape)

  Add a mechanical check — sibling to `.claude/hooks/check_gated_test_drift.py`
  (which already catches a test-staleness class prompts miss). Scan staged (or
  all) test files for `git show HEAD:` / `subprocess...['show', 'HEAD:...']` /
  `HEAD:{...}` reconstruction. On a match, BLOCK-WITH-ACK (like the drift
  gate):
    "Test <X> reconstructs the pre-change code from `git show HEAD` (F043).
     This passes THIS gate and breaks the NEXT build once HEAD advances.
     Retire it (a within-build transition check is done once its transition
     commits) or freeze a golden-value table. If it is a deliberate
     within-build check being removed before merge, ack:
       GATED_HEAD_ORACLE_ACK=Class.test_method"
  Advisory-then-block, same threshold idiom as the librarian-backlog guard.

  ## The deeper alternative (more robust, more invasive — decide at scope)

  Make the tree gate run against the POST-COMMIT state: temp-commit the staged
  changes, point HEAD at them, run the suite, reset. Then ANY HEAD-relative
  breakage (not just the grep pattern) fails in the SAME build that introduces
  it. Catches the whole class generically, but changes gate semantics broadly
  and adds a stash/reset dance — riskier. The targeted check is the 80/20;
  this is the principled ceiling. Prefer the targeted check first; consider
  this only if HEAD-relative breakage recurs in a form the grep misses.

  ## Acceptance
  - A test added with `git show HEAD:` reconstruction FAILS the pre-commit
    check in the build that adds it (not two builds later), with the F043
    message and the ack escape hatch.
  - The check is advisory below a small count and blocking above it, or
    block-with-ack; it does NOT fire on non-test files or on legitimate
    acked within-build transition checks.
  - Under `.claude/hooks/` or `.claude/sdk/` (agent-only infra, excluded from
    sync_to_main).
