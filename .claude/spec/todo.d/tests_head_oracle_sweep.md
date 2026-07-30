---
section: Backlog
---

- **RETIRE THE REMAINING `git show HEAD` TEST ORACLES — 4 files, 10 call
  sites** `[housekeeping]` — F043 recorded the antipattern, F045 recorded that
  it spreads by REUSE and that leaving a retired test as a skipped ghost is
  what keeps the helper alive. `test_lensing_surrogate_training.py` is now
  clean (helpers and all callers deleted 2026-07-30). Four files still carry
  one, and each will detonate in whichever future build first touches the
  module it pins:

  | file | helper | call sites | what it pulls from HEAD |
  |---|---|---|---|
  | `test_lensing_caustic_cusps.py` | `_head_module_source` | 331, 363 | `surrogate_training.py` source text |
  | `test_lensing_farfield_envelope.py` | `_head_git_default` | 1546 | a `field: float = <v>` dataclass DEFAULT, by regex |
  | `test_lensing_ghost.py` | `_head_geometry` | 1346, 1379, 1404, 1425, 1603 | `geometry.py` imported side-by-side (numba, real temp file) |
  | `test_lensing_levers.py` | `_head_source` | 149, 181 | `geometry.py` / `likelihood.py` source text |

  CLASSIFY BEFORE DELETING — these are not all the same construct, and one is
  explicitly PROTECTED:
  - A **cross-version NUMERICAL comparison** (current vs HEAD values) is the
    F043 antipattern outright. `_head_git_default` is the clearest case: it
    reads a bar's previous default and compares, so the first build that
    changes that default and commits makes the assertion compare a value to
    itself. Retire, or freeze the old default as a literal.
  - A **source-TEXT scan** of HEAD (`_head_module_source`, `_head_source`) is
    the same rot with a longer fuse: it asserts something about the previous
    revision's text, which stops being the previous revision the moment the
    build commits. If the claim is about the CURRENT tree ("the oracle does not
    import the implementation"), it should read the WORKTREE, not HEAD — that
    version never rots and is strictly what the test means. Check each of the
    four sites: some may be worktree claims written against HEAD by accident.
  - `test_lensing_ghost.py`'s FD oracle **is on the protected list** in
    [[lensing_analytic_derivatives]] ("an AST guard whose PURPOSE is to prove
    the oracle is independent of the implementation"). Protecting the
    INDEPENDENCE ARGUMENT does not protect the HEAD fetch: verify whether those
    five sites need a past revision at all, or whether the worktree module
    serves the same purpose. Do not delete a genuine independence guard —
    re-point it.

  ACCEPTANCE: no test in `cogwheel/tests/` reconstructs code or constants from
  `git show HEAD`; the pre-commit guard
  (`.claude/hooks/check_head_relative_tests.py`) reports clean with an empty
  ack; every retired claim is either re-expressed against the worktree, frozen
  as a golden literal, or deleted with a one-line reason. NO test is left as a
  `@unittest.skip` shell — that is the specific mistake F045 records, since a
  skipped body still keeps its helper alive for the next build to reuse.
