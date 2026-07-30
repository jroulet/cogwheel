---
date: 2026-07-30
section: SDK
---

### SDK guard: HEAD-relative test oracles (F043)

`.claude/hooks/check_head_relative_tests.py`, wired into `pre-commit` beside
the gated-test-drift check. Blocks a commit that ADDS a test reconstructing
pre-change code via `git show HEAD:` — the antipattern that broke three tests
across two builds on 2026-07-29/30.

The hole it closes is structural: the tree-wide gate runs BEFORE the commit,
so such a test reads the OLD HEAD and PASSES; it goes red in the NEXT build's
gate, attributed to a build that never touched it. Reviewer vigilance cannot
close a hole that is green in its own gate.

Scans only ADDED lines of staged test files (a pre-existing oracle does not
re-fire on unrelated edits), reports file:line and the enclosing def/class,
and offers the same ack idiom as the drift gate:
`GATED_HEAD_ORACLE_ACK="Class,Other.test_method"`. Verified on the real F043
pattern (caught), with ack (passes), on an ordinary subprocess-using test (no
false positive), and with nothing staged (clean).

Remedy text points at the three fixes: retire it (a within-build transition
check is done once its transition commits), freeze a golden-value table, or
pin an explicit commit SHA.
