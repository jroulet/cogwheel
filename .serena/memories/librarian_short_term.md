## 2026-07-29 post-commit sync (backlog through d61b893)

Scope: 6 queued commits in .claude/sync_issues.json (3fe5b96, d0aadf7,
f0cb9dd, a9a1374, 0eddcad, d61b893) covering the Born saddle carrier
attribution split, F027 rationale correction, F028/F029 (fold-arm wrong
values + authoritative one-home branch gate in operator.py), and two
gates.py fixes. Outcome: **no doc-surface fixes needed** — everything
was already synced by the driver/prior builds. Verified, didn't just
trust the claim:

- SPEC.md's Build 8e "uniform arms (certified)" text: confirmed already
  rewritten in 0eddcad's own commit (grep "CORRECTED by the
  authoritative-gate build" — present, matches F028/F029 and the
  select_branch one-home predicate).
- FINDINGS.md F027: already stated the correct (parity-independent)
  rationale *before* f0cb9dd; that commit only synced the channels.py
  code comment to match — confirmed the current docstring
  (channels.py ~L1456-1480) now cites F027 correctly. No FINDINGS
  change was ever needed for f0cb9dd despite the commit touching no
  doc file.
- d0aadf7's code diff (_born.py/channels.py/surrogate_census.py/
  test_lensing_born.py) is NOT new content — commit message itself
  (in f0cb9dd) explains a driver `git add -A` swept the live Born-carrier
  build's files into this spec commit while the build was still running;
  the real doc sync for that feature already happened in 416558d
  ("docs: update documentation after build", predates d0aadf7 by commit
  time despite being for the same feature). Cross-checked: confirmed via
  `git show --stat 416558d` that it already touched SPEC.md/TODO.md/
  CHANGELOG.md for the Born carrier.
- COVERAGE_DESIGN.md (new file, d0aadf7): a standalone spec analysis doc,
  same tier as FINDINGS.md/DATA_CONTRACTS.yaml but NOT part of the
  Knowledge Anchoring list and NOT cross-linked from SPEC.md — confirmed
  this matches existing convention (SPEC.md has no "related docs" section
  linking sibling spec/*.md files at all, so its absence there is not
  staleness).
- todo.d/lensing_fold_arm_serves_wrong_values.md: already edited to mark
  "defect 1 of 2 (admission routing) CLOSED by the authoritative-gate
  build", i.e. already reflects 0eddcad's fix — not stale.
- scripts/sync_derived_docs.py clean: only advisory noise (4 test-only
  consumers of lens_amplification_surrogate not in DATA_CONTRACTS.yaml —
  pre-existing, Build 8a surrogate, unrelated to this window, and
  test-only consumers are conventionally left off per prior sessions).
- docs/source/**: zero hits for uniform-arm/select_branch/Pearcey/8e
  terms — confirms overview.rst stays architecture-level and this
  implementation-detail window needed no Sphinx changes (no docs/source
  file touched -> no rebuild required either).

SURPRISE / pattern worth flagging forward: found an UNRELATED staged
(uncommitted) change sitting in the index when I started —
`.claude/spec/TODO.md` (render), a new fragment
`.claude/spec/todo.d/tests_consolidate_duplicate_routing_pins.md`, and
an `AGENTS.md`/`CLAUDE.md` addition ("Assert VALUES, not code paths").
Confirmed via reading scripts/sync_derived_docs.py source that none of
its check_* functions write files — so this was NOT produced by my
`sync_derived_docs.py` run, it predates it. This content is outside the
6-commit range in sync_issues.json (not part of any commit yet). Left
it untouched per the standing rule (don't touch other agents'
concurrent in-flight uncommitted work) and made sure my own commit used
an explicit pathspec (`git commit -- .claude/sync_issues.json ...`) so
it couldn't sweep this unrelated staged content in. Next Librarian
session: if this is STILL just staged and uncommitted, it's probably
an orphaned Tidier/driver artifact worth asking the owner about rather
than silently absorbing into an unrelated commit.

No fixes -> no code/doc edits this session; only sync_issues.json deleted.
