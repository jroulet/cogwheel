## 2026-07-29 post-commit sync (backlog through d61b893) — final

Scope: 6 queued commits in .claude/sync_issues.json (3fe5b96, d0aadf7,
f0cb9dd, a9a1374, 0eddcad, d61b893). Outcome: no doc-surface fixes
needed — everything already synced by the driver/prior builds. Verified
each claim rather than trusting it:

- SPEC.md's Build 8e text: confirmed already rewritten in 0eddcad
  ("CORRECTED by the authoritative-gate build", one-home `select_branch`
  predicate documented).
- FINDINGS.md F027: already stated the parity-independent rationale
  BEFORE f0cb9dd; that commit only synced the channels.py code comment
  to match (confirmed current docstring ~L1456-1480 cites F027
  correctly). No FINDINGS edit was ever needed despite f0cb9dd touching
  no doc file.
- d0aadf7's code diff (_born.py/channels.py/surrogate_census.py/
  test_lensing_born.py) is not new content — a driver `git add -A`
  swept the live Born-carrier build's files into this spec commit while
  it was still running (explained in f0cb9dd's own message); the real
  doc sync for that feature landed earlier in 416558d
  ("docs: update documentation after build" — confirmed via `git show
  --stat` it already touched SPEC.md/TODO.md/CHANGELOG.md).
- COVERAGE_DESIGN.md (new file, d0aadf7): standalone spec analysis doc,
  not in the Knowledge Anchoring list, not cross-linked from SPEC.md —
  confirmed SPEC.md has no "related docs" section linking sibling
  spec/*.md at all, so the absence is not staleness.
- todo.d/lensing_fold_arm_serves_wrong_values.md: already marks
  "defect 1 of 2 (admission routing) CLOSED by the authoritative-gate
  build" — reflects 0eddcad, not stale.
- sync_derived_docs.py clean except pre-existing advisory noise (4
  test-only consumers of lens_amplification_surrogate not in
  DATA_CONTRACTS.yaml — Build 8a surrogate, unrelated to this window;
  test-only consumers conventionally stay off per prior sessions).
- docs/source/**: zero hits for uniform-arm/select_branch/Pearcey/8e —
  overview.rst stays architecture-level; no docs/source file touched
  this window so no Sphinx rebuild needed.

GIT MECHANICS GOTCHA (new, worth remembering): found unrelated
already-staged content in the index at session start — TODO.md render,
new fragment .claude/spec/todo.d/tests_consolidate_duplicate_routing_pins.md,
and an AGENTS.md/CLAUDE.md addition ("Assert VALUES, not code paths").
Confirmed scripts/sync_derived_docs.py's check_* functions are all
read-only (no file writes), so this predated my session, not caused by
running it. Tried to protect my no-op sync commit by pathspec-
restricting to just the memory file: `git commit -m "..." --
.serena/memories/librarian_short_term.md`. **This did NOT work as
"only this path" the way I expected** — the repo's `.claude/hooks/
pre-commit` hook unconditionally runs `git add ... .claude/spec/TODO.md
...` (render_fragments step) and, on any sync_derived_docs.py
auto-fix, `git add docs/ .claude/spec/ README.md` (step 8) — a
directory-glob `git add` that swallows ANY staged/modified file under
`.claude/spec/`, including one that had nothing to do with my pathspec.
The resulting commit picked up TODO.md + the new fragment (both under
`.claude/spec/`) but correctly left AGENTS.md out (outside that glob).
Net effect: `git commit -- <one file>` is NOT a hard restriction in
this repo when the pre-commit hook's own `git add <dir>` calls run
first — pathspec only guarantees your named file is included, not that
OTHER already-staged files under hook-globbed directories are excluded.
Lesson for next time: before a "no fixes, just the memory write" no-op
commit, run `git status --short` immediately before AND after
`git commit --dry-run` (or just accept the hook may staple in
`.claude/spec/**` content and check `git show --stat HEAD` right after
committing) rather than assuming pathspec = isolation. I closed the
loop this session by committing the orphaned AGENTS.md counterpart
in a small follow-up (same fragment's lesson, pre-existing content,
nothing invented) rather than leaving it half-landed — see commit
b4c0777.

No SPEC/FINDINGS/DATA_CONTRACTS/docs-source edits were needed or made
this session. Both commits (53d3d36, b4c0777) only touched TODO.md,
the pre-existing todo.d fragment, AGENTS.md, and the memory file —
none of it newly authored by me.
