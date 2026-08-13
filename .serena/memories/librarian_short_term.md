Post-commit sync 2026-08-13, range a281d23..252e7c2 (19 queued commits, HEAD 252e7c2).

SCOPE: large lensing backlog (F066-F073 filed, F071 retracted-then-refiled).
Most commits self-synced their own spec fragments (crew already wrote
FINDINGS/TODO fragments in the same commit as the code fix) — Librarian's
real work was the RESIDUE: staleness introduced by a LATER commit that
fixed something an EARLIER commit's fragment had described as open/proposed.

FIXED:
- SPEC.md line 53 (the giant Microlensing-engine row): the F070 low-w-clamp
  paragraph said "the guard is free at the serve site... and should mirror
  the w_trust split" (proposal language) even though 8dfb8ca shipped exactly
  that guard 4 minutes after the commit that wrote the SPEC sentence. Same
  staleness in todo.d/lensing_slow_tier_fixtures_left_their_served_domains.md
  OPEN 5. Both corrected to say FIXED (serve side), with the still-open
  training-side gap named and pointed at its own fragment. New pattern for
  the "SPEC STATUS SENTENCES STALE SILENTLY" family: this time the staleness
  window was FOUR MINUTES between two commits in the same backlog, not weeks.
- OPEN 4 in the same fixture-rot fragment named
  known_failures.txt's "single entry" — b5a09e3 (same day, later) emptied
  the file by FIXING that test, not just deselecting it. Marked CLOSED with
  the fix attribution.
- Retired two todo.d fragments to completed.d whose acceptance criteria were
  met by same-day-later commits but never marked done:
  lensing_cusp_tie_guard_watches_the_wrong_side.md (fixed by 252e7c2 — verified
  by reading the actual _merging_fold_pair body, tie_count now maxes both
  sides of the selected pair, matches FINDINGS F072's fix description
  exactly) and lensing_schwinger_certified_band_is_narrower_than_150.md
  (both actions — ORACLE_MAXDEGREE 5->6, _MP_PANEL_ORDER 32->40 — verified
  live in code, matches 5451ab9). Both followed the established in-repo
  retirement convention: append a dated RESOLVED section preserving the
  original text, then move file with an inline mv (no rename tool needed,
  used create_text_file + rm since Serena has no file-move primitive).

NOT FIXED, reported instead: `check_wiki_links()` in
scripts/render_fragments.py (added 42328c9) ONLY resolves `[[stem]]` against
todo.d/completed.d filenames. Today's fragments introduced a NEW convention,
`[[FINDINGS F0xx]]`, that this checker was never taught — it reports these
as dangling on every render (8 of them, stable count across my edits,
confirmed by manually verifying every F066-F073 section exists in
FINDINGS.md and every referenced todo.d/completed.d stem exists as a file).
This is non-blocking (pre-commit calls render_fragments.py WITHOUT --check,
so exit code stays 0) but is a genuine tooling gap: either check_wiki_links
needs to also parse FINDINGS.md's `## F0xx` headers as valid targets, or the
convention should revert to unbracketed "F0xx" prose (the pre-42328c9
style, still used elsewhere e.g. "F023, see [[stem]]"). This needs a build
decision (touches scripts/render_fragments.py, a code file, out of
Librarian's edit scope) — flag for the driver, don't invent a fix.

SKIPPED per triage: all test-only commits (73ec561, cc082b1, e083546's test
files, f0e08cb, b5a09e3, a9e2fa9's test churn) — no doc surface describes
test internals. a281d23 tidy pass — AST-verified no-op by its own commit
message, mechanical style only. 0de8379 — already a librarian-style
docstring fix (Fields->Attributes), nothing left to do. Several commits
(c74f514, 97179fe, 71a5051, a290814, db47bc2, 24b79c9, 8453795, 926483e)
bundled their own spec fragment updates in the same commit — verified
self-consistent, no propagation gap found beyond the two items above.

SURPRISE: a concurrent agent had uncommitted work in-flight on
cogwheel/lensing/chang_refsdal/geometry.py, cogwheel/tests/test_lensing_ghost.py
and .claude/spec/CONSUMER_GRAPH.json throughout this whole session (162/48/99
line diffs) — not mine, not part of the 19-commit backlog, never touched.
Staged and committed ONLY my own files by explicit path list, never `git add
-A`. Reconfirms the "don't touch concurrent in-flight changes" rule from
last time, now with a concrete instance of what it looks like from inside a
post-commit sync (unrelated dirty files sitting alongside the ones you
actually need to touch).

docs/source/overview.rst checked against the Schwinger w<=60 ceiling and
chang_refsdal narrative — already consistent, no edit. No new/deleted
cogwheel/ modules in this backlog, so api.rst/SPEC module-list unaffected.
