# Librarian Short-Term Observations (2026-07-27)

Scope: cleared the 15+ commit backlog (dc984c1..d47422a). Documented ONLY
the scope the driver named — bc27d39 (Build 8h-b4) as SPEC/changelog
content, plus a rolled-up agent-infra changelog line. Left untouched (by
design, not oversight):
- b14df4b (Build 8h-b2 ghost-kernel machinery) — Inspector/Professor PASS,
  looks production-real, but the driver's task explicitly scoped "the
  substantive PRODUCTION change to document is Build 8h-b4" and the open
  todo.d fragment `surrogate_component-representation-8hb.md` frames
  ghost-kernel/component-representation as still "In progress" (P1/P2
  probes, not landed levers). Don't backfill this on the next pass without
  re-confirming with the driver/todo state — it may be intentionally held
  until the wider 8h-b program closes.
- 497ecf1/75bf00f (WIP checkpoints, explicitly "PRE-GATE do not build on")
  and c28408b (WIP, "KNOWN REMAINING" gamma=1 bug later fixed for real in
  bc27d39) — all superseded by bc27d39, correctly not documented separately.

## New pattern: HARD FENCE + sync_derived_docs.py interaction
When another agent is concurrently editing cogwheel/tests/ (or any fenced
dir), running `scripts/sync_derived_docs.py` can print "N checks run, some
issues auto-fixed" and `git status` afterward will show files change in the
fenced dir — but check with `git diff --stat` on JUST your own edited
files (e.g. SPEC.md) before assuming the sync script touched the fenced
files. In this run, `git status` showed 4 modified cogwheel/tests/*.py
files after sync_derived_docs.py ran, but a targeted `git diff --stat
.claude/spec/SPEC.md` proved my own edit was a clean single-line diff — the
test-file changes were the concurrent agent's own uncommitted work
(content matched their exact feature: coverage-band docstrings/numbers for
the same exterior-admission build I was documenting), not something the
sync script wrote. Lesson: NEVER assume dirty state under a fenced path is
your own tool's side effect — diff-check your own targets specifically,
and leave the fenced files alone either way (do not revert, do not
"clean up").  sync_derived_docs.py's only real output here was 4 stale
DATA_CONTRACTS.yaml consumer-list warnings, all test-file-only callers of
`LensAmplificationSurrogate.load` — left off per the established
test-file-only-consumers convention (long-term memory already covers
this).

## Refined: SPEC_CHANGELOG.md version-ordering quirk (precise mechanism)
Previously flagged as "alphabetical filename order" — more precisely, it
is a TWO-TIER sort: `_render_versioned_changelog` sorts new (bump-only)
fragments by `(meta.get("date",""), filename)`. Fragments that carry NO
`date:` frontmatter field (the majority — e.g. all of Build 8b through
8h-b4, i.e. everything from `2026-07-20_build8b-levers.md` onward except
a handful) sort with key `("", filename)`, and `""` sorts BEFORE any real
date string. So EVERY undated fragment — regardless of how recent its
underlying build actually is — gets a LOWER derived version number than
EVERY dated fragment, and within each tier ordering is by filename only.
My new fragment `2026-07-27_build8hb4-exterior-admission.md` (undated,
matching sibling convention) landed at derived version `0.9.0`, sorted
behind `farfield-tiling-eps-gate` (`0.8.0`) and ahead of nothing higher —
even though it is chronologically the newest fragment in the directory,
because the ~10 OLDER fragments that happen to carry an explicit `date:`
field (Builds 3-7b era) all out-rank it. Content integrity was still
100%: 23 fragments in, 23 distinct entries out (0.1.0..0.19.0, no gaps/
dupes), final `spec_version` (0.19.0) genuinely is the max of all 23.
Confirmed this is a pure display/versioning artifact, not a content bug —
flag, don't fix (per precedent), and don't retrofit a `date:` field onto
new fragments just to "escape" the low tier — that would be inconsistent
with how every recent sibling fragment (8b through 8g) already looks.

## last_updated frontmatter never advances
`SPEC.md`'s `last_updated:` field is only rewritten by
`update_spec_version()` when the WINNING (highest-version) fragment has a
non-empty `date` meta field. Because the highest-version fragment is
currently `2026-07-20_surrogate-8a.md` (an OLD dated fragment, per the
two-tier quirk above) and not the most recently-added one, `last_updated`
has been stuck at `2026-07-20` across at least two Librarian passes now
(confirmed unchanged before/after this run) despite real content changes
on 2026-07-22 and 2026-07-27. Same "flag don't fix" call — this will keep
happening until either (a) a future fragment happens to out-rank
surrogate-8a's tier, or (b) the render script's sort key is fixed
upstream (out of Librarian scope; it's a `scripts/` change, not a
`.claude/spec/` doc).

## Skipped by design
- todo.d/completed.d: left alone — the crew prompt's deliverable list did
  not include closing out `surrogate_component-representation-8hb.md` or
  `surrogate_farfield-envelope-v2.md`, and neither fragment's listed scope
  is fully covered by bc27d39 alone (component-representation especially
  is explicitly a bigger, still-open program).
- DATA_CONTRACTS.yaml: untouched — commit message and code comments both
  say chart npz schema is unchanged by Build 8h-b4; sync_derived_docs.py's
  only complaint was the known test-file-only consumer gap (not a real
  contract change).
- docs/source/: grepped for far-field/exterior-admission/caustic_reach
  terms, zero hits — the Sphinx narrative pages sit above this
  implementation-detail layer (consistent with the standing "don't
  manufacture a perf/impl blurb" note), nothing to sync there.
