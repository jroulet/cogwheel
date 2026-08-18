# Librarian Short-Term Observations

## 2026-08-18 (post-commit sync, --post-commit 733b7ef, backlog 1a12559..733b7ef)

Scope: the ~13-commit lensing backlog (beat-free tube, c3 band-split +
census re-gate, low-w diffractive rungs) plus trailing housekeeping.
`sync_derived_docs.py` reported "5 checks run, all OK" (no diff).

WHAT WAS ALREADY CURRENT (verified, not duplicated):
- SPEC.md already carries the full LOW-W DIFFRACTIVE RUNGS paragraph
  (build `low_w_diffractive_rung`, `_diffractive.py` added to the
  Microlensing-engine module list) and spec_version was already bumped
  to 0.48.0 with a matching spec_changelog.d fragment -- this landed
  IN-BAND with the build commit (733b7ef), not by a prior librarian pass.
- COMPLETED.md/completed.d already has both
  `2026-08-17_saddle_c3_band_split_serving.md` and
  `2026-08-18_low_w_diffractive_analytic_rung.md` with the full headline
  numbers.
- `docs/source/api.rst` needs no manual entry for `_diffractive.py` --
  confirmed (again) it uses bare `:recursive:` autosummary over
  `cogwheel`, same as `_born.py`/`_schwinger.py` (no explicit listing).
  `docs/source/overview.rst`'s microlensing blurb is architecture-level
  prose with no stale "not yet" claim -- nothing to propagate.

WHAT WAS ACTUALLY STALE (fixed this pass):
1. ROOT `changelog.d/` had NO user-facing entries for the c3 band-split
   + census re-gate (commit 6958f0c/b097ce1) or the low-w diffractive
   rungs (733b7ef) -- the most recent entries were 2026-08-17 (census
   corrected demand, tube beat-free). Added
   `changelog.d/2026-08-17_saddle_c3_band_split_serving.md` and
   `changelog.d/2026-08-18_low_w_diffractive_rungs.md`, quoting the
   measured route percentages from the completed.d records.
2. SPEC.md's "ENGINE-FREE SERVE-ROUTE DEMAND CENSUS" paragraph (the
   Microlensing-engine table row) still described `0.32% saddle_c3` as
   "unreachable... under the physical 20 Hz prior" and pointed at
   `todo.d/lensing_saddle_c3_band_split_serving` -- a PLAIN-TEXT
   (non-`[[...]]`) reference to a fragment that had already been
   completed and moved to `completed.d/2026-08-17_saddle_c3_band_split_
   serving.md` (deleted from todo.d in commit b097ce1). Same family as
   the "plain-text fragment-name references are invisible to the
   dangling-link checker" pattern already in long-term memory, but this
   is the FIRST instance caught where the reference lives inside SPEC.md
   itself (not a fragment) -- worth generalizing that rule to "any doc
   surface", not just fragments. Fixed in place (FIXED-note style
   matching the adjacent F070 paragraph), with a
   `spec_changelog.d/2026-08-18_c3_band_split_pointer_correction.md`
   patch-bump fragment.
3. `render_fragments.py`'s dangling-`[[link]]` checker flagged TWO real
   hits inside the in-scope backlog (of 7 total; the other 5 are the
   long-standing FINDINGS-F0xx-not-a-target gap, already tracked,
   untouched): `todo.d/lensing_born_farfield_completion.md` linked
   `[[lensing_deltoid_farfield_coordinate_redesign]]` (target HARD-
   DELETED outright in 6a1e33a, never archived to completed.d -- dropped
   the brackets to plain backticked text per the checker's own "or drop
   the link" suggestion, since there is no valid resolution target) and
   `[[lensing_low_w_diffractive_analytic_rung]]` (target's real stem is
   date-prefixed `2026-08-18_low_w_diffractive_analytic_rung` once moved
   to completed.d -- the checker matches on EXACT stem, so a same-day
   `[[bare_slug]]` backlink written before completion silently dangles
   the moment the fragment is archived; repointed).

FRAGILE CROSS-REFERENCES noted for future passes:
- Any `[[bare_slug]]` link written INTO a todo.d fragment pointing at
  another OPEN todo.d fragment will dangle the instant that target
  fragment completes and gets date-prefixed into completed.d -- the
  link must be repointed at completion time, not just `depends_on`
  frontmatter (existing rule) but body-prose `[[...]]` links too.
- SPEC.md's own historical-measurement sentences (not just fragment
  prose) can embed a plain-text pointer to a todo.d fragment path --
  grep SPEC.md for literal `todo.d/` substrings on every sync pass
  touching a lensing row, not just `[[...]]` syntax.

Not touched (out of scope / not mine): `.claude/handoff/
born_farfield_completion.md` (untracked, written live by a concurrent
build-brief agent mid-session -- left alone). `.claude/tidy_advisory.json`
picked up the known `sync_derived_docs.py`/`render_fragments.py` side-
effect diff -- reverted via `git checkout --`, not committed (per
standing rule). The 5 pre-existing FINDINGS-F0xx dangling links are
unrelated to this backlog and already tracked by a prior escalation --
left alone.
