## Last session: 2026-08-06 post-commit sync (--post-commit 4962a26)

Scope: 10-commit backlog 1840958..4962a26 (mpmath-band F061 test demotions,
conftest.py fast-tier ceiling, SDK known_failures gate, six lensing todo.d
measurement fragments culminating in the interior-delay-coordinate ODE
validation). Only 4 test files + new conftest.py touched `cogwheel/`; zero
`cogwheel/lensing/**` changes, zero docs/source/** references (confirmed via
targeted grep for conftest|f_schwinger|known_failures and InteriorWedgeChart|
astroid interior|wedge — all empty) — Sphinx rebuild correctly skipped.

### What was stale
1. DATA_CONTRACTS.yaml `lens_amplification_surrogate`: the FarFieldChart
   sentence's parenthetical "the astroid interior is InteriorWedgeChart's
   domain" stated settled coverage. `todo.d/lensing_wedge_charts_fail_the_
   eps_bar.md` (measured same day, first completed production run on the
   wedge path) found 0/12 wedge interior charts pass the 5e-2 eps bar
   (median 5.38e-1 vs retired `ffin`'s 106/106 at 3.42e-4) — interior is
   CURRENTLY UNSERVED, falls to the serving ladder. Softened to "nominal
   domain" + gate-failure note, WITHOUT reverting to describe `ffin` (the
   recommended revert has not landed — driver was explicit not to
   pre-empt this). contracts_changelog.d/2026-08-06_wedge-interior-
   unserved.md (bump patch). This is the second contract sentence about
   this same field/region corrected in three days (2026-08-04 factual
   flip: interior moved to wedge; 2026-08-06 coverage flip: wedge doesn't
   actually work) — the field is unusually fast-moving, worth extra
   scrutiny on future passes until the interior story stabilizes.
2. Nothing else. SPEC.md's conftest.py paragraph (added upstream in this
   same backlog with its own spec_changelog fragment, bump patch) was
   verified word-for-word against the actual `pytest_configure` body —
   already accurate (900s ceiling, all 4 slow-tier env vars, --timeout
   override, no-op-without-plugin) — no edit needed, a genuine no-op not
   a missed check.

### Cross-reference verification (the driver's specific ask)
All four named links resolve: lensing_wedge_charts_fail_the_eps_bar.md,
lensing_farfield_sd_coordinate_degenerates.md,
lensing_interior_delay_coordinate_validated.md,
lensing_farfield_name_spans_three_regimes.md all exist as files and their
`[[...]]` backlinks (checked via full-repo grep, not per-file) target real
fragments. F061 renders correctly in FINDINGS.md; all F061 cross-references
in cogwheel/tests/*.py resolve to it (grep confirmed 5 test files + conftest
all point at the same finding).

### Surprise: pre-existing dangling link OUTSIDE this session's range
`todo.d/lensing_caustic_relative_coordinates.md:298` still links
`[[lensing_born_b1_derivation]]`, but that fragment file was deleted
(completed) before 1840958 — predates this backlog by several commits (visible
only because the conversation's initial `git status` showed it staged-deleted
from an earlier session). Flagged, NOT fixed — out of the assigned commit
range per scope discipline; leave for whichever session's range actually
contains that deletion, or a dedicated dangling-link sweep.

### Process notes
- `sync_derived_docs.py` reported the same 4 test-only-caller consumer-graph
  warnings for `lens_amplification_surrogate` as prior sessions (2026-08-06
  1749eed session, and earlier) — THIRD session seeing this exact unchanged
  warning set. Per my own prior note this is worth escalating to the
  contract owner to silence rather than re-triaging; flagging again here
  since I didn't act on my own prior advice.
- render_fragments.py's out-of-chronological-order rendering (0.2.6 above
  0.2.5, both below the file's top 0.4.1 entry) reconfirmed harmless —
  same quirk as documented in librarian_knowledge, content intact.
- No stray `.claude/tidy_advisory.json` / `foreman_lite.json` diff this time
  (unlike a previously logged occurrence) — apparently intermittent, not
  guaranteed on every render_fragments.py run.
- Caught my own typo before committing: first draft of the new changelog
  fragment had a garbled quote ("interior[`s] domain") — re-read fragment
  bodies after writing them, before running render_fragments a second time,
  rather than trusting the first draft.
